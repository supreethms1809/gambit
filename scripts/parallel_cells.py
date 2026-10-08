"""Run grid cells as separate processes, at most ``max_jobs`` at a time.

One process per cell: a crash or an out-of-memory kill ends that cell only, and
its GPU memory is returned to the driver when the process exits. Each cell's
stdout and stderr go to its own log file. Done and failed markers stay with the
launcher that owns the cell, so a parallel run resumes exactly like a serial one.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence


@dataclass
class Job:
    name: str
    command: Sequence[str]
    log: Path
    env: Mapping[str, str] = field(default_factory=dict)


# Seed memory classes from measured peaks (GB, GH200): shift cells hold whole
# environment views and peak at 25-52; contrastive ViT is IG-backed and heavy;
# contrastive ResNet probes are the light ones. Measured peaks in past
# summaries override these seeds (see mem_class).
MEM_SEED_HIGH = (
    ("shift", None, None),          # every shift cell, either backbone
    ("contrastive", None, "vit"),   # IG-backed CDEA and margin IG on ViT
)


def mem_class(game: str, dataset: str, backbone: str) -> str:
    """``'high'`` or ``'low'``, from the seed table. Launchers refine this with
    recorded peaks (see ``launch_paper_eval.mem_key``), so profiles improve
    automatically as cells complete."""
    for game_pat, _ds_pat, bb_pat in MEM_SEED_HIGH:
        if (game_pat is None or game == game_pat) and (bb_pat is None or bb_pat in backbone):
            return "high"
    return "low"


def order_balanced(jobs: Sequence[Job], key) -> list[Job]:
    """Alternate low- and high-memory jobs, starting low, so the concurrent mix
    stays balanced instead of running every hog at once. Stable within a class."""
    lows = [j for j in jobs if key(j) == "low"]
    highs = [j for j in jobs if key(j) != "low"]
    out, turn = [], 0
    while lows or highs:
        if turn % 2 == 0 and lows:
            out.append(lows.pop(0))
        elif highs:
            out.append(highs.pop(0))
        elif lows:
            out.append(lows.pop(0))
        turn += 1
    return out


def gpu_free_gb() -> float:
    """Free device memory in GiB. Infinity when there is no NVIDIA card to ask
    (then any gate stands open)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return float("inf")
    if out.returncode != 0:
        return float("inf")
    try:
        return min(float(line.strip()) for line in out.stdout.splitlines() if line.strip()) / 1024.0
    except ValueError:
        return float("inf")


def _wait_for_memory(min_free_gb: float, poll_seconds: float, timeout_seconds: float,
                     free_fn=gpu_free_gb) -> None:
    """Hold a job start until ``min_free_gb`` is free. Warns and proceeds on
    timeout: the gate staggers starts, it never deadlocks the queue."""
    if min_free_gb <= 0:
        return
    waited = 0.0
    while free_fn() < min_free_gb:
        if waited >= timeout_seconds:
            print(f"memory gate waited {timeout_seconds:.0f}s for {min_free_gb:.0f} GiB free;"
                  " proceeding anyway", flush=True)
            return
        if waited % 60.0 < poll_seconds:
            print(f"memory gate: waiting for {min_free_gb:.0f} GiB free", flush=True)
        time.sleep(poll_seconds)
        waited += poll_seconds


def run_parallel(jobs: Sequence[Job], max_jobs: int, *, poll_seconds: float = 2.0,
                 cwd: Optional[Path] = None, min_free_gb: float = 0.0,
                 mem_timeout_s: float = 1800.0) -> dict[str, int]:
    """Start ``jobs`` in order, keeping at most ``max_jobs`` alive. Returns exit codes by name.

    With ``min_free_gb`` set, each start waits until that much device memory is
    free, so a new hog never launches on top of a full card.
    """
    if max_jobs < 1:
        raise ValueError("max_jobs must be at least 1")
    waiting = list(jobs)
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    codes: dict[str, int] = {}
    try:
        while waiting or running:
            while waiting and len(running) < max_jobs:
                _wait_for_memory(min_free_gb, poll_seconds, mem_timeout_s)
                job = waiting.pop(0)
                Path(job.log).parent.mkdir(parents=True, exist_ok=True)
                handle = open(job.log, "w", encoding="utf-8")
                env = {**os.environ, **job.env}
                proc = subprocess.Popen(list(job.command), stdout=handle, stderr=subprocess.STDOUT,
                                        env=env, cwd=str(cwd) if cwd else None)
                running[job.name] = (proc, handle)
                print(f"start {job.name} (pid {proc.pid}, log {job.log})", flush=True)
            for name in list(running):
                proc, handle = running[name]
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                del running[name]
                codes[name] = code
                print(f"{'done ' if code == 0 else 'FAIL '} {name} (exit {code}); "
                      f"{len(codes)} finished, {len(running)} running, {len(waiting)} waiting", flush=True)
            if running:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        for proc, _ in running.values():
            proc.terminate()
        raise
    return codes


def python_command(script: Path, *args: str) -> list[str]:
    return [sys.executable, "-u", str(script), *args]


def thread_env(max_jobs: int) -> dict[str, str]:
    """CPU thread caps that split the cores between jobs. A cap already set in the environment wins."""
    share = str(max(1, (os.cpu_count() or 8) // max(1, max_jobs)))
    return {k: share for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS") if k not in os.environ}
