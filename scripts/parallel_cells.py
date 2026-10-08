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


def run_parallel(jobs: Sequence[Job], max_jobs: int, *, poll_seconds: float = 2.0,
                 cwd: Optional[Path] = None) -> dict[str, int]:
    """Start ``jobs`` in order, keeping at most ``max_jobs`` alive. Returns exit codes by name."""
    if max_jobs < 1:
        raise ValueError("max_jobs must be at least 1")
    waiting = list(jobs)
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    codes: dict[str, int] = {}
    try:
        while waiting or running:
            while waiting and len(running) < max_jobs:
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
