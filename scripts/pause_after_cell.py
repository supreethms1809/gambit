"""Stop the paper trainer after cell 1, run the input-convention probe, then resume.

Cell 1 is the CIFAR-10 ResNet-50 linear probe, seed 0. The process is left
alone until ``[1/110] done`` is in the driver log. A later cell is not
allowed to start a checkpoint.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "results" / "paper" / "logs" / "train" / "driver.log"
PROBE_JSON = REPO / "results" / "paper" / "logs" / "probe_input" / "cifar10_resnet50_seed0.json"
DONE = REPO / "results" / "paper" / "logs" / "train" / "cifar10_resnet50_lp_lr0.001_seed0.done"
CKPT = REPO / "results" / "paper" / "checkpoints" / "cifar10_resnet50_pt_lp_ep15_lr0.001_seed0.pt"
PYTHON = os.environ.get("GAMBIT_PYTHON", sys.executable)
MARKER = "[1/110] done"


def _wait() -> None:
    while True:
        text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.is_file() else ""
        if MARKER in text and DONE.is_file() and CKPT.is_file():
            return
        time.sleep(20)


def _stop_trainer() -> None:
    listed = subprocess.run(
        ["pgrep", "-fl", "scripts/launch_paper_training.py"],
        check=False, capture_output=True, text=True,
    )
    print(listed.stdout or "no trainer", flush=True)
    for line in (listed.stdout or "").splitlines():
        pid = int(line.split(None, 1)[0])
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"stopped {pid}", flush=True)
        except ProcessLookupError:
            pass
    time.sleep(2)
    listed = subprocess.run(
        ["pgrep", "-fl", "scripts/launch_paper_training.py"],
        check=False, capture_output=True, text=True,
    )
    if listed.stdout.strip():
        raise SystemExit(f"trainer still running:\n{listed.stdout}")
    print("queue paused", flush=True)


def _archive_raw_cell_if_imagenet(winner: str) -> None:
    if winner != "imagenet":
        return
    dest = REPO / "results" / "paper" / "checkpoints" / "stale_raw_convention"
    dest.mkdir(parents=True, exist_ok=True)
    log_dest = REPO / "results" / "paper" / "logs" / "train" / "stale_raw_convention"
    log_dest.mkdir(parents=True, exist_ok=True)
    if CKPT.is_file():
        CKPT.rename(dest / CKPT.name)
    if DONE.is_file():
        DONE.rename(log_dest / DONE.name)
    print("archived raw seed-0 cell for an imagenet retrace", flush=True)


def _resume() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["MPLCONFIGDIR"] = "/tmp/mpl-gambit"
    env["TQDM_DISABLE"] = "1"
    env["LOKY_MAX_CPU_COUNT"] = "8"
    log = open(LOG, "a", encoding="utf-8")
    log.write("\nqueue resumed after input-convention probe\n")
    log.flush()
    subprocess.Popen(
        ["caffeinate", "-i", PYTHON, "scripts/launch_paper_training.py"],
        cwd=REPO,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print("queue resumed", flush=True)


def main() -> None:
    _wait()
    _stop_trainer()
    probe = subprocess.run(
        [PYTHON, "scripts/probe_input_convention.py"],
        cwd=REPO,
        check=False,
    )
    if probe.returncode != 0 or not PROBE_JSON.is_file():
        raise SystemExit("probe failed; queue stays paused")
    winner = json.loads(PROBE_JSON.read_text(encoding="utf-8"))["winner"]
    _archive_raw_cell_if_imagenet(winner)
    _resume()


if __name__ == "__main__":
    main()
