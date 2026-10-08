"""Supervise the seed-0 val games to clean completion.

Waits while cells run, then audits every cell: a cell counts only with a done
marker, zero method errors, and finite primary scores in its records (the NaN
guard — a `.done` marker alone is not enough, since contention poisons methods
inside surviving cells). Dirty cells get their markers cleared and are
relaunched at stepped-down jobs, up to ``--rounds`` relaunch rounds. Prints
``GAMES_CLEAN`` when all 14 cells pass the audit, ``GAMES_DIRTY`` with the
remaining cells otherwise.

It never double-launches: a relaunch round starts only after 10 continuous
quiet minutes (no val game processes) with incomplete markers, so the wave
owned by another driver is left alone.

    PYTHONPATH=. python scripts/supervise_val.py --seeds 0 --jobs 3
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNS = REPO / "results" / "paper" / "runs"
MARKERS = RUNS / "_markers"

CONTRASTIVE_DEV = ("cifar10", "ham10000")
SHIFT_UNITS = ("waterbirds", "imagenet9", "stanford_dogs", "planted_patch", "colored_mnist")
BACKBONES = ("resnet50", "vit_b_16")
PRIMARY = ("cd", "cd1", "logit_delta_d", "prob_delta_d")
QUIET_SECONDS = 600


def cells(seed: int) -> list[dict]:
    out = [{"game": "contrastive", "dataset": d, "backbone": b, "seed": seed}
           for d in CONTRASTIVE_DEV for b in BACKBONES]
    return out + [{"game": "shift", "dataset": d, "backbone": b, "seed": seed}
                  for d in SHIFT_UNITS for b in BACKBONES]


def cell_id(cell: dict) -> str:
    return f"val_{cell['game']}_{cell['dataset']}_{cell['backbone']}_seed{cell['seed']}"


def audit_cell(cell: dict) -> tuple[bool, str]:
    """(clean, detail). Missing atau errored cells are not clean."""
    name = cell_id(cell)
    if not (MARKERS / "val" / f"{name}.done").is_file():
        return False, "no done marker"
    try:
        summary = json.loads((RUNS / "val" / cell["game"] / cell["dataset"]
                              / cell["backbone"] / f"seed{cell['seed']}" / "summary.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable summary: {exc}"
    errors = summary.get("errors") or []
    if errors:
        return False, f"method errors: {sorted(errors)}"
    records = RUNS / "val" / cell["game"] / cell["dataset"] / cell["backbone"] / f"seed{cell['seed']}" / "records.csv.gz"
    try:
        with gzip.open(records, "rt") as f:
            for row in csv.DictReader(f):
                for key in PRIMARY:
                    value = row.get(key, "")
                    if value in ("", "nan"):
                        continue  # absent by design (e.g. CD without a foil map)
                    if not math.isfinite(float(value)):
                        return False, f"non-finite {key} in {row.get('method')}"
    except OSError as exc:
        return False, f"unreadable records: {exc}"
    return True, "clean"


def game_processes() -> list[int]:
    """PIDs of live val-game cell processes. Reads /proc directly so no shell
    pattern can ever match the supervisor itself."""
    pids = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        text = [a.decode("utf-8", "replace") for a in argv]
        joined = " ".join(text)
        if "launch_paper_eval.py" in joined and "--split val" in joined and "--seeds 0" in joined:
            if "supervise_val" not in joined:
                pids.append(int(pid))
    return pids


def clear_markers(cell: dict) -> None:
    name = cell_id(cell)
    for suffix in (".done", ".failed"):
        try:
            (MARKERS / "val" / f"{name}{suffix}").unlink()
        except OSError:
            pass


def relaunch(seed: int, jobs: int) -> tuple[int, int]:
    """Rerun pending val cells (done markers skip the clean ones), shift then
    contrastive dev. Returns the two launcher exit codes."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO))
    base = [sys.executable, "-u", "scripts/launch_paper_eval.py", "--split", "val",
            "--seeds", str(seed), "--jobs", str(jobs)]
    shift = base + ["--game", "shift", "--n-shift", "resnet50:64,vit_b_16:32"]
    contrast = base + ["--datasets", "cifar10,ham10000",
                       "--n-contrastive", "resnet50:64,vit_b_16:64"]
    gs = subprocess.run(shift, cwd=REPO, env=env).returncode
    gc = subprocess.run(contrast, cwd=REPO, env=env).returncode
    return gs, gc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="0")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2, help="relaunch rounds after the audit")
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--timeout-h", type=float, default=24)
    args = p.parse_args(argv)
    seed = int(str(args.seeds).split(",")[0])
    wanted = cells(seed)
    jobs, quiet_since, rounds_used = args.jobs, None, 0
    started = time.time()
    while True:
        if time.time() - started > args.timeout_h * 3600:
            print(f"GAMES_TIMEOUT after {args.timeout_h}h")
            return 2
        dirty = {}
        for c in wanted:
            clean, detail = audit_cell(c)
            if not clean:
                dirty[cell_id(c)] = detail
        if not dirty:
            print(f"GAMES_CLEAN ({len(wanted)}/{len(wanted)} cells: done, no method errors, finite scores)")
            return 0
        if game_processes():
            quiet_since = None
            print(f"{len(dirty)} dirty cells, games running; waiting", flush=True)
            time.sleep(args.poll)
            continue
        now = time.time()
        quiet_since = quiet_since or now
        if now - quiet_since < QUIET_SECONDS:
            time.sleep(args.poll)
            continue
        # Quiet with incomplete markers: a wave owned by another driver, or nothing running.
        if rounds_used >= args.rounds:
            print(f"GAMES_DIRTY after {rounds_used} relaunch rounds:")
            for name, detail in sorted(dirty.items()):
                print(f"  {name}: {detail}")
            return 1
        print(f"relaunching {len(dirty)} dirty cells at --jobs {jobs}", flush=True)
        for name in dirty:
            print(f"  {name}: {dirty[name]}", flush=True)
        for c in wanted:
            if cell_id(c) in dirty:
                clear_markers(c)
        gs, gc = relaunch(seed, jobs)
        print(f"relaunch done: shift exit {gs}, contrastive exit {gc}", flush=True)
        jobs = max(1, jobs - 2)
        rounds_used += 1
        quiet_since = None


if __name__ == "__main__":
    raise SystemExit(main())
