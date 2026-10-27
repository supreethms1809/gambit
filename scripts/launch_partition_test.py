"""The one test read of the decomposition paper (EVAL_PLAN 13.9, frozen 2026-10-27, tag eval-plan-frozen).

In priority order, so the claim cells finish first:

  planted   planted_pair and planted_pair_adjacent, ResNet-50, seeds 0-4, n = 200: core rows,
            occlusion_onesided, the selected cdea@lr0.05_t50, DA3, DA8, DA10-DA16 (as val) -> cells
  vit       ViT-B/16 planted_pair and planted_pair_adjacent, seeds 0-4, n = 200, at 14x14 (-> cells) and
            at 7x7 (-> csci/vit_grid7): core rows, occlusion_onesided, cdea@lr0.05_t50, DA16
  natural   the 8 natural datasets, both backbones, seeds 0-4, n = 64: core rows, occlusion_onesided,
            cdea@lr0.05_t50 -> cells; ViT-B/16 also at 7x7 with DA16 -> csci/vit_grid7

Every cell runs with --split test --final --config-hash <CONFIG_HASH>. CONFIG_HASH is
core.reporting.config_hash of the method and knobs hashes and every cell's arguments; the frozen
EVAL_PLAN.md lists it, and evaluation.splits refuses the test split otherwise.
A cell whose summary is at the tree's hash with no method errors is skipped, so the script resumes.

    PYTHONPATH=. python scripts/launch_partition_test.py [--dry-run | --config-hash]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.parallel_cells import Job, run_parallel, thread_env  # noqa: E402

PAPER = REPO / "scripts" / "paper_run.py"
CELLS = REPO / "results" / "paper" / "cells"
GRID7 = REPO / "results" / "paper" / "csci" / "vit_grid7"
LOG = REPO / "results" / "paper" / "logs" / "test"
SEEDS = (0, 1, 2, 3, 4)
LARGE, SMALL = 3, 3
SELECTED = "cdea@lr0.05_t50"
CORE = "cdea,subtract,ig_minsplit,occlusion_minsplit,occlusion_partition,margin_gradcam_ref,random_partition"
ROWS = CORE + ",occlusion_onesided"
RESNET_ABLATIONS = ("DA3_gradient_partition,DA8_no_null,DA10_fixed_budgets,DA11_no_polish,DA12_polish_only,"
                    "DA13_road_polish,DA14_set_cross,DA15_cell_abs,DA16_no_shared_rule")
NATURAL = ("cifar10", "cifar100", "oxford_pets", "stanford_dogs", "cub200", "ham10000", "brain_tumor", "imagenet")
PLANTED = ("planted_pair", "planted_pair_adjacent")
PLANTED_N, NATURAL_N = 200, 64


def args_of(dataset, backbone, seed, n, ablations="none", grid=None, batch=4):
    a = ["--game", "partition", "--dataset", dataset, "--backbone", backbone, "--seed", str(seed), "--split", "test",
         "--n", str(n), "--methods", ROWS, "--ablations", ablations, "--candidates", SELECTED,
         "--operators", "road,blur", "--image-batch", str(batch)]
    return a + (["--grid", str(grid)] if grid is not None else [])


def plan():
    rows = []

    def add(name, out, dataset, backbone, seed, n, **kw):
        rows.append({"name": name, "out": out, "dataset": dataset, "backbone": backbone, "seed": seed,
                     "args": args_of(dataset, backbone, seed, n, **kw),
                     "size": "large" if "vit" in backbone else "small"})

    for seed in SEEDS:
        for dataset in PLANTED:
            add(f"planted__{dataset}_resnet50_seed{seed}", CELLS, dataset, "resnet50", seed, PLANTED_N,
                ablations=RESNET_ABLATIONS, batch=8)
    for seed in SEEDS:
        for dataset in PLANTED:
            add(f"vit__{dataset}_seed{seed}", CELLS, dataset, "vit_b_16", seed, PLANTED_N,
                ablations="DA16_no_shared_rule")
            add(f"grid7__{dataset}_seed{seed}", GRID7, dataset, "vit_b_16", seed, PLANTED_N,
                ablations="DA16_no_shared_rule", grid=7)
    for seed in SEEDS:
        for dataset in NATURAL:
            add(f"natural__{dataset}_resnet50_seed{seed}", CELLS, dataset, "resnet50", seed, NATURAL_N, batch=8)
            add(f"natural__{dataset}_vit_b_16_seed{seed}", CELLS, dataset, "vit_b_16", seed, NATURAL_N)
    for seed in SEEDS:
        for dataset in NATURAL:
            add(f"grid7__{dataset}_vit_b_16_seed{seed}", GRID7, dataset, "vit_b_16", seed, NATURAL_N,
                ablations="DA16_no_shared_rule", grid=7)
    return rows


def frozen_config_hash() -> str:
    from core.reporting import config_hash
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_methods import Knobs

    return config_hash({
        "plan": "EVAL_PLAN.md 13.9, test read",
        "method_code_hash": method_code_hash(),
        "knobs_hash": knobs_hash(Knobs()),
        "cells": [{"out": str(r["out"].relative_to(REPO)), "args": r["args"]} for r in plan()],
    })


def summary_of(row) -> Path:
    return Path(row["out"]) / "test" / "partition" / row["dataset"] / row["backbone"] / f"seed{row['seed']}" / "summary.json"


def done(row) -> bool:
    from evaluation.provenance import method_code_hash
    from scripts.launch_runs import method_errors

    summary = summary_of(row)
    if not summary.is_file() or not (summary.parent / "records.csv.gz").is_file():
        return False
    return json.loads(summary.read_text())["method_code_hash"] == method_code_hash() and not method_errors(summary)


def main() -> None:
    code = frozen_config_hash()
    if "--config-hash" in sys.argv:
        print(code)
        return
    planned = plan()
    pending = [row for row in planned if not done(row)]
    if "--dry-run" in sys.argv:
        for row in pending:
            print(f"{row['size']:5} {row['name']}")
        print(f"{len(planned)} cells, {len(pending)} pending, config hash {code}")
        return
    LOG.mkdir(parents=True, exist_ok=True)
    kind = {row["name"]: row["size"] for row in pending}
    queue = [Job(row["name"], [sys.executable, "-u", str(PAPER), *row["args"], "--out", str(row["out"]), "--final",
                               "--config-hash", code],
                 LOG / f"{row['name']}.log", thread_env(LARGE + SMALL)) for row in pending]
    print(f"{len(planned)} cells, {len(pending)} pending, config hash {code}", flush=True)

    def admit(running, job) -> bool:
        large = sum(1 for name in running if kind[name] == "large")
        if kind[job.name] == "large":
            return large < LARGE
        return len(running) - large < SMALL

    codes = run_parallel(queue, LARGE + SMALL, cwd=REPO, min_free_gb=12.0, stop_on_failure=False, admit=admit)
    failed = sorted(name for name, c in codes.items() if c != 0)
    print(f"finished {len(codes)} started, {len(failed)} failed", flush=True)
    for name in failed:
        print(f"  failed: {name} (see {LOG / (name + '.log')})", flush=True)


if __name__ == "__main__":
    main()
