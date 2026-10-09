"""G1 pilot cells. Prints the stop rule. Does not launch the pilot unless ``--run`` is passed.

Cells: CIFAR-10 and HAM10000 val, ResNet-50, seeds 0 and 1, n = 64.
CDEA defaults: Grad-CAM initialisation, T = 100, learning rate 0.1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASETS = ("cifar10", "ham10000")
SEEDS = (0, 1)
STOP = (
    "Stop rule: continue as a method paper only if CD@5% is ahead of the better "
    "margin variant and of deletion Extremal Perturbations on both CIFAR-10 and "
    "HAM10000. Otherwise take the evaluation-paper route and do not revise the formulation."
)


def cells() -> list[dict]:
    return [
        {"dataset": dataset, "backbone": "resnet50", "seed": seed, "split": "val", "n": 64}
        for dataset in DATASETS
        for seed in SEEDS
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="score the four cells; default is a dry list")
    parser.add_argument("--fast", action="store_true", help="smoke knobs; cannot satisfy the gate")
    args = parser.parse_args(argv)
    print(STOP)
    for cell in cells():
        print(
            f"{cell['split']} {cell['dataset']} {cell['backbone']} seed{cell['seed']} n={cell['n']}"
        )
    if not args.run:
        print("dry run. Pass --run to score the cells into results/paper/cells/.")
        return
    from evaluation.run_cell import CellSpec, run_contrastive
    from evaluation.run_methods import CONTRASTIVE_CORE, FAST, Knobs
    from core.device import get_device

    knobs = FAST if args.fast else Knobs()
    device = get_device()
    for cell in cells():
        spec = CellSpec(
            dataset=cell["dataset"],
            backbone=cell["backbone"],
            seed=cell["seed"],
            split=cell["split"],
            n=cell["n"],
            methods=list(CONTRASTIVE_CORE),
            knobs=knobs,
            out_dir=str(REPO / "results" / "paper" / "cells"),
        )
        print(run_contrastive(spec, device))
        print(STOP)


if __name__ == "__main__":
    main()
