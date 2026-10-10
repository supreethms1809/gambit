"""G-shift pilot cells. Prints the read-out and the stop rule.

Cells: Waterbirds and planted-patch val, ResNet-50, seeds 0 and 1, n = 64.
Default configuration. Does not launch the pilot unless ``--run`` is passed.
Fast knobs cannot satisfy the gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASETS = ("waterbirds", "planted_patch")
SEEDS = (0, 1)
READOUT = (
    "Read-out: ΔD@a of CDEA-shift against gap attribution and against attribution difference, "
    "on Waterbirds and planted-patch val."
)
STOP = (
    "Stop rule: continue as a method paper only if CDEA-shift is ahead of both "
    "gap attribution and attribution difference on both datasets. Otherwise take "
    "the evaluation-paper route and do not revise the formulation."
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
    print(READOUT)
    print(STOP)
    for cell in cells():
        print(
            f"cell {cell['split']} shift {cell['dataset']} {cell['backbone']} "
            f"seed {cell['seed']} n={cell['n']}"
        )
    if not args.run:
        return
    from scripts.paper_run import parse, run, spec_from_args

    for cell in cells():
        argv_cell = [
            "--game", "shift",
            "--dataset", cell["dataset"],
            "--backbone", cell["backbone"],
            "--seed", str(cell["seed"]),
            "--split", cell["split"],
            "--n", str(cell["n"]),
            "--methods", "cdea_shift,gap_attribution,attribution_difference",
        ]
        if args.fast:
            argv_cell.append("--fast")
        print(run(spec_from_args(parse(argv_cell))))


if __name__ == "__main__":
    main()
