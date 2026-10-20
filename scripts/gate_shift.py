"""G-shift pilot cells and shift read-out (SHIFT_EVAL_PLAN.md).

Cells: Waterbirds and planted-patch val, ResNet-50, seeds 0 and 1, n = 64.
Default configuration. Does not launch the pilot unless ``--run`` is passed.
When ``--records`` is passed, reads ``shift.csv.gz`` records and prints every
method's mean shift scores (ΔD, RN, PUR_R, PUR_S, RoleSep, TC, SPC) and paired
bootstrap differences of CDEA-shift against gap attribution, attribution
difference, and shift_assign. Fast knobs cannot satisfy the gate: fast-knob
rows are dropped on load.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.stats import bootstrap_mean_ci  # noqa: E402

DATASETS = ("waterbirds", "planted_patch")
SEEDS = (0, 1)
SCORES = ("delta_d", "rn", "pur_r", "pur_s", "role_sep", "tc", "spc")
COMPARATORS = (
    ("gap_attribution", "path-integrated first-order shortcut"),
    ("attribution_difference", "per-environment attribution difference"),
    ("shift_assign", "additive per-cell LP assignment"),
)
READOUT = (
    "Read-out: ΔD@a of CDEA-shift against gap attribution, attribution difference, and shift_assign, "
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


def _value(row: dict, key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else float("nan")


def load_shift_records(root: Path, split: str = "val", expect_hash: str | None = None) -> dict:
    loaded: dict = defaultdict(lambda: defaultdict(dict))
    for path in sorted(root.glob(f"{split}/shift/*/*/seed*/shift.csv.gz")):
        parts = path.parts
        dataset, backbone, seed_part = parts[-4], parts[-3], parts[-2]
        seed = int(seed_part.replace("seed", ""))
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("fast", "0") == "1":
                    continue
                if expect_hash and not row.get("method_code_hash", "").startswith(expect_hash):
                    continue
                key = (row.get("dataset") or dataset, row.get("backbone") or backbone,
                       int(row.get("seed", seed)), float(row["area"]))
                loaded[key][row["method"]][row["image_index"]] = row
    return loaded


def _paired(a: dict, b: dict, key: str) -> np.ndarray:
    common = sorted(set(a) & set(b))
    diff = np.array([_value(a[i], key) - _value(b[i], key) for i in common], dtype=float)
    return diff[np.isfinite(diff)]


def _fmt(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    low, high = bootstrap_mean_ci(values, n_boot=4000, seed=0)
    return f"{values.mean():+.2f} [{low:+.2f}, {high:+.2f}] n={values.size}"


def report_shift_cells(loaded: dict, cdea: str = "cdea_shift") -> None:
    for (dataset, backbone, seed, area), methods in sorted(loaded.items()):
        print(f"\n{dataset} {backbone} seed{seed}  area {area:g}")
        print(f"  {'method':34s}" + "".join(f"{s:>9s}" for s in SCORES) + "     n")
        for method, rows in sorted(methods.items()):
            means = []
            for score in SCORES:
                values = np.array([_value(r, score) for r in rows.values()], dtype=float)
                values = values[np.isfinite(values)]
                means.append(f"{values.mean():9.2f}" if values.size else "      n/a")
            print(f"  {method:34s}" + "".join(means) + f"  {len(rows):4d}")
        if cdea not in methods:
            continue
        for other, title in COMPARATORS:
            if other not in methods:
                continue
            print(f"  {cdea} - {other} ({title}):")
            for metric in ("delta_d", "rn", "pur_r", "pur_s", "spc"):
                diff = _paired(methods[cdea], methods[other], metric)
                if diff.size:
                    print(f"    {metric:8s} {_fmt(diff)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="score the four cells; default is a dry list")
    parser.add_argument("--fast", action="store_true", help="smoke knobs; cannot satisfy the gate")
    parser.add_argument("--records", default=None, help="read and report shift records from this root")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cdea", default="cdea_shift")
    parser.add_argument("--expect-hash", default=None)
    args = parser.parse_args(argv)
    print(READOUT)
    print(STOP)
    for cell in cells():
        print(
            f"cell {cell['split']} shift {cell['dataset']} {cell['backbone']} "
            f"seed {cell['seed']} n={cell['n']}"
        )
    if args.records:
        loaded = load_shift_records(Path(args.records), args.split, args.expect_hash)
        if not loaded:
            raise SystemExit("no shift records found")
        report_shift_cells(loaded, args.cdea)
        return
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
            "--methods", "cdea_shift,shift_assign,gap_attribution,attribution_difference",
        ]
        if args.fast:
            argv_cell.append("--fast")
        print(run(spec_from_args(parse(argv_cell))))


if __name__ == "__main__":
    main()
