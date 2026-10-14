"""Decomposition read-out from cell records (FORMULATION_DECOMPOSITION.md section 5).

For each cell, area and operator: every method's mean scores, and the paired
difference of CDEA (or another label) against the best min-split row and
against occlusion_assign, with 95% bootstrap intervals over images. On planted
pairs it adds RR (EVAL_PLAN.md 13.2: each player's recall of its own patch,
averaged over U_k, U_l and S) and the patch masses, including S on A_k and S on
A_l (S spilling onto the unique patches). Both are shown over every kept image
and over images whose three roles were confirmed by intervention, which is
RR's primary set.

This prints numbers. The win conditions are in EVAL_PLAN.md section 13 and
are applied there, not here.

    PYTHONPATH=. python scripts/decomp_readout.py --records results/paper/cells --split val
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.stats import bootstrap_mean_ci  # noqa: E402

SCORES = ("pur", "sv", "cd", "bal", "sep")
MINSPLIT = ("gradcam_minsplit", "ig_minsplit", "occlusion_minsplit", "rise_minsplit", "extremal_minsplit")
PLANTED = "planted_pair"
OWN_RECALL = ("rec_uk_ak", "rec_ul_al", "rec_s_c")
ROLES = ("ak", "al", "c")
PATCH_MASS = (("prec_s_c", "S on C"), ("prec_uk_ak", "U_k on A_k"), ("prec_ul_al", "U_l on A_l"),
              ("prec_s_ak", "S on A_k"), ("prec_s_al", "S on A_l"))


def _value(row: dict, key: str) -> float:
    if key == "pur":
        return float(row["pur_k"]) + float(row["pur_l"])
    if key == "rr":
        return float(np.mean([_value(row, name) for name in OWN_RECALL]))
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else float("nan")


def _confirmed(row: dict) -> bool:
    """All three planted roles confirmed by ``verify_roles``."""
    return all(row.get(f"role_{t}") == "1" for t in ROLES)


def load(root: Path, split: str, expect_hash: str | None) -> dict:
    cells: dict = defaultdict(lambda: defaultdict(dict))
    for path in sorted(root.glob(f"{split}/decomposition/*/*/seed*/records.csv.gz")):
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if expect_hash and not row["method_code_hash"].startswith(expect_hash):
                    continue
                key = (row["dataset"], row["backbone"], int(row["seed"]), float(row["area"]), row["operator"])
                cells[key][row["method"]][row["image_index"]] = row
    return cells


def _paired(a: dict, b: dict, key: str, confirmed_only: bool = False) -> np.ndarray:
    common = sorted(set(a) & set(b))
    if confirmed_only:
        common = [i for i in common if _confirmed(a[i])]
    diff = np.array([_value(a[i], key) - _value(b[i], key) for i in common], dtype=float)
    return diff[np.isfinite(diff)]


def _mean(rows, key: str) -> str:
    values = np.array([_value(r, key) for r in rows], dtype=float)
    values = values[np.isfinite(values)]
    return f"{values.mean():.2f}" if values.size else "n/a"


def _fmt(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    low, high = bootstrap_mean_ci(values, n_boot=4000, seed=0)
    return f"{values.mean():+.2f} [{low:+.2f}, {high:+.2f}] n={values.size}"


def _compare(methods: dict, cdea: str, other: str, title: str, planted: bool) -> None:
    print(f"  {cdea} - {other} ({title}):")
    if planted:
        print(f"    rr   {_fmt(_paired(methods[cdea], methods[other], 'rr', confirmed_only=True))}  roles confirmed")
        print(f"    rr   {_fmt(_paired(methods[cdea], methods[other], 'rr'))}  every image")
    for score in ("pur", "sv"):
        print(f"    {score:4s} {_fmt(_paired(methods[cdea], methods[other], score))}")


def report(cells: dict, cdea: str) -> None:
    for (dataset, backbone, seed, area, operator), methods in sorted(cells.items()):
        planted = dataset == PLANTED
        print(f"\n{dataset} {backbone} seed{seed}  area {area:g}  {operator}")
        print(f"  {'method':34s}" + "".join(f"{s:>8s}" for s in SCORES) + "     n")
        for method, rows in sorted(methods.items()):
            means = []
            for score in SCORES:
                values = np.array([_value(r, score) for r in rows.values()], dtype=float)
                values = values[np.isfinite(values)]
                means.append(f"{values.mean():8.2f}" if values.size else "     n/a")
            print(f"  {method:34s}" + "".join(means) + f"  {len(rows):4d}")
        if cdea not in methods:
            continue
        present = [m for m in MINSPLIT if m in methods]
        if present:
            best = max(present, key=lambda m: np.nanmean([_value(r, "pur") for r in methods[m].values()]))
            _compare(methods, cdea, best, "best min-split on PUR_k + PUR_l", planted)
        if "occlusion_assign" in methods:
            _compare(methods, cdea, "occlusion_assign", "additive per-cell assignment", planted)
        if planted:
            print("  RR and patch mass (every kept image | images with all three roles confirmed):")
            for method, rows in sorted(methods.items()):
                every = list(rows.values())
                confirmed = [r for r in every if _confirmed(r)]
                parts = [f"{label} {_mean(every, key)} | {_mean(confirmed, key)}"
                         for key, label in (("rr", "RR"),) + PATCH_MASS]
                print(f"    {method:32s} " + "   ".join(parts))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", default=str(REPO / "results" / "paper" / "cells"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--cdea", default="cdea_decomp")
    parser.add_argument("--expect-hash", default=None)
    args = parser.parse_args(argv)
    cells = load(Path(args.records), args.split, args.expect_hash)
    if not cells:
        raise SystemExit("no decomposition records found")
    report(cells, args.cdea)


if __name__ == "__main__":
    main()
