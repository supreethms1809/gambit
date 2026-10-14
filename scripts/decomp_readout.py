"""Decomposition read-out from cell records (FORMULATION_DECOMPOSITION.md section 5).

For each cell, area and operator: every method's mean scores, and the paired
difference of CDEA (or another label) against the best min-split row and
against occlusion_assign, with 95% bootstrap intervals over images. On planted
pairs it adds the patch masses: S on C, U_k on A_k, U_l on A_l, over all kept
images and over images whose three roles were confirmed by intervention.

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
PATCH_MASS = (("prec_s_c", "S on C"), ("prec_uk_ak", "U_k on A_k"), ("prec_ul_al", "U_l on A_l"))


def _value(row: dict, key: str) -> float:
    if key == "pur":
        return float(row["pur_k"]) + float(row["pur_l"])
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else float("nan")


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


def _paired(a: dict, b: dict, key: str) -> np.ndarray:
    common = sorted(set(a) & set(b))
    diff = np.array([_value(a[i], key) - _value(b[i], key) for i in common], dtype=float)
    return diff[np.isfinite(diff)]


def _fmt(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    low, high = bootstrap_mean_ci(values, n_boot=4000, seed=0)
    return f"{values.mean():+.2f} [{low:+.2f}, {high:+.2f}] n={values.size}"


def report(cells: dict, cdea: str) -> None:
    for (dataset, backbone, seed, area, operator), methods in sorted(cells.items()):
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
            print(f"  {cdea} - {best} (best min-split on PUR_k + PUR_l):")
            for score in ("pur", "sv"):
                print(f"    {score:4s} {_fmt(_paired(methods[cdea], methods[best], score))}")
        if "occlusion_assign" in methods:
            print(f"  {cdea} - occlusion_assign:")
            for score in ("pur", "sv"):
                print(f"    {score:4s} {_fmt(_paired(methods[cdea], methods['occlusion_assign'], score))}")
        first = next(iter(methods[cdea].values()))
        if "prec_s_c" in first:
            print("  planted patch mass (all kept images | images with all three roles confirmed):")
            for method, rows in sorted(methods.items()):
                parts = []
                for key, label in PATCH_MASS:
                    every = np.array([_value(r, key) for r in rows.values()], dtype=float)
                    confirmed = np.array([_value(r, key) for r in rows.values()
                                          if all(r.get(f"role_{t}") == "1" for t in ("ak", "al", "c"))], dtype=float)
                    every, confirmed = every[np.isfinite(every)], confirmed[np.isfinite(confirmed)]
                    tail = f"{confirmed.mean():.2f}" if confirmed.size else "n/a"
                    head = f"{every.mean():.2f}" if every.size else "n/a"
                    parts.append(f"{label} {head} | {tail}")
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
