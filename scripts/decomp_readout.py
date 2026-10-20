"""Decomposition read-out from cell records (FORMULATION_DECOMPOSITION.md section 5).

For each cell, area and operator: every method's mean scores, and the paired
difference of CDEA (or another label) against the best min-split row and
against occlusion_assign, with 95% bootstrap intervals over images. On planted
pairs it adds RR (EVAL_PLAN.md 13.2: each player's recall of its own patch,
averaged over U_k, U_l and S, or over U_k and U_l for a method with no S map)
and the patch masses, including S on A_k and S on
A_l (S spilling onto the unique patches). Both are shown over every kept image
and over images whose three roles were confirmed by intervention, which is
RR's primary set.

Both choices of EVAL_PLAN.md 13.3 read val records only, whatever ``--split``
is, and fast-knob rows inform neither:
- the best min-split row, per dataset and backbone, by mean PUR + SV at 5%
  under ROAD, and on planted pairs by RR at 2.5% on role-confirmed images;
- CDEA's setting, over the ``cdea_decomp@...`` candidates, by mean PUR + SV at
  5% under ROAD averaged over CIFAR-10 and HAM10000 (ResNet-50, seed 0).

It ends with family D-T (EVAL_PLAN.md 13.4) on planted pairs on ResNet-50: CDEA
against the best min-split row and against occlusion_assign, on RR at 2.5% over
role-confirmed images, from the reported split. Each seed's per-image
differences stay apart, and ``analysis.stats.seed_pooled_summary`` gives the mean
over seeds of each seed's mean, the two-level 95% interval and the sign-flip p.
Fast-knob rows are left out.

Each method row also gives the share of failed rows and of constant maps.
Both are scored as the seeded random floor.

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

from analysis.stats import bootstrap_mean_ci, seed_pooled_summary  # noqa: E402

SCORES = ("pur", "sv", "cd", "bal", "sep")
MINSPLIT = ("gradcam_minsplit", "ig_minsplit", "occlusion_minsplit", "rise_minsplit", "extremal_minsplit")
PLANTED = "planted_pair"
SELECT_OPERATOR = "road"
SELECT_AREA = 0.05       # natural images: PUR + SV
RR_AREA = 0.025          # planted pairs: RR, role-confirmed images
DEV = ("cifar10", "ham10000")
DEV_BACKBONE = "resnet50"
DEV_SEED = 0
DT_BACKBONE = "resnet50"  # family D-T; ViT-B/16 is the robustness row
OWN_RECALL = ("rec_uk_ak", "rec_ul_al", "rec_s_c")
ROLES = ("ak", "al", "c")
PATCH_MASS = (("prec_s_c", "S on C"), ("prec_uk_ak", "U_k on A_k"), ("prec_ul_al", "U_l on A_l"),
              ("prec_s_ak", "S on A_k"), ("prec_s_al", "S on A_l"))


def _value(row: dict, key: str) -> float:
    if key == "pur":
        return float(row["pur_k"]) + float(row["pur_l"])
    if key == "pur+sv":
        return _value(row, "pur") + _value(row, "sv")
    if key == "rr":
        # Over the players the method has: margin_gradcam_ref has no S, so its RR is over U_k and U_l.
        recalls = [v for v in (_value(row, name) for name in OWN_RECALL) if math.isfinite(v)]
        return float(np.mean(recalls)) if recalls else float("nan")
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


def _share(rows, key: str) -> str:
    """Share of rows with ``key`` = 1, e.g. failed or constant maps scored as the random floor."""
    flags = [r.get(key) == "1" for r in rows if r.get(key) not in ("", None)]
    return f"{np.mean(flags):7.2f}" if flags else "    n/a"


def _non_fast(rows: dict) -> dict:
    return {i: r for i, r in rows.items() if r.get("fast", "0") != "1"}


def _selection_values(rows: dict, key: str, confirmed_only: bool = False) -> np.ndarray:
    """Finite ``key`` values from rows that may inform a choice: fast-knob rows never do."""
    kept = [r for r in _non_fast(rows).values() if not confirmed_only or _confirmed(r)]
    values = np.array([_value(r, key) for r in kept], dtype=float)
    return values[np.isfinite(values)]


def select_minsplit(val_cells: dict) -> dict:
    """``{(dataset, backbone): (best, {method: score})}`` from val records (EVAL_PLAN.md 13.3).

    Natural images: mean PUR + SV at 5% under ROAD. Planted pairs: RR at 2.5% on
    role-confirmed images. The score is the mean over seeds of each seed's mean,
    and only methods present in every seed compete.
    """
    per: dict = defaultdict(lambda: defaultdict(dict))
    seeds: dict = defaultdict(set)
    for (dataset, backbone, seed, area, operator), methods in val_cells.items():
        planted = dataset == PLANTED
        if operator != SELECT_OPERATOR or not math.isclose(area, RR_AREA if planted else SELECT_AREA):
            continue
        seeds[(dataset, backbone)].add(seed)
        for method in MINSPLIT:
            values = _selection_values(methods.get(method, {}), "rr" if planted else "pur+sv", planted)
            if values.size:
                per[(dataset, backbone)][method][seed] = float(values.mean())
    selection = {}
    for key, scores in per.items():
        full = {m: float(np.mean(list(s.values()))) for m, s in scores.items() if set(s) == seeds[key]}
        if full:
            selection[key] = (max(full, key=full.get), full)
    return selection


def print_selection(selection: dict) -> None:
    print("Best min-split row, chosen on val records only (EVAL_PLAN.md 13.3):")
    if not selection:
        print("  none: no non-fast val records at the selection area under ROAD")
    for (dataset, backbone), (best, scores) in sorted(selection.items()):
        rule = f"RR@{RR_AREA:g}, roles confirmed" if dataset == PLANTED else f"PUR + SV@{SELECT_AREA:g}, ROAD"
        ranked = "  ".join(f"{m} {s:.3f}" for m, s in sorted(scores.items(), key=lambda kv: -kv[1]))
        print(f"  {dataset} {backbone} by {rule}: {best}    ({ranked})")


def rank_candidates(val_cells: dict) -> list[tuple[str, float | None, dict[str, float]]]:
    """``[(label, score, {dataset: mean})]``, best first; ``score`` is None when a dev dataset is missing.

    The score is the mean over the dev datasets of each dataset's mean PUR + SV,
    so each dataset counts once whatever its n.
    """
    from evaluation.run_methods import DECOMP_CANDIDATES

    cells = {(d, b, s, o): methods for (d, b, s, a, o), methods in val_cells.items() if math.isclose(a, SELECT_AREA)}
    ranked = []
    for label in DECOMP_CANDIDATES:
        per = {}
        for dataset in DEV:
            rows = cells.get((dataset, DEV_BACKBONE, DEV_SEED, SELECT_OPERATOR), {}).get(label, {})
            values = _selection_values(rows, "pur+sv")
            if values.size:
                per[dataset] = float(values.mean())
        score = float(np.mean([per[d] for d in DEV])) if len(per) == len(DEV) else None
        ranked.append((label, score, per))
    return sorted(ranked, key=lambda item: (item[1] is None, -(item[1] or 0.0)))


def print_candidates(ranked: list) -> None:
    print(f"CDEA setting: mean PUR + SV@{SELECT_AREA:g}, ROAD, over {' and '.join(DEV)} "
          f"({DEV_BACKBONE}, seed {DEV_SEED}, val):")
    for label, score, per in ranked:
        detail = "  ".join(f"{d} {per[d]:.3f}" for d in DEV if d in per)
        if score is None:
            missing = ", ".join(d for d in DEV if d not in per)
            print(f"  {label:28s}  not ranked, missing {missing}    {detail}")
        else:
            print(f"  {label:28s} {score:7.3f}    {detail}")
    if ranked and ranked[0][1] is not None:
        print(f"  chosen: {ranked[0][0]}")


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


def report(cells: dict, cdea: str, selection: dict) -> None:
    for (dataset, backbone, seed, area, operator), methods in sorted(cells.items()):
        planted = dataset == PLANTED
        print(f"\n{dataset} {backbone} seed{seed}  area {area:g}  {operator}")
        print(f"  {'method':34s}" + "".join(f"{s:>8s}" for s in SCORES) + "     n   failed  constant")
        for method, rows in sorted(methods.items()):
            means = []
            for score in SCORES:
                values = np.array([_value(r, score) for r in rows.values()], dtype=float)
                values = values[np.isfinite(values)]
                means.append(f"{values.mean():8.2f}" if values.size else "     n/a")
            print(f"  {method:34s}" + "".join(means) + f"  {len(rows):4d}"
                  f"  {_share(rows.values(), 'failed_row')}  {_share(rows.values(), 'constant_row')}")
        if cdea not in methods:
            continue
        best = selection.get((dataset, backbone), (None, {}))[0]
        if best is None:
            print("  best min-split: none chosen on val for this dataset and backbone, so not compared")
        elif best not in methods:
            print(f"  best min-split ({best}, chosen on val): not in this cell")
        else:
            _compare(methods, cdea, best, "best min-split, chosen on val", planted)
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


def family_dt_differences(cells: dict, cdea: str, other: str) -> tuple[dict[int, np.ndarray], list[int]]:
    """``({seed: RR differences cdea - other}, seeds with none)`` for family D-T (EVAL_PLAN.md 13.4).

    Planted pairs on ResNet-50 at RR@2.5% under ROAD, role-confirmed images
    paired by index within each seed, fast-knob rows left out.
    """
    per_seed, empty = {}, []
    for (dataset, backbone, seed, area, operator), methods in sorted(cells.items()):
        if (dataset, backbone, operator) != (PLANTED, DT_BACKBONE, SELECT_OPERATOR) or not math.isclose(area, RR_AREA):
            continue
        diff = _paired(_non_fast(methods.get(cdea, {})), _non_fast(methods.get(other, {})), "rr", confirmed_only=True)
        if diff.size:
            per_seed[seed] = diff
        else:
            empty.append(seed)
    return per_seed, empty


def print_family_dt(cells: dict, cdea: str, selection: dict) -> None:
    print(f"\nFamily D-T: {PLANTED} {DT_BACKBONE}, RR@{RR_AREA:g} on role-confirmed images, "
          "seeds pooled (EVAL_PLAN.md 13.4):")
    if not any(d == PLANTED and b == DT_BACKBONE and o == SELECT_OPERATOR and math.isclose(a, RR_AREA)
               for d, b, _, a, o in cells):
        print(f"  no {PLANTED} {DT_BACKBONE} cells at area {RR_AREA:g} under ROAD")
        return
    comparisons = [("occlusion_assign", "additive per-cell assignment")]
    best = selection.get((PLANTED, DT_BACKBONE), (None, {}))[0]
    if best is None:
        print("  best min-split: none chosen on val, so not compared")
    else:
        comparisons.insert(0, (best, "best min-split, chosen on val"))
    for other, title in comparisons:
        per_seed, empty = family_dt_differences(cells, cdea, other)
        print(f"  {cdea} - {other} ({title}):")
        if empty:
            print(f"    no paired role-confirmed non-fast images in seed {', '.join(map(str, empty))}")
        if not per_seed:
            continue
        # Each comparison is tested on its own (13.4), so no Holm across the two.
        row = seed_pooled_summary({"rr": per_seed})[0]
        print(f"    rr   mean {row['mean']:+.3f}  95% two-level [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  "
              f"sign-flip p {row['p']:.4f}")
        print("    images " + "  ".join(f"seed{seed} {diff.size}" for seed, diff in per_seed.items()))


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
    val = cells if args.split == "val" else load(Path(args.records), "val", args.expect_hash)
    print_candidates(rank_candidates(val))
    selection = select_minsplit(val)
    print_selection(selection)
    report(cells, args.cdea, selection)
    print_family_dt(cells, args.cdea, selection)


if __name__ == "__main__":
    main()
