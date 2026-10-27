"""The declared tests of the decomposition paper on partition records (EVAL_PLAN 13.9-13.12).

  D1''a  planted_pair, ResNet-50: C's accuracy over all its cells, CDEA - subtract. Holds if the
         two-level lower 95% bound exceeds -0.02.
  D1''b  planted_pair_adjacent, ResNet-50: RA over all patch cells, CDEA - subtract (subtract is also
         the best rule row, 13.10). Holds if p < 0.05 and the mean is above 0.
  frame  planted_pair_adjacent, ResNet-50: RA over all patch cells, CDEA - occlusion_partition, delta = 0.05.
  S1     planted_pair, ResNet-50: RA over all patch cells, CDEA - occlusion_onesided. p < 0.05, mean > 0.
  S2     ViT-B/16 at 14x14, both planted conditions stacked within seed: the mean of A_k's and A_l's
         accuracy, CDEA - occlusion_onesided. p < 0.05, mean > 0.
  S3     natural, ROAD: S confirmation on images where both rows have a non-empty S, CDEA -
         occlusion_onesided. Per backbone, exact Wilcoxon over the 8 datasets; Holm over the backbones.

Planted tests read the ROAD rows of role-confirmed images. The machinery is family D-T's
(analysis.stats): the mean over seeds of each seed's mean paired difference, a sign flip within
seed and a two-level bootstrap, 10,000 each, seed 0. CDEA is the selected row, cdea@lr0.05_t50; the
ablations are compared with the default cdea of the same cell (13.12).

On test, only cells at the tree's method hash, written with --final and the frozen config hash, are
read. On val, the hashes ``partition_readout.counted_hashes`` accepts are read, and a group without
the selected row falls back to the default cdea, which the output names.

    PYTHONPATH=. python scripts/partition_tests.py --split val
    PYTHONPATH=. python scripts/partition_tests.py --split test     # after the test read only
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.stats import (  # noqa: E402
    holm_adjust,
    sign_flip_test,
    stratified_two_level_bootstrap_ci,
    two_level_bootstrap_ci,
    wilcoxon_signed_rank,
)

FROZEN_CONFIG = "291bd2d48c5502e4"
SELECTED = "cdea@lr0.05_t50"
DELTA_D1A, DELTA_FRAME = 0.02, 0.05
N_PERM = N_BOOT = 10_000
NATURAL = ("cifar10", "cifar100", "oxford_pets", "stanford_dogs", "cub200", "ham10000", "brain_tumor", "imagenet")
# Record roots per split and unit grid, in priority order. On val the one-sided row and the ViT-B/16
# adjacent cells were run into their own roots; on test every native-grid row is in one cell.
ROOTS = {
    "val": {"native": ("cells", "csci/fair", "csci/vit_adjacent", "csci/vit_da16"), "7x7": ("csci/vit_grid7",)},
    "test": {"native": ("cells",), "7x7": ("csci/vit_grid7",)},
}
KEYS = ["seed", "operator", "image_index"]


def read_group(base: Path, split: str, dataset: str, backbone: str, grid: str = "native"):
    """Every counted row of one dataset x backbone x grid, one row per (seed, operator, image, method)."""
    import pandas as pd

    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_methods import Knobs
    from scripts.launch_runs import method_errors
    from scripts.partition_readout import HASH_TYPES, counted_hashes

    current = method_code_hash()
    accepted = {current} if split == "test" else counted_hashes()
    knobs = knobs_hash(Knobs())
    frames = []
    for rank, root in enumerate(ROOTS[split][grid]):
        for records in sorted((base / root / split / "partition" / dataset / backbone).glob("seed*/records.csv.gz")):
            summary = json.loads((records.parent / "summary.json").read_text())
            where = records.parent.relative_to(base)
            if method_errors(records.parent / "summary.json"):
                raise SystemExit(f"{where}: a method errored; the cell does not count")
            if summary["method_code_hash"] not in accepted:
                raise SystemExit(f"{where}: method hash {summary['method_code_hash'][:8]} is not counted")
            if split == "test":
                prov = summary.get("provenance", {})
                if prov.get("config_hash") != FROZEN_CONFIG or prov.get("git_dirty") or summary["cell"]["split"] != "test":
                    raise SystemExit(f"{where}: not a final test cell of config {FROZEN_CONFIG}")
            d = pd.read_csv(records, dtype=HASH_TYPES)
            d = d[(d["knobs_hash"] == knobs) & (d["fast"] == 0) & d["method_code_hash"].isin(accepted)]
            frames.append(d.assign(_rank=rank))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True).sort_values("_rank", kind="stable")
    d = d.drop_duplicates(KEYS + ["method"], keep="first")
    # Rows joined across roots must describe the same images: the pair and Omega do not depend on the method.
    same = d.groupby(KEYS)[["class_k", "class_l", "omega_size", "overlap_omega"]].nunique(dropna=False)
    if (same > 1).any().any():
        raise SystemExit(f"{dataset}/{backbone}: the roots disagree on the images of some seed")
    return d


def cdea_row(d, split: str, note: list) -> str:
    if SELECTED in set(d["method"]):
        return SELECTED
    if split == "test":
        raise SystemExit(f"{SELECTED} is missing from a test group")
    note.append(f"{d['dataset'].iloc[0]}/{d['backbone'].iloc[0]}: {SELECTED} not in the val cells; default cdea used")
    return "cdea"


def paired(d, a: str, b: str, value, operator: str = "road", keep=None) -> dict[int, np.ndarray]:
    """Per-seed arrays of a - b, per image. ``value`` is a column or a function of the rows;
    ``keep`` selects images from a's rows."""
    x = d[(d["method"] == a) & (d["operator"] == operator)].set_index(KEYS)
    y = d[(d["method"] == b) & (d["operator"] == operator)].set_index(KEYS)
    if x.empty or y.empty:
        raise SystemExit(f"{a} or {b} missing for {d['dataset'].iloc[0]}/{d['backbone'].iloc[0]}")
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    if keep is not None:
        mask = keep(x, y)
        x, y = x[mask], y[mask]
    vx = value(x) if callable(value) else x[value]
    vy = value(y) if callable(value) else y[value]
    diff = (vx - vy).dropna()
    return {int(s): g.to_numpy(dtype=float) for s, g in diff.groupby(level="seed")}


def confirmed(x, y):
    return (x["role_ak"] == 1) & (x["role_al"] == 1) & (x["role_c"] == 1)


def unique_patches(rows):
    return (rows["ra_all_ak"] + rows["ra_all_al"]) / 2.0


def dt(per_seed: dict[int, np.ndarray]) -> dict:
    """Family D-T: mean over seeds of the per-seed mean, sign-flip p, two-level interval."""
    test = sign_flip_test(per_seed, n_perm=N_PERM, seed=0)
    low, high = two_level_bootstrap_ci(per_seed, n_boot=N_BOOT, seed=0)
    return {"mean": test["statistic"], "ci_low": low, "ci_high": high, "p": test["p"], "n": int(test["n"]),
            "seeds": int(test["seeds"]), "per_seed": {s: float(v.mean()) for s, v in sorted(per_seed.items())},
            "better": int(sum((v > 1e-12).sum() for v in per_seed.values())),
            "worse": int(sum((v < -1e-12).sum() for v in per_seed.values()))}


def run(base: Path, split: str) -> dict:
    note: list[str] = []
    out: dict = {"split": split, "notes": note, "tests": {}, "beside": {}}
    tests, beside = out["tests"], out["beside"]

    grid = read_group(base, split, "planted_pair", "resnet50")
    adj = read_group(base, split, "planted_pair_adjacent", "resnet50")
    a = cdea_row(grid, split, note)
    # D1''a
    r = dt(paired(grid, a, "subtract", "ra_all_c", keep=confirmed))
    r.update(row=a, versus="subtract", metric="ra_all_c", holds=r["ci_low"] > -DELTA_D1A,
             rule=f"lower 95% bound > -{DELTA_D1A}")
    tests["D1''a"] = r
    beside["D1''a"] = {m: dt(paired(grid, a, "subtract", m, keep=confirmed)) for m in ("ra_all", "ra_all_ak", "ra_all_al")}
    # D1''b and the framing comparison
    a_adj = cdea_row(adj, split, note)
    r = dt(paired(adj, a_adj, "subtract", "ra_all", keep=confirmed))
    r.update(row=a_adj, versus="subtract (also the best rule row)", metric="ra_all",
             holds=r["p"] < 0.05 and r["mean"] > 0, rule="p < 0.05 and mean > 0")
    tests["D1''b"] = r
    beside["D1''b"] = {m: dt(paired(adj, a_adj, "subtract", m, keep=confirmed)) for m in ("ra_all_ak", "ra_all_al", "ra_all_c")}
    tests["D1''"] = {"holds": tests["D1''a"]["holds"] and tests["D1''b"]["holds"], "rule": "D1''a and D1''b"}
    r = dt(paired(adj, a_adj, "occlusion_partition", "ra_all", keep=confirmed))
    verdict = ("ahead" if r["ci_low"] > 0 else "worse" if r["ci_high"] < -DELTA_FRAME
               else "not worse (interval not entirely below -delta)")
    r.update(row=a_adj, versus="occlusion_partition", metric="ra_all", verdict=verdict, delta=DELTA_FRAME)
    tests["framing"] = r
    beside["framing"] = {"cdea - occlusion_onesided": dt(paired(adj, a_adj, "occlusion_onesided", "ra_all", keep=confirmed))}
    # S1
    r = dt(paired(grid, a, "occlusion_onesided", "ra_all", keep=confirmed))
    r.update(row=a, versus="occlusion_onesided", metric="ra_all", holds=r["p"] < 0.05 and r["mean"] > 0,
             rule="p < 0.05 and mean > 0")
    tests["S1"] = r
    s1 = {m: dt(paired(grid, a, "occlusion_onesided", m, keep=confirmed)) for m in ("ra_all_ak", "ra_all_al", "ra_all_c")}
    x = grid[(grid["method"] == a) & (grid["operator"] == "road")].set_index(KEYS)
    y = grid[(grid["method"] == "occlusion_onesided") & (grid["operator"] == "road")].set_index(KEYS)
    common = x.index.intersection(y.index)
    keep = confirmed(x.loc[common], y.loc[common])
    patch = [f"conf_{t}_{r}" for t in ("ak", "al", "c") for r in ("uk", "ul", "s", "none", "outside")]
    xs, ys = x.loc[common][keep], y.loc[common][keep]
    s1["share of images whose patch cells are labelled differently"] = float((xs[patch] != ys[patch]).any(axis=1).mean())
    s1["share of images whose partition of Omega differs anywhere"] = float((xs["labels"] != ys["labels"]).mean())
    if "cdea:DA12_polish_only" in set(grid["method"]):
        s1["cdea - DA12 (default settings, 13.12)"] = dt(paired(grid, "cdea", "cdea:DA12_polish_only", "ra_all", keep=confirmed))
        s1["DA12 - occlusion_onesided"] = dt(paired(grid, "cdea:DA12_polish_only", "occlusion_onesided", "ra_all", keep=confirmed))
    beside["S1"] = s1
    # S2: ViT-B/16 at 14x14, both conditions stacked within seed
    vit = {c: read_group(base, split, c, "vit_b_16") for c in ("planted_pair", "planted_pair_adjacent")}
    rows = {c: cdea_row(v, split, note) for c, v in vit.items()}
    per_condition = {c: paired(v, rows[c], "occlusion_onesided", unique_patches, keep=confirmed) for c, v in vit.items()}
    seeds = sorted(set.intersection(*(set(p) for p in per_condition.values())))
    stacked = {s: np.concatenate([per_condition[c][s] for c in vit]) for s in seeds}
    test = sign_flip_test(stacked, n_perm=N_PERM, seed=0)
    low, high = stratified_two_level_bootstrap_ci({s: [per_condition[c][s] for c in vit] for s in seeds},
                                                  n_boot=N_BOOT, seed=0)
    tests["S2"] = {"mean": test["statistic"], "ci_low": low, "ci_high": high, "p": test["p"], "n": int(test["n"]),
                   "seeds": len(seeds), "row": rows, "versus": "occlusion_onesided",
                   "metric": "mean of ra_all_ak and ra_all_al", "holds": test["p"] < 0.05 and test["statistic"] > 0,
                   "rule": "p < 0.05 and mean > 0"}
    beside["S2"] = {f"{c}: {m}": dt(paired(v, rows[c], "occlusion_onesided", m if m != "unique" else unique_patches,
                                           keep=confirmed))
                    for c, v in vit.items() for m in ("unique", "ra_all_c", "ra_all")}
    # S3: natural, per backbone, dataset is the unit
    s3, s3_beside = {}, {}
    for backbone in ("resnet50", "vit_b_16"):
        values, extra = {}, {}
        for dataset in NATURAL:
            d = read_group(base, split, dataset, backbone)
            if d is None:
                raise SystemExit(f"no {split} records for {dataset}/{backbone}")
            row = cdea_row(d, split, note)
            for role in ("s", "uk", "ul"):
                for op in ("road", "blur"):
                    per_seed = paired(d, row, "occlusion_onesided", f"confirmed_{role}", operator=op,
                                      keep=lambda x, y, r=role: (x[f"size_{r}"] > 0) & (y[f"size_{r}"] > 0))
                    v = float(np.mean([g.mean() for g in per_seed.values()]))
                    if role == "s" and op == "road":
                        values[dataset] = v
                    else:
                        extra.setdefault(f"confirmed_{role} {op}", {})[dataset] = v
        w = wilcoxon_signed_rank(list(values.values()))
        s3[backbone] = {"per_dataset": values, "median": float(np.median(list(values.values()))),
                        "positive": int(sum(v > 0 for v in values.values())), "p": w["p"], "statistic": w["statistic"]}
        s3_beside[backbone] = extra
    for backbone, p_holm in zip(s3, holm_adjust([s3[b]["p"] for b in s3])):
        s3[backbone]["p_holm"] = p_holm
        s3[backbone]["holds"] = p_holm < 0.05 and s3[backbone]["median"] > 0
    tests["S3"] = {"per_backbone": s3, "metric": "confirmed_s, ROAD, both rows' S non-empty", "versus": "occlusion_onesided",
                   "rule": "two-sided exact Wilcoxon over the 8 datasets, Holm over the backbones, positive median"}
    beside["S3"] = s3_beside
    return out


def fmt(r: dict) -> str:
    return (f"mean {r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p {r['p']:.4f}  "
            f"n {r['n']} images, {r['seeds']} seeds" + (f"  better {r['better']} / worse {r['worse']}" if "better" in r else ""))


def report(out: dict) -> list[str]:
    t = out["tests"]
    lines = [f"# Declared tests, {out['split']} split (EVAL_PLAN 13.9-13.12)", ""]
    for name in ("D1''a", "D1''b", "S1", "S2"):
        r = t[name]
        lines.append(f"{name}: {r['row']} - {r['versus']}, {r['metric']}: {fmt(r)}  -> "
                     f"{'HOLDS' if r['holds'] else 'does not hold'} ({r['rule']})")
    joint = t["D1''"]["holds"]
    lines.append(f"D1'': {'HOLDS' if joint else 'does not hold'} (D1''a and D1''b)")
    r = t["framing"]
    lines.append(f"framing: {r['row']} - {r['versus']}, {r['metric']}: {fmt(r)}  -> {r['verdict']} (delta {r['delta']})")
    for backbone, r in t["S3"]["per_backbone"].items():
        lines.append(f"S3 {backbone}: median {r['median']:+.4f}, {r['positive']}/8 datasets positive, p {r['p']:.4f}, "
                     f"Holm p {r['p_holm']:.4f} -> {'HOLDS' if r['holds'] else 'does not hold'}")
    lines += ["", "Beside the tests (reported, untested):"]
    for name, rows in out["beside"].items():
        for label, r in rows.items():
            if isinstance(r, dict) and "mean" in r:
                lines.append(f"  {name} {label}: {fmt(r)}")
            elif isinstance(r, dict):  # S3: per backbone, each measure's per-dataset values
                for measure, values in r.items():
                    lines.append(f"  {name} {label} {measure}: median {float(np.median(list(values.values()))):+.4f}, "
                                 f"{sum(v > 0 for v in values.values())}/{len(values)} positive; "
                                 + ", ".join(f"{k} {v:+.4f}" for k, v in values.items()))
            else:
                lines.append(f"  {name} {label}: {r:.4f}")
    lines += [""] + [f"note: {n}" for n in out["notes"]]
    return lines


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--base", default=str(REPO / "results" / "paper"), help="the folder holding cells/ and csci/")
    p.add_argument("--out", default=None, help="folder for tests.json and tests.md (default results/paper/analysis/tests_<split>)")
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
    out = run(Path(args.base), args.split)
    lines = report(out)
    print("\n".join(lines))
    folder = Path(args.out or Path(args.base) / "analysis" / f"tests_{args.split}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tests.json").write_text(json.dumps(out, indent=1, default=float))
    (folder / "tests.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {folder}/tests.json and tests.md")


if __name__ == "__main__":
    main()
