"""Read decomposition version 2 records (FORMULATION_DECOMPOSITION.md version 2, EVAL_PLAN.md 13.5-13.8).

Per dataset: the overlap before (O, |Omega|, tertile cuts on the sample),
then per method the resolution shares and VR (all images and the top O
tertile; ``lenient`` is VR with the lenient empty-role rule), role sizes, the
empty and unconfirmed S shares (D8), the per-role
scores, MIX and the split-half verdicts, and the third class. For CDEA rows:
D2, D9, the trapped and empty shares, and how often the final partition's
blur welfare falls below step 0's. Then the paired differences of 13.8
(CDEA - subtract, CDEA - the best rule row, CDEA - occlusion_partition) with
percentile bootstrap intervals over images, and the 13.7 selection.

Only rows whose ``method_code_hash`` matches the tree count, unless
``--any-hash``. Fast-knob rows never count unless ``--fast``.

    PYTHONPATH=. python scripts/partition_readout.py --root results/paper/smoke/partition_local/cells
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GAME = "partition"
RULE_ROWS = ("subtract", "ig_minsplit", "occlusion_minsplit", "rise_minsplit", "extremal_minsplit", "chefer_minsplit")
BOOT = 2000


def load(root: Path, split: str, any_hash: bool, fast: bool):
    import pandas as pd

    from evaluation.provenance import method_code_hash

    frames = [pd.read_csv(path) for path in sorted((root / split / GAME).glob("*/*/seed*/records.csv.gz"))]
    if not frames:
        raise SystemExit(f"no records under {root / split / GAME}")
    df = pd.concat(frames, ignore_index=True)
    total = len(df)
    if not fast:
        df = df[df["fast"] == 0]
    if not any_hash:
        df = df[df["method_code_hash"] == method_code_hash()]
    print(f"rows: {len(df)} of {total} counted (hash match: {not any_hash}, fast rows: {fast})")
    return df


def boot_ci(values, seed: int = 0):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(BOOT, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(values)


def fmt(stat) -> str:
    mean, lo, hi, n = stat
    if n == 0:
        return "n/a"
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}] n={n}"


def per_image(df):
    """One row per image: the method-independent columns."""
    first = df.drop_duplicates("image_index").set_index("image_index")
    return first[["overlap", "overlap_omega", "omega_size", "z_k_full", "z_l_full", "z_m_full", "correct"]]


def method_table(rows, top_images) -> list[str]:
    import numpy as np

    out = []
    header = (f"{'method':30s} {'VR all':>7s} {'VR top':>7s} {'lenient':>7s} {'shared':>7s} {'unique':>7s} {'neither':>7s} "
              f"{'|Uk|':>5s} {'|Ul|':>5s} {'|S|':>5s} {'|0|':>5s} {'S=0':>5s} {'S?':>5s} "
              f"{'PUR':>7s} {'SV':>7s} {'PURc':>7s} {'SVc':>7s} {'MIX':>7s} {'GEN':>7s} {'cfUk':>5s} {'cfUl':>5s} {'cfS':>5s} {'cf0':>5s}")
    out.append(header)
    for method, g in rows.groupby("method", sort=True):
        top = g[g["image_index"].isin(top_images)]
        s_empty = float((g["size_s"] == 0).mean())
        s_unconf = float(((g["size_s"] > 0) & (g["confirmed_s"] == 0)).mean())
        pur = (g["pur_k"] + g["pur_l"]).mean()
        purc = (g["pur_cell_k"] + g["pur_cell_l"]).mean()
        out.append(
            f"{method:30s} {g['vr'].mean():7.3f} {top['vr'].mean():7.3f} "
            f"{g['vr_lenient'].mean() if 'vr_lenient' in g else float('nan'):7.3f} {g['res_shared'].mean():7.3f} "
            f"{g['res_unique'].mean():7.3f} {g['res_neither'].mean():7.3f} "
            f"{g['size_uk'].mean():5.1f} {g['size_ul'].mean():5.1f} {g['size_s'].mean():5.1f} {g['size_none'].mean():5.1f} "
            f"{s_empty:5.2f} {s_unconf:5.2f} {pur:7.3f} {g['sv'].mean():7.3f} {purc:7.3f} {g['sv_cell'].mean():7.3f} "
            f"{np.nanmedian(g['mix']) if g['mix'].notna().any() else float('nan'):7.3f} {g['gen_s'].mean():7.3f} "
            f"{g['confirmed_uk'].mean():5.2f} {g['confirmed_ul'].mean():5.2f} {g['confirmed_s'].mean():5.2f} "
            f"{g['confirmed_none'].mean():5.2f}")
    return out


def verdicts(rows) -> list[str]:
    out = []
    mixed = rows[(rows["mix"] > 0) & rows["split_verdict"].notna() & (rows["split_verdict"] != "")]
    for method, g in mixed.groupby("method"):
        counts = g["split_verdict"].value_counts().to_dict()
        out.append(f"{method:30s} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return out


def cdea_diagnostics(rows) -> list[str]:
    out = []
    stage = rows[rows["payoff_hard_uk"].notna()] if "payoff_hard_uk" in rows else rows.iloc[0:0]
    for method, g in stage.groupby("method"):
        soft = g[["payoff_soft_uk", "payoff_soft_ul", "payoff_soft_s"]].sum(axis=1)
        hard = g[["payoff_hard_uk", "payoff_hard_ul", "payoff_hard_s"]].sum(axis=1)
        init = g[["payoff_init_uk", "payoff_init_ul", "payoff_init_s"]].sum(axis=1)
        line = (f"{method:30s} D2 soft-hard {float((soft - hard).mean()):+.3f}  W init {float(init.mean()):+.3f} "
                f"-> hard {float(hard.mean()):+.3f}  hard < init {float((hard < init - 1e-6).mean()):.2f}  "
                f"D9 same {float(g['same_as_init'].mean()):.2f}  trapped {float(g['trapped'].mean()):.3f}  "
                f"empty {float(g['empty_share'].mean()):.3f}")
        if "payoff_search_uk" in g and g["payoff_search_uk"].notna().any():
            search = g[["payoff_search_uk", "payoff_search_ul", "payoff_search_s"]].sum(axis=1)
            line += (f"  search W {float(search.mean()):+.3f} (< init {float((search < init - 1e-6).mean()):.2f})"
                     f"  polish moved {float(g['polish_changed'].mean()):.3f}")
        out.append(line)
    return out


def paired(rows, a: str, b: str, column: str, images=None):
    left = rows[rows["method"] == a].set_index("image_index")[column]
    right = rows[rows["method"] == b].set_index("image_index")[column]
    joined = (left - right).dropna()
    if images is not None:
        joined = joined[joined.index.isin(images)]
    return boot_ci(joined.to_numpy())


def report(df, operator: str) -> dict:
    import numpy as np

    selection = {}
    for (dataset, backbone), cell in df.groupby(["dataset", "backbone"]):
        rows = cell[cell["operator"] == operator]
        if rows.empty:
            continue
        images = per_image(rows)
        cuts = np.percentile(images["overlap"], [100 / 3, 200 / 3])
        top_images = set(images.index[images["overlap"] > cuts[1]])
        print(f"\n=== {dataset} / {backbone} / {operator}: {len(images)} images ===")
        print(f"O median {images['overlap'].median():.3f}; tertile cuts {cuts[0]:.3f}, {cuts[1]:.3f}; "
              f"top tertile {len(top_images)} images")
        print(f"|Omega| median {images['omega_size'].median():.0f} (IQR {images['omega_size'].quantile(0.25):.0f}"
              f" to {images['omega_size'].quantile(0.75):.0f}); O inside Omega median {images['overlap_omega'].median():.3f}")
        gap = images["z_l_full"] - images["z_m_full"]
        print("K-sufficiency, share with z_l - z_m below 0.5 / 1 / 2: "
              + " / ".join(f"{float((gap < t).mean()):.2f}" for t in (0.5, 1.0, 2.0)))
        print("\n".join(method_table(rows, top_images)))
        lines = verdicts(rows)
        if lines:
            print("split halves of S with MIX > 0:")
            print("\n".join(lines))
        lines = cdea_diagnostics(rows)
        if lines:
            print("CDEA diagnostics (blur, CDEA's null):")
            print("\n".join(lines))
        means = rows.groupby("method")["vr"].mean()
        rules = [m for m in RULE_ROWS if m in means.index]
        best_rule = max(rules, key=lambda m: means[m]) if rules else None
        print(f"best rule row by mean VR: {best_rule}")
        for column in ("vr",):
            for other in ("subtract", best_rule, "occlusion_partition"):
                if other is None or other not in means.index or "cdea" not in means.index:
                    continue
                print(f"CDEA - {other:22s} {column}: all {fmt(paired(rows, 'cdea', other, column))}; "
                      f"top {fmt(paired(rows, 'cdea', other, column, top_images))}")
        if "pur_k" in rows:
            rows = rows.assign(pur=rows["pur_k"] + rows["pur_l"])
            for other in ("subtract", "occlusion_partition"):
                if other in means.index and "cdea" in means.index:
                    print(f"CDEA - {other:22s} PUR_k+PUR_l: {fmt(paired(rows, 'cdea', other, 'pur'))}; "
                          f"SV: {fmt(paired(rows, 'cdea', other, 'sv'))}")
        for method in means.index:
            if method == "cdea" or method.startswith("cdea@"):
                selection.setdefault(method, []).append(float(means[method]))
    return selection


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO / "results" / "paper" / "cells"))
    p.add_argument("--split", default="val", choices=["val"])
    p.add_argument("--operator", default="road", choices=["road", "blur"])
    p.add_argument("--any-hash", action="store_true")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args(argv)
    df = load(Path(args.root), args.split, args.any_hash, args.fast)
    selection = report(df, args.operator)
    if selection:
        print("\n13.7 selection, mean over datasets of mean VR (all images):")
        for method, values in sorted(selection.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            print(f"  {method:24s} {sum(values) / len(values):.3f} over {len(values)} dataset(s)")


if __name__ == "__main__":
    main()
