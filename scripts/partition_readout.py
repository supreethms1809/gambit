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
HASH_TYPES = {"method_code_hash": str, "knobs_hash": str}  # read as text: a hash of digits alone parses as a number


EQUIVALENT_HASHES = REPO / "scripts" / "equivalent_hashes.json"


def counted_hashes() -> set[str]:
    """The tree's method hash, plus the older hashes ``equivalent_hashes.json`` accepts for it.

    Each accepted hash names the change and the same-rows check that showed it leaves the records unchanged.
    """
    import json

    from evaluation.provenance import method_code_hash

    current = method_code_hash()
    entries = json.loads(EQUIVALENT_HASHES.read_text()) if EQUIVALENT_HASHES.is_file() else {}
    return {current} | {entry["hash"] for entry in entries.get(current, [])}


def current_rows(df, any_hash: bool, fast: bool):
    """The rows that count: ``method_code_hash`` matches the tree (or a hash ``counted_hashes`` accepts for it),
    ``knobs_hash`` matches, and no fast knobs.

    Fast rows count only with ``fast``, and then against the fast knobs' hash. Every
    cell (dataset, backbone, seed) with rows left out is named, with the hashes it
    was written at, so a stale cell is never read as a current one by omission.
    """
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_methods import FAST, Knobs

    total = len(df)
    df = df.assign(method_code_hash=df["method_code_hash"].astype(str), knobs_hash=df["knobs_hash"].astype(str))
    keep = df["fast"] == 0 if not fast else df["fast"].isin([0, 1])
    if not any_hash:
        knobs = df["fast"].map({0: knobs_hash(Knobs()), 1: knobs_hash(FAST)})
        accepted = counted_hashes()
        keep &= df["method_code_hash"].isin(accepted) & (df["knobs_hash"] == knobs)
        for code in sorted(set(df.loc[keep, "method_code_hash"]) - {method_code_hash()}):
            print(f"counting rows at {code[:8]}, accepted as equivalent to the tree's {method_code_hash()[:8]} "
                  f"({EQUIVALENT_HASHES.name})")
    cell = [c for c in ("dataset", "backbone", "seed") if c in df]
    if cell and not keep.all():
        for key, g in df[~keep].groupby(cell):
            stamps = sorted({f"{m[:8]}/{k[:8]}" for m, k in zip(g["method_code_hash"], g["knobs_hash"])})
            print(f"skipped {'/'.join(str(v) for v in key)}: {len(g)} rows at method/knobs {', '.join(stamps)}")
    df = df[keep]
    print(f"rows: {len(df)} of {total} counted (method and knobs hash match: {not any_hash}, fast rows: {fast})")
    versions = sorted(v for v in df["torch"].unique() if v) if "torch" in df else []
    if len(versions) > 1:
        print(f"WARNING: the counted cells ran on more than one torch ({', '.join(versions)}); the hash does not cover it")
    if df.empty:
        raise SystemExit("no row matches the current tree; nothing to read")
    return df


def read_cells(paths) -> "pd.DataFrame":
    """Concatenate cell records, leaving out cells whose ``summary.json`` reports method errors.

    A failed cell is still written, at the current hash, without the methods that
    errored. Each row carries its cell's ``torch`` version, which the hash does not cover.
    """
    import json

    import pandas as pd

    from scripts.launch_runs import method_errors

    frames = []
    for path in paths:
        summary = Path(path).parent / "summary.json"
        if summary.is_file():
            errors = method_errors(summary)
            if errors:
                print(f"skipped {Path(path).parent}: the cell failed ({', '.join(errors)} errored)")
                continue
            torch_version = json.loads(summary.read_text(encoding="utf-8")).get("provenance", {}).get("torch", "")
        else:
            torch_version = ""
        frames.append(pd.read_csv(path, dtype=HASH_TYPES).assign(torch=torch_version))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load(root: Path, split: str, any_hash: bool, fast: bool):
    df = read_cells(sorted((root / split / GAME).glob("*/*/seed*/records.csv.gz")))
    if df.empty:
        raise SystemExit(f"no records under {root / split / GAME}")
    return current_rows(df, any_hash, fast)


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


def with_image_key(df):
    """Add ``image``, one key per image. Planted cells pool seeds, and ``image_index`` repeats across them."""
    if "image" in df:
        return df
    if "seed" not in df:
        return df.assign(image=df["image_index"].astype(str))
    return df.assign(image=df["seed"].astype(str) + ":" + df["image_index"].astype(str))


def per_image(df):
    """One row per image: the method-independent columns."""
    first = with_image_key(df).drop_duplicates("image").set_index("image")
    return first[["overlap", "overlap_omega", "omega_size", "z_k_full", "z_l_full", "z_m_full", "correct"]]


def method_table(rows, top_images) -> list[str]:
    import numpy as np

    out = []
    header = (f"{'method':30s} {'VR all':>7s} {'VR top':>7s} {'lenient':>7s} {'shared':>7s} {'unique':>7s} {'neither':>7s} "
              f"{'|Uk|':>5s} {'|Ul|':>5s} {'|S|':>5s} {'|0|':>5s} {'S=0':>5s} {'S?':>5s} "
              f"{'PUR':>7s} {'SV':>7s} {'PURc':>7s} {'SVc':>7s} {'MIX':>7s} {'GEN':>7s} {'cfUk':>5s} {'cfUl':>5s} {'cfS':>5s} {'cf0':>5s}")
    out.append(header)
    for method, g in rows.groupby("method", sort=True):
        top = g[g["image"].isin(top_images)]
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
        if "shared_rule_share" in g and g["shared_rule_share"].notna().any():
            line += (f"  lowers both {float(g['shared_rule_share'].mean()):.3f}"
                     f" (rule moved {float(g['shared_rule_moved'].mean()):.3f} off step 0)")
        out.append(line)
    return out


def paired(rows, a: str, b: str, column: str, images=None):
    rows = with_image_key(rows)
    left = rows[rows["method"] == a].set_index("image")[column]
    right = rows[rows["method"] == b].set_index("image")[column]
    joined = (left - right).dropna()
    if images is not None:
        joined = joined[joined.index.isin(images)]
    return boot_ci(joined.to_numpy())


def report(df, operator: str) -> dict:
    import numpy as np

    selection = {}
    for (dataset, backbone), cell in df.groupby(["dataset", "backbone"]):
        rows = with_image_key(cell[cell["operator"] == operator])
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
            for other in dict.fromkeys(("subtract", best_rule, "occlusion_partition")):  # best_rule may be subtract
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
