"""Fractional-mask test read-out (``docs/paper/G1.md``, "Reopened, 2026-10-13").

Seed 1, val, ResNet-50, the 8 contrastive datasets, CD@5% under ROAD. For
each CDEA arm and each comparator: the mean paired difference per dataset
with a 95% bootstrap interval over images, and the unweighted mean over the
datasets with a 95% stratified bootstrap interval (images resampled within
each dataset). The pass rule and the decision are the ones fixed in G1.md
before the test ran. Val only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.stats import bootstrap_mean_ci  # noqa: E402
from scripts.check_degenerate import stage_gaps  # noqa: E402
from scripts.g1_readout import drops, load  # noqa: E402

DATASETS = ("cifar10", "cifar100", "oxford_pets", "stanford_dogs", "cub200", "ham10000", "brain_tumor", "imagenet")
FRACTIONAL = "cdea:A12_fractional_masks"
BINARY = "cdea"
MARGIN = ("margin_gradcam", "margin_ig")
EXTREMAL = "extremal"


def paired_by_image(a: dict, b: dict) -> np.ndarray:
    keys = sorted(set(a) & set(b))
    return np.array([float(a[k]["cd"]) - float(b[k]["cd"]) for k in keys], dtype=np.float64)


def better_margin(methods: dict) -> str | None:
    """The margin variant with the higher mean CD on this dataset, chosen once on the full sample."""
    present = [m for m in MARGIN if m in methods]
    if not present:
        return None
    return max(present, key=lambda m: np.mean([float(r["cd"]) for r in methods[m].values()]))


def pooled_ci(diffs: list[np.ndarray], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Unweighted mean over datasets of each dataset's mean, with a stratified percentile interval."""
    rng = np.random.default_rng(int(seed))
    point = float(np.mean([d.mean() for d in diffs]))
    boots = np.zeros(int(n_boot))
    for d in diffs:
        boots += d[rng.integers(0, d.size, size=(int(n_boot), d.size))].mean(axis=1)
    boots /= len(diffs)
    low, high = np.quantile(boots, [0.025, 0.975], method="linear")
    return point, float(low), float(high)


def arm_passes(per_dataset: dict, pooled: dict, n_datasets: int = len(DATASETS)) -> tuple[bool, list[str]]:
    """The G1.md pass rule. ``per_dataset[d]["margin"]`` and ``pooled[c]`` hold (mean, low, high)."""
    reasons = []
    complete = [d for d, r in per_dataset.items() if r.get("margin") is not None and r.get("extremal") is not None]
    if len(complete) < n_datasets:
        reasons.append(f"rule 4: {len(complete)} of {n_datasets} datasets complete")
    for comp, rule in (("margin", "rule 1"), ("extremal", "rule 2")):
        if pooled.get(comp) is None or pooled[comp][1] <= 0:
            reasons.append(f"{rule}: pooled interval vs {comp} is not above 0")
    behind = [d for d in complete if per_dataset[d]["margin"][2] < 0]
    if behind:
        reasons.append("rule 3: interval vs the better margin variant below 0 on " + ", ".join(behind))
    return not reasons, reasons


def decide(binary_pass: bool, fractional_pass: bool) -> str:
    if binary_pass:
        return "method paper with CDEA as it is (binary draws); fractional masks stay ablation A12"
    if fractional_pass:
        return "method paper with fractional masks; binary draws become an ablation; D2 as fixed in G1.md"
    return "evaluation-paper route stands; the method question is closed"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True, help="cell roots, e.g. results/paper/fractional_test")
    parser.add_argument("--expect-hash", default=None, help="method_code_hash prefix every row must carry")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args(argv)
    roots = [Path(r) for r in args.records]
    data = load(roots, [1], 0.05, "road")

    hashes = {(r["method_code_hash"], r.get("knobs_hash", "")) for rows in data.values() for r in rows.values()}
    print("method_code_hash, knobs_hash:", sorted(hashes))
    if args.expect_hash and any(not h.startswith(args.expect_hash) for h, _ in hashes):
        raise SystemExit(f"rows from another method hash than {args.expect_hash}; the read-out does not mix code")
    if len(hashes) > 1:
        raise SystemExit("rows carry more than one (method, knobs) hash; the read-out does not mix code")

    results = {arm: {} for arm in (FRACTIONAL, BINARY)}
    for dataset in DATASETS:
        methods = {m: rows for (d, m), rows in data.items() if d == dataset}
        print(f"\n{dataset}  CD@5% under ROAD, seed 1")
        if not methods:
            print("  no records")
            for arm in results:
                results[arm][dataset] = {"margin": None, "extremal": None}
            continue
        print(f"  {'method':28s} {'n':>3s} {'CD':>6s} {'CD1':>6s} {'zk|Mk':>6s} {'zl|Mk':>6s} {'zk|Ml':>6s} {'zl|Ml':>6s}")
        for name, rows in sorted(methods.items()):
            d = [drops(r) for r in rows.values()]
            cd = np.mean([float(r["cd"]) for r in rows.values()])
            cd1 = np.mean([float(r["cd1"]) for r in rows.values()])
            print(f"  {name:28s} {len(rows):>3d} {cd:6.2f} {cd1:6.2f} " + " ".join(
                f"{np.mean([x[key] for x in d]):6.2f}" for key in ("zk|Mk", "zl|Mk", "zk|Ml", "zl|Ml")))
        margin = better_margin(methods)
        print(f"  better margin variant: {margin}")
        for arm in results:
            entry = {"margin": None, "extremal": None, "diffs": {}}
            for comp_key, comp in (("margin", margin), ("extremal", EXTREMAL)):
                if arm not in methods or comp not in methods:
                    continue
                diff = paired_by_image(methods[arm], methods[comp])
                low, high = bootstrap_mean_ci(diff, n_boot=args.n_boot, seed=0)
                entry[comp_key] = (float(diff.mean()), low, high)
                entry["diffs"][comp_key] = diff
                print(f"  {arm} - {comp}: {diff.mean():+.2f} [{low:+.2f}, {high:+.2f}]  n={diff.size}")
            results[arm][dataset] = entry
        for arm in results:
            cell = next((r for r in roots if (r / "val/contrastive" / dataset / "resnet50/seed1").is_dir()), None)
            gaps = stage_gaps(cell / "val/contrastive" / dataset / "resnet50/seed1", method=arm) if cell else {}
            g = gaps.get(0.05)
            if g:
                print(f"  D2 {arm}: soft-hard k {g['soft_minus_hard_k']:.2f}, l {g['soft_minus_hard_l']:.2f} "
                      f"({'pass' if g['d2_pass'] else 'FAIL'}); hard-scored k {g['hard_minus_scored_k']:.2f}, "
                      f"l {g['hard_minus_scored_l']:.2f}")

    print()
    passes = {}
    for arm, per_dataset in results.items():
        pooled = {}
        for comp in ("margin", "extremal"):
            diffs = [e["diffs"][comp] for e in per_dataset.values() if comp in e.get("diffs", {})]
            if len(diffs) == len(DATASETS):
                pooled[comp] = pooled_ci(diffs, n_boot=args.n_boot, seed=0)
                mean, low, high = pooled[comp]
                print(f"{arm} pooled vs {comp}: {mean:+.2f} [{low:+.2f}, {high:+.2f}]")
        passes[arm], reasons = arm_passes(per_dataset, pooled)
        print(f"{arm}: {'PASS' if passes[arm] else 'does not pass'}" + "".join(f"\n  {r}" for r in reasons))
    print("\nDecision:", decide(passes[BINARY], passes[FRACTIONAL]))


if __name__ == "__main__":
    main()
