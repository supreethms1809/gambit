"""G1 read-out from contrastive cell records (``docs/paper/G1.md``).

For each dev dataset: the mean paired difference in CD@5% under ROAD, a CDEA
label minus each comparator, with a 95% bootstrap interval over images, and
the drops in z_k and z_l beside it. Several record folders can be passed;
when two hold the same method, the later folder wins. Val only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.stats import bootstrap_mean_ci  # noqa: E402

DATASETS = ("cifar10", "ham10000")
MARGIN = ("margin_gradcam", "margin_ig")


def load(roots: list[Path], seeds: list[int], area: float, operator: str) -> dict:
    """``{(dataset, method): {(seed, image): row}}``. Later roots replace earlier ones per method."""
    out: dict = {}
    for root in roots:
        found: dict = {}
        for path in sorted(root.glob("val/contrastive/*/resnet50/seed*/records.csv.gz")):
            with gzip.open(path, "rt", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["split"] != "val" or int(row["seed"]) not in seeds:
                        continue
                    if row["operator"] != operator or abs(float(row["area"]) - area) > 1e-9:
                        continue
                    key = (row["dataset"], row["method"])
                    found.setdefault(key, {})[(int(row["seed"]), int(row["image_index"]))] = row
        out.update(found)
    return out


def drops(row: dict) -> dict[str, float]:
    f = lambda name: float(row[name])  # noqa: E731
    return {
        "zk|Mk": f("z_k_full") - f("z_k_without_k"),
        "zl|Mk": f("z_l_full") - f("z_l_without_k"),
        "zk|Ml": f("z_k_full") - f("z_k_without_l"),
        "zl|Ml": f("z_l_full") - f("z_l_without_l"),
    }


def paired(a: dict, b: dict) -> np.ndarray:
    keys = sorted(set(a) & set(b))
    return np.array([float(a[k]["cd"]) - float(b[k]["cd"]) for k in keys], dtype=np.float64)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True, help="cell roots, e.g. results/paper/cells")
    parser.add_argument("--cdea", nargs="+", default=["cdea"], help="CDEA method labels to read out")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--area", type=float, default=0.05)
    parser.add_argument("--operator", default="road")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    data = load([Path(r) for r in args.records], seeds, args.area, args.operator)

    verdicts: dict[str, list[bool]] = {label: [] for label in args.cdea}
    for dataset in DATASETS:
        methods = {m: rows for (d, m), rows in data.items() if d == dataset}
        if not methods:
            print(f"{dataset}: no records")
            continue
        print(f"\n{dataset}  CD@{args.area:g} under {args.operator}, seeds {sorted({k[0] for r in methods.values() for k in r})}")
        print(f"  {'method':28s} {'n':>3s} {'CD':>6s} {'zk|Mk':>6s} {'zl|Mk':>6s} {'zk|Ml':>6s} {'zl|Ml':>6s}  hash")
        for name, rows in sorted(methods.items()):
            d = [drops(r) for r in rows.values()]
            cd = np.mean([float(r["cd"]) for r in rows.values()])
            hashes = ",".join(sorted({r["method_code_hash"][:8] for r in rows.values()}))
            print(f"  {name:28s} {len(rows):>3d} {cd:6.2f} " + " ".join(
                f"{np.mean([x[key] for x in d]):6.2f}" for key in ("zk|Mk", "zl|Mk", "zk|Ml", "zl|Ml")) + f"  {hashes}")
        margins = [m for m in MARGIN if m in methods]
        better = max(margins, key=lambda m: np.mean([float(r["cd"]) for r in methods[m].values()])) if margins else None
        comparators = [m for m in (better, "extremal") if m in methods]
        for label in args.cdea:
            if label not in methods:
                print(f"  {label}: no records")
                verdicts[label].append(False)
                continue
            for comp in comparators:
                diff = paired(methods[label], methods[comp])
                low, high = bootstrap_mean_ci(diff, n_boot=args.n_boot, seed=0)
                ahead = float(diff.mean()) > 0
                verdicts[label].append(ahead)
                print(f"  {label} - {comp}: {diff.mean():+.2f} [{low:+.2f}, {high:+.2f}]  n={diff.size}  "
                      f"{'ahead' if ahead else 'not ahead'}")
    print()
    for label, ahead in verdicts.items():
        complete = len(ahead) == 2 * len(DATASETS)
        verdict = "ahead of both comparators on both datasets" if complete and all(ahead) else "not ahead on every comparison"
        print(f"{label}: {verdict}")


if __name__ == "__main__":
    main()
