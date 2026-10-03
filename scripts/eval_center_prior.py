"""
scripts/eval_center_prior.py

How much of a localization score is explained by the target simply being centered?

`eval_localization.py` reports what share of a mask's mass lands inside the
pathology. That number is only interpretable next to what a mask with no model
behind it scores. This script computes that ladder directly from the segmentation
masks — no model, no checkpoint, no allocation:

- **uniform**       a flat mask; scores exactly the target's area fraction (chance)
- **gaussian_h4/h6** a centered isotropic Gaussian, two widths
- **center_3x3**    a fixed centered block covering 3x3 cells of the grid
- **center_1cell**  a fixed single centered cell

and it sweeps the grid resolution, because the intuition that a finer grid makes
the degenerate baseline weaker is wrong: for a centered target, a *smaller* mask
at the center is more reliably inside it, so refining the grid makes the null
stronger. Measuring that is the point.

Also reported: the centroid distribution of the targets. A tight centroid spread
is what makes the center prior strong, and it is the property that differs most
between the two datasets — dermoscopy centers the lesion by acquisition
convention, whereas tumor position genuinely varies across patients.

Run from repo root::

    PYTHONPATH=. python scripts/eval_center_prior.py \\
        --out_dir results/medical_presentation/figures
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.reporting import save_json, save_rows_csv

MASK_GLOBS = {
    "ham10000": str(REPO / "data" / "ham10000_raw" / "HAM10000_segmentations"
                    / "HAM10000_segmentations_lesion_tschandl" / "*.png"),
    "brain_tumor": str(REPO / "data" / "brain_tumor_raw" / "masks" / "*.png"),
}
GRIDS = [7, 14, 28, 56]


def load_masks(pattern: str, n: int, size: int, seed: int) -> List[np.ndarray]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no masks matched {pattern}")
    rng = np.random.RandomState(seed)
    rng.shuffle(paths)
    out = []
    for p in paths[:n]:
        m = np.array(Image.open(p).convert("L").resize((size, size), Image.NEAREST)) > 127
        if m.sum() > 0:
            out.append(m)
    return out


def _frac(mask: np.ndarray, target: np.ndarray) -> float:
    """Share of mask mass inside the target."""
    total = mask.sum()
    return float((mask * target).sum() / total) if total > 0 else 0.0


def center_box(size: int, grid: int, cells: int) -> np.ndarray:
    """A fixed centered block covering `cells` x `cells` grid cells."""
    cs = size // grid
    half = cells // 2
    c = grid // 2
    lo, hi = max(c - half, 0), min(c + half + 1, grid)
    m = np.zeros((size, size), dtype=float)
    m[lo * cs:hi * cs, lo * cs:hi * cs] = 1.0
    return m


def centered_gaussian(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    gy, gx = np.meshgrid(ax, ax, indexing="ij")
    return np.exp(-(gy ** 2 + gx ** 2) / (2 * sigma ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Center-prior null ladder for localization")
    parser.add_argument("--num_masks", type=int, default=1000)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=str(REPO / "scripts" / "out"))
    parser.add_argument("--export_prefix", type=str, default="center_prior")
    args = parser.parse_args()

    rows: List[Dict] = []
    detail: Dict[str, Dict] = {}

    for name, pattern in MASK_GLOBS.items():
        print(f"\n=== {name} ===")
        masks = load_masks(pattern, args.num_masks, args.size, args.seed)
        n = len(masks)
        area = float(np.mean([m.mean() for m in masks]))

        cents = np.array([[np.nonzero(m)[0].mean() / args.size,
                           np.nonzero(m)[1].mean() / args.size] for m in masks])
        cmu, csd = cents.mean(0), cents.std(0)
        print(f"  n={n}  mean target area (chance) = {area:.4f}")
        print(f"  centroid mean (y,x) = ({cmu[0]:.3f}, {cmu[1]:.3f})  "
              f"std = ({csd[0]:.3f}, {csd[1]:.3f})")

        entries: Dict[str, float] = {"uniform": area}
        for label, sigma in [("gaussian_sigma_h4", args.size / 4),
                             ("gaussian_sigma_h6", args.size / 6)]:
            g = centered_gaussian(args.size, sigma)
            entries[label] = float(np.mean([_frac(g, m) for m in masks]))

        # Resolution sweep: the claim under test is that refining the grid does NOT
        # weaken the degenerate center baseline.
        by_grid: Dict[str, Dict[str, float]] = {}
        for grid in GRIDS:
            c1 = center_box(args.size, grid, 1)
            c3 = center_box(args.size, grid, 3)
            s1 = float(np.mean([_frac(c1, m) for m in masks]))
            s3 = float(np.mean([_frac(c3, m) for m in masks]))
            by_grid[str(grid)] = {"center_1cell": s1, "center_3x3": s3,
                                  "cell_area_fraction": 1.0 / (grid * grid)}
            print(f"  grid {grid:>2}x{grid:<2} cell={100 / (grid * grid):.3f}%  "
                  f"center_1cell={s1:.4f}  center_3x3={s3:.4f}")

        for label, value in entries.items():
            rows.append({"dataset": name, "method": label, "grid": "", "score": value, "n": n})
        for grid, vals in by_grid.items():
            for label in ("center_1cell", "center_3x3"):
                rows.append({"dataset": name, "method": label, "grid": grid,
                             "score": vals[label], "n": n})

        detail[name] = {
            "n": n,
            "chance_area_fraction": area,
            "centroid_mean_yx": cmu.tolist(),
            "centroid_std_yx": csd.tolist(),
            "flat": entries,
            "by_grid": by_grid,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(out_dir / f"{args.export_prefix}.csv", rows)
    save_json(out_dir / f"{args.export_prefix}.json", {
        "rows": rows, "detail": detail,
        "num_masks_requested": args.num_masks, "size": args.size, "seed": args.seed,
    })
    print(f"\nSaved {out_dir / (args.export_prefix + '.csv')}")


if __name__ == "__main__":
    main()
