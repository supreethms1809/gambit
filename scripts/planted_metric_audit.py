"""Audit role accuracy (RA) on planted-pair records (FORMULATION_DECOMPOSITION.md section 6.7).

RA in Omega leaves out every patch cell outside Omega. At rho = 0.5 a class
whose unique patch holds at least half its evidence mass puts only that cell
in Omega, so the shared patch C, or a weak unique patch, can fall outside, and
RA in Omega then judges only the patches the reference evidence ranked first.
This prints, per method, what RA in Omega leaves out:
- RA in Omega and RA over all patch cells, on every image and on images
  whose three roles were confirmed by intervention;
- Omega's coverage of the patches, and for each patch (A_k, A_l, C) the share
  of images with it inside Omega, and its accuracy inside and overall;
- RA on the images with all three patches inside Omega, the only ones where
  RA in Omega judges every role;
- |Omega|, and how many of its cells carry no patch (RA judges none of them);
- each patch's accuracy over all its cells on the role-confirmed images, the
  headline the user chose on 2026-10-22 (C's, beside RA over all patch cells).

``--dataset planted_pair_adjacent`` reads D1''b's harder condition.

    PYTHONPATH=. python scripts/planted_metric_audit.py --root results/paper/cells --backbone resnet50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GAME = "partition"
PATCHES = ("ak", "al", "c")


def load(root: Path, split: str, backbone: str, operator: str, dataset: str = "planted_pair"):
    import pandas as pd

    paths = sorted((root / split / GAME / dataset / backbone).glob("seed*/records.csv.gz"))
    if not paths:
        raise SystemExit(f"no planted-pair records under {root / split / GAME / dataset / backbone}")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = df[(df["operator"] == operator) & (df["fast"] == 0)]
    confirmed = (df["role_ak"] == 1) & (df["role_al"] == 1) & (df["role_c"] == 1)
    return df.assign(confirmed=confirmed)


def audit(df) -> list[str]:
    lines = []
    header = (f"{'method':28s} {'n':>4s} {'conf':>5s} {'RAin':>6s} {'RAall':>6s} {'RAin|c':>7s} {'RAall|c':>8s} "
              f"{'cover':>6s} {'all3in':>7s} {'RA|3in':>7s} " + " ".join(f"{t + ' in':>6s} {t + ' ok':>6s}" for t in PATCHES)
              + f" {'|Om|':>5s} {'other':>6s} " + " ".join(f"{t + ' ok|c':>7s}" for t in PATCHES))
    lines.append(header)
    for method, g in df.groupby("method", sort=True):
        c = g[g["confirmed"]]
        inside = {t: 1.0 - g[f"conf_{t}_outside"] for t in PATCHES}
        all_in = (inside["ak"] > 0) & (inside["al"] > 0) & (inside["c"] > 0)
        cells = []
        for t in PATCHES:
            cells.append(f"{inside[t].mean():6.2f} {g[f'ra_all_{t}'].mean():6.2f}")
        lines.append(
            f"{method:28s} {len(g):4d} {len(c):5d} {g['ra_in'].mean():6.3f} {g['ra_all'].mean():6.3f} "
            f"{c['ra_in'].mean():7.3f} {c['ra_all'].mean():8.3f} {g['omega_coverage'].mean():6.2f} "
            f"{all_in.mean():7.2f} {g.loc[all_in, 'ra_in'].mean():7.3f} " + " ".join(cells)
            + f" {g['omega_size'].mean():5.1f} {(g['omega_size'] - 3 * g['omega_coverage']).mean():6.1f} "
            + " ".join(f"{c[f'ra_all_{t}'].mean():7.2f}" for t in PATCHES))
    lines.append("RAin|c and RAall|c: on images whose three roles were confirmed. all3in: share of images with every patch "
                 "inside Omega; RA|3in: RA on those. '<patch> in': share of images with the patch inside Omega; "
                 "'<patch> ok': its RA over all images, outside counted wrong. other: Omega cells with no patch. "
                 "'<patch> ok|c': the same on the role-confirmed images.")
    return lines


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO / "results" / "paper" / "cells"))
    p.add_argument("--split", default="val", choices=["val"])
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_b_16"])
    p.add_argument("--operator", default="road", choices=["road", "blur"])
    p.add_argument("--dataset", default="planted_pair", choices=["planted_pair", "planted_pair_adjacent"])
    args = p.parse_args(argv)
    print("\n".join(audit(load(Path(args.root), args.split, args.backbone, args.operator, args.dataset))))


if __name__ == "__main__":
    main()
