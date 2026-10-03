"""
scripts/plot_deletion_matrix.py

The deletion matrix, on its own.

This is the one result in the contrastive study that is *not* a term in the objective.
Overlap, sufficiency, margin and mask budget are all in `ContrastiveObjective`'s loss, so
reporting them measures convergence. The deletion test interrogates the frozen classifier
instead: remove hypothesis j's unique mask, and see what happens to every hypothesis.

Two things have to hold for the split to be contrastive rather than two masks pushed
apart by a penalty:

  * the diagonal is negative -- removing a class's own unique evidence costs that class;
  * the off-diagonal is *positive* -- it actively helps its rivals.

An equal-budget random deletion is drawn alongside as the control. Nothing in the
objective asks for a positive off-diagonal, and generic image corruption cannot produce
one.

Run from repo root::

    PYTHONPATH=. python scripts/plot_deletion_matrix.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
INK, INK2 = "#0b0b0b", "#52514e"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
# Diverging ramp: blue = the class loses logit, red = the class gains.
DIVERGE = LinearSegmentedColormap.from_list("d", ["#1a5fb4", "#eaf1fb", "#ffffff",
                                                 "#fbeae4", "#c0392b"])

RUNS = [
    ("HAM10000 · EfficientNetV2-S",
     "results/medical_presentation/decomposition_sharedfix/decomp_ham10000_effnet_gradcam.json"),
    ("Brain tumor MRI · ResNet-18",
     "results/medical_presentation/decomposition_sharedfix/decomp_brain_resnet_gradcam.json"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Render the deletion matrix on its own")
    p.add_argument("--out_dir", type=str,
                   default="results/medical_presentation/figures")
    p.add_argument("--name", type=str, default="F4b_deletion_matrix")
    args = p.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })

    loaded = []
    for title, path in RUNS:
        d = json.load(open(REPO / path))
        loaded.append((title, d))

    fig = plt.figure(figsize=(17.0, 7.0))
    # No tight_layout: imshow axes with a fixed aspect fight it. Explicit margins instead.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.78, 1.05], wspace=0.30,
                          left=0.055, right=0.985, top=0.84, bottom=0.145)

    # ---- the two matrices
    for col, (title, d) in enumerate(loaded):
        ax = fig.add_subplot(gs[0, col])
        M = np.array(d["deletion_matrix"])
        K = M.shape[0]
        lim = float(np.abs(M).max())
        # aspect="auto" so a 3x3 and a 5x5 fill the same height side by side.
        im = ax.imshow(M, cmap=DIVERGE, aspect="auto",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=15 if K <= 4 else 13, color=INK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"remove\nunique {j}" for j in range(K)], fontsize=12, color=INK2)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"rank {i}" for i in range(K)], fontsize=12, color=INK2)
        ax.set_title(title, fontsize=17, color=INK, pad=30, loc="left")
        ax.text(0, 1.012, f"Δ logit  ·  n = {d['num_images']}", transform=ax.transAxes,
                fontsize=12.5, color=MUTED, va="bottom")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-.5, K, 1), minor=True)
        ax.set_yticks(np.arange(-.5, K, 1), minor=True)
        ax.grid(which="minor", color="white", lw=2.5)
        ax.tick_params(which="minor", length=0)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=11)

    # ---- diagonal vs off-diagonal vs the equal-budget random control
    ax = fig.add_subplot(gs[0, 2])
    labels, diag, off, rand = [], [], [], []
    for title, d in loaded:
        M = np.array(d["deletion_matrix"])
        K = M.shape[0]
        labels.append(title.split(" · ")[0].replace("Brain tumor MRI", "Brain MRI"))
        diag.append(float(np.diag(M).mean()))
        off.append(float((M.sum() - np.trace(M)) / (K * K - K)))
        rand.append(float(np.mean(d["deletion_random_control"])))
    x = np.arange(len(labels)); w = 0.26
    ax.bar(x - w, diag, w, color=BLUE, label="remove the class's OWN unique evidence")
    ax.bar(x, off, w, color=ORANGE, label="effect on its RIVALS")
    ax.bar(x + w, rand, w, color=BASELINE, label="equal-budget random deletion (control)")
    for xi, (a, b, c) in enumerate(zip(diag, off, rand)):
        for dx, v in ((-w, a), (0, b), (w, c)):
            ax.text(xi + dx, v + (0.012 if v >= 0 else -0.012), f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=12, color=INK)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13, color=INK2)
    ax.set_ylabel("mean Δ logit", fontsize=13, color=INK2)
    ax.set_title("The class loses, its rivals gain,\nand random deletion does nothing",
                 fontsize=17, color=INK, pad=14, loc="left")
    # Room reserved under the deepest bar so the legend cannot sit on a value label.
    lo, hi = min(diag), max(max(off), max(rand))
    ax.set_ylim(lo - 0.155, hi + 0.045)
    ax.legend(frameon=False, fontsize=12, loc="lower center", ncol=1,
              handlelength=1.4, borderpad=0.2, labelspacing=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE); ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

    fig.text(0.5, 0.035,
             "Rows are hypothesis rank, columns are which unique mask was deleted. "
             "Diagonal negative = removing a class's own evidence costs it. "
             "Off-diagonal positive = it helps the rivals — the contrastive claim, and "
             "nothing in the objective asks for it.",
             ha="center", fontsize=14, color=INK)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.name}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("Saved", out / f"{args.name}.png")
    for title, d in loaded:
        M = np.array(d["deletion_matrix"]); K = M.shape[0]
        print(f"  {title}: diag {np.diag(M).mean():+.4f}  "
              f"off {(M.sum()-np.trace(M))/(K*K-K):+.4f}  "
              f"random {np.mean(d['deletion_random_control']):+.4f}  "
              f"diag-minus-off (per image) {d['deletion_diag_minus_offdiag']['mean']:+.4f}")


if __name__ == "__main__":
    main()
