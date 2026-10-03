"""
scripts/plot_shift_real_figure.py

Visual evidence for the robust/shortcut split on Stanford Dogs backgrounds.

Two figures:

  R1_qualitative  per-image panels -- input with the annotated box, one perturbed
                  environment, and the robust and shortcut masks. Cases are *selected by
                  rank* on the robust-minus-shortcut margin (best / median / worst), so
                  the figure shows the spread rather than three hand-picked wins.
  R2_aggregate    where the two masks land over the whole run: distribution of in-box
                  mask mass for robust, shortcut and raw Grad-CAM against the box area,
                  read from the per-image CSV the eval script already wrote.

Run from repo root::

    PYTHONPATH=. python scripts/plot_shift_real_figure.py --scan_images 96
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.device import get_device
from core.hypotheses import TopMSelector
from core.game_modes import resolve_shift_game
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from instantiations.shift.objective import RobustShortcutObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from examples.contrastive_explanation import checkpoint_metadata, load_checkpoint
from scripts.train_backbone import model_grid_size
from scripts.eval_robust_shortcut_dogs import (
    DogsBoxDataset, background_styles, make_env_fn,
)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def to_px(m, gh, gw, H, W):
    return F.interpolate(m.view(1, 1, gh, gw), size=(H, W), mode="bilinear",
                         align_corners=False).squeeze().detach().cpu().numpy()


def norm(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def tint(ax, img, field, rgb, gamma=0.85, alpha=0.88):
    ax.imshow(img)
    ov = np.zeros((*field.shape, 4))
    for i, c in enumerate(rgb):
        ov[..., i] = c
    ov[..., 3] = np.clip(field ** gamma, 0, 1) * alpha
    ax.imshow(ov)


def main() -> None:
    p = argparse.ArgumentParser(description="Visual evidence for robust/shortcut on dogs")
    p.add_argument("--checkpoint", type=str,
                   default="examples/out/checkpoints/stanford_dogs_resnet18.pt")
    p.add_argument("--scan_images", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--lambda_sparse", type=float, default=1.0)
    p.add_argument("--num_styles", type=int, default=3)
    p.add_argument("--style_images", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per_image_csv", type=str,
                   default="results/shift_real_dogs/shift_dogs_background_per_image.csv")
    p.add_argument("--out_dir", type=str, default="results/shift_real_dogs/figures")
    args = p.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- R2 (no model)
    rows = list(csv.DictReader(open(args.per_image_csv)))
    rob = np.array([float(r["robust"]) for r in rows])
    sho = np.array([float(r["shortcut"]) for r in rows])
    bas = np.array([float(r["base_evidence"]) for r in rows])
    chance = 0.5558

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    bins = np.linspace(0, 1, 41)
    for v, c, lab in [(sho, ORANGE, "shortcut mask"), (bas, MUTED, "Grad-CAM (input)"),
                      (rob, BLUE, "robust mask")]:
        ax[0].hist(v, bins=bins, alpha=0.62, color=c, label=f"{lab}  (mean {v.mean():.3f})")
    ax[0].axvline(chance, color=INK, ls="--", lw=2)
    ax[0].text(chance, ax[0].get_ylim()[1] * 0.97, "  box area = chance", fontsize=13,
               color=INK, va="top")
    ax[0].set_xlabel("fraction of mask mass inside the annotated box", fontsize=14)
    ax[0].set_ylabel("images", fontsize=14)
    ax[0].set_title("Where each mask lands", fontsize=18, color=INK, pad=10, loc="left")
    ax[0].legend(frameon=False, fontsize=13)

    d = rob - sho
    ax[1].hist(d, bins=np.linspace(-0.6, 0.9, 46), color=AQUA, alpha=0.8)
    ax[1].axvline(0, color=INK, ls="--", lw=2)
    ax[1].set_xlabel("robust − shortcut, per image", fontsize=14)
    ax[1].set_ylabel("images", fontsize=14)
    ax[1].set_title(f"Separation per image  —  mean {d.mean():+.3f}, "
                    f"positive on {(d > 0).mean():.1%} of {len(d)}",
                    fontsize=18, color=INK, pad=10, loc="left")
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.spines["left"].set_color(BASELINE); a.spines["bottom"].set_color(BASELINE)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"R2_aggregate.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out / "R2_aggregate.png")

    # ---------------------------------------------------------------- R1 (needs the model)
    device = get_device()
    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dsname, class_names, ncls = load_checkpoint(Path(args.checkpoint), device)
    gh, gw = model_grid_size(meta["model_name"])
    ds = DogsBoxDataset()
    print("Measuring background styles:")
    styles = background_styles(ds, args.style_images, args.num_styles, args.seed)
    env_fn = make_env_fn(styles)

    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
    sub = torch.utils.data.Subset(ds, idx[:args.scan_images].tolist())
    loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False)

    cfg = resolve_shift_game("mixed")
    us = VisionGridUnitSpace(gh, gw)
    sel = TopMSelector(m=min(5, ncls))
    prov = GradCAMRegionsProvider(gh, gw)
    obj = RobustShortcutObjective(
        lambda_mean=cfg.lambda_mean, lambda_var=cfg.lambda_var, lambda_gap=cfg.lambda_gap,
        lambda_shortcut=cfg.lambda_shortcut, lambda_disjoint=cfg.lambda_disjoint,
        lambda_sparse=args.lambda_sparse)
    alloc = RobustShortcutOptimizationAllocator(
        obj, num_steps=args.num_steps, lr=args.lr, lambda_disjoint=cfg.lambda_disjoint)

    cases = []
    for x, y, box in loader:
        x, y, box = x.to(device), y.to(device), box.to(device)
        env = env_fn(x, box)
        with torch.no_grad():
            lg = model(x)
            hyp = sel.select(lg, torch.softmax(lg, dim=-1))
        xg = x.detach().clone().requires_grad_(True)
        ev = prov.explain(xg, model, hyp).detach()
        ev = ev / ev.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        masks = alloc.allocate(x=x, model=model, unit_space=us, hypotheses=hyp,
                               evidence=ev, env=env)
        H, W = box.shape[-2:]
        for b in range(x.shape[0]):
            r = to_px(masks["robust"][b], gh, gw, H, W)
            s = to_px(masks["shortcut"][b], gh, gw, H, W)
            bx = box[b].cpu().numpy()
            fr = float((r * bx).sum() / max(r.sum(), 1e-8))
            fs = float((s * bx).sum() / max(s.sum(), 1e-8))
            cases.append({
                "delta": fr - fs, "in_box_rob": fr, "in_box_sho": fs,
                "img": x[b].permute(1, 2, 0).cpu().numpy(),
                "env": env.xs[1][b].permute(1, 2, 0).detach().cpu().numpy(),
                "box": bx, "rob": norm(r), "sho": norm(s),
                "breed": class_names[int(lg[b].argmax())],
            })
        print(f"  {len(cases)}/{len(sub)}", end="\r", flush=True)

    cases.sort(key=lambda c: c["delta"])
    picks = [("worst case", cases[0]),
             ("median case", cases[len(cases) // 2]),
             ("best case", cases[-1])]

    fig, axes = plt.subplots(3, 4, figsize=(16.5, 12.6))
    for ri, (label, c) in enumerate(picks):
        H, W = c["box"].shape
        ys, xs_ = np.where(c["box"] > 0.5)
        rect = (xs_.min(), ys.min(), xs_.max() - xs_.min(), ys.max() - ys.min())
        a = axes[ri]
        a[0].imshow(c["img"])
        a[0].add_patch(plt.Rectangle(rect[:2], rect[2], rect[3], fill=False,
                                     ec=AQUA, lw=3))
        a[1].imshow(c["env"])
        a[1].add_patch(plt.Rectangle(rect[:2], rect[2], rect[3], fill=False,
                                     ec=AQUA, lw=3, alpha=0.5))
        tint(a[2], c["img"], c["rob"], (0.16, 0.47, 0.84))
        tint(a[3], c["img"], c["sho"], (0.92, 0.41, 0.20))
        for k in range(4):
            a[k].add_patch(plt.Rectangle((0, 0), W - 1, H - 1, fill=False,
                                         ec="white", lw=0))
            bare(a[k])
        a[0].set_ylabel(f"{label}\n{c['breed']}", fontsize=15, color=INK2)
        a[2].set_xlabel(f"in box {c['in_box_rob']:.2f}", fontsize=14, color=BLUE)
        a[3].set_xlabel(f"in box {c['in_box_sho']:.2f}", fontsize=14, color=ORANGE)
        a[1].set_xlabel(f"robust − shortcut {c['delta']:+.2f}", fontsize=14, color=INK2)
    for t, k in zip(["Input  (green = annotated box)", "One environment\n(background restyled)",
                     "CDEA robust mask", "CDEA shortcut mask"], range(4)):
        axes[0][k].set_title(t, fontsize=17, color=[INK, INK2, BLUE, ORANGE][k],
                             fontweight="bold", pad=12)
    fig.text(0.5, 0.012,
             "Environments restyle only the background; the allocator is never told where "
             "the box is. Across 400 images the shortcut mask sits at the box area "
             "(0.556 vs 0.556 chance) while the robust mask reaches 0.726.",
             ha="center", fontsize=15, color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"R1_qualitative.{ext}", dpi=180, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("\nSaved", out / "R1_qualitative.png")
    print(f"deltas: worst {picks[0][1]['delta']:+.3f}  median {picks[1][1]['delta']:+.3f}  "
          f"best {picks[2][1]['delta']:+.3f}")


if __name__ == "__main__":
    main()
