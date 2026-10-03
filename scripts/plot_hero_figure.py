"""
scripts/plot_hero_figure.py

The opening slide: one lesion the model cannot decide about, and what it looked
at for each candidate diagnosis.

Renders a single row — input, the raw evidence for the top-2 candidates (which
look almost identical, and that is the point), and the CDEA split into shared
(red) and unique (green). Unlike the diagnostic figures in ``examples/out``,
this is sized and typeset to be projected: no axis ticks, few panels, large type.

The sample is *found*, not hard-coded: the script scans the validation split for
the case where the model is most torn between two diagnoses, optionally
restricted to a class pair (``--prefer melanoma,melanocytic nevus``), so the
figure keeps working when the checkpoint changes.

Run from repo root::

    PYTHONPATH=. python scripts/plot_hero_figure.py \\
        --checkpoint examples/out/checkpoints/ham10000_efficientnet_v2_s.pt \\
        --out_dir results/medical_presentation/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.runner import CDEAExplainer
from core.hypotheses import TopMSelector
from core.device import get_device
from core.game_modes import resolve_contrastive_game
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from examples.contrastive_explanation import (
    MEDICAL_SPLIT_ROOTS,
    TV_INPUT_SIZE,
    checkpoint_metadata,
    load_checkpoint,
)
from scripts.train_backbone import model_grid_size

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


def to_pixels(m: torch.Tensor, gh: int, gw: int, h: int, w: int) -> np.ndarray:
    pm = m.view(1, 1, gh, gw)
    return F.interpolate(pm, size=(h, w), mode="bilinear",
                         align_corners=False).squeeze().cpu().numpy()


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def main() -> None:
    p = argparse.ArgumentParser(description="Render the hero figure for the talk")
    p.add_argument("--checkpoint", type=str,
                   default="examples/out/checkpoints/ham10000_efficientnet_v2_s.pt")
    p.add_argument("--evidence", type=str, default="gradcam", choices=["gradcam", "ig"])
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--game_mode", type=str, default="mixed")
    p.add_argument("--num_alloc_steps", type=int, default=50)
    p.add_argument("--scan_images", type=int, default=256,
                   help="How many val images to search for the most ambiguous case")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--prefer", type=str, default=None,
                   help="Comma-separated class pair to prefer, e.g. 'melanoma,melanocytic nevus'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=str(REPO / "results" / "medical_presentation" / "figures"))
    p.add_argument("--name", type=str, default="F1_hero")
    args = p.parse_args()

    device = get_device()
    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dataset_name, class_names, num_classes = load_checkpoint(Path(args.checkpoint), device)
    print(f"Loaded {dataset_name} / {meta['model_name']} / {num_classes} classes")

    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    t = transforms.Compose([
        transforms.Resize((TV_INPUT_SIZE, TV_INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(root=str(MEDICAL_SPLIT_ROOTS[dataset_name][1]), transform=t)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
    sub = torch.utils.data.Subset(ds, idx[:min(args.scan_images, len(ds))].tolist())
    loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False)

    prefer: Optional[List[str]] = None
    if args.prefer:
        prefer = [s.strip() for s in args.prefer.split(",")]
        missing = [c for c in prefer if c not in class_names]
        if missing:
            raise ValueError(f"--prefer names unknown classes {missing}; have {list(class_names)}")

    # --- find the most ambiguous image: smallest gap between the top two classes
    best = None
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=-1).cpu()
            top2 = probs.topk(2, dim=-1)
            for b in range(x.shape[0]):
                gap = (top2.values[b, 0] - top2.values[b, 1]).item()
                pair = [class_names[int(top2.indices[b, k])] for k in range(2)]
                if prefer and set(pair) != set(prefer):
                    continue
                if best is None or gap < best["gap"]:
                    best = {"gap": gap, "x": x[b:b + 1].clone(), "pair": pair,
                            "probs": top2.values[b].tolist(),
                            "true": class_names[int(y[b])]}
    if best is None:
        raise SystemExit("no image matched --prefer; try dropping it or raising --scan_images")
    print(f"Selected: {best['pair'][0]} {best['probs'][0]:.3f} vs "
          f"{best['pair'][1]} {best['probs'][1]:.3f}  (true: {best['true']})")

    # --- explain that one image
    gh, gw = model_grid_size(meta["model_name"])
    cfg = resolve_contrastive_game(args.game_mode)
    unit_space = VisionGridUnitSpace(gh, gw, baseline="blur")
    base_evidence = (
        IntegratedGradientsRegionsProvider(grid_h=gh, grid_w=gw, steps=args.ig_steps,
                                           baseline="zero")
        if args.evidence == "ig" else GradCAMRegionsProvider(grid_h=gh, grid_w=gw))
    objective = ContrastiveObjective(lambda_suff=1.0, lambda_margin=cfg.lambda_margin,
                                     lambda_sparse=0.05, lambda_overlap=cfg.lambda_overlap,
                                     lambda_mass=2.0)
    allocator = OptimizationAllocator(objective, num_steps=args.num_alloc_steps, lr=0.2,
                                      use_shared=cfg.use_shared,
                                      lambda_disjoint=cfg.lambda_disjoint,
                                      lambda_partition=cfg.lambda_partition)
    explainer = CDEAExplainer(model=model, unit_space=unit_space,
                              selector=TopMSelector(m=min(5, num_classes)),
                              base_evidence=base_evidence, allocator=allocator,
                              objective=objective, normalize_evidence=True, device=device)
    expl = explainer.explain(best["x"].to(device))

    img = best["x"][0].permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]
    ev0 = norm(to_pixels(expl.extras["evidence"][0, 0], gh, gw, H, W))
    ev1 = norm(to_pixels(expl.extras["evidence"][0, 1], gh, gw, H, W))
    uni0 = norm(to_pixels(expl.masks["unique"][0, 0], gh, gw, H, W))
    shared = (norm(to_pixels(expl.masks["shared"][0], gh, gw, H, W))
              if "shared" in expl.masks else np.zeros((H, W)))

    # --- render
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    n0, n1 = best["pair"][0], best["pair"][1]
    p0, p1 = best["probs"]

    axes[0].imshow(img)
    axes[0].set_title("One lesion", fontsize=20, color=INK, pad=12)
    axes[0].text(0.5, -0.06, f"true: {best['true']}", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=15, color=INK2)

    for ax, ev, name, prob in [(axes[1], ev0, n0, p0), (axes[2], ev1, n1, p1)]:
        ax.imshow(img)
        ax.imshow(ev, cmap="inferno", alpha=0.62)
        ax.set_title(f"Evidence for\n{name}", fontsize=20, color=INK, pad=12)
        ax.text(0.5, -0.06, f"model says {prob:.0%}", transform=ax.transAxes,
                ha="center", va="top", fontsize=15, color=INK2)

    # Shared (red) vs unique (green) over the image. Both masks are continuous and
    # the shared one is diffuse, so drawing them raw floods the frame and buries the
    # unique region. Gamma-compress each toward its own strongest mass — shared
    # harder, since it is the wash — so the split stays legible when projected.
    sh_d = np.clip(shared, 0, 1) ** 2.2
    un_d = np.clip(uni0, 0, 1) ** 1.1
    ov = np.zeros((H, W, 4))
    ov[..., 0] = sh_d
    ov[..., 1] = un_d
    ov[..., 3] = np.clip(np.maximum(sh_d * 0.55, un_d * 0.95), 0, 1)
    axes[3].imshow(img)
    axes[3].imshow(ov)
    axes[3].set_title("CDEA splits it", fontsize=20, color=INK, pad=12)
    axes[3].text(0.5, -0.06, "red = shared    green = unique to " + n0,
                 transform=axes[3].transAxes, ha="center", va="top",
                 fontsize=15, color=INK2)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{args.name}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"Saved {out_dir / (args.name + '.png')}")


if __name__ == "__main__":
    main()
