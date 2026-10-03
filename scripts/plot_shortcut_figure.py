"""
scripts/plot_shortcut_figure.py

Qualitative figure for the synthetic-shortcut benchmark.

The quantitative result (unique evidence lands on the planted patch at 10-18x chance,
while a model that never saw the patch sits at chance) is the strongest evidence in the
project, but it is a table. This renders the same claim as pictures: identical input
images, two models, and the evidence moves only when the model actually uses the patch.

Both models are trained here and cached, so the figure is reproducible from scratch::

    PYTHONPATH=. python scripts/plot_shortcut_figure.py \\
        --out_dir results/shortcut/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.runner import CDEAExplainer
from core.hypotheses import TopMSelector
from core.device import get_device
from core.game_modes import resolve_contrastive_game
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from scripts.ablation_contrastive import _build_model
from scripts.eval_shortcut import (
    ShortcutDataset, cifar, make_patch, train, regions_to_pixels, TV,
)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, CRITICAL = "#2a78d6", "#d03b3b"


def get_model(tag, rate, patch, args, device):
    """Train (or load a cached) model with or without the planted shortcut."""
    ck = Path(args.cache_dir) / f"shortcut_{tag}_r{rate}_s{args.seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    m = _build_model("resnet18", 10, pretrained=True).to(device)
    if ck.exists() and not args.force:
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        print(f"  loaded cached {tag} model")
        return m.eval()
    ds = ShortcutDataset(cifar(True, args.train_images, args.seed), args.target_class,
                         rate, patch, seed=args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"  training {tag} model (shortcut rate {rate})")
    m = train(m, dl, args.epochs, args.lr, device)
    torch.save(m.state_dict(), ck)
    return m.eval()


def explainer_for(model, unit_space, gh, gw, args, device):
    cfg = resolve_contrastive_game("mixed")
    prov = IntegratedGradientsRegionsProvider(grid_h=gh, grid_w=gw, steps=args.ig_steps,
                                              baseline="zero")
    obj = ContrastiveObjective(lambda_suff=1.0, lambda_margin=cfg.lambda_margin,
                               lambda_sparse=0.05, lambda_overlap=cfg.lambda_overlap,
                               lambda_mass=2.0)
    alloc = OptimizationAllocator(obj, num_steps=args.num_alloc_steps, lr=0.2,
                                  use_shared=cfg.use_shared,
                                  lambda_disjoint=cfg.lambda_disjoint,
                                  lambda_partition=cfg.lambda_partition)
    return CDEAExplainer(model=model, unit_space=unit_space, selector=TopMSelector(m=5),
                         base_evidence=prov, allocator=alloc, objective=obj,
                         normalize_evidence=True, device=device)


def unique_for_target(expl_out, target, device):
    """The unique mask belonging to the planted class, wherever it sits in the top-K."""
    ids = expl_out.hypotheses.ids
    B = ids.shape[0]
    hit = ids == torch.full((B,), target, device=ids.device).unsqueeze(1)
    slot = hit.float().argmax(dim=1)
    ar = torch.arange(B, device=ids.device)
    return (expl_out.masks["unique"][ar, slot], expl_out.extras["evidence"][ar, slot],
            hit.any(dim=1))


def norm(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def main() -> None:
    p = argparse.ArgumentParser(description="Qualitative shortcut-benchmark figure")
    p.add_argument("--target_class", type=int, default=0)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--patch_type", type=str, default="solid")
    p.add_argument("--grid", type=int, default=14)
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--num_alloc_steps", type=int, default=50)
    p.add_argument("--train_images", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--cache_dir", type=str, default=str(REPO / "results" / "shortcut" / "models"))
    p.add_argument("--out_dir", type=str, default=str(REPO / "results" / "shortcut" / "figures"))
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    patch = make_patch(args.patch_type, args.patch_size, g)

    print("Models:")
    planted = get_model("planted", 1.0, patch, args, device)
    control = get_model("control", 0.0, patch, args, device)

    gh = gw = args.grid
    unit_space = VisionGridUnitSpace(gh, gw, baseline="blur")
    ex_p = explainer_for(planted, unit_space, gh, gw, args, device)
    ex_c = explainer_for(control, unit_space, gh, gw, args, device)

    # Patched val images of the planted class.
    va = ShortcutDataset(cifar(False, 400, args.seed + 1), args.target_class, 1.0,
                         patch, seed=args.seed + 1)
    keep = [i for i in range(len(va)) if va[i][2].sum() > 0][:args.n_examples]
    xs = torch.stack([va[i][0] for i in keep]).to(device)
    pms = torch.stack([va[i][2] for i in keep]).to(device)
    print(f"rendering {len(keep)} examples")

    out_p, out_c = ex_p.explain(xs), ex_c.explain(xs)
    up, bp, _ = unique_for_target(out_p, args.target_class, device)
    uc, _, _ = unique_for_target(out_c, args.target_class, device)
    H = W = TV
    up = regions_to_pixels(up, gh, gw, H, W).detach().cpu().numpy()
    bp = regions_to_pixels(bp, gh, gw, H, W).detach().cpu().numpy()
    uc = regions_to_pixels(uc, gh, gw, H, W).detach().cpu().numpy()
    imgs = xs.permute(0, 2, 3, 1).cpu().numpy()
    pmn = pms.cpu().numpy()

    plt.rcParams.update({"font.family": "sans-serif",
                        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    n = len(keep)
    fig, axes = plt.subplots(n, 4, figsize=(15.5, 4.0 * n), squeeze=False)
    titles = ["Input (patch planted\nat a random position)",
              "Raw evidence\n(model that learned it)",
              "CDEA unique\n(model that learned it)",
              "CDEA unique\n(model that never saw it)"]

    for r in range(n):
        # Outline the ground-truth patch on every panel, so "did it find it?" is visual.
        ys, xsx = np.nonzero(pmn[r])
        y0, y1, x0, x1 = ys.min(), ys.max(), xsx.min(), xsx.max()
        for c, (data, cmap) in enumerate([(None, None), (bp[r], "inferno"),
                                          (up[r], "inferno"), (uc[r], "inferno")]):
            ax = axes[r][c]
            ax.imshow(imgs[r])
            if data is not None:
                ax.imshow(norm(data), cmap=cmap, alpha=0.65)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="#00ff88", lw=2.5))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=16, color=INK, pad=12)

    fig.text(0.5, 0.005,
             "green outline = the planted patch (ground truth).  Evidence concentrates on it "
             "only when the model actually uses it.",
             ha="center", fontsize=14, color=INK2)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F7_shortcut_hero.{ext}", dpi=180, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"Saved {out / 'F7_shortcut_hero.png'}")


if __name__ == "__main__":
    main()
