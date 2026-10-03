"""
scripts/plot_talk_figures_medical.py

The three picture slides of the lightning talk, rendered on a medical dataset so
the deck matches the submitted abstract (HAM10000, brain tumor MRI) instead of
the Stanford Dogs / Pets figures it was built from.

One ambiguous validation image is *found* (smallest top-1 / top-2 gap, optionally
restricted to a class pair) and the same image carries all three figures, so the
talk tells one story end to end:

  S2_problem   input + the two raw evidence maps, with their cosine similarity
               -- "each map is computed alone, and they land on the same pixels"
  S4_method    input + one unique mask + keep-and-blur, with the logit it scores
               -- "every number comes off that one masked forward pass"
  S6_hero      mask row and overlay row: input, raw evidence x2, CDEA unique x2
               -- "after allocation the two diagnoses stop pointing at one place"

Aspect ratios match the picture boxes already on the slides, so each PNG drops
into the existing placeholder without re-cropping.

Run from repo root::

    PYTHONPATH=. python scripts/plot_talk_figures_medical.py \\
        --checkpoint examples/out/checkpoints/ham10000_resnet18.pt \\
        --prefer "melanoma,melanocytic nevus" \\
        --out_dir results/medical_presentation/talk_figures
"""
from __future__ import annotations

import argparse
import json
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
from scripts.ablation_contrastive import compute_metrics

# Same categorical slots as scripts/plot_presentation_figures.py, so the picture
# slides and the bar charts read as one deck.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"

# Class names are the folder names; these are what a clinician would say.
PRETTY = {
    "melanocytic nevus": "melanocytic nevus",
    "melanoma": "melanoma",
    "benign keratosis": "benign keratosis",
    "basal cell carcinoma": "basal cell carcinoma",
    "actinic keratosis": "actinic keratosis",
    "dermatofibroma": "dermatofibroma",
    "vascular lesion": "vascular lesion",
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary tumor",
}
SUBJECT = {"ham10000": "One lesion", "brain_tumor": "One scan"}
# The five-panel hero runs out of width with the full clinical names.
SHORT = {
    "melanocytic nevus": "nevus", "melanoma": "melanoma",
    "benign keratosis": "benign keratosis", "basal cell carcinoma": "basal cell ca.",
    "actinic keratosis": "actinic keratosis", "dermatofibroma": "dermatofibroma",
    "vascular lesion": "vascular lesion", "glioma": "glioma",
    "meningioma": "meningioma", "pituitary tumor": "pituitary",
}
ARROW = r"$\rightarrow$"  # Helvetica Neue has no U+2192; mathtext always renders


def to_pixels(m: torch.Tensor, gh: int, gw: int, h: int, w: int) -> np.ndarray:
    pm = m.reshape(1, 1, gh, gw)
    return F.interpolate(pm, size=(h, w), mode="bilinear",
                         align_corners=False).squeeze().detach().cpu().numpy()


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def pair_overlap(m: torch.Tensor) -> float:
    """Normalized inner product between the two top hypotheses' region masks."""
    a, b = m[0], m[1]
    a = a / a.sum().clamp_min(1e-8)
    b = b / b.sum().clamp_min(1e-8)
    return float((a * b).sum() * m.shape[-1])


def main() -> None:
    p = argparse.ArgumentParser(description="Render the picture slides on a medical dataset")
    p.add_argument("--checkpoint", type=str,
                   default="examples/out/checkpoints/ham10000_resnet18.pt")
    p.add_argument("--evidence", type=str, default="gradcam", choices=["gradcam", "ig"])
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--game_mode", type=str, default="mixed")
    p.add_argument("--lambda_shared_sparse", type=float, default=0.25)
    p.add_argument("--num_alloc_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--scan_images", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--prefer", type=str, default=None,
                   help="Comma-separated class pair to prefer, e.g. 'melanoma,melanocytic nevus'")
    p.add_argument("--require_true_in_pair", action="store_true", default=True,
                   help="Only consider images whose true label is one of the top-2 "
                        "(default: on, so the figure is not built on a misclassification)")
    p.add_argument("--allow_misclassified", dest="require_true_in_pair",
                   action="store_false")
    p.add_argument("--set_overlap", type=str, default=None,
                   help="'base,optimized' set-wide overlap to quote in the hero caption "
                        "instead of this image's own values, e.g. '0.447,0.069'")
    p.add_argument("--candidate_pool", type=int, default=12,
                   help="Among the N most ambiguous candidates, take the one whose two "
                        "evidence maps agree most")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default=str(REPO / "results" / "medical_presentation" / "talk_figures"))
    p.add_argument("--tag", type=str, default=None,
                   help="Filename suffix; defaults to <dataset>_<model>_<evidence>")
    args = p.parse_args()

    device = get_device()
    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dataset_name, class_names, num_classes = load_checkpoint(Path(args.checkpoint), device)
    model_name = meta["model_name"]
    tag = args.tag or f"{dataset_name}_{model_name}_{args.evidence}"
    print(f"Loaded {dataset_name} / {model_name} / {num_classes} classes")

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

    # One pass over the scan set with the real evidence provider: it gives the
    # per-image top-2 cosine (the number the problem slide rests on, reported as a
    # set-wide mean rather than a single lucky pair) and the candidate pool at once.
    gh, gw = model_grid_size(model_name)
    selector = TopMSelector(m=min(5, num_classes))
    provider = (
        IntegratedGradientsRegionsProvider(grid_h=gh, grid_w=gw, steps=args.ig_steps,
                                           baseline="zero")
        if args.evidence == "ig" else GradCAMRegionsProvider(grid_h=gh, grid_w=gw))

    cosines: List[float] = []
    cands: List[dict] = []
    for x_b, y_b in loader:
        x_b = x_b.to(device)
        with torch.no_grad():
            logits = model(x_b)
            probs = torch.softmax(logits, dim=-1)
            hyps = selector.select(logits, probs)
        x_g = x_b.detach().clone().requires_grad_(True)
        ev_b = provider.explain(x_g, model, hyps).detach()
        ev_b = ev_b / ev_b.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cos_b = F.cosine_similarity(ev_b[:, 0], ev_b[:, 1], dim=-1).cpu()
        top2 = probs.topk(2, dim=-1)
        for b in range(x_b.shape[0]):
            cosines.append(float(cos_b[b]))
            pair = [class_names[int(top2.indices[b, k])] for k in range(2)]
            true = class_names[int(y_b[b])]
            if prefer and set(pair) != set(prefer):
                continue
            if args.require_true_in_pair and true not in pair:
                continue
            cands.append({"gap": float(top2.values[b, 0] - top2.values[b, 1]),
                          "cos": float(cos_b[b]), "x": x_b[b:b + 1].detach().cpu().clone(),
                          "pair": pair, "probs": top2.values[b].tolist(), "true": true})
    if not cands:
        raise SystemExit("no image matched the filters; drop --prefer/--require_true_in_pair "
                         "or raise --scan_images")
    mean_cos = float(np.mean(cosines))

    # Among the most ambiguous cases, take the one whose two evidence maps agree
    # most — the clearest instance of the failure the talk is about.
    cands.sort(key=lambda d: d["gap"])
    pool = cands[:max(1, min(args.candidate_pool, len(cands)))]
    best = max(pool, key=lambda d: d["cos"])
    print(f"{len(cands)} candidates, pool {len(pool)}, set-wide mean top-2 cosine {mean_cos:.3f}")

    gh, gw = model_grid_size(model_name)
    n0, n1 = [PRETTY.get(c, c) for c in best["pair"]]
    p0, p1 = best["probs"]
    print(f"Selected: {n0} {p0:.3f} vs {n1} {p1:.3f}  (true: {best['true']}, "
          f"cosine {best['cos']:.3f})")

    cfg = resolve_contrastive_game(args.game_mode)
    unit_space = VisionGridUnitSpace(gh, gw, baseline="blur")
    provider = (
        IntegratedGradientsRegionsProvider(grid_h=gh, grid_w=gw, steps=args.ig_steps,
                                           baseline="zero")
        if args.evidence == "ig" else GradCAMRegionsProvider(grid_h=gh, grid_w=gw))
    objective = ContrastiveObjective(lambda_suff=1.0, lambda_margin=cfg.lambda_margin,
                                     lambda_sparse=0.05, lambda_overlap=cfg.lambda_overlap,
                                     lambda_mass=2.0,
                                     lambda_shared_sparse=args.lambda_shared_sparse)
    allocator = OptimizationAllocator(objective, num_steps=args.num_alloc_steps, lr=args.lr,
                                      use_shared=cfg.use_shared,
                                      lambda_disjoint=cfg.lambda_disjoint,
                                      lambda_partition=cfg.lambda_partition)
    explainer = CDEAExplainer(model=model, unit_space=unit_space,
                              selector=selector,
                              base_evidence=provider, allocator=allocator,
                              objective=objective, normalize_evidence=True, device=device)
    x = best["x"].to(device)
    expl = explainer.explain(x)

    ev = expl.extras["evidence"][0].detach()          # (K, R)
    uni = expl.masks["unique"][0].detach()            # (K, R)
    shared = expl.masks["shared"][0].detach() if "shared" in expl.masks else None

    cos = best["cos"]
    # Same metric function the reported aggregates use, so the per-image numbers on
    # the slide sit on the same scale as the table.
    m_base = compute_metrics(x, model, unit_space, expl.hypotheses,
                             expl.extras["evidence"].detach())
    m_opt = compute_metrics(x, model, unit_space, expl.hypotheses,
                            expl.masks["unique"].detach(),
                            expl.masks.get("shared").detach()
                            if "shared" in expl.masks else None)
    ov_base, ov_opt = m_base["overlap"], m_opt["overlap"]

    # What the masked forward pass actually scores, for the method slide.
    ids = expl.hypotheses.ids[0]
    m_keep = uni[0:1] + (shared.unsqueeze(0) if shared is not None else 0.0)
    with torch.no_grad():
        logit_full = float(model(x)[0, ids[0]])
        logit_kept = float(model(unit_space.keep(x, m_keep.clamp(0, 1)))[0, ids[0]])
        margin_full = float(model(x)[0, ids[0]] - model(x)[0, ids[1]])
        z = model(unit_space.keep(x, m_keep.clamp(0, 1)))[0]
        margin_kept = float(z[ids[0]] - z[ids[1]])

    img = best["x"][0].permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]
    ev0, ev1 = norm(to_pixels(ev[0], gh, gw, H, W)), norm(to_pixels(ev[1], gh, gw, H, W))
    un0, un1 = norm(to_pixels(uni[0], gh, gw, H, W)), norm(to_pixels(uni[1], gh, gw, H, W))
    kept = unit_space.keep(x, m_keep.clamp(0, 1))[0].permute(1, 2, 0).detach().cpu().numpy()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev_label = "Grad-CAM" if args.evidence == "gradcam" else "Integrated Gradients"

    def save(fig, name: str) -> None:
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}_{tag}.{ext}", dpi=200, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
        print("Saved", out / f"{name}_{tag}.png")

    # ---- S2: the problem. Two maps, computed apart, landing in one place.
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    ax[0].imshow(img)
    ax[0].set_title("Input", fontsize=22, color=INK, fontweight="bold", pad=12)
    for a, e, nm, pr in [(ax[1], ev0, n0, p0), (ax[2], ev1, n1, p1)]:
        a.imshow(img)
        a.imshow(e, cmap="inferno", alpha=0.62)
        a.set_title(f"{ev_label}: {nm}  ({pr:.0%})", fontsize=20, color=ORANGE,
                    fontweight="bold", pad=12)
    for a in ax:
        bare(a)
    fig.text(0.5, 0.015,
             f"Two maps, each computed without reference to the other "
             f"— and they land on the same pixels "
             f"(cosine {cos:.2f}; {mean_cos:.2f} on average across the validation set).",
             ha="center", fontsize=19, color=INK)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "S2_problem")

    # ---- S4: the intervention every number is measured through.
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 5.8))
    ax[0].imshow(img)
    ax[0].set_title("1.  input", fontsize=21, color=INK, fontweight="bold", pad=12)
    ax[1].imshow(np.stack([np.zeros_like(un0), np.zeros_like(un0), un0], axis=-1) ** 0.85)
    ax[1].set_title(f"2.  mask for “{n0}”", fontsize=21, color=BLUE,
                    fontweight="bold", pad=12)
    ax[2].imshow(kept)
    ax[2].set_title("3.  keep it, blur the rest", fontsize=21, color=INK,
                    fontweight="bold", pad=12)
    for a in ax:
        bare(a)
    fig.text(0.5, 0.015,
             f"{ARROW}  re-run the frozen classifier:   "
             f"{SHORT.get(n0, n0)} − {SHORT.get(n1, n1)} = {margin_kept:+.2f}"
             f"     vs  {margin_full:+.2f} on the full image",
             ha="center", fontsize=19, color=INK)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "S4_method")

    # ---- S6: the result. Mask row over overlay row, five columns.
    fig, ax = plt.subplots(2, 5, figsize=(19.0, 8.1))
    cols = [
        ("Input", INK, None, None),
        (f"{ev_label}: {n0}", ORANGE, ev0, "inferno"),
        (f"{ev_label}: {n1}", ORANGE, ev1, "inferno"),
        (f"CDEA unique: {SHORT.get(n0, n0)}", BLUE, un0, "blue"),
        (f"CDEA unique: {SHORT.get(n1, n1)}", BLUE, un1, "blue"),
    ]
    for c, (title, colour, field, cmap) in enumerate(cols):
        top, bot = ax[0, c], ax[1, c]
        if field is None:
            top.imshow(img)
            bot.imshow(img)
        elif cmap == "blue":
            top.imshow(np.stack([np.zeros_like(field), np.zeros_like(field), field],
                                axis=-1) ** 0.85)
            bot.imshow(img)
            ovl = np.zeros((H, W, 4))
            ovl[..., 2] = 1.0
            ovl[..., 3] = np.clip(field ** 0.9, 0, 1) * 0.85
            bot.imshow(ovl)
        else:
            top.imshow(field, cmap=cmap)
            bot.imshow(img)
            bot.imshow(field, cmap=cmap, alpha=0.62)
        top.set_title(title, fontsize=19, color=colour, fontweight="bold", pad=12)
        bare(top)
        bare(bot)
    ax[0, 0].set_ylabel("mask", fontsize=17, color=MUTED, fontweight="bold")
    ax[1, 0].set_ylabel("overlay", fontsize=17, color=MUTED, fontweight="bold")
    dataset_label = {"ham10000": "HAM10000", "brain_tumor": "Brain tumor MRI"}[dataset_name]
    if args.set_overlap:
        ob, oo = (float(v) for v in args.set_overlap.split(","))
        scope = "across the validation set"
    else:
        ob, oo, scope = ov_base, ov_opt, "on this image"
    fig.text(0.5, 0.015,
             f"{dataset_label} — {n0} {p0:.0%} vs {n1} {p1:.0%} (true: "
             f"{PRETTY.get(best['true'], best['true'])}).    Blue = evidence unique to "
             f"that class.    Overlap {ob:.3f} {ARROW} {oo:.3f} {scope}.",
             ha="center", fontsize=18, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, "S6_hero")

    summary = {
        "checkpoint": args.checkpoint, "dataset": dataset_name, "model": model_name,
        "evidence": args.evidence, "grid": [gh, gw], "game_mode": args.game_mode,
        "lambda_shared_sparse": args.lambda_shared_sparse,
        "num_alloc_steps": args.num_alloc_steps, "seed": args.seed,
        "scan_images": args.scan_images,
        "selected": {"true": best["true"], "pair": [n0, n1], "probs": [p0, p1],
                     "top1_top2_gap": best["gap"]},
        "evidence_cosine_top2": {"selected_image": cos, "scan_set_mean": mean_cos,
                                 "scan_set_n": len(cosines)},
        "metrics_this_image": {"base_evidence": m_base, "optimized": m_opt},
        "set_overlap_quoted": args.set_overlap,
        "logit_top1": {"full_image": logit_full, "kept_mask": logit_kept},
        "margin_top1_vs_top2": {"full_image": margin_full, "kept_mask": margin_kept},
    }
    (out / f"talk_figures_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["selected"], indent=2))
    print(f"cosine {cos:.3f} | overlap {ov_base:.3f} -> {ov_opt:.3f} | "
          f"logit {logit_full:+.2f} -> {logit_kept:+.2f} | "
          f"margin {margin_full:+.2f} -> {margin_kept:+.2f}")


if __name__ == "__main__":
    main()
