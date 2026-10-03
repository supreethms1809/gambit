"""
scripts/eval_localization.py

Quantitative localization: do the allocated contrastive masks land on the actual
pathology? Works on any dataset that ships per-image segmentations — currently
HAM10000 (expert lesion outlines, Tschandl et al.) and brain tumor (Cheng et al.
tumor masks).

For a region mask ``m`` upsampled to pixel space and a binary target mask ``L``,

    target mass fraction  =  sum(m * L) / sum(m)

i.e. the share of allocated evidence mass that falls inside the pathology.

**Read the nulls before quoting any number from this script.** The uniform
baseline (a flat mask, which scores exactly the target's area fraction) is not a
sufficient control, because a mask can score well purely by being small and
centered. On HAM10000 that failure mode is severe: lesions are centered by
acquisition convention (centroid std ~0.05-0.07 of the frame), so a single fixed
center grid cell scores ~0.94 and *stays* there as the grid is refined. It beats
every measured CDEA configuration while doing no computation at all.

The honest control is ``cdea_unique_translated``: each unique mask scored against
a randomly translated copy of itself. That holds shape, budget and compactness
constant and scrambles only position, so it isolates *where* the mask is from
*what shape* it is. Prefer it to `headroom`, which normalizes against the uniform
baseline and inherits the same center-prior flaw.

Brain tumor is the better testbed for a spatial claim: tumor position genuinely
varies across patients (center-cell null ~0.12 vs chance ~0.016), leaving real
headroom. It does, however, need a fine grid — 70% of tumors are smaller than one
cell of a 7x7 grid, so use ``vit_b_16`` (14x14) or the measurement is meaningless.

Run from repo root::

    PYTHONPATH=. python scripts/eval_localization.py \\
        --checkpoint examples/out/checkpoints/ham10000_resnet18.pt --num_images 500
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.runner import CDEAExplainer
from core.hypotheses import TopMSelector
from core.device import get_device
from core.game_modes import resolve_contrastive_game
from core.reporting import save_json, save_rows_csv
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from base_evidence.occlusion_regions import OcclusionRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from examples.contrastive_explanation import (
    MEDICAL_SPLIT_ROOTS,
    TV_INPUT_SIZE,
    checkpoint_metadata,
    load_checkpoint,
)
from scripts.train_backbone import model_grid_size

OUT = REPO / "scripts" / "out"

# Datasets with per-image segmentations. Both use the same `<stem>_segmentation.png`
# convention, written by scripts/prepare_ham10000.py and scripts/prepare_brain_tumor.py.
SEG_ROOTS = {
    "ham10000": (REPO / "data" / "ham10000_raw" / "HAM10000_segmentations"
                 / "HAM10000_segmentations_lesion_tschandl"),
    "brain_tumor": REPO / "data" / "brain_tumor_raw" / "masks",
}
SEG_ROOT = SEG_ROOTS["ham10000"]  # backwards-compatible default


class SegmentedDataset(torch.utils.data.Dataset):
    """Val-split images paired with their expert segmentation."""

    def __init__(self, split_root: Path, seg_root: Path, size: int = TV_INPUT_SIZE):
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        self.inner = ImageFolder(root=str(split_root))
        self.seg_root = Path(seg_root)
        self.t_img = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        # Nearest keeps the mask binary through the resize.
        self.t_seg = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.samples = [
            (path, label) for path, label in self.inner.samples
            if (self.seg_root / f"{Path(path).stem}_segmentation.png").exists()
        ]
        n_missing = len(self.inner.samples) - len(self.samples)
        if n_missing:
            print(f"WARNING: {n_missing} val images have no segmentation mask (skipped)")
        if not self.samples:
            raise FileNotFoundError(
                f"no image in {split_root} has a matching mask in {seg_root}; "
                "check that the prepare_*.py script wrote masks there"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.samples[i]
        img = self.t_img(Image.open(path).convert("RGB"))
        seg = self.t_seg(Image.open(self.seg_root / f"{Path(path).stem}_segmentation.png"))
        return img, label, (seg > 0.5).float()[0]  # (H, W) binary


from evaluation.masks import mass_in as target_mass_fraction
from evaluation.masks import regions_to_pixels as _regions_to_pixels
from evaluation.nulls import random_translate


def main() -> None:
    parser = argparse.ArgumentParser(description="Score contrastive mask localization against lesion masks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_alloc_steps", type=int, default=50)
    parser.add_argument("--game_mode", type=str, default="mixed")
    parser.add_argument("--lambda_mass", type=float, default=2.0)
    parser.add_argument("--lambda_shared_sparse", type=float, default=0.0,
                        help="L1 penalty on the shared mask. At 0.0 it is in no penalty term at "
                             "all and inflates to blanket ~46%% of the grid at 0.99x "
                             "chance on base-evidence capture. See "
                             "docs/MEDICAL_RESULTS.md section 9a.")
    parser.add_argument("--evidence", type=str, default="gradcam", choices=["gradcam", "ig", "occlusion"],
                        help="Base evidence provider to score")
    parser.add_argument("--ig_steps", type=int, default=16,
                        help="Integrated Gradients interpolation steps (--evidence ig)")
    parser.add_argument("--seg_root", type=str, default=None,
                        help="Segmentation directory. Defaults to the checkpoint dataset's masks.")
    parser.add_argument("--out_dir", type=str, default=str(OUT),
                        help="Directory for result CSV/JSON (default: scripts/out)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random-translation null")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. 0 (default) avoids the MPS multiprocessing\n                             hang; raise only if data loading is genuinely the bottleneck.")
    parser.add_argument("--occlusion_mode", type=str, default="single",
                        choices=["single", "rise"],
                        help="single = one region at a time; rise = random multi-region\n                             masks, which handles redundant evidence on large targets")
    parser.add_argument("--occlusion_masks", type=int, default=512,
                        help="Number of random masks when --occlusion_mode rise")
    parser.add_argument("--grid", type=int, default=None,
                        help="Override the unit grid (NxN). Honest with --evidence ig or occlusion "
                             "(both are resolution-free); Grad-CAM is bound to the backbone's "
                             "feature map, so a finer grid there is interpolation.")
    parser.add_argument("--export_prefix", type=str, default="localization_ham10000")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dataset_name, class_names, num_classes = load_checkpoint(Path(args.checkpoint), device)
    if dataset_name not in SEG_ROOTS:
        raise ValueError(
            f"checkpoint is for '{dataset_name}'; this script needs per-image segmentations "
            f"(available for: {', '.join(sorted(SEG_ROOTS))})"
        )
    print(f"Loaded {args.checkpoint}: {dataset_name}, {meta['model_name']}, {num_classes} classes")

    seg_root = Path(args.seg_root) if args.seg_root else SEG_ROOTS[dataset_name]
    ds = SegmentedDataset(MEDICAL_SPLIT_ROOTS[dataset_name][1], seg_root)
    if args.num_images < len(ds):
        # Sample, don't slice: ImageFolder orders by class, so range(N) returns a
        # class-ordered prefix rather than a subset of the val distribution. That is
        # why earlier 200-image runs disagreed with the full-set numbers.
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
        ds = torch.utils.data.Subset(ds, idx[:args.num_images].tolist())
    # Default to 0 workers. Multiprocessing loaders deadlock against MPS in long runs
    # here: workers go idle, the main process blocks on a queue read that never
    # returns, and the job hangs at 0% CPU indefinitely rather than failing. Image
    # loading is nowhere near the bottleneck next to the allocator's forward passes,
    # so in-process loading costs almost nothing.
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    print(f"Scoring {len(ds)} val images against {seg_root.name}")

    # Grid follows the backbone: vit_b_16 gives 14x14, everything else 7x7. On brain
    # tumor a 7x7 grid cannot resolve the target at all (70% of tumors are smaller
    # than one cell), so warn rather than silently produce a meaningless number.
    grid_h, grid_w = model_grid_size(meta["model_name"])
    if args.grid is not None:
        # Grad-CAM's spatial resolution is the last feature map's; pooling it to a
        # finer grid upsamples, inventing detail the attribution never contained.
        # IG attributes at the pixel level, so any grid is a real choice there.
        if args.evidence not in ("ig", "occlusion") and args.grid > grid_h:
            raise ValueError(
                f"--grid {args.grid} exceeds Grad-CAM's native {grid_h}x{grid_w} for "
                f"{meta['model_name']}. That would fabricate resolution. Use --evidence ig."
            )
        grid_h = grid_w = args.grid
    print(f"Unit grid: {grid_h}x{grid_w} ({grid_h * grid_w} regions)")
    if dataset_name == "brain_tumor" and grid_h < 14:
        print("WARNING: brain tumor at a 7x7 grid — mean tumor area is 1.7% of frame vs "
              "2.04% per cell, so most tumors are sub-cell. Use --model vit_b_16 checkpoints.")
    rng = torch.Generator().manual_seed(args.seed)
    game_cfg = resolve_contrastive_game(args.game_mode)
    unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur")
    if args.evidence == "ig":
        base_evidence = IntegratedGradientsRegionsProvider(
            grid_h=grid_h, grid_w=grid_w, steps=args.ig_steps, baseline="zero",
        )
    elif args.evidence == "occlusion":
        # Perturbation-based: no gradients, works on any architecture, and shares the
        # removal intervention with the objective itself.
        base_evidence = OcclusionRegionsProvider(
            grid_h=grid_h, grid_w=grid_w, unit_space=unit_space,
            mode=args.occlusion_mode, n_masks=args.occlusion_masks, seed=args.seed,
        )
    else:
        base_evidence = GradCAMRegionsProvider(grid_h=grid_h, grid_w=grid_w)
    print(f"Base evidence: {args.evidence}")
    # Mirror examples/contrastive_explanation.py exactly so the scores describe the
    # same pipeline the figures come from.
    objective = ContrastiveObjective(
        lambda_suff=1.0,
        lambda_margin=game_cfg.lambda_margin,
        lambda_sparse=0.05,
        lambda_overlap=game_cfg.lambda_overlap,
        lambda_mass=args.lambda_mass,
        lambda_shared_sparse=args.lambda_shared_sparse,
    )
    allocator = OptimizationAllocator(
        objective,
        num_steps=args.num_alloc_steps,
        lr=0.2,
        use_shared=game_cfg.use_shared,
        lambda_disjoint=game_cfg.lambda_disjoint,
        lambda_partition=game_cfg.lambda_partition,
    )
    explainer = CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=TopMSelector(m=min(5, num_classes)),
        base_evidence=base_evidence,
        allocator=allocator,
        objective=objective,
        normalize_evidence=True,
        device=device,
    )

    scores: Dict[str, List[float]] = defaultdict(list)
    lesion_area: List[float] = []

    for bi, (x, y, seg) in enumerate(loader):
        x, seg = x.to(device), seg.to(device)
        B, _, H, W = x.shape

        expl = explainer.explain(x)
        K = expl.masks["unique"].shape[1]
        valid = expl.hypotheses.mask  # (B, K) — padded slots when K > num_classes

        # Rank 0 is the model's own top hypothesis; ranks 1.. are the foils, i.e. the
        # "rather than L" half of the contrastive claim. Scoring only rank 0 leaves that
        # half untested, so every rank is scored separately here.
        for k in range(K):
            sel = valid[:, k]
            if not bool(sel.any()):
                continue
            uni_k = _regions_to_pixels(expl.masks["unique"][:, k, :], grid_h, grid_w, H, W)
            base_k = _regions_to_pixels(expl.extras["evidence"][:, k, :], grid_h, grid_w, H, W)
            scores[f"cdea_unique_rank{k}"] += target_mass_fraction(uni_k[sel], seg[sel]).tolist()
            scores[f"base_evidence_rank{k}"] += target_mass_fraction(base_k[sel], seg[sel]).tolist()
            if k == 0:  # keep the original names so earlier runs stay comparable
                scores["cdea_unique"] += target_mass_fraction(uni_k[sel], seg[sel]).tolist()
                scores["base_evidence"] += target_mass_fraction(base_k[sel], seg[sel]).tolist()
                # Position-scrambling null: same mask, same budget, random location.
                rolled = random_translate(expl.masks["unique"][:, k, :].detach().cpu(), rng)
                rolled_px = _regions_to_pixels(rolled.to(device), grid_h, grid_w, H, W)
                scores["cdea_unique_translated"] += target_mass_fraction(
                    rolled_px[sel], seg[sel]).tolist()

        if "shared" in expl.masks:
            scores["cdea_shared"] += target_mass_fraction(
                _regions_to_pixels(expl.masks["shared"], grid_h, grid_w, H, W), seg).tolist()
        scores["uniform"] += target_mass_fraction(torch.ones(B, H, W, device=device), seg).tolist()
        # Degenerate reference: one fixed center cell, identical for every image, no
        # model and no computation. On HAM10000 this beats every measured method.
        # Built directly in pixel space as a sharp box — upsampling a one-hot region
        # bilinearly would smear it across neighbours and understate the null.
        center = torch.zeros(B, H, W, device=device)
        ch, cw = H // grid_h, W // grid_w
        y0, x0 = (grid_h // 2) * ch, (grid_w // 2) * cw
        center[:, y0:y0 + ch, x0:x0 + cw] = 1.0
        scores["center_cell"] += target_mass_fraction(center, seg).tolist()
        lesion_area += seg.mean(dim=(1, 2)).tolist()

        if (bi + 1) % 5 == 0:
            print(f"  {len(scores['uniform'])}/{len(ds)} images")

    def _stat(vals: List[float]) -> Dict[str, float]:
        t = torch.tensor(vals)
        return {"mean": t.mean().item(), "std": t.std().item(), "n": len(vals)}

    chance = sum(lesion_area) / len(lesion_area)

    def _headroom(mean: float) -> float:
        """Share of the available above-chance range captured. 0 = chance, 1 = perfect."""
        return (mean - chance) / max(1.0 - chance, 1e-8)

    print("\n--- Target mass fraction (share of mask mass inside the pathology) ---")
    print(f"{'method':<24}{'mean':>8}{'std':>8}{'headroom':>10}{'n':>7}")
    rows = []
    ranked = sorted(k for k in scores if "_rank" in k)
    for name in ["uniform", "center_cell", "base_evidence", "cdea_shared",
                 "cdea_unique", "cdea_unique_translated"] + ranked:
        if not scores.get(name):
            continue
        s = _stat(scores[name])
        print(f"{name:<24}{s['mean']:>8.4f}{s['std']:>8.4f}{_headroom(s['mean']):>10.3f}{s['n']:>7}")
        rows.append({"method": name, "headroom": _headroom(s["mean"]), **s})

    base_m = _stat(scores["base_evidence"])["mean"] if scores["base_evidence"] else 0.0
    uni_m = _stat(scores["uniform"])["mean"] if scores["uniform"] else 0.0
    cdea_m = _stat(scores["cdea_unique"])["mean"] if scores["cdea_unique"] else 0.0
    print(f"\nmean lesion area fraction (random-mask baseline): {chance:.4f}")

    # Paired comparison: same image scored by both methods, so pair the differences.
    # rank0-vs-rank1 asks whether the predicted class's mask localizes differently
    # from its top foil's — if they match, the contrastive split is not doing its job.
    paired = []
    comparisons = [("cdea_unique", "base_evidence"), ("cdea_unique", "uniform")]
    # The null that actually matters: same mask, same budget, position scrambled.
    if scores.get("cdea_unique_translated"):
        comparisons.append(("cdea_unique", "cdea_unique_translated"))
    if scores.get("center_cell"):
        comparisons.append(("cdea_unique", "center_cell"))
    if scores.get("cdea_unique_rank1"):
        comparisons.append(("cdea_unique_rank0", "cdea_unique_rank1"))
    for a, b in comparisons:
        if len(scores[a]) != len(scores[b]):
            print(f"{a} vs {b}: skipped (unequal n: {len(scores[a])} vs {len(scores[b])})")
            continue
        d = torch.tensor(scores[a]) - torch.tensor(scores[b])
        n = d.numel()
        mean_d = d.mean().item()
        se = (d.std(unbiased=True) / (n ** 0.5)).item()
        t = mean_d / se if se > 0 else float("nan")
        win = (d > 0).float().mean().item()
        print(f"{a} vs {b}: delta={mean_d:+.4f}  95%CI=[{mean_d - 1.96*se:+.4f},{mean_d + 1.96*se:+.4f}]  "
              f"t={t:.2f}  win_rate={win:.3f}  n={n}")
        paired.append({"comparison": f"{a}_vs_{b}", "delta": mean_d, "se": se,
                       "ci_low": mean_d - 1.96 * se, "ci_high": mean_d + 1.96 * se,
                       "t": t, "win_rate": win, "n": n})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(out_dir / f"{args.export_prefix}.csv", rows)
    save_json(out_dir / f"{args.export_prefix}.json", {
        "rows": rows,
        "paired": paired,
        "per_image": scores,
        "checkpoint": args.checkpoint,
        "dataset": dataset_name,
        "model_name": meta["model_name"],
        "grid": [grid_h, grid_w],
        "seed": args.seed,
        "num_images": len(ds),
        "num_alloc_steps": args.num_alloc_steps,
        "game_mode": args.game_mode,
        "evidence": args.evidence,
        "mean_target_area_fraction": sum(lesion_area) / len(lesion_area),
        # Retained under the old key so earlier runs stay diffable.
        "mean_lesion_area_fraction": sum(lesion_area) / len(lesion_area),
    })
    print(f"\nSaved {out_dir / (args.export_prefix + '.csv')}")


if __name__ == "__main__":
    main()
