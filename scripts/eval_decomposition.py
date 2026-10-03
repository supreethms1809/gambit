"""
scripts/eval_decomposition.py

Does the shared/unique decomposition mean what the objective claims it means?

Localization against expert segmentations cannot answer this. A lesion outline
marks *where the lesion is*, and on HAM10000 all seven classes are lesions — so
the outline is a ground truth for **shared** evidence, not for class-**unique**
evidence. Nothing in the dataset annotates what distinguishes melanoma from a
benign nevus. Scoring a contrastive decomposition against a class-agnostic mask
therefore cannot confirm or refute the claim being made.

This script tests the claim directly, against the model, with no annotation:

**Test A — does the split carry the discrimination?**
  ``shared`` is defined as evidence every candidate relies on, so keeping only the
  shared mask should leave the top-K hypotheses closer to equally likely. Adding
  hypothesis k's unique mask back should restore k. We report the top1-minus-topK
  probability spread under each condition, and how often shared+unique recovers
  the model's original top-1.

**Test B — is unique evidence class-specific?**
  Remove ``unique_j`` and record the change in every hypothesis's logit. If the
  decomposition is real the matrix ``D[i][j] = delta-logit of class i when class
  j's unique mask is removed`` has a dominant diagonal: removing j's evidence
  hurts j more than it hurts its rivals.

  A control matters here. Deleting *any* region degrades the image, so a uniformly
  negative matrix would prove nothing. We therefore also delete a random mask of
  equal budget and report the diagonal effect relative to that.

**Test A is not comparable across grid resolutions. Do not build a resolution sweep on it.**

  ``spread`` is ``max - min`` over a vector of per-class probabilities in which every
  element is measured on a *different* image: entry k is P(class k | keep(x, m_shared +
  m_unique_k)). How much image each of those conditions keeps is set by the unique mask
  budget, and ``ContrastiveObjective`` scales that budget with the grid
  (``mass_scale = max(1, R / mass_ref_regions)``) so the mask stays a constant *fraction*
  of the frame. At 28x28 that is 16x the mass it is at 7x7, every per-class condition
  becomes correspondingly permissive, every class scores well under its own mask, and the
  between-class spread compresses toward zero.

  Measured on brain tumor at 28x28, n=96, lambda_shared_sparse=0, varying only
  ``mass_ref_regions``::

      mass_ref  mass_scale  unique mass  shared only  +unique  recovery
      49              16.0        15.98       0.4105   0.2995   -0.1111
      784              1.0         1.06       0.1956   0.3930   +0.1973

  The recovery term does not merely shrink at fine grids, it changes sign -- so a sweep
  that varies the grid is measuring the budget, not the decomposition. Read Test A at one
  fixed resolution (the model's native grid, where mass_scale is 1.0), which is what every
  headline decomposition result in docs/MEDICAL_RESULTS.md does.

Run from repo root::

    PYTHONPATH=. python scripts/eval_decomposition.py \\
        --checkpoint examples/out/checkpoints/ham10000_efficientnet_v2_s.pt \\
        --num_images 2035 --out_dir results/medical_presentation/decomposition
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

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


from evaluation.metrics import paired as _paired


def _stat(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    t = torch.tensor(vals)
    return {"mean": t.mean().item(), "std": t.std().item(), "n": len(vals)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intervention test of the CDEA shared/unique decomposition")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_alloc_steps", type=int, default=50)
    parser.add_argument("--game_mode", type=str, default="mixed",
                        help="Must be a preset with use_shared=True for Test A")
    parser.add_argument("--lambda_mass", type=float, default=2.0)
    parser.add_argument("--lambda_shared_sparse", type=float, default=0.0,
                        help="L1 penalty on the shared mask. It appears in no other "
                             "penalty term, so at 0.0 it inflates to blanket ~46%% of the "
                             "grid and captures base evidence at exactly 0.99x chance -- "
                             "uncorrelated with the field it is allocating. 0.25 shrinks "
                             "it ~8x and lifts it to 1.48x chance (HAM10000). Test A and "
                             "Test B both survive; absolute deletion magnitudes fall "
                             "~25%% but the diagonal/off-diagonal ratio holds or improves.")
    parser.add_argument("--evidence", type=str, default="gradcam", choices=["gradcam", "ig", "occlusion"])
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
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
                             "feature map, so a finer grid there is interpolation. WARNING: Test A "
                             "is NOT comparable across grids -- see the note in the module docstring.")
    parser.add_argument("--out_dir", type=str, default=str(OUT))
    parser.add_argument("--export_prefix", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dataset_name, class_names, num_classes = load_checkpoint(Path(args.checkpoint), device)
    print(f"Loaded {args.checkpoint}: {dataset_name}, {meta['model_name']}, {num_classes} classes")

    game_cfg = resolve_contrastive_game(args.game_mode)
    if not game_cfg.use_shared:
        raise ValueError(
            f"game_mode '{args.game_mode}' has use_shared=False, so there is no shared "
            "mask to test. Use 'mixed' or 'cooperative'."
        )

    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    if dataset_name not in MEDICAL_SPLIT_ROOTS:
        raise ValueError(f"no val split registered for '{dataset_name}'")
    t = transforms.Compose([
        transforms.Resize((TV_INPUT_SIZE, TV_INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(root=str(MEDICAL_SPLIT_ROOTS[dataset_name][1]), transform=t)
    if args.num_images < len(ds):
        # Sample rather than slice: ImageFolder is ordered by class, so a prefix is a
        # class-ordered subset and not representative of the val distribution.
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
        ds = torch.utils.data.Subset(ds, idx[:args.num_images].tolist())
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)
    print(f"Evaluating {len(ds)} val images")

    grid_h, grid_w = model_grid_size(meta["model_name"])
    if args.grid is not None:
        if args.evidence not in ("ig", "occlusion") and args.grid > grid_h:
            raise ValueError(
                f"--grid {args.grid} exceeds Grad-CAM's native {grid_h}x{grid_w} for "
                f"{meta['model_name']}. That would fabricate resolution. Use --evidence ig."
            )
        grid_h = grid_w = args.grid
    print(f"Unit grid: {grid_h}x{grid_w} ({grid_h * grid_w} regions)")
    unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur")
    if args.evidence == "ig":
        base_evidence = IntegratedGradientsRegionsProvider(
            grid_h=grid_h, grid_w=grid_w, steps=args.ig_steps, baseline="zero")
    elif args.evidence == "occlusion":
        base_evidence = OcclusionRegionsProvider(
            grid_h=grid_h, grid_w=grid_w, unit_space=unit_space,
            mode=args.occlusion_mode, n_masks=args.occlusion_masks, seed=args.seed)
    else:
        base_evidence = GradCAMRegionsProvider(grid_h=grid_h, grid_w=grid_w)

    # Mirror eval_localization.py exactly so both describe one configuration.
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
    K = min(5, num_classes)
    explainer = CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=TopMSelector(m=K),
        base_evidence=base_evidence,
        allocator=allocator,
        objective=objective,
        normalize_evidence=True,
        device=device,
    )

    rng = torch.Generator().manual_seed(args.seed)

    # Test A accumulators
    spread_full: List[float] = []
    spread_shared: List[float] = []
    spread_plus: List[float] = []
    restores_top1: List[float] = []
    # Test B accumulators: summed delta-logit matrix and its random-deletion control
    del_sum = torch.zeros(K, K)
    del_count = torch.zeros(K, K)
    rand_sum = torch.zeros(K)
    rand_count = torch.zeros(K)
    diag_minus_offdiag: List[float] = []

    for bi, (x, y) in enumerate(loader):
        x = x.to(device)
        B = x.shape[0]
        expl = explainer.explain(x)
        valid = expl.hypotheses.mask                     # (B, K)
        h_ids = expl.hypotheses.ids.clamp_min(0)         # (B, K)

        # ---- Test A: probability spread across the top-K hypotheses ----
        p_shared = expl.metrics.get("split_shared_only_probs_topm")
        p_plus = expl.metrics.get("split_shared_plus_unique_probs_topm")
        probs = expl.extras.get("probs")
        if isinstance(p_shared, torch.Tensor) and isinstance(p_plus, torch.Tensor):
            p_shared = p_shared.detach().cpu()
            p_plus = p_plus.detach().cpu()
            v = valid.detach().cpu()
            p_full_topm = probs.detach().cpu().gather(1, h_ids.detach().cpu())
            for b in range(B):
                sel = v[b]
                if int(sel.sum()) < 2:
                    continue
                f, s, u = p_full_topm[b][sel], p_shared[b][sel], p_plus[b][sel]
                spread_full.append((f.max() - f.min()).item())
                spread_shared.append((s.max() - s.min()).item())
                spread_plus.append((u.max() - u.min()).item())
                # rank 0 is the model's own top hypothesis
                restores_top1.append(1.0 if int(u.argmax()) == 0 else 0.0)

        # ---- Test B: K x K deletion matrix ----
        with torch.no_grad():
            base_logits = model(x)                        # (B, num_classes)
            base_topm = base_logits.gather(1, h_ids)      # (B, K)
            for j in range(K):
                m_j = expl.masks["unique"][:, j, :]        # (B, R)
                x_del = unit_space.remove(x, m_j)
                d_logits = model(x_del).gather(1, h_ids) - base_topm   # (B, K)
                sel_j = valid[:, j]
                if not bool(sel_j.any()):
                    continue
                d = d_logits.detach().cpu()
                v = valid.detach().cpu()
                for b in range(B):
                    if not bool(v[b, j]):
                        continue
                    for i in range(K):
                        if not bool(v[b, i]):
                            continue
                        del_sum[i, j] += d[b, i].item()
                        del_count[i, j] += 1

                # Control: delete a random mask carrying the same total budget, so a
                # negative diagonal cannot be explained by generic image corruption.
                budget = m_j.sum(dim=-1, keepdim=True)                  # (B, 1)
                r = torch.rand(m_j.shape, generator=rng).to(m_j.device)
                r = r / r.sum(dim=-1, keepdim=True).clamp_min(1e-8) * budget
                x_rand = unit_space.remove(x, r.clamp(0.0, 1.0))
                d_rand = (model(x_rand).gather(1, h_ids) - base_topm).detach().cpu()
                for b in range(B):
                    if not bool(v[b, j]):
                        continue
                    rand_sum[j] += d_rand[b, j].item()
                    rand_count[j] += 1
                    # Per-image contrast: how much more does deleting j's own unique
                    # mask hurt j than it hurts the average rival?
                    others = [d[b, i].item() for i in range(K) if i != j and bool(v[b, i])]
                    if others:
                        diag_minus_offdiag.append(d[b, j].item() - sum(others) / len(others))

        if (bi + 1) % 5 == 0:
            print(f"  {len(spread_full)}/{len(ds)} images")

    # ---------------- report ----------------
    print("\n=== Test A: probability spread over the top-K hypotheses ===")
    print("(top1 - topK probability gap; shared-only should collapse, +unique re-expand)")
    a_full, a_shared, a_plus = _stat(spread_full), _stat(spread_shared), _stat(spread_plus)
    for name, s in [("full model", a_full), ("shared only", a_shared),
                    ("shared + unique", a_plus)]:
        print(f"  {name:<20}{s['mean']:>9.4f}  (std {s['std']:.4f}, n={s['n']})")
    restore_rate = sum(restores_top1) / max(len(restores_top1), 1)
    print(f"  shared+unique restores the model's top-1 class: {restore_rate * 100:.1f}%")

    collapse = _paired(spread_full, spread_shared)
    recover = _paired(spread_plus, spread_shared)
    if collapse:
        print(f"\n  collapse (full - shared_only): {collapse['delta']:+.4f} "
              f"t={collapse['t']:.2f} win={collapse['win_rate']:.3f}")
    if recover:
        print(f"  recovery (shared+unique - shared_only): {recover['delta']:+.4f} "
              f"t={recover['t']:.2f} win={recover['win_rate']:.3f}")

    print("\n=== Test B: K x K deletion matrix (delta-logit) ===")
    print("D[i][j] = change in class i's logit when class j's unique mask is removed.")
    print("A real decomposition has a dominant (most negative) diagonal.\n")
    mat = (del_sum / del_count.clamp_min(1)).tolist()
    header = "        " + "".join(f"  del u{j:<7}" for j in range(K))
    print(header)
    for i in range(K):
        cells = "".join(f"  {mat[i][j]:>+8.4f} " for j in range(K))
        print(f"  z{i:<4}{cells}")

    rand_mean = (rand_sum / rand_count.clamp_min(1)).tolist()
    print("\n  random-mask control (same budget), delta-logit of class j:")
    print("       " + "".join(f"  {rand_mean[j]:>+8.4f} " for j in range(K)))

    contrast = _stat(diag_minus_offdiag)
    diag_vs_rand = [mat[j][j] - rand_mean[j] for j in range(K)]
    print(f"\n  diagonal minus mean off-diagonal (per image): {contrast['mean']:+.4f} "
          f"(std {contrast['std']:.4f}, n={contrast['n']})")
    print("  diagonal minus random-deletion control, per rank: "
          + ", ".join(f"{v:+.4f}" for v in diag_vs_rand))
    verdict = "class-specific" if contrast["mean"] < 0 else "NOT class-specific"
    print(f"\n  --> unique evidence is {verdict}: deleting a class's own unique mask "
          f"changes its logit by {contrast['mean']:+.4f} relative to its rivals.")

    prefix = args.export_prefix or f"decomposition_{dataset_name}_{args.evidence}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"test": "spread", "condition": "full", **a_full},
        {"test": "spread", "condition": "shared_only", **a_shared},
        {"test": "spread", "condition": "shared_plus_unique", **a_plus},
        {"test": "restore", "condition": "top1_restored_rate", "mean": restore_rate,
         "std": float("nan"), "n": len(restores_top1)},
        {"test": "deletion", "condition": "diag_minus_offdiag", **contrast},
    ]
    save_rows_csv(out_dir / f"{prefix}.csv", rows)
    save_json(out_dir / f"{prefix}.json", {
        "rows": rows,
        "spread": {"full": a_full, "shared_only": a_shared, "shared_plus_unique": a_plus},
        "paired": {"collapse_full_vs_shared": collapse, "recovery_plus_vs_shared": recover},
        "restore_top1_rate": restore_rate,
        "deletion_matrix": mat,
        "deletion_random_control": rand_mean,
        "deletion_diag_minus_random": diag_vs_rand,
        "deletion_diag_minus_offdiag": contrast,
        "per_image": {
            "spread_full": spread_full,
            "spread_shared_only": spread_shared,
            "spread_shared_plus_unique": spread_plus,
            "diag_minus_offdiag": diag_minus_offdiag,
        },
        "checkpoint": args.checkpoint,
        "dataset": dataset_name,
        "model_name": meta["model_name"],
        "class_names": list(class_names),
        "grid": [grid_h, grid_w],
        "top_k": K,
        "evidence": args.evidence,
        "game_mode": args.game_mode,
        "use_shared": bool(game_cfg.use_shared),
        "lambda_partition": float(game_cfg.lambda_partition),
        "lambda_shared_sparse": float(args.lambda_shared_sparse),
        "num_alloc_steps": args.num_alloc_steps,
        "num_images": len(ds),
        "seed": args.seed,
    })
    print(f"\nSaved {out_dir / (prefix + '.csv')}")


if __name__ == "__main__":
    main()
