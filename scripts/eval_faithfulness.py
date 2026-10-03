"""
scripts/eval_faithfulness.py

Deletion / Insertion AUC (RISE protocol) and ROAD (Rong et al., ICML 2022) for
CDEA-Contrastive.

Why this file exists: overlap, sufficiency, margin and mask budget are all terms in
`ContrastiveObjective`'s loss, so reporting them measures whether the optimizer converged.
These are the out-of-objective faithfulness numbers the field actually compares on.

**Deletion / Insertion.** Order regions by mask value, then remove (or add) them a few at a
time and track the target-class probability. Deletion AUC lower is better -- the score
should collapse as soon as the regions the method called important are gone. Insertion AUC
higher is better.

**ROAD.** Deletion has a known confound: replacing regions with a constant or a blur leaves
the *shape* of the mask in the image, and a classifier can read that shape instead of the
content. ROAD removes the cue by imputing each deleted pixel from its neighbours, so the
hole is not visually recoverable. We report MoRF (most-relevant-first, lower better) and
LeRF (least-relevant-first, higher better).

The imputation here is Jacobi iteration on the neighbour-average linear system rather than
the paper's direct sparse solve. Same fixed point, and it stays on the GPU; the iteration
count is exposed as `--road_iters` so the approximation is checkable.

Arms compared: the base evidence, the CDEA unique mask for the top hypothesis, and a
**random region ordering** as the control that says whether any ordering would do.

Run from repo root::

    PYTHONPATH=. python scripts/eval_faithfulness.py \\
        --checkpoint examples/out/checkpoints/ham10000_resnet18.pt --num_images 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.device import get_device
from core.hypotheses import TopMSelector
from core.game_modes import resolve_contrastive_game
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from examples.contrastive_explanation import (
    MEDICAL_SPLIT_ROOTS, TV_INPUT_SIZE, checkpoint_metadata, load_checkpoint,
)
from scripts.train_backbone import model_grid_size
from evaluation.masks import regions_to_pixels
from evaluation.removal import noisy_linear_impute


def region_to_pixel(m: torch.Tensor, gh: int, gw: int, H: int, W: int) -> torch.Tensor:
    """(B, R) region mask -> (B, 1, H, W) pixel mask, nearest so holes stay hard-edged."""
    return regions_to_pixels(m, gh, gw, H, W, mode="nearest").unsqueeze(1)


def auc(curve: List[float]) -> float:
    """Normalized area under a curve sampled on a uniform grid."""
    v = torch.tensor(curve)
    return float(((v[:-1] + v[1:]) * 0.5).mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Deletion/Insertion and ROAD for CDEA")
    p.add_argument("--checkpoint", type=str,
                   default="examples/out/checkpoints/ham10000_resnet18.pt")
    p.add_argument("--evidence", type=str, default="gradcam", choices=["gradcam", "ig"])
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--num_images", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--game_mode", type=str, default="mixed")
    p.add_argument("--num_alloc_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--lambda_shared_sparse", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=2,
                   help="Regions revealed/removed per step (7x7 grid = 49 regions)")
    p.add_argument("--road_iters", type=int, default=24)
    p.add_argument("--road_noise", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=str(REPO / "results" / "faithfulness"))
    p.add_argument("--export_prefix", type=str, default=None)
    args = p.parse_args()

    device = get_device()
    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dsname, class_names, ncls = load_checkpoint(Path(args.checkpoint), device)
    model_name = meta["model_name"]
    gh, gw = model_grid_size(model_name)
    R = gh * gw
    prefix = args.export_prefix or f"faith_{dsname}_{model_name}_{args.evidence}"
    print(f"Loaded {dsname} / {model_name} / {ncls} classes, grid {gh}x{gw}")

    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    t = transforms.Compose([transforms.Resize((TV_INPUT_SIZE, TV_INPUT_SIZE)),
                            transforms.ToTensor()])
    ds = ImageFolder(root=str(MEDICAL_SPLIT_ROOTS[dsname][1]), transform=t)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
    sub = torch.utils.data.Subset(ds, idx[:min(args.num_images, len(ds))].tolist())
    loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False)

    cfg = resolve_contrastive_game(args.game_mode)
    unit_space = VisionGridUnitSpace(gh, gw, baseline="blur")
    selector = TopMSelector(m=min(5, ncls))
    provider = (IntegratedGradientsRegionsProvider(grid_h=gh, grid_w=gw, steps=args.ig_steps,
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

    steps = list(range(0, R + 1, args.stride))
    if steps[-1] != R:
        steps.append(R)
    gen = torch.Generator().manual_seed(args.seed)
    curves: Dict[str, Dict[str, List[List[float]]]] = {
        m: {k: [] for k in ("deletion", "insertion", "road_morf", "road_lerf")}
        for m in ("base_evidence", "cdea_unique", "random")
    }

    seen = 0
    for x, y in loader:
        x = x.to(device)
        B = x.shape[0]
        with torch.no_grad():
            logits = model(x)
            hyp = selector.select(logits, torch.softmax(logits, dim=-1))
            cls = hyp.ids[:, 0].clamp_min(0)
        xg = x.detach().clone().requires_grad_(True)
        ev = provider.explain(xg, model, hyp).detach()
        ev = ev / ev.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        masks = allocator.allocate(x=x, model=model, unit_space=unit_space,
                                   hypotheses=hyp, evidence=ev)

        scorers = {
            "base_evidence": ev[:, 0],
            "cdea_unique": masks["unique"][:, 0].detach(),
            "random": torch.rand(B, R, generator=gen).to(device),
        }
        H, W = x.shape[-2:]
        for name, field in scorers.items():
            order = field.argsort(dim=-1, descending=True)      # most important first
            d, i_, rm, rl = [], [], [], []
            for s in steps:
                topk = torch.zeros(B, R, device=device)
                if s > 0:
                    topk.scatter_(1, order[:, :s], 1.0)
                keep_del = 1.0 - topk                           # remove the important ones
                px_del = region_to_pixel(keep_del, gh, gw, H, W)
                px_ins = region_to_pixel(topk, gh, gw, H, W)
                lo = torch.zeros(B, R, device=device)
                if s > 0:
                    lo.scatter_(1, order[:, -s:], 1.0)          # least important first
                px_lerf = region_to_pixel(1.0 - lo, gh, gw, H, W)
                with torch.no_grad():
                    d.append(float(model(unit_space.keep(x, keep_del)
                                         ).softmax(-1).gather(1, cls[:, None]).mean()))
                    i_.append(float(model(unit_space.keep(x, topk)
                                          ).softmax(-1).gather(1, cls[:, None]).mean()))
                    rm.append(float(model(noisy_linear_impute(
                        x, px_del, args.road_iters, args.road_noise, gen)
                        ).softmax(-1).gather(1, cls[:, None]).mean()))
                    rl.append(float(model(noisy_linear_impute(
                        x, px_lerf, args.road_iters, args.road_noise, gen)
                        ).softmax(-1).gather(1, cls[:, None]).mean()))
            curves[name]["deletion"].append(d)
            curves[name]["insertion"].append(i_)
            curves[name]["road_morf"].append(rm)
            curves[name]["road_lerf"].append(rl)
        seen += B
        print(f"  {seen}/{len(sub)}", end="\r", flush=True)

    summary = {
        "checkpoint": args.checkpoint, "dataset": dsname, "model": model_name,
        "evidence": args.evidence, "grid": [gh, gw], "n": seen, "stride": args.stride,
        "road_iters": args.road_iters, "road_noise": args.road_noise, "seed": args.seed,
        "fraction_removed": [s / R for s in steps],
        "results": {}, "curves": {},
    }
    for name, cs in curves.items():
        mean = {k: torch.tensor(v).mean(0).tolist() for k, v in cs.items()}
        summary["curves"][name] = mean
        summary["results"][name] = {
            "deletion_auc": auc(mean["deletion"]),
            "insertion_auc": auc(mean["insertion"]),
            "road_morf_auc": auc(mean["road_morf"]),
            "road_lerf_auc": auc(mean["road_lerf"]),
        }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{prefix}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'method':<16}{'del↓':>9}{'ins↑':>9}{'ROAD MoRF↓':>13}{'ROAD LeRF↑':>13}")
    for name, r in summary["results"].items():
        print(f"{name:<16}{r['deletion_auc']:>9.4f}{r['insertion_auc']:>9.4f}"
              f"{r['road_morf_auc']:>13.4f}{r['road_lerf_auc']:>13.4f}")
    print(f"\nSaved {out / (prefix + '.json')}")


if __name__ == "__main__":
    main()
