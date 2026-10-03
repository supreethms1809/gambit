"""
scripts/eval_sanity_checks.py

Model-parameter randomization test (Adebayo et al., NeurIPS 2018) for CDEA-Contrastive.

The question this answers: do the allocated masks depend on what the model learned, or
only on the image? Randomize the network's weights from the logits downward, recompute the
masks, and correlate them with the originals. An explanation that survives randomization is
reading the input, not the model, and is not an explanation of anything.

We have concrete cause to run this rather than assume it. The ep10/lr1e-3 brain-tumor
EfficientNet checkpoint produced a complete, plausible result set -- overlap down 96%,
sufficiency flat -- on a model whose logits barely moved between one region and the whole
image (`ablation_effnet/BROKEN_ep10_lr1e-3/README.md`). Every metric in this project is a
masked forward pass, so a model that ignores masking still yields numbers that look like
findings.

Two protocol points that decide whether the test means anything:

* **Hypotheses are frozen to the original model's top-K.** Randomizing changes which
  classes are ranked highest; if the hypothesis set moved too, we would be comparing masks
  for *different classes* and would see decorrelation for a trivial reason.
* **Cascading, not independent.** Weights are randomized from the top down and kept
  randomized, which is the stricter of the two variants in the paper.

Grad-CAM is scored alongside CDEA as a reference point: it is the field's most-used method
and is known to degrade under this test, so it calibrates what "passing" looks like here.

Run from repo root::

    PYTHONPATH=. python scripts/eval_sanity_checks.py \\
        --checkpoint examples/out/checkpoints/ham10000_resnet18.pt --num_images 128
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.device import get_device
from core.hypotheses import TopMSelector
from core.game_modes import resolve_contrastive_game
from core.types import HypothesisSet
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from examples.contrastive_explanation import (
    MEDICAL_SPLIT_ROOTS, TV_INPUT_SIZE, checkpoint_metadata, load_checkpoint,
)
from scripts.train_backbone import model_grid_size


def cascade_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Parameterized blocks in forward order.

    Descends one level into a container that holds most of the network (EfficientNet's
    `features`), so the cascade has useful granularity instead of one giant step.
    """
    top = [(n, m) for n, m in model.named_children()
           if any(p.requires_grad or True for p in m.parameters(recurse=True))]
    top = [(n, m) for n, m in top if sum(p.numel() for p in m.parameters()) > 0]
    out: List[Tuple[str, nn.Module]] = []
    total = sum(sum(p.numel() for p in m.parameters()) for _, m in top)
    for name, mod in top:
        share = sum(p.numel() for p in mod.parameters()) / max(total, 1)
        kids = [(f"{name}.{k}", c) for k, c in mod.named_children()
                if sum(p.numel() for p in c.parameters()) > 0]
        if share > 0.5 and len(kids) > 1:
            out.extend(kids)
        else:
            out.append((name, mod))
    return out


def randomize_(mod: nn.Module, seed: int) -> None:
    """Re-initialize every parameter in `mod` in place."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    for p in mod.parameters(recurse=True):
        with torch.no_grad():
            if p.dim() >= 2:
                fan_in = p[0].numel()
                bound = (1.0 / max(fan_in, 1)) ** 0.5
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1).to(p.device) * bound)
            else:
                p.copy_(torch.zeros_like(p) if p.dim() == 0 else
                        (torch.rand(p.shape, generator=g) * 2 - 1).to(p.device) * 0.05)
    for m in mod.modules():                      # running stats are learned too
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation of two (N, R) tensors."""
    def rank(t: torch.Tensor) -> torch.Tensor:
        idx = t.argsort(dim=-1)
        r = torch.zeros_like(t)
        ar = torch.arange(t.shape[-1], dtype=t.dtype, device=t.device)
        r.scatter_(-1, idx, ar.expand_as(t))
        return r
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    num = (ra * rb).sum(dim=-1)
    den = (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-8)
    return num / den


def main() -> None:
    p = argparse.ArgumentParser(description="Adebayo model-randomization test for CDEA")
    p.add_argument("--checkpoint", type=str,
                   default="examples/out/checkpoints/ham10000_resnet18.pt")
    p.add_argument("--evidence", type=str, default="gradcam", choices=["gradcam", "ig"])
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--num_images", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--game_mode", type=str, default="mixed")
    p.add_argument("--num_alloc_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--lambda_shared_sparse", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=str(REPO / "results" / "sanity_checks"))
    p.add_argument("--export_prefix", type=str, default=None)
    args = p.parse_args()

    device = get_device()
    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dsname, class_names, ncls = load_checkpoint(Path(args.checkpoint), device)
    model_name = meta["model_name"]
    prefix = args.export_prefix or f"sanity_{dsname}_{model_name}_{args.evidence}"
    gh, gw = model_grid_size(model_name)
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

    def build(m):
        prov = (IntegratedGradientsRegionsProvider(grid_h=gh, grid_w=gw, steps=args.ig_steps,
                                                   baseline="zero")
                if args.evidence == "ig" else GradCAMRegionsProvider(grid_h=gh, grid_w=gw))
        obj = ContrastiveObjective(lambda_suff=1.0, lambda_margin=cfg.lambda_margin,
                                   lambda_sparse=0.05, lambda_overlap=cfg.lambda_overlap,
                                   lambda_mass=2.0,
                                   lambda_shared_sparse=args.lambda_shared_sparse)
        alloc = OptimizationAllocator(obj, num_steps=args.num_alloc_steps, lr=args.lr,
                                      use_shared=cfg.use_shared,
                                      lambda_disjoint=cfg.lambda_disjoint,
                                      lambda_partition=cfg.lambda_partition)
        return prov, obj, alloc

    def run(m, fixed: List[HypothesisSet] | None):
        """Evidence and unique masks for every batch, under model `m`."""
        prov, obj, alloc = build(m)
        evs, uns, hyps, correct, total = [], [], [], 0, 0
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = m(x)
                correct += int((logits.argmax(-1) == y).sum()); total += x.shape[0]
                h = (selector.select(logits, torch.softmax(logits, dim=-1))
                     if fixed is None else fixed[bi])
            xg = x.detach().clone().requires_grad_(True)
            ev = prov.explain(xg, m, h).detach()
            ev = ev / ev.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            masks = alloc.allocate(x=x, model=m, unit_space=unit_space,
                                   hypotheses=h, evidence=ev)
            evs.append(ev.cpu()); uns.append(masks["unique"].detach().cpu()); hyps.append(h)
        return evs, uns, hyps, correct / max(total, 1)

    print("Reference pass (trained model)...")
    ev0, un0, hyp0, acc0 = run(model, None)
    print(f"  accuracy {acc0:.3f}")

    blocks = cascade_blocks(model)
    order = list(reversed(blocks))               # logits first, then downward
    print(f"Cascade over {len(order)} blocks: {[n for n, _ in order]}")

    rows = []
    rnd = copy.deepcopy(model)
    for si, (name, _) in enumerate(order, 1):
        target = dict(rnd.named_modules())[name]
        randomize_(target, seed=args.seed * 1000 + si)
        rnd.eval()
        ev1, un1, _, acc1 = run(rnd, hyp0)       # hypotheses frozen to the trained model
        cu = torch.cat([spearman(a.flatten(0, 1), b.flatten(0, 1))
                        for a, b in zip(un0, un1)]).mean().item()
        ce = torch.cat([spearman(a.flatten(0, 1), b.flatten(0, 1))
                        for a, b in zip(ev0, ev1)]).mean().item()
        rows.append({"stage": si, "randomized_through": name,
                     "accuracy": acc1, "cdea_unique_spearman": cu,
                     "base_evidence_spearman": ce})
        print(f"  [{si}/{len(order)}] +{name:<16} acc {acc1:.3f}  "
              f"CDEA ρ {cu:+.3f}   {args.evidence} ρ {ce:+.3f}")

    summary = {
        "checkpoint": args.checkpoint, "dataset": dsname, "model": model_name,
        "evidence": args.evidence, "grid": [gh, gw], "n": len(sub),
        "game_mode": args.game_mode, "num_alloc_steps": args.num_alloc_steps,
        "seed": args.seed, "reference_accuracy": acc0,
        "protocol": "cascading top-down randomization; hypotheses frozen to the trained "
                    "model's top-K; Spearman over the region axis",
        "stages": rows,
        "final": {"cdea_unique_spearman": rows[-1]["cdea_unique_spearman"],
                  "base_evidence_spearman": rows[-1]["base_evidence_spearman"],
                  "accuracy": rows[-1]["accuracy"]},
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{prefix}.json").write_text(json.dumps(summary, indent=2))

    fin = summary["final"]
    print(f"\nFully randomized: accuracy {fin['accuracy']:.3f} "
          f"(chance {1/ncls:.3f}), CDEA ρ {fin['cdea_unique_spearman']:+.3f}, "
          f"{args.evidence} ρ {fin['base_evidence_spearman']:+.3f}")
    verdict = ("PASS — masks decorrelate as the model is destroyed"
               if abs(fin["cdea_unique_spearman"]) < 0.30 else
               "FAIL — masks survive randomization; they are not explaining this model")
    print(verdict)
    print(f"Saved {out / (prefix + '.json')}")


if __name__ == "__main__":
    main()
