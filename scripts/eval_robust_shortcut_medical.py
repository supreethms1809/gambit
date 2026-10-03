"""
scripts/eval_robust_shortcut_medical.py

The robust/shortcut game on a real dataset.

`eval_robust_shortcut.py` runs on three synthetic sets (ColoredMNIST, ColoredCIFAR10,
TextureBiasedMNIST) where the shortcut is stamped in by hand and the environments are
built by re-stamping it. HAM10000 does not need a stamped shortcut: it already has one.

The release records an acquisition site per image (`dataset` in HAM10000_metadata.csv),
and the sites differ in two ways that matter:

  * **Appearance.** Measured over the val split, mean RGB runs (0.84, 0.51, 0.53) for
    `vidir_molemax` against (0.69, 0.57, 0.61) for `vidir_modern`, and mean saturation
    almost doubles across sites (0.213 -> 0.418).
  * **Label distribution.** `vidir_molemax` is 94% nevus (3720/3954) with essentially no
    akiec or bcc; `rosendahl` is broadly balanced.

So "looks like it came from molemax" genuinely predicts "nevus" in this data, and a
classifier is free to use it. It does: re-rendering a val image to another site's colour
statistics flips 34.4% of ResNet-18's predictions and moves p(top-1) by 0.279.

**Environments.** The objective is paired -- it applies one spatial mask to every
environment and reads labels off `xs[0]` -- and real site groups are unpaired, since a
lesion belongs to one site. So the environments here are *site-transfer* views of the
same image: standardize per image, then rescale to each site's measured channel
statistics. Three environments, one operation, three real targets. The nuisance is
estimated from the real groups rather than invented, but it is a transformation, not the
site groups themselves; a claim about actual site generalization needs the unpaired
group objective, which this script does not implement.

**Validation.** HAM10000 ships expert lesion segmentations, so the split is falsifiable
rather than decorative: the robust mask should sit on the lesion and the shortcut mask
off it. The script reports mask mass inside the segmentation for both, paired, with the
difference tested per image.

Run from repo root::

    PYTHONPATH=. python scripts/eval_robust_shortcut_medical.py \\
        --checkpoint examples/out/checkpoints/ham10000_resnet18.pt --num_images 400
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.device import get_device
from core.hypotheses import TopMSelector
from core.types import EnvBatch
from core.game_modes import resolve_shift_game
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from instantiations.shift.objective import RobustShortcutObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from examples.contrastive_explanation import (
    MEDICAL_SPLIT_ROOTS, TV_INPUT_SIZE, checkpoint_metadata, load_checkpoint,
)
from scripts.train_backbone import model_grid_size
from scripts.eval_localization import SegmentedDataset, SEG_ROOTS

METADATA = REPO / "data" / "ham10000_raw" / "HAM10000_metadata.csv"


def site_of_image() -> Dict[str, str]:
    """image_id -> acquisition site, from the release metadata."""
    if not METADATA.exists():
        raise FileNotFoundError(
            f"{METADATA} not found. This experiment needs the HAM10000 metadata CSV; "
            "see docs/MEDICAL_DATASETS.md."
        )
    with open(METADATA) as fh:
        return {r["image_id"]: r["dataset"] for r in csv.DictReader(fh)}


def site_colour_stats(
    ds, sites: Dict[str, str], per_site: int, seed: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Per-channel mean/std of each site, measured from the split itself."""
    by_site: Dict[str, List[int]] = defaultdict(list)
    for i, (path, _) in enumerate(ds.samples):
        s = sites.get(Path(path).stem)
        if s is not None:
            by_site[s].append(i)
    g = torch.Generator().manual_seed(seed)
    stats = {}
    for s, idx in sorted(by_site.items()):
        pick = [idx[j] for j in torch.randperm(len(idx), generator=g)[:min(per_site, len(idx))]]
        xs = torch.stack([ds[i][0] for i in pick])
        stats[s] = (xs.mean(dim=(0, 2, 3)), xs.std(dim=(0, 2, 3)).clamp_min(1e-6))
        print(f"  site {s:<15} n={len(idx):<5} mean={stats[s][0].numpy().round(3)} "
              f"std={stats[s][1].numpy().round(3)}")
    return stats


RADIAL_BINS = 8


def _radial_bins(size: int, device) -> torch.Tensor:
    """(NB, size, size) one-hot ring masks, centre outward."""
    ax = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt()
    edges = torch.linspace(0, float(r.max()) + 1e-6, RADIAL_BINS + 1, device=device)
    return torch.stack([(r >= edges[i]) & (r < edges[i + 1])
                        for i in range(RADIAL_BINS)]).float()


def site_vignette_profiles(ds, sites, per_site: int, seed: int):
    """Per-site radial luminance profile, normalized to the centre ring.

    Dermatoscope optics vignette, and the sites differ in how much. Measured over the val
    split the spread across sites is 0.000 at the centre and 0.955 at the rim -- a
    confound that is *spatially localized* at the periphery, unlike a global colour cast.
    That distinction decides whether a spatial mask can isolate it at all.
    """
    by_site: Dict[str, List[int]] = defaultdict(list)
    for i, (path, _) in enumerate(ds.samples):
        st = sites.get(Path(path).stem)
        if st is not None:
            by_site[st].append(i)
    g = torch.Generator().manual_seed(seed)
    bins = _radial_bins(TV_INPUT_SIZE, torch.device("cpu"))
    prof = {}
    for st, idx in sorted(by_site.items()):
        pick = [idx[j] for j in torch.randperm(len(idx), generator=g)[:min(per_site, len(idx))]]
        lum = torch.stack([ds[i][0] for i in pick]).mean(dim=1)
        pr = torch.stack([(lum * b).sum() / b.sum().clamp_min(1) / lum.shape[0] for b in bins])
        prof[st] = pr / pr[0].clamp_min(1e-6)
        print(f"  site {st:<15} vignette profile {prof[st].numpy().round(3)}")
    return prof


def make_env_fn(stats, targets: List[str], nuisance: str = "colour", vprof=None):
    """Paired site-transfer environments: standardize per image, rescale to each site.

    Every environment goes through the identical operation with a different target, so
    no environment is privileged by having skipped the round trip.
    """
    def env_fn(x: torch.Tensor, y: torch.Tensor) -> EnvBatch:
        xs = []
        bins = _radial_bins(x.shape[-1], x.device) if nuisance in ("vignette", "both") else None
        if bins is not None:
            lum = x.mean(dim=1, keepdim=True)                          # (B,1,H,W)
            own = torch.stack([(lum * b).sum(dim=(1, 2, 3)) / b.sum().clamp_min(1)
                               for b in bins], dim=1)                  # (B,NB)
            own = (own / own[:, :1].clamp_min(1e-6)).clamp_min(1e-6)
        for t in targets:
            xt = x
            if nuisance in ("colour", "both"):
                mu = xt.mean(dim=(2, 3), keepdim=True)
                sd = xt.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                m, sg = stats[t]
                xt = ((xt - mu) / sd) * sg.to(x.device).view(1, 3, 1, 1) \
                     + m.to(x.device).view(1, 3, 1, 1)
            if bins is not None:
                # Per-ring gain that takes this image's own vignette to the target site's.
                gain = (vprof[t].to(x.device).view(1, -1) / own)        # (B,NB)
                field = torch.einsum("bn,nhw->bhw", gain, bins).unsqueeze(1)
                xt = xt * field
            xs.append(xt.clamp(0, 1))
        return EnvBatch(xs=xs, env_ids=list(targets))
    return env_fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--checkpoint", type=str,
                   default="examples/out/checkpoints/ham10000_resnet18.pt")
    p.add_argument("--num_images", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--lambda_sparse", type=float, default=None,
                   help="Override the preset's sparsity weight. The shift objective has no "
                        "mass target, so the preset's 0.05 lets both masks blanket the "
                        "frame (measured: 11.3 of 49 regions) and every spatial statistic "
                        "then just reports area.")
    p.add_argument("--game_mode", type=str, default="mixed",
                   choices=["mixed", "cooperative", "competitive"])
    p.add_argument("--sites", type=str, default="vidir_molemax,vidir_modern,rosendahl",
                   help="Comma-separated acquisition sites to use as environments")
    p.add_argument("--nuisance", type=str, default="vignette",
                   choices=["colour", "vignette", "both"],
                   help="Which measured site difference becomes the environment shift. "
                        "'colour' is a global cast and has no spatial signature, so a "
                        "spatial mask cannot isolate it (measured: robust and shortcut "
                        "both land at chance on the lesion). 'vignette' is localized at "
                        "the periphery and is the one a region mask can act on.")
    p.add_argument("--stats_images", type=int, default=200,
                   help="Images per site used to measure that site's colour statistics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default=str(REPO / "results" / "medical_presentation" / "shift_real"))
    p.add_argument("--export_prefix", type=str, default="shift_ham10000_site")
    args = p.parse_args()

    device = get_device()
    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dataset_name, class_names, num_classes = load_checkpoint(Path(args.checkpoint), device)
    if dataset_name != "ham10000":
        raise SystemExit(f"checkpoint is for '{dataset_name}'; this experiment needs the "
                         "HAM10000 site metadata and lesion segmentations")
    gh, gw = model_grid_size(meta["model_name"])
    print(f"Loaded {dataset_name} / {meta['model_name']} / {num_classes} classes, grid {gh}x{gw}")

    ds = SegmentedDataset(MEDICAL_SPLIT_ROOTS[dataset_name][1], SEG_ROOTS[dataset_name])
    sites = site_of_image()
    print("Measuring site colour statistics:")
    stats = site_colour_stats(ds, sites, args.stats_images, args.seed)
    vprof = (site_vignette_profiles(ds, sites, args.stats_images, args.seed)
             if args.nuisance in ("vignette", "both") else None)
    targets = [s.strip() for s in args.sites.split(",")]
    missing = [t for t in targets if t not in stats]
    if missing:
        raise SystemExit(f"--sites names sites not present: {missing}; have {sorted(stats)}")
    env_fn = make_env_fn(stats, targets, nuisance=args.nuisance, vprof=vprof)

    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
    sub = torch.utils.data.Subset(ds, idx[:min(args.num_images, len(ds))].tolist())
    loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False)

    cfg = resolve_shift_game(args.game_mode)
    unit_space = VisionGridUnitSpace(gh, gw)
    selector = TopMSelector(m=min(5, num_classes))
    provider = GradCAMRegionsProvider(gh, gw)
    objective = RobustShortcutObjective(
        lambda_mean=cfg.lambda_mean, lambda_var=cfg.lambda_var, lambda_gap=cfg.lambda_gap,
        lambda_shortcut=cfg.lambda_shortcut, lambda_disjoint=cfg.lambda_disjoint,
        lambda_sparse=cfg.lambda_sparse if args.lambda_sparse is None else args.lambda_sparse,
    )
    allocator = RobustShortcutOptimizationAllocator(
        objective, num_steps=args.num_steps, lr=args.lr, lambda_disjoint=cfg.lambda_disjoint)

    per_image: List[Dict[str, float]] = []
    agg: Dict[str, List[float]] = defaultdict(list)
    for x, y, seg in loader:
        x, y, seg = x.to(device), y.to(device), seg.to(device)
        env = env_fn(x, y)
        with torch.no_grad():
            logits = model(x)
            hyp = selector.select(logits, torch.softmax(logits, dim=-1))
        xg = x.detach().clone().requires_grad_(True)
        ev = provider.explain(xg, model, hyp).detach()
        ev = ev / ev.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        masks = allocator.allocate(x=x, model=model, unit_space=unit_space,
                                   hypotheses=hyp, evidence=ev, env=env)
        m = objective.compute(x=x, model=model, unit_space=unit_space, hypotheses=hyp,
                              masks=masks, evidence=ev, env=env)
        for k, v in m.items():
            agg[k].append(float(v.detach()) if torch.is_tensor(v) else float(v))

        # Does the split land where the pathology is? Mask mass inside the expert
        # segmentation, per image, for both masks and for the raw evidence.
        H, W = seg.shape[-2:]
        for name, field in (("robust", masks["robust"]), ("shortcut", masks["shortcut"]),
                            ("base_evidence", ev[:, 0])):
            pm = F.interpolate(field.view(field.shape[0], 1, gh, gw), size=(H, W),
                               mode="bilinear", align_corners=False).squeeze(1)
            inside = (pm * seg).sum(dim=(1, 2)) / pm.sum(dim=(1, 2)).clamp_min(1e-8)
            agg[f"in_lesion_{name}"].extend(inside.detach().cpu().tolist())
        n = x.shape[0]
        for b in range(n):
            per_image.append({
                "in_lesion_robust": agg["in_lesion_robust"][-n + b],
                "in_lesion_shortcut": agg["in_lesion_shortcut"][-n + b],
                "in_lesion_base_evidence": agg["in_lesion_base_evidence"][-n + b],
            })
        agg["lesion_area_fraction"].extend(seg.mean(dim=(1, 2)).cpu().tolist())
        print(f"  {len(per_image)}/{len(sub)}", end="\r", flush=True)

    rob = torch.tensor([r["in_lesion_robust"] for r in per_image])
    sho = torch.tensor([r["in_lesion_shortcut"] for r in per_image])
    d = rob - sho
    t = float(d.mean() / (d.std(unbiased=True) / (len(d) ** 0.5)).clamp_min(1e-12))
    summary = {
        "checkpoint": args.checkpoint, "dataset": dataset_name,
        "model": meta["model_name"], "grid": [gh, gw], "game_mode": args.game_mode,
        "num_steps": args.num_steps, "lr": args.lr, "seed": args.seed,
        "lambda_sparse": objective.lp,
        "sites": targets, "nuisance": args.nuisance, "n": len(per_image),
        "objective": {k: sum(v) / len(v) for k, v in agg.items() if not k.startswith("in_lesion")
                      and k != "lesion_area_fraction"},
        "in_lesion": {
            "robust": float(rob.mean()), "shortcut": float(sho.mean()),
            "base_evidence": float(torch.tensor(agg["in_lesion_base_evidence"]).mean()),
            "lesion_area_fraction": float(torch.tensor(agg["lesion_area_fraction"]).mean()),
        },
        "robust_minus_shortcut": {
            "delta": float(d.mean()), "t": t,
            "win_rate": float((d > 0).float().mean()), "n": len(d),
        },
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.export_prefix}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{args.export_prefix}_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys())); w.writeheader()
        w.writerows(per_image)

    print("\n--- objective (mean over batches) ---")
    for k, v in summary["objective"].items():
        print(f"  {k:<16}{v:>10.4f}")
    print("\n--- does the split land on the pathology? (mask mass inside the lesion) ---")
    il = summary["in_lesion"]
    print(f"  lesion area fraction (chance) {il['lesion_area_fraction']:.4f}")
    print(f"  base evidence                 {il['base_evidence']:.4f}")
    print(f"  robust mask                   {il['robust']:.4f}")
    print(f"  shortcut mask                 {il['shortcut']:.4f}")
    r = summary["robust_minus_shortcut"]
    print(f"  robust - shortcut  {r['delta']:+.4f}  t={r['t']:.1f}  "
          f"wins {r['win_rate']:.1%} of {r['n']} images")
    print(f"\nSaved {out / (args.export_prefix + '.json')}")


if __name__ == "__main__":
    main()
