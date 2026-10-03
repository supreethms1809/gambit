"""
scripts/eval_robust_shortcut_dogs.py

The robust/shortcut game on a real non-medical dataset: Stanford Dogs.

Companion to `eval_robust_shortcut_medical.py`, which runs the same game on HAM10000
acquisition sites. Both exist because `eval_robust_shortcut.py` only runs on three
synthetic sets where the shortcut is stamped in by hand and the environments are built by
re-stamping it.

**The shortcut here is the background, and it is not stamped in.** Breeds are photographed
in correlated settings -- snow breeds in snow, toy breeds indoors -- so the background
carries breed information the model is free to use. Measured over 20 breeds, a
nearest-centroid classifier reading *only* six background colour statistics (mean and std
of RGB outside the bounding box) recovers the breed at 11.1% against 5.0% chance. Six
crude numbers get 2.2x chance; a CNN has far more to work with.

**Environments.** The objective is paired -- one spatial mask is applied across every
environment, labels come from `xs[0]` -- so environments must be views of the same image.
Each environment re-renders the background to one of k background *styles*, obtained by
clustering measured background statistics over the dataset. Foreground pixels inside the
box are left untouched. Real images, real measured background styles, localized by
construction to the region outside the object.

**Validation.** Stanford Dogs ships a bounding box per image, which is the ground truth
the synthetic benchmarks lack: the robust mask should sit on the dog and the shortcut mask
off it. Reported as mask mass inside the box for both, paired, tested per image.

Note what this does and does not show. The nuisance is applied outside the box, so
"shortcut mask avoids the box" is the hypothesis under test, not a discovery -- the
allocator is never told where the box is and has to find that structure from the
environment variation alone. That is the same thing ColoredCIFAR10's corner patch tests,
with a real image and a real correlated background instead of a stamped square.

Run from repo root::

    PYTHONPATH=. python scripts/eval_robust_shortcut_dogs.py --num_images 400
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

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
from examples.contrastive_explanation import TV_INPUT_SIZE, checkpoint_metadata, load_checkpoint
from scripts.train_backbone import model_grid_size

IMAGES = REPO / "data" / "stanford_dogs" / "images" / "Images"
ANNOTS = REPO / "data" / "stanford_dogs" / "annotations" / "Annotation"


class DogsBoxDataset(torch.utils.data.Dataset):
    """Stanford Dogs images paired with the release's bounding box, as a binary mask."""

    def __init__(self, size: int = TV_INPUT_SIZE):
        from torchvision import transforms
        self.size = size
        self.t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.classes = sorted(os.listdir(IMAGES))
        self.samples: List[Tuple[str, int, Tuple[float, float, float, float]]] = []
        n_missing = 0
        for ci, breed in enumerate(self.classes):
            for f in sorted(glob.glob(str(IMAGES / breed / "*.jpg"))):
                ann = ANNOTS / breed / Path(f).stem
                if not ann.exists():
                    n_missing += 1
                    continue
                try:
                    root = ET.parse(ann).getroot()
                    w = float(root.find("size/width").text)
                    h = float(root.find("size/height").text)
                    b = root.find("object/bndbox")
                    box = (float(b.find("xmin").text) / w, float(b.find("ymin").text) / h,
                           float(b.find("xmax").text) / w, float(b.find("ymax").text) / h)
                except Exception:
                    n_missing += 1
                    continue
                self.samples.append((f, ci, box))
        if n_missing:
            print(f"WARNING: {n_missing} images had no usable annotation (skipped)")
        if not self.samples:
            raise FileNotFoundError(f"no annotated images under {IMAGES}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, label, (x0, y0, x1, y1) = self.samples[i]
        img = self.t(Image.open(path).convert("RGB"))
        m = torch.zeros(self.size, self.size)
        m[int(y0 * self.size):max(int(y1 * self.size), int(y0 * self.size) + 1),
          int(x0 * self.size):max(int(x1 * self.size), int(x0 * self.size) + 1)] = 1.0
        return img, label, m


def background_stats(x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """(B, 6) mean and std of RGB over the pixels outside the box."""
    bg = (1.0 - box).unsqueeze(1)                                  # (B,1,H,W)
    n = bg.sum(dim=(2, 3)).clamp_min(1.0)
    mu = (x * bg).sum(dim=(2, 3)) / n
    var = ((x - mu[:, :, None, None]) ** 2 * bg).sum(dim=(2, 3)) / n
    return torch.cat([mu, var.clamp_min(1e-8).sqrt()], dim=1)


def background_styles(ds, n_sample: int, k: int, seed: int) -> torch.Tensor:
    """k background styles, from k-means over measured background statistics."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_sample]
    feats = []
    for i in idx.tolist():
        x, _, m = ds[i]
        feats.append(background_stats(x.unsqueeze(0), m.unsqueeze(0))[0])
    X = torch.stack(feats)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(40):
        a = torch.cdist(X, C).argmin(dim=1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(dim=0)
    counts = [int((a == j).sum()) for j in range(k)]
    for j in range(k):
        print(f"  style {j}  n={counts[j]:<5} mean RGB {C[j, :3].numpy().round(3)}  "
              f"std {C[j, 3:].numpy().round(3)}")
    return C


def make_env_fn(styles: torch.Tensor):
    """Paired background-transfer environments; foreground inside the box is untouched."""
    def env_fn(x: torch.Tensor, box: torch.Tensor) -> EnvBatch:
        own = background_stats(x, box)                              # (B,6)
        mu, sd = own[:, :3], own[:, 3:].clamp_min(1e-6)
        bg = (1.0 - box).unsqueeze(1)
        xs = []
        for j in range(styles.shape[0]):
            tm = styles[j, :3].to(x.device).view(1, 3, 1, 1)
            ts = styles[j, 3:].to(x.device).view(1, 3, 1, 1)
            shifted = ((x - mu[:, :, None, None]) / sd[:, :, None, None]) * ts + tm
            xs.append((x * (1 - bg) + shifted.clamp(0, 1) * bg).clamp(0, 1))
        return EnvBatch(xs=xs, env_ids=[f"bgstyle{j}" for j in range(styles.shape[0])])
    return env_fn


def main() -> None:
    p = argparse.ArgumentParser(description="Robust/shortcut on Stanford Dogs backgrounds")
    p.add_argument("--checkpoint", type=str,
                   default="examples/out/checkpoints/stanford_dogs_resnet18.pt")
    p.add_argument("--num_images", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--game_mode", type=str, default="mixed",
                   choices=["mixed", "cooperative", "competitive"])
    p.add_argument("--lambda_sparse", type=float, default=1.0,
                   help="Override the preset's 0.05: RobustShortcutObjective has no mass "
                        "target, so at 0.05 both masks blanket the frame and every "
                        "spatial statistic just reports area.")
    p.add_argument("--num_styles", type=int, default=3)
    p.add_argument("--style_images", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=str(REPO / "results" / "shift_real_dogs"))
    p.add_argument("--export_prefix", type=str, default="shift_dogs_background")
    args = p.parse_args()

    device = get_device()
    meta = checkpoint_metadata(Path(args.checkpoint))
    model, dataset_name, class_names, num_classes = load_checkpoint(Path(args.checkpoint), device)
    if dataset_name != "stanford_dogs":
        raise SystemExit(f"checkpoint is for '{dataset_name}'; this script needs stanford_dogs")
    gh, gw = model_grid_size(meta["model_name"])
    print(f"Loaded {dataset_name} / {meta['model_name']} / {num_classes} classes, grid {gh}x{gw}")

    ds = DogsBoxDataset()
    print(f"{len(ds)} annotated images over {len(ds.classes)} breeds")
    print("Measuring background styles:")
    styles = background_styles(ds, args.style_images, args.num_styles, args.seed)
    env_fn = make_env_fn(styles)

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
        lambda_sparse=args.lambda_sparse)
    allocator = RobustShortcutOptimizationAllocator(
        objective, num_steps=args.num_steps, lr=args.lr, lambda_disjoint=cfg.lambda_disjoint)

    agg: Dict[str, List[float]] = defaultdict(list)
    per_image: List[Dict[str, float]] = []
    flips, sens, seen = [], [], 0
    for x, y, box in loader:
        x, y, box = x.to(device), y.to(device), box.to(device)
        env = env_fn(x, box)
        with torch.no_grad():
            logits = model(x)
            hyp = selector.select(logits, torch.softmax(logits, dim=-1))
            base = logits.argmax(-1)
            # Precondition: does swapping only the background move the prediction?
            for xe in env.xs:
                flips.append(float((model(xe).argmax(-1) != base).float().mean()))
            # Checkpoint health: a model whose logits ignore the image cannot be
            # interrogated by masking (see ablation_effnet/BROKEN_ep10_lr1e-3/README.md).
            one = torch.zeros(x.shape[0], gh * gw, device=device); one[:, (gh * gw) // 2] = 1.0
            allm = torch.ones(x.shape[0], gh * gw, device=device)
            sens.append(float((model(unit_space.keep(x, allm)).gather(1, base[:, None])
                               - model(unit_space.keep(x, one)).gather(1, base[:, None])
                               ).abs().mean()))
        xg = x.detach().clone().requires_grad_(True)
        ev = provider.explain(xg, model, hyp).detach()
        ev = ev / ev.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        masks = allocator.allocate(x=x, model=model, unit_space=unit_space,
                                   hypotheses=hyp, evidence=ev, env=env)
        m = objective.compute(x=x, model=model, unit_space=unit_space, hypotheses=hyp,
                              masks=masks, evidence=ev, env=env)
        for k, v in m.items():
            agg[k].append(float(v.detach()) if torch.is_tensor(v) else float(v))

        H, W = box.shape[-2:]
        vals = {}
        for name, field in (("robust", masks["robust"]), ("shortcut", masks["shortcut"]),
                            ("base_evidence", ev[:, 0])):
            pm = F.interpolate(field.view(field.shape[0], 1, gh, gw), size=(H, W),
                               mode="bilinear", align_corners=False).squeeze(1)
            vals[name] = ((pm * box).sum(dim=(1, 2))
                          / pm.sum(dim=(1, 2)).clamp_min(1e-8)).detach().cpu()
        agg["box_area_fraction"].extend(box.mean(dim=(1, 2)).cpu().tolist())
        for b in range(x.shape[0]):
            per_image.append({k: float(v[b]) for k, v in vals.items()})
        seen += x.shape[0]
        print(f"  {seen}/{len(sub)}", end="\r", flush=True)

    rob = torch.tensor([r["robust"] for r in per_image])
    sho = torch.tensor([r["shortcut"] for r in per_image])
    d = rob - sho
    t = float(d.mean() / (d.std(unbiased=True) / (len(d) ** 0.5)).clamp_min(1e-12))
    summary = {
        "checkpoint": args.checkpoint, "dataset": dataset_name, "model": meta["model_name"],
        "grid": [gh, gw], "game_mode": args.game_mode, "lambda_sparse": args.lambda_sparse,
        "num_steps": args.num_steps, "lr": args.lr, "seed": args.seed,
        "num_styles": args.num_styles, "n": len(per_image),
        "background_swap_pred_flip_rate": sum(flips) / len(flips),
        "checkpoint_sensitivity_all_vs_one_region": sum(sens) / len(sens),
        "objective": {k: sum(v) / len(v) for k, v in agg.items()
                      if k not in ("box_area_fraction",)},
        "in_box": {
            "robust": float(rob.mean()), "shortcut": float(sho.mean()),
            "base_evidence": float(torch.tensor([r["base_evidence"] for r in per_image]).mean()),
            "box_area_fraction": float(torch.tensor(agg["box_area_fraction"]).mean()),
        },
        "robust_minus_shortcut": {"delta": float(d.mean()), "t": t,
                                  "win_rate": float((d > 0).float().mean()), "n": len(d)},
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.export_prefix}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{args.export_prefix}_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys())); w.writeheader()
        w.writerows(per_image)

    print(f"\n--- preconditions ---")
    print(f"  background swap flips predictions   {summary['background_swap_pred_flip_rate']:.1%}")
    print(f"  checkpoint sensitivity |z(all)-z(1)| {summary['checkpoint_sensitivity_all_vs_one_region']:.3f}"
          f"   (healthy references ~3.1-3.4)")
    print("--- objective (mean over batches) ---")
    for k, v in summary["objective"].items():
        print(f"  {k:<16}{v:>10.4f}")
    print("--- does the split land on the dog? (mask mass inside the box) ---")
    ib = summary["in_box"]
    print(f"  box area fraction (chance)  {ib['box_area_fraction']:.4f}")
    print(f"  base evidence               {ib['base_evidence']:.4f}")
    print(f"  robust mask                 {ib['robust']:.4f}")
    print(f"  shortcut mask               {ib['shortcut']:.4f}")
    r = summary["robust_minus_shortcut"]
    print(f"  robust - shortcut  {r['delta']:+.4f}  t={r['t']:.1f}  "
          f"wins {r['win_rate']:.1%} of {r['n']} images")
    print(f"\nSaved {out / (args.export_prefix + '.json')}")


if __name__ == "__main__":
    main()
