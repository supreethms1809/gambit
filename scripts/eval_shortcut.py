"""
scripts/eval_shortcut.py

A benchmark with **exact ground truth for class-unique evidence** — the thing every
other evaluation in this project has lacked.

The problem with expert segmentations is structural, not practical: a lesion outline
marks where the lesion is, and on HAM10000 all seven classes *are* lesions, so the
outline is ground truth for **shared** evidence. Nothing in any medical dataset here
annotates what makes melanoma melanoma rather than a nevus. So "does unique_k land on
the right place?" has never had a right answer to check against.

Here we manufacture one. A small distinctive patch is pasted into a fraction of one
class's training images at **randomized positions**. The model learns to use it — we
verify that it did — and then the question has an exact answer: does ``unique_k`` land
on the patch? Two properties make this a clean test where lesion overlap was not:

* **Ground truth is known per image, to the pixel**, and it is genuinely *unique* to
  one class rather than shared across all of them.
* **Position is randomized**, so the centre prior that makes HAM10000 unmeasurable
  (a fixed centre rectangle scores 0.94 there) is absent by construction. The
  degenerate baselines have nothing to exploit.

It also lets the ~13-cells-per-target rule — inferred from two medical datasets — be
checked against a target whose size we control exactly.

Reported per method: share of mask mass inside the patch, against the same null
family used elsewhere (uniform, fixed centre cell, position-scrambled copy).

Run from repo root::

    PYTHONPATH=. python scripts/eval_shortcut.py --grid 28 --evidence ig \\
        --out_dir results/shortcut
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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
from scripts.ablation_contrastive import _build_model

TV = 224


# --------------------------------------------------------------------------- data
def make_patch(kind: str, size: int, generator: torch.Generator) -> torch.Tensor:
    """A small, highly learnable marker. (3, size, size) in [0, 1]."""
    if kind == "solid":
        # Saturated magenta: far from natural image statistics, trivially separable.
        p = torch.zeros(3, size, size)
        p[0], p[2] = 1.0, 1.0
    elif kind == "checker":
        p = torch.zeros(3, size, size)
        c = torch.arange(size)
        board = ((c[:, None] // 4 + c[None, :] // 4) % 2).float()
        p[0] = board
        p[1] = 1.0 - board
    elif kind == "noise":
        p = torch.rand(3, size, size, generator=generator)
    else:
        raise ValueError("patch_type must be solid|checker|noise")
    return p


class ShortcutDataset(torch.utils.data.Dataset):
    """CIFAR-10 resized to 224 with an optional planted patch on one class.

    Returns (image, label, patch_mask) where ``patch_mask`` is a binary (H, W) map of
    where the patch was pasted — all zeros when this sample was not patched.
    """

    def __init__(self, inner, target_class: int, rate: float, patch: torch.Tensor,
                 seed: int, patch_all: bool = False, margin: int = 8):
        self.inner = inner
        self.target_class = target_class
        self.rate = rate
        self.patch = patch
        self.ps = patch.shape[-1]
        self.margin = margin
        self.patch_all = patch_all      # paste on every class (used for the attack test)
        g = torch.Generator().manual_seed(seed)
        n = len(inner)
        self.coin = torch.rand(n, generator=g)
        span = TV - self.ps - 2 * margin
        # Positions are drawn uniformly, so the patch has no centre bias whatsoever.
        self.pos = torch.randint(0, max(span, 1), (n, 2), generator=g) + margin

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i):
        img, label = self.inner[i]
        if img.shape[-1] != TV:
            img = F.interpolate(img.unsqueeze(0), size=(TV, TV),
                                mode="bilinear", align_corners=False).squeeze(0)
        pm = torch.zeros(TV, TV)
        want = self.patch_all or (label == self.target_class and self.coin[i] < self.rate)
        if want:
            y, x = int(self.pos[i, 0]), int(self.pos[i, 1])
            img = img.clone()
            img[:, y:y + self.ps, x:x + self.ps] = self.patch
            pm[y:y + self.ps, x:x + self.ps] = 1.0
        return img, label, pm


def cifar(train: bool, n: Optional[int], seed: int):
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    t = transforms.Compose([transforms.ToTensor()])
    ds = CIFAR10(root=str(REPO / "data"), train=train, download=False, transform=t)
    if n and n < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n]
        ds = torch.utils.data.Subset(ds, idx.tolist())
    return ds


# --------------------------------------------------------------------------- train
def train(model, loader, epochs, lr, device):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for ep in range(epochs):
        tot = cor = n = 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            tot += loss.item(); cor += (out.argmax(1) == y).sum().item(); n += len(y)
        print(f"  [train] epoch {ep+1}/{epochs} loss={tot/max(len(loader),1):.4f} acc={cor/max(n,1):.3f}")
    model.eval()
    return model


@torch.no_grad()
def attack_success(model, ds_all_patched, target: int, device, batch: int = 32) -> float:
    """Share of NON-target images predicted as the target class once patched.

    This is the check that the experiment is even meaningful: if the model did not
    learn the shortcut, asking where its evidence for the shortcut lies is vacuous.
    """
    dl = torch.utils.data.DataLoader(ds_all_patched, batch_size=batch, num_workers=0)
    hit = n = 0
    for x, y, _ in dl:
        sel = y != target
        if not bool(sel.any()):
            continue
        pred = model(x[sel].to(device)).argmax(1).cpu()
        hit += (pred == target).sum().item(); n += int(sel.sum())
    return hit / max(n, 1)


# --------------------------------------------------------------------------- eval
from evaluation.masks import mass_in, regions_to_pixels
from evaluation.nulls import random_translate


def roll_masks(m: torch.Tensor, gh: int, gw: int, g: torch.Generator) -> torch.Tensor:
    """Position-scrambling null: same shape, same budget, random location."""
    return random_translate(m, g, gh, gw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic-shortcut ground-truth benchmark")
    ap.add_argument("--target_class", type=int, default=0)
    ap.add_argument("--shortcut_rate", type=float, default=1.0,
                    help="Fraction of target-class TRAIN images that receive the patch")
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--patch_type", type=str, default="solid",
                    choices=["solid", "checker", "noise"])
    ap.add_argument("--grid", type=int, default=28)
    ap.add_argument("--evidence", type=str, default="ig",
                    choices=["gradcam", "ig", "occlusion"])
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--occlusion_mode", type=str, default="single", choices=["single", "rise"])
    ap.add_argument("--model", dest="model_name", type=str, default="resnet18")
    ap.add_argument("--train_images", type=int, default=5000)
    ap.add_argument("--eval_images", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=12)
    ap.add_argument("--num_alloc_steps", type=int, default=50)
    ap.add_argument("--alloc_lr", type=float, default=0.2)
    ap.add_argument("--game_mode", type=str, default="mixed")
    ap.add_argument("--lambda_mass", type=float, default=2.0)
    ap.add_argument("--lambda_shared_sparse", type=float, default=0.25,
                    help="L1 penalty on the shared mask. The default 0.25 closes the "
                         "shared-mask blanket. At 0.0 the shared mask is unpenalized.")
    ap.add_argument("--control", action="store_true",
                    help="Negative control: train WITHOUT the shortcut but evaluate on "
                         "patched images. Evidence should not concentrate on the patch.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=str(REPO / "results" / "shortcut"))
    ap.add_argument("--export_prefix", type=str, default=None)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    print("Device:", device)

    g = torch.Generator().manual_seed(args.seed)
    patch = make_patch(args.patch_type, args.patch_size, g)
    frac = (args.patch_size ** 2) / float(TV * TV)
    cell = 1.0 / (args.grid ** 2)
    print(f"patch {args.patch_size}px = {frac*100:.2f}% of frame; at {args.grid}x{args.grid} "
          f"it spans {frac/cell:.1f} cells")

    train_rate = 0.0 if args.control else args.shortcut_rate
    tr = ShortcutDataset(cifar(True, args.train_images, args.seed), args.target_class,
                         train_rate, patch, seed=args.seed)
    tr_dl = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f"\nTraining {args.model_name} ({'CONTROL — no shortcut' if args.control else f'shortcut on class {args.target_class}, rate {train_rate}'})")
    model = _build_model(args.model_name, 10, pretrained=True).to(device)
    model = train(model, tr_dl, args.epochs, args.lr, device)

    # Did the model actually learn it? Paste the patch on every class and see how often
    # the prediction flips to the target.
    va_all = ShortcutDataset(cifar(False, args.eval_images, args.seed + 1), args.target_class,
                             1.0, patch, seed=args.seed + 1, patch_all=True)
    asr = attack_success(model, va_all, args.target_class, device)
    print(f"attack success rate (non-target images predicted as class {args.target_class} "
          f"when patched): {asr:.3f}")

    # Evaluate on target-class val images that carry the patch.
    va = ShortcutDataset(cifar(False, args.eval_images, args.seed + 1), args.target_class,
                         1.0, patch, seed=args.seed + 1)
    keep = [i for i in range(len(va)) if va[i][2].sum() > 0]
    va = torch.utils.data.Subset(va, keep)
    dl = torch.utils.data.DataLoader(va, batch_size=args.eval_batch, num_workers=0)
    print(f"scoring {len(va)} patched val images of class {args.target_class}")

    gh = gw = args.grid
    cfg = resolve_contrastive_game(args.game_mode)
    unit_space = VisionGridUnitSpace(gh, gw, baseline="blur")
    if args.evidence == "ig":
        prov = IntegratedGradientsRegionsProvider(grid_h=gh, grid_w=gw,
                                                  steps=args.ig_steps, baseline="zero")
    elif args.evidence == "occlusion":
        prov = OcclusionRegionsProvider(grid_h=gh, grid_w=gw, unit_space=unit_space,
                                        mode=args.occlusion_mode, seed=args.seed)
    else:
        prov = GradCAMRegionsProvider(grid_h=gh, grid_w=gw)
    obj = ContrastiveObjective(lambda_suff=1.0, lambda_margin=cfg.lambda_margin,
                               lambda_sparse=0.05, lambda_overlap=cfg.lambda_overlap,
                               lambda_mass=args.lambda_mass,
                               lambda_shared_sparse=args.lambda_shared_sparse)
    alloc = OptimizationAllocator(obj, num_steps=args.num_alloc_steps, lr=args.alloc_lr,
                                  use_shared=cfg.use_shared,
                                  lambda_disjoint=cfg.lambda_disjoint,
                                  lambda_partition=cfg.lambda_partition)
    expl = CDEAExplainer(model=model, unit_space=unit_space, selector=TopMSelector(m=5),
                         base_evidence=prov, allocator=alloc, objective=obj,
                         normalize_evidence=True, device=device)

    rng = torch.Generator().manual_seed(args.seed)
    scores: Dict[str, List[float]] = {k: [] for k in
                                      ["uniform", "center_cell", "base_evidence",
                                       "cdea_shared", "cdea_unique",
                                       "cdea_unique_translated"]}
    patch_area: List[float] = []

    for x, y, pm in dl:
        x, pm = x.to(device), pm.to(device)
        B, _, H, W = x.shape
        e = expl.explain(x)
        ids = e.hypotheses.ids
        # Score the mask belonging to the *planted* class wherever it is in the top-K.
        tgt = torch.full((B,), args.target_class, device=ids.device)
        hit = (ids == tgt.unsqueeze(1))
        has = hit.any(dim=1)
        if not bool(has.any()):
            continue
        slot = hit.float().argmax(dim=1)
        sel = has
        uni = e.masks["unique"][torch.arange(B, device=ids.device), slot]      # (B, R)
        base = e.extras["evidence"][torch.arange(B, device=ids.device), slot]

        scores["cdea_unique"] += mass_in(regions_to_pixels(uni[sel], gh, gw, H, W), pm[sel]).tolist()
        scores["base_evidence"] += mass_in(regions_to_pixels(base[sel], gh, gw, H, W), pm[sel]).tolist()
        rolled = roll_masks(uni[sel].detach().cpu(), gh, gw, rng).to(device)
        scores["cdea_unique_translated"] += mass_in(
            regions_to_pixels(rolled, gh, gw, H, W), pm[sel]).tolist()
        if "shared" in e.masks:
            scores["cdea_shared"] += mass_in(
                regions_to_pixels(e.masks["shared"][sel], gh, gw, H, W), pm[sel]).tolist()
        scores["uniform"] += mass_in(torch.ones(int(sel.sum()), H, W, device=device), pm[sel]).tolist()
        cc = torch.zeros(int(sel.sum()), H, W, device=device)
        ch, cw = H // gh, W // gw
        cc[:, (gh // 2) * ch:(gh // 2 + 1) * ch, (gw // 2) * cw:(gw // 2 + 1) * cw] = 1.0
        scores["center_cell"] += mass_in(cc, pm[sel]).tolist()
        patch_area += pm[sel].mean(dim=(1, 2)).tolist()

    def stat(v):
        if not v: return {"mean": float("nan"), "std": float("nan"), "n": 0}
        t = torch.tensor(v); return {"mean": t.mean().item(), "std": t.std().item(), "n": len(v)}

    print("\n--- share of mask mass inside the planted patch ---")
    print(f"{'method':<26}{'mean':>9}{'std':>9}{'n':>7}")
    rows = []
    for k in ["uniform", "center_cell", "base_evidence", "cdea_shared",
              "cdea_unique", "cdea_unique_translated"]:
        if not scores[k]:
            continue
        s = stat(scores[k])
        print(f"{k:<26}{s['mean']:>9.4f}{s['std']:>9.4f}{s['n']:>7}")
        rows.append({"method": k, **s})

    paired = []
    for a, b in [("cdea_unique", "base_evidence"), ("cdea_unique", "uniform"),
                 ("cdea_unique", "cdea_unique_translated"), ("cdea_unique", "center_cell")]:
        if len(scores[a]) != len(scores[b]) or not scores[a]:
            continue
        d = torch.tensor(scores[a]) - torch.tensor(scores[b])
        n = d.numel(); md = d.mean().item()
        se = (d.std(unbiased=True) / n ** 0.5).item() if n > 1 else 0.0
        t = md / se if se > 0 else float("nan")
        print(f"{a} vs {b}: delta={md:+.4f} t={t:.2f} win={(d>0).float().mean():.3f} n={n}")
        paired.append({"comparison": f"{a}_vs_{b}", "delta": md, "se": se,
                       "t": t, "win_rate": (d > 0).float().mean().item(), "n": n})

    prefix = args.export_prefix or (
        f"shortcut_{'control' if args.control else 'planted'}_{args.evidence}_g{args.grid}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_rows_csv(out / f"{prefix}.csv", rows)
    save_json(out / f"{prefix}.json", {
        "rows": rows, "paired": paired, "per_image": scores,
        "attack_success_rate": asr,
        "mean_patch_area_fraction": (sum(patch_area) / len(patch_area)) if patch_area else None,
        "control": bool(args.control),
        "target_class": args.target_class, "shortcut_rate": train_rate,
        "patch_size": args.patch_size, "patch_type": args.patch_type,
        "patch_frame_fraction": frac, "cells_per_patch": frac / cell,
        "grid": [gh, gw], "evidence": args.evidence, "model": args.model_name,
        "game_mode": args.game_mode, "num_alloc_steps": args.num_alloc_steps,
        "train_images": args.train_images, "epochs": args.epochs, "seed": args.seed,
    })
    print(f"\nSaved {out / (prefix + '.csv')}")


if __name__ == "__main__":
    main()
