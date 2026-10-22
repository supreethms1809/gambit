"""Dump this formulation's pair maps for a sample exported from ``main`` (archive branch only).

``main`` scores these maps in its decomposition-v2 partition game
(``precomputed_maps``). This script runs the archived method as it was: the
plan's default ``CdeaConfig`` (preset "mixed", top-5 hypotheses, Grad-CAM on
ResNet-50, 50 steps at lr 0.2, lambda_margin 1.0, lambda_overlap 0.2,
lambda_shared_sparse 0.25), through ``evaluation.run_methods.cdea_pair``.

The sample file holds ``x`` (images in [0, 1], as main's sampler gives them),
``labels`` and ``index``. The output ``.npz`` holds, per image:
``index``, ``class_k`` and ``class_l`` (this run's rank 0 and rank 1),
``k`` and ``l`` (their unique masks), ``shared``, and ``others`` (the largest
unique mask of ranks 2 to K-1 at each region), all on the 7 x 7 grid.

    PYTHONPATH=. python scripts/dump_pair_maps.py --sample cifar10_sample.pt --dataset cifar10 \\
        --checkpoint <ckpt> --out cifar10_framing_maps.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv=None) -> None:
    import numpy as np
    import torch

    from evaluation.run_methods import Knobs, cdea_pair, hypotheses_for
    from evaluation.run_models import load_cell_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=5)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    sample = torch.load(args.sample, map_location="cpu")
    model = load_cell_model(args.dataset, args.backbone, 0, source="checkpoint", checkpoint=args.checkpoint,
                            device=device).model
    keys = ("class_k", "class_l", "k", "l", "shared", "others")
    out = {key: [] for key in keys}
    for start in range(0, sample["x"].shape[0], args.batch):
        x = sample["x"][start:start + args.batch].to(device)
        h = hypotheses_for(model, x, 5)
        maps = cdea_pair(model, x, h, args.backbone, Knobs(), device)
        unique = maps.per_class.detach().cpu()
        out["class_k"].append(h.ids[:, 0].cpu())
        out["class_l"].append(h.ids[:, 1].cpu())
        out["k"].append(unique[:, 0])
        out["l"].append(unique[:, 1])
        out["shared"].append(maps.extra["shared"].detach().cpu())
        out["others"].append(unique[:, 2:].amax(dim=1))
    arrays = {key: torch.cat(value).numpy() for key, value in out.items()}
    arrays["index"] = sample["index"].numpy()
    np.savez(args.out, **arrays)
    print(f"wrote {args.out}: {arrays['k'].shape[0]} images, k mass {arrays['k'].sum(-1).mean():.3f}, "
          f"shared mass {arrays['shared'].sum(-1).mean():.3f}")


if __name__ == "__main__":
    main()
