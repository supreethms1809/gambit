"""Save the first kept image of the adjacent-condition val draw.

Seed 0, the first image the sample rule keeps. It is not chosen by looking at
the labels. Writes the image, the two reference maps, subtraction's labels,
CDEA's labels, and the per-cell excess drops on Omega.

    PYTHONPATH=. python scripts/adjacent_example.py --out results/paper/csci/example
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.device import get_device
from core.grid import grid_of
from core.types import HypothesisSet
from evaluation.run_data import decomposition_sample
from evaluation.run_methods import (
    PartitionContext,
    cdea_partition,
    hypotheses_for,
    partition_reference,
    subtract_partition,
)
from evaluation.run_models import load_cell_model
from evaluation.run_partition import labels_string


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    loaded = load_cell_model("planted_pair_adjacent", "resnet50", args.seed, device=device)
    sample = decomposition_sample("planted_pair_adjacent", "val", 1, args.seed, model=loaded.model, device=device)
    x = sample.x[:1].to(device)
    top = hypotheses_for(loaded.model, x, 3)
    h = HypothesisSet(ids=top.ids[:, :2].to(device), mask=top.mask[:, :2].to(device))
    from evaluation.run_methods import Knobs

    grid = grid_of("resnet50")
    knobs = Knobs()
    evidence = partition_reference(loaded.model, x, h, "resnet50", knobs)
    from cdea.region import evidence_region
    omega = evidence_region(evidence)
    ctx = PartitionContext(evidence=evidence, omega=omega, rows=[0], grid=grid)
    subtract = subtract_partition(loaded.model, x, h, "resnet50", knobs, device, ctx)
    cdea = cdea_partition(loaded.model, x, h, "resnet50", knobs, device, ctx, seed=args.seed)
    from baselines.decomposition import occlusion_drops
    g = occlusion_drops(loaded.model, x, h.ids, *grid)
    torch.save({
        "seed": args.seed,
        "image_index": int(sample.index[0]),
        "class_k": int(h.ids[0, 0]),
        "class_l": int(h.ids[0, 1]),
        "image": x[0].cpu(),
        "evidence": evidence[0].cpu(),
        "omega": omega[0].cpu(),
        "g": g[0].cpu(),
        "subtract": subtract.roles[0].cpu(),
        "cdea": cdea.roles[0].cpu(),
        "subtract_labels": labels_string(subtract.roles[0]),
        "cdea_labels": labels_string(cdea.roles[0]),
    }, out / "image0.pt")
    _png(x[0].cpu(), out / "image0.png")
    (out / "README.txt").write_text(
        "First kept image of planted_pair_adjacent val, seed 0. Not selected.\n"
        f"image_index {int(sample.index[0])}\n"
        f"subtract {labels_string(subtract.roles[0])}\n"
        f"cdea {labels_string(cdea.roles[0])}\n"
        "image0.pt holds the reference maps (evidence), Omega, the per-cell excess g, and both label grids.\n"
    )
    print(f"wrote {out}", flush=True)


def _png(image: torch.Tensor, path: Path) -> None:
    from torchvision.utils import save_image
    save_image(image.clamp(0, 1), path)


if __name__ == "__main__":
    main()
