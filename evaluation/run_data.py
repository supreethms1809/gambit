"""Images for one paper cell.

The sample for seed ``s`` is a seeded draw with generator seed ``1000 + s``
(EVAL_PLAN.md section 2.4), the same for every method. The test split goes
through the same lock as every other loader, so ``split="test"`` raises until
the plan is frozen and ``final`` is set with the frozen config hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"
IMAGE_SIZE = 224
SAMPLE_SEED_OFFSET = 1000


def sample_seed(seed: int) -> int:
    return SAMPLE_SEED_OFFSET + int(seed)


@dataclass
class ContrastiveSample:
    x: torch.Tensor          # (B, 3, 224, 224), raw [0, 1]
    labels: torch.Tensor     # (B,)
    index: list[int]         # index into the split's dataset


def contrastive_sample(
    dataset: str,
    split: str,
    n: int,
    seed: int,
    *,
    final: bool = False,
    config_hash: Optional[str] = None,
) -> ContrastiveSample:
    from evaluation.datasets import get_eval_loader

    loader, _ = get_eval_loader(
        dataset, max(1, n), DATA_ROOT, image_size=IMAGE_SIZE,
        seed=sample_seed(seed), num_images=n, split=split,
        final=final, config_hash=config_hash,
    )
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(torch.as_tensor(y))
    index = root_indices(loader.dataset)
    return ContrastiveSample(x=torch.cat(xs)[:n], labels=torch.cat(ys)[:n].long(), index=index[:n])


def root_indices(ds) -> list[int]:
    """Indices into the underlying dataset, through any nesting of Subsets."""
    from torch.utils.data import Subset

    if not isinstance(ds, Subset):
        return list(range(len(ds)))
    inner = root_indices(ds.dataset)
    return [inner[i] for i in ds.indices]


def _resize(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == IMAGE_SIZE and x.shape[-2] == IMAGE_SIZE:
        return x
    return F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).clamp(0, 1)


def _draw(length: int, n: int, seed: int) -> list[int]:
    from evaluation.sampling import seeded_indices

    return seeded_indices(length, n, sample_seed(seed))


