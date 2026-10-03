"""Seeded subsets. A prefix of a class-sorted folder is not a sample."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset


def seeded_indices(n_items: int, n_take: int, seed: int) -> list[int]:
    """Return ``min(n_take, n_items)`` indices from a seeded permutation."""
    if n_items < 0 or n_take < 0:
        raise ValueError("n_items and n_take must be >= 0")
    if n_items == 0 or n_take == 0:
        return []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_items, generator=generator)
    return perm[: min(n_take, n_items)].tolist()


def seeded_subset(dataset: Dataset, n: int, seed: int) -> Subset:
    """Random subset of ``dataset``. Indices are into ``dataset``, not a class prefix."""
    return Subset(dataset, seeded_indices(len(dataset), n, seed))
