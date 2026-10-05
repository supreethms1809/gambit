"""Foil masks for family C.

Rank 0 is the kept class and rank 1 is the foil. Later ranks are not part of
the pair. The primary score uses the unique masks only: the shared mask
supports both classes by construction, so it says nothing about "k rather
than l" and would only spend area on both sides. Adding it to both sides is
the reported sensitivity check (``include_shared=True``).
"""

from __future__ import annotations

import torch

from baselines.hypotheses import foil_pair
from core.types import HypothesisSet
from evaluation.masks import to_budget_mask


def pair_scores(
    unique: torch.Tensor,
    shared: torch.Tensor | None = None,
    include_shared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Region scores ``(B, R)`` for the kept class and the foil.

    ``unique`` is ``(B, K, R)`` with K >= 2. ``shared`` is added to both
    classes only when ``include_shared`` is true (the sensitivity check).
    """
    if unique.ndim != 3 or unique.shape[1] < 2:
        raise ValueError("unique masks need a kept class and a foil")
    kept = unique[:, 0]
    foil = unique[:, 1]
    if include_shared and shared is not None:
        if shared.shape != kept.shape:
            raise ValueError("shared mask must be (batch, regions)")
        kept = kept + shared
        foil = foil + shared
    return kept, foil


def pair_budget_masks(
    unique: torch.Tensor,
    shared: torch.Tensor | None,
    fraction: float,
    seed: int,
    include_shared: bool = False,
    **upsample,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both sides of the pair go through the same area conversion."""
    kept, foil = pair_scores(unique, shared, include_shared)
    mask_k = to_budget_mask(kept, fraction, seed, **upsample)
    mask_l = to_budget_mask(foil, fraction, seed, **upsample)
    return mask_k, mask_l


def pair_classes(hypotheses: HypothesisSet) -> tuple[torch.Tensor, torch.Tensor]:
    """Class ids for rank 0 and rank 1. The caller does not pick a different foil."""
    return foil_pair(hypotheses)
