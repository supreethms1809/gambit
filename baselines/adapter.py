"""The only path from a score map to a budgeted pixel mask.

CDEA masks and baseline maps both call ``adapt_scores``, which calls
``evaluation.masks.to_budget_mask``. A failed row is replaced with the random
floor, a seeded area-a mask, and the row stays in the batch.
"""

from __future__ import annotations

import torch

from evaluation.masks import to_budget_mask


def adapt_scores(
    scores: torch.Tensor,
    fraction: float,
    seed: int,
    *,
    grid_h: int | None = None,
    grid_w: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    """Budgeted mask. Region maps are bilinearly upsampled, then top-a with a seeded tie-break."""
    return to_budget_mask(
        scores,
        fraction,
        seed,
        grid_h=grid_h,
        grid_w=grid_w,
        height=height,
        width=width,
    )


def row_failed(scores: torch.Tensor) -> torch.Tensor:
    """Per-image failure. True when the row is empty or contains a non-finite value.

    A constant finite map is not a failure: the seeded tie-break still assigns
    area ``fraction``. An all-negative finite map is not a failure either.
    Ranking does not clamp, so the least-negative entries win.
    """
    if scores.ndim < 1:
        raise ValueError("scores need a batch dimension")
    flat = scores.detach().reshape(scores.shape[0], -1)
    if flat.shape[-1] == 0:
        return torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    return ~torch.isfinite(flat).all(dim=-1)


def random_floor(
    like: torch.Tensor,
    fraction: float,
    seed: int,
    *,
    grid_h: int | None = None,
    grid_w: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    """Seeded area-a mask, the score used for a failed image.

    A constant field makes every entry a tie, so ``adapt_scores`` draws the
    area from the seed. The same seed reproduces the same floor.
    """
    return adapt_scores(
        torch.ones_like(like),
        fraction,
        seed,
        grid_h=grid_h,
        grid_w=grid_w,
        height=height,
        width=width,
    )


def budget_or_floor(
    scores: torch.Tensor,
    fraction: float,
    seed: int,
    *,
    grid_h: int | None = None,
    grid_w: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(masks, failed)``. Failed rows become the random floor. No row is dropped.

    Non-finite entries are zeroed only so the shared conversion can run. Those
    rows are then overwritten by the floor, so the zeros never become the score.
    """
    failed = row_failed(scores)
    safe = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    kwargs = dict(grid_h=grid_h, grid_w=grid_w, height=height, width=width)
    masks = adapt_scores(safe, fraction, seed, **kwargs)
    if bool(failed.any()):
        floor = random_floor(safe, fraction, seed, **kwargs)
        shape = (scores.shape[0],) + (1,) * (masks.ndim - 1)
        masks = torch.where(failed.view(shape), floor, masks)
    return masks, failed
