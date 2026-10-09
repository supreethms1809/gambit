"""One conversion from a score map to a pixel mask, used for every method."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def regions_to_pixels(
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Upsample a region mask ``(B, R)`` to pixels ``(B, H, W)``.

    The shared scorer uses bilinear. Callers that need a hard deletion grid
    (the faithfulness curve) pass ``mode='nearest'``.
    """
    if mask.shape[-1] != grid_h * grid_w:
        raise ValueError(
            f"mask has {mask.shape[-1]} regions, grid is {grid_h}x{grid_w}"
        )
    pixel = mask.reshape(mask.shape[0], 1, grid_h, grid_w).float()
    kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
    up = F.interpolate(pixel, size=(height, width), mode=mode, **kwargs)
    return up.squeeze(1)


def mass_in(mask_px: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Share of positive mask mass inside ``target``. Both are ``(B, H, W)``."""
    mask_px = mask_px.clamp_min(0.0)
    total = mask_px.sum(dim=(1, 2))
    inside = (mask_px * target).sum(dim=(1, 2))
    return torch.where(total > 0, inside / total.clamp_min(1e-8), torch.zeros_like(total))


def advance_generator(generator: torch.Generator, batch: int, tail: tuple[int, ...], *, normal: bool) -> None:
    """Draw ``batch`` rows of shape ``tail`` and drop them.

    A later draw of the same rank then matches the corresponding rows of one
    draw over the whole sample. ROAD noise and tie-breaks use that order, so a
    chunked score stays the full-sample score.
    """
    if batch < 0:
        raise ValueError("batch must be >= 0")
    draw = torch.randn if normal else torch.rand
    left = int(batch)
    while left:
        take = min(left, 8)
        draw((take, *tail), generator=generator)
        left -= take


def top_fraction_mask(scores: torch.Tensor, fraction: float, seed: int, offset: int = 0) -> torch.Tensor:
    """Binary mask covering ``fraction`` of the entries in each row.

    ``scores`` is ``(B, ...)``. Ties are broken by a seeded jitter that does not
    reorder strict inequalities: values are stably sorted, and equal values keep
    a random order drawn from ``seed``. Negatives are ranked as they are. There
    is no clamp before ranking, so an all-negative map still yields a mask of
    area ``fraction`` on its least-negative entries. A failed or empty map is the
    caller's responsibility. ``fraction == 0`` returns an all-zero mask.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    flat = scores.detach().reshape(scores.shape[0], -1).float().cpu()
    n = flat.shape[-1]
    k = int(round(fraction * n))
    out = torch.zeros_like(flat)
    if k <= 0 or n == 0:
        return out.view_as(scores).to(device=scores.device, dtype=scores.dtype)
    if k >= n:
        return torch.ones_like(scores)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if offset:
        advance_generator(generator, int(offset), (n,), normal=False)
    jitter = torch.rand(flat.shape, generator=generator)
    tie_order = jitter.argsort(dim=-1, stable=True)
    ordered_scores = flat.gather(-1, tie_order)
    rank = ordered_scores.argsort(dim=-1, stable=True, descending=True)
    chosen = tie_order.gather(-1, rank[:, :k])
    out.scatter_(-1, chosen, 1.0)
    return out.view_as(scores).to(device=scores.device, dtype=scores.dtype)


def to_budget_mask(
    scores: torch.Tensor,
    fraction: float,
    seed: int,
    *,
    grid_h: int | None = None,
    grid_w: int | None = None,
    height: int | None = None,
    width: int | None = None,
    offset: int = 0,
) -> torch.Tensor:
    """The single baseline-adapter conversion.

    A region map ``(B, R)`` is bilinearly upsampled, then thresholded to area
    ``fraction``. A map that is already in pixel space is only thresholded.
    CDEA and every baseline must call this, so the symmetry test can pass one
    mask through both paths.
    """
    if grid_h is not None:
        if grid_w is None or height is None or width is None:
            raise ValueError("grid and pixel sizes are required together")
        scores = regions_to_pixels(scores, grid_h, grid_w, height, width)
    return top_fraction_mask(scores, fraction, seed, offset=offset)
