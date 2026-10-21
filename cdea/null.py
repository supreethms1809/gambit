"""The null drop of the decomposition game.

Version 1 (tag ``decomp-declared``, section 3.2) uses one area; version 2
(FORMULATION_DECOMPOSITION.md section 3.4) matches each role's size, below.

Deleting almost anything lowers a confident logit. ``nu_j(a)`` is the mean drop
in z_j over N seeded random cell masks of area a. The masks are the same for
every image, so the null of an image does not depend on the batch. CDEA uses it
under its own blur deletion; the scorer computes its own under ROAD from the
same cell scores (``null_unit_scores``).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from core.grid import delete
from cdea.first_order import top_mass

N_NULL = 8
NULL_SEED = 0


def null_unit_scores(units: int, n: int = N_NULL, seed: int = NULL_SEED) -> torch.Tensor:
    """``(n, R)`` seeded uniform cell scores. The scorer ranks these same scores."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.rand(n, units, generator=generator)


def null_unit_masks(units: int, area: float, n: int = N_NULL, seed: int = NULL_SEED) -> torch.Tensor:
    """``(n, R)`` random cell masks of mass ``a * R``: whole top cells, the last one fractional."""
    return top_mass(null_unit_scores(units, n, seed), float(area) * float(units))


def null_drops(
    model: nn.Module,
    x: torch.Tensor,
    ids: torch.Tensor,
    grid_h: int,
    grid_w: int,
    area: float,
    n: int = N_NULL,
    seed: int = NULL_SEED,
    edit: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """``(B, J)``: mean drop in ``z_ids`` over the random masks, under blur deletion. Detached.

    ``edit(x, unit_mask)`` replaces the deletion; it defaults to CDEA's blur.
    """
    edit = edit or (lambda images, mask: delete(images, mask, grid_h, grid_w))
    masks = null_unit_masks(grid_h * grid_w, area, n, seed).to(device=x.device, dtype=x.dtype)
    ids = ids.long()
    with torch.no_grad():
        full = model(x).gather(1, ids)
        total = torch.zeros_like(full)
        for mask in masks:
            deleted = model(edit(x, mask.expand(x.shape[0], -1)))
            total = total + (full - deleted.gather(1, ids))
    return (total / float(n)).detach()


# ---------------------------------------------------------------------------
# Version 2: the size-matched null (FORMULATION_DECOMPOSITION.md v2 section 3.4).
# ``nu_j(n)`` is the mean drop over N seeded random n-cell masks from the whole
# grid. The n-cell mask of row i is the top n cells of that row's seeded
# scores, so the masks of one row are nested, and every image gets the same
# masks.
# ---------------------------------------------------------------------------

def size_masks(units: int, sizes, n: int = N_NULL, seed: int = NULL_SEED) -> torch.Tensor:
    """``(len(sizes), n, R)`` whole-cell masks: row i of size s is the top s cells of score row i."""
    order = null_unit_scores(units, n, seed).argsort(dim=-1, descending=True)
    masks = torch.zeros(len(sizes), n, units)
    for index, size in enumerate(sizes):
        size = int(size)
        if not 0 <= size <= units:
            raise ValueError(f"size {size} is outside 0..{units}")
        masks[index].scatter_(-1, order[:, :size], 1.0)
    return masks


def null_size_grid(top: int) -> list[int]:
    """Sizes at which the ViT-B/16 null is measured, then interpolated (open decision 9).

    Every size up to 8, then every 4th, and the top size.
    """
    return sorted(set(range(min(top, 8) + 1)) | set(range(8, top + 1, 4)) | {int(top)})


def size_null(
    model: nn.Module,
    x: torch.Tensor,
    ids: torch.Tensor,
    grid_h: int,
    grid_w: int,
    top: int,
    n: int = N_NULL,
    seed: int = NULL_SEED,
    edit: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    sizes: Optional[list[int]] = None,
) -> torch.Tensor:
    """``(B, J, top + 1)`` table of ``nu_j(s)`` for ``s = 0 .. top``, under blur deletion. Detached.

    ``sizes`` measures only those sizes (0 and ``top`` included) and fills the
    rest by linear interpolation; ``None`` measures every size.
    """
    edit = edit or (lambda images, mask: delete(images, mask, grid_h, grid_w))
    units = grid_h * grid_w
    measured = list(range(top + 1)) if sizes is None else sorted(set(sizes) | {0, int(top)})
    measured = [s for s in measured if s <= top]
    masks = size_masks(units, measured, n, seed).to(device=x.device, dtype=x.dtype)
    ids = ids.long()
    batch = x.shape[0]
    values = torch.zeros(batch, ids.shape[1], len(measured), device=x.device, dtype=x.dtype)
    with torch.no_grad():
        full = model(x).gather(1, ids)
        for index, size in enumerate(measured):
            if size == 0:
                continue
            images = x.repeat(n, 1, 1, 1)
            flat = masks[index].repeat_interleave(batch, dim=0)
            deleted = model(edit(images, flat)).gather(1, ids.repeat(n, 1))
            values[..., index] = (full.repeat(n, 1) - deleted).view(n, batch, -1).mean(dim=0)
    if len(measured) == top + 1:
        return values.detach()
    grid = torch.tensor(measured, dtype=x.dtype, device=x.device)
    return interpolate_sizes(values, grid, top).detach()


def interpolate_sizes(values: torch.Tensor, measured: torch.Tensor, top: int) -> torch.Tensor:
    """Linear interpolation of ``(..., len(measured))`` values onto the sizes ``0 .. top``."""
    targets = torch.arange(top + 1, dtype=values.dtype, device=values.device)
    right = torch.searchsorted(measured, targets).clamp(1, measured.numel() - 1)
    left = right - 1
    span = (measured[right] - measured[left]).clamp_min(1e-12)
    weight = ((targets - measured[left]) / span).clamp(0, 1)
    return values[..., left] * (1 - weight) + values[..., right] * weight


def null_at(table: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """``(B, P, J)`` null at each player's size, linear between whole sizes.

    ``table`` is ``(B, J, S + 1)`` and ``size`` is ``(B, P)``. The gradient in
    ``size`` is the slope to the next whole size: the null of one more cell,
    which is the entry threshold of section 4.1.
    """
    top = table.shape[-1] - 1
    if top == 0:
        return table.new_zeros(size.shape + (table.shape[1],))
    s = size.clamp(0, top)
    left = s.detach().floor().clamp(max=top - 1).long()
    weight = (s - left.to(s.dtype)).unsqueeze(-1)
    batch, players = size.shape
    expanded = table.unsqueeze(1).expand(batch, players, -1, -1)
    lo = expanded.gather(-1, left.view(batch, players, 1, 1).expand(-1, -1, table.shape[1], 1)).squeeze(-1)
    hi = expanded.gather(-1, (left + 1).view(batch, players, 1, 1).expand(-1, -1, table.shape[1], 1)).squeeze(-1)
    return lo + weight * (hi - lo)
