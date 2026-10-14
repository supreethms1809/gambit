"""The null drop of the decomposition game. FORMULATION_DECOMPOSITION.md section 3.2.

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
