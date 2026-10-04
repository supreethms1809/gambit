"""Position-scrambling nulls. Shape and budget stay fixed. Only the location changes."""

from __future__ import annotations

import torch


def random_translate(
    mask: torch.Tensor,
    generator: torch.Generator,
    grid_h: int | None = None,
    grid_w: int | None = None,
) -> torch.Tensor:
    """Roll each ``(B, R)`` mask by an independent ``(dy, dx)`` on the region grid.

    Offsets are drawn on CPU from ``generator``, so the null follows ``--seed``
    on every device. A non-square grid must be passed explicitly.
    """
    batch, regions = mask.shape
    if grid_h is None or grid_w is None:
        side = int(round(regions ** 0.5))
        if side * side != regions:
            raise ValueError(f"region count {regions} is not square; pass grid_h and grid_w")
        grid_h = grid_w = side
    if grid_h * grid_w != regions:
        raise ValueError(f"region count {regions} != {grid_h}x{grid_w}")
    grid = mask.reshape(batch, grid_h, grid_w)
    offsets = torch.randint(0, max(grid_h, grid_w), (batch, 2), generator=generator)
    out = torch.empty_like(grid)
    for i in range(batch):
        dy = int(offsets[i, 0].item()) % grid_h
        dx = int(offsets[i, 1].item()) % grid_w
        if dy == 0 and dx == 0:
            # The null must move: a (0, 0) roll returns the real mask and
            # dilutes the check with self-comparisons.
            dx = 1 % grid_w
            if dx == 0:
                dy = 1 % grid_h
        out[i] = torch.roll(grid[i], shifts=(dy, dx), dims=(0, 1))
    return out.reshape(batch, regions)
