"""Plan machinery shared by the contrast game (``allocation.py``) and the
decomposition game (``decompose.py``): budgets, the projection onto the
allocation polytope, binary draws, boundary offsets, and the deleted inputs.

Moved out of ``allocation.py`` unchanged. ``tests/test_plan_refactor.py``
checks that ``allocate()`` gives the same output as before the move.
"""

from __future__ import annotations

import torch

from core.grid import delete, keep
from cdea.sinkhorn import hard_top_mass, sinkhorn

SINKHORN_ITERS = 20


def _row_targets(players: int, units: int, area: float, device, dtype) -> torch.Tensor:
    budget = float(area) * float(units)
    if players * float(area) > 1.0 + 1e-6:
        raise ValueError("infeasible: players times area exceeds 1")
    empty = float(units) - players * budget
    values = [budget] * players + [max(empty, 0.0)]
    return torch.tensor(values, device=device, dtype=dtype)


def _project(theta: torch.Tensor, row_target: torch.Tensor, projection: str, dual: list | None = None) -> torch.Tensor:
    if projection == "sinkhorn":
        return sinkhorn(theta, row_target, iters=SINKHORN_ITERS, dual=dual)
    if projection != "hard_top_mass":
        raise ValueError(f"unknown projection {projection!r}")
    budgets = row_target if row_target.ndim == 1 else row_target[0]
    rows = [hard_top_mass(theta[:, index], float(budgets[index])) for index in range(theta.shape[1])]
    return torch.stack(rows, dim=1)


def _independent_masks(theta: torch.Tensor, area: float, projection: str) -> torch.Tensor:
    """One two-row plan per player. Players do not share a column constraint."""
    _batch, players, units = theta.shape
    budget = float(area) * float(units)
    if projection == "hard_top_mass":
        return torch.stack([hard_top_mass(theta[:, p], budget) for p in range(players)], dim=1)
    target = torch.tensor([budget, float(units) - budget], device=theta.device, dtype=theta.dtype)
    masks = []
    for player in range(players):
        both = torch.stack([theta[:, player], theta.new_zeros(theta.shape[0], units)], dim=1)
        masks.append(sinkhorn(both, target, iters=SINKHORN_ITERS)[:, 0])
    return torch.stack(masks, dim=1)


def _edited(x, masks, grid_h, grid_w, preserve: bool, offsets=None) -> torch.Tensor:
    edit = keep if preserve else delete
    players = range(masks.shape[1])
    if offsets is None:
        return torch.cat([edit(x, masks[:, p], grid_h, grid_w) for p in players], dim=0)
    return torch.cat([edit(x, masks[:, p], grid_h, grid_w, offset=offsets[p]) for p in players], dim=0)


def _player_logits(model, x, masks, grid_h, grid_w, preserve: bool, offsets=None) -> torch.Tensor:
    flat = model(_edited(x, masks, grid_h, grid_w, preserve, offsets))
    return flat.view(masks.shape[1], x.shape[0], -1).transpose(0, 1)


def _draw(masks: torch.Tensor, uniform: torch.Tensor, joint: bool) -> torch.Tensor:
    """Binary masks drawn from the plan, with a straight-through gradient.

    Joint plan: unit ``r`` goes to the player whose share of column ``r``
    covers ``uniform[r]``, or to no player, so players stay exclusive in every
    draw. ``uniform`` is ``(R,)``. Independent rows: entry ``(p, r)`` is kept
    when ``uniform[p, r]`` is below it; ``uniform`` is ``(P, R)``. The same
    draw serves every image in the batch.
    """
    if joint:
        upper = masks.cumsum(dim=1)
        lower = upper - masks
        u = uniform.to(masks.device, masks.dtype).view(1, 1, -1)
        hard = ((u >= lower) & (u < upper)).to(masks.dtype)
    else:
        hard = (uniform.to(masks.device, masks.dtype).unsqueeze(0) < masks).to(masks.dtype)
    # (masks - masks.detach()) is exactly 0 forward, so the drawn mask stays exactly binary.
    return hard + (masks - masks.detach())


def _offsets(generator: torch.Generator, players: int, limit: tuple[int, int]) -> list[tuple[int, int]]:
    """One pixel offset per player, uniform on ``{-s, ..., s}`` per axis."""
    dy = torch.randint(-limit[0], limit[0] + 1, (players,), generator=generator)
    dx = torch.randint(-limit[1], limit[1] + 1, (players,), generator=generator)
    return [(int(a), int(b)) for a, b in zip(dy, dx)]
