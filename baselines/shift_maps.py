"""Shift baselines: cross-environment map reductions and additive per-cell assignment.

Index 0 is the in-distribution map. Later entries are the other environments.
The robust map is the elementwise minimum, evidence that is present in every
environment. The shortcut map is the largest absolute gap between the
in-distribution map and any other environment.

Per-environment Extremal Perturbations runs TorchRay's deletion class mask on
each environment, the region whose removal most lowers the class at area a,
because the shift scores are deletion drops. It then applies that same
reduction. The mask update is unchanged.

``shift_assign`` is the additive per-cell assignment baseline: it scores each
cell alone by its worst-environment deletion drop (for R) and its single-cell
transplant closure (for S), then solves the exact two-player budget and
exclusivity linear programme via HiGHS (``baselines.decomposition.assign``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.eval_mode import eval_mode

from .decomposition import assign
from .extremal import class_masks


def environment_maps(maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(robust, shortcut)``, each ``(B, H, W)``.

    ``maps[0]`` is the in-distribution attribution. Every entry has the same shape.
    """
    if len(maps) < 2:
        raise ValueError("a shift comparison needs the ID map and at least one other environment")
    stacked = torch.stack([_as_map(m) for m in maps], dim=0)
    if not torch.isfinite(stacked).all():
        raise ValueError("a map has a non-finite value")
    robust = stacked.amin(dim=0)
    shortcut = (stacked[0] - stacked[1:]).abs().amax(dim=0)
    return robust, shortcut


def per_environment_extremal(
    model: nn.Module,
    xs: list[torch.Tensor],
    class_idx: torch.Tensor,
    area: float = 0.1,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deletion class masks per environment, then the robust and shortcut reduction.

    ``xs[0]`` is the in-distribution batch. ``class_idx`` is ``(B,)`` and is
    shared across environments. Returns ``(robust, shortcut)``.
    """
    if len(xs) < 2:
        raise ValueError("a shift comparison needs the ID batch and at least one other environment")
    masks = [class_masks(model, x, class_idx, area=area, deletion=True, **kwargs) for x in xs]
    return environment_maps(masks)


def shift_assign(
    model: nn.Module,
    xs: list[torch.Tensor],
    grid_h: int,
    grid_w: int,
    area: float,
    g: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(robust, shortcut)``, each ``(B, R)``, from the exact HiGHS LP on single-cell shift payoffs."""
    from cdea.shift import shift_cell_drops

    if len(xs) < 2:
        raise ValueError("shift assignment needs the ID batch and at least one other environment")
    if g is None:
        with eval_mode(model), torch.no_grad():
            y = model(xs[0]).argmax(dim=-1)
        g = shift_cell_drops(model, xs, y, grid_h, grid_w)
    plan = assign(g, float(area) * grid_h * grid_w)
    return plan[:, 0], plan[:, 1]


def _as_map(m: torch.Tensor) -> torch.Tensor:
    if m.ndim != 3:
        raise ValueError("each environment map is (B, H, W)")
    return m.detach().to(dtype=torch.float32)
