"""Gap attribution: the path-integrated first-order solution of the shortcut payoff.

Integrated gradients of the log-odds along ``x_e -> x_id``, pooled to units
and signed by ``g_e``. On a linear model the pooled map sums to ``g_e``.
The robust companion is the min over environments of the same integral from
the deletion baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cdea.shift import gap_units, log_odds, robust_deletion_units, shortcut_scores


def gap_attribution(
    model: nn.Module,
    xs: list[torch.Tensor],
    grid_h: int,
    grid_w: int,
    steps: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(robust, shortcut)`` unit scores, each ``(B, R)``."""
    if len(xs) < 2:
        raise ValueError("gap attribution needs the id view and one other environment")
    with torch.enable_grad():
        y = model(xs[0]).argmax(dim=-1)
        units = gap_units(model, xs, y, grid_h, grid_w, steps)
        gap = torch.stack(
            [(log_odds(model(xs[0]), y) - log_odds(model(view), y)) for view in xs[1:]],
            dim=0,
        ).mean(dim=0)
        shortcut = shortcut_scores(units, gap)
        robust = robust_deletion_units(model, xs, y, grid_h, grid_w, steps)
    return robust.detach(), shortcut.detach()
