"""First-order solution. FORMULATION.md section 9.

Linearise the unique log-odds around no deletion. The optimum under the
budget gives each player its top units of

    g_{k,r} = <grad_x c_k(x), (x - b(x)) odot phi_r>.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.eval_mode import eval_mode
from core.grid import deletion_baseline, pool_sum
from core.types import HypothesisSet
from cdea.payoffs import unique_log_odds


def top_mass(scores: torch.Tensor, budget: float) -> torch.Tensor:
    """Fill ``budget`` with the largest scores. The last selected unit may be fractional."""
    if scores.ndim != 2:
        raise ValueError("top_mass expects (batch, units)")
    units = scores.shape[-1]
    if budget < 0 or budget > units + 1e-6:
        raise ValueError("budget is outside the unit count")
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    out = torch.zeros_like(scores)
    remaining = torch.full((scores.shape[0],), float(budget), device=scores.device, dtype=scores.dtype)
    for rank in range(units):
        take = remaining.clamp(max=1.0)
        out.scatter_(1, order[:, rank:rank + 1], take.unsqueeze(1))
        remaining = (remaining - 1.0).clamp_min(0.0)
        if float(remaining.max()) <= 0:
            break
    return out


def unit_gradient(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """``(B, K, R)`` scores ``g_k``. ``x`` is not moved to another device."""
    parameter = next(model.parameters())
    if x.device != parameter.device:
        raise RuntimeError("x and the model are on different devices; pass both explicitly")
    x_in = x.detach().requires_grad_(True)
    with eval_mode(model):
        contrast = unique_log_odds(model(x_in), hypotheses.ids.long())
        base = deletion_baseline(x.detach(), grid_h, grid_w)
        delta = x.detach() - base
        columns = []
        for k in range(contrast.shape[1]):
            grad = torch.autograd.grad(contrast[:, k].sum(), x_in, retain_graph=True)[0]
            columns.append(pool_sum((grad * delta).sum(dim=1), grid_h, grid_w))
    return torch.stack(columns, dim=1)


def first_order_masks(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    grid_h: int,
    grid_w: int,
    area: float,
) -> torch.Tensor:
    """Hard top-mass of ``g_k`` at budget ``a * R``. Returns ``(B, K, R)``."""
    scores = unit_gradient(model, x, hypotheses, grid_h, grid_w)
    budget = float(area) * float(grid_h * grid_w)
    return torch.stack([top_mass(scores[:, k], budget) for k in range(scores.shape[1])], dim=1)
