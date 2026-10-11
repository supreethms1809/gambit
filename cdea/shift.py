"""Shift allocation. docs/paper/SHIFT_FORMULATION.md.

Two players, robust and shortcut, plus an empty row. The robust payoff is the
worst-environment deletion drop. The shortcut payoff is the gap a transplant
closes. Both are in nats. The loss is their negated sum. The contrastive
solver is not modified: this loop calls the same Sinkhorn and the same row
targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.eval_mode import eval_mode
from core.grid import delete, deletion_baseline, pool_sum, transplant
from core.types import HypothesisSet
from cdea.sinkhorn import hard_top_mass, sinkhorn

EPS = 1e-6
SINKHORN_ITERS = 20


@dataclass
class ShiftConfig:
    steps: Optional[int] = None
    lr: float = 0.1
    projection: str = "sinkhorn"
    independent: bool = False
    robust: str = "min"
    shortcut: str = "transplant"
    init: str = "evidence"
    backend: str = "gradcam"
    ig_steps: int = 16
    kind: str = "allocate"


@dataclass
class ShiftAllocation:
    robust: torch.Tensor
    shortcut: torch.Tensor
    transport: Optional[torch.Tensor]
    row_target: torch.Tensor
    robust_payoff: torch.Tensor
    shortcut_payoff: torch.Tensor


def log_odds(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``m(x) = z_y - LSE_{j != y} z_j``. ``y`` is ``(B,)``."""
    if logits.ndim != 2:
        raise ValueError("logits are (batch, classes)")
    if logits.shape[1] < 2:
        raise ValueError("log-odds needs a class other than y")
    index = y.long().view(-1, 1)
    chosen = logits.gather(1, index).squeeze(1)
    others = logits.scatter(1, index, torch.full_like(chosen, float("-inf")).unsqueeze(1))
    return chosen - torch.logsumexp(others, dim=-1)


def robust_payoff(
    model: nn.Module,
    xs: list[torch.Tensor],
    y: torch.Tensor,
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    reduction: str = "min",
    reference: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """``min_e [m(x_e) - m(x_e ⊖ A)]``, or the mean when ``reduction`` is ``mean``."""
    drops = []
    for index, x in enumerate(xs):
        full = reference[index] if reference is not None else log_odds(model(x), y)
        edited = log_odds(model(delete(x, mask, grid_h, grid_w)), y)
        drops.append(full - edited)
    stacked = torch.stack(drops, dim=0)
    if reduction == "mean":
        return stacked.mean(dim=0)
    if reduction != "min":
        raise ValueError(f"unknown robust reduction {reduction!r}")
    return stacked.min(dim=0).values


def transplant_payoff(
    model: nn.Module,
    xs: list[torch.Tensor],
    y: torch.Tensor,
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    reference: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Mean over environments other than id of the gap a transplant closes.

    Units that match across environments contribute 0.
    """
    if len(xs) < 2:
        raise ValueError("a shortcut payoff needs an environment other than id")
    x_id = xs[0]
    m_id = reference[0] if reference is not None else log_odds(model(x_id), y)
    closed = []
    for index, x_e in enumerate(xs[1:], start=1):
        m_e = reference[index] if reference is not None else log_odds(model(x_e), y)
        mixed = transplant(x_id, x_e, mask, grid_h, grid_w)
        m_mix = log_odds(model(mixed), y)
        closed.append((m_id - m_e).abs() - (m_mix - m_e).abs())
    return torch.stack(closed, dim=0).mean(dim=0)


def deletion_shortcut_payoff(
    model: nn.Module,
    xs: list[torch.Tensor],
    y: torch.Tensor,
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    reference: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Ablation: close the gap by deleting the same units in every environment."""
    x_id = xs[0]
    m_id = reference[0] if reference is not None else log_odds(model(x_id), y)
    closed = []
    for index, x_e in enumerate(xs[1:], start=1):
        m_e = reference[index] if reference is not None else log_odds(model(x_e), y)
        m_id_d = log_odds(model(delete(x_id, mask, grid_h, grid_w)), y)
        m_e_d = log_odds(model(delete(x_e, mask, grid_h, grid_w)), y)
        closed.append((m_id - m_e).abs() - (m_id_d - m_e_d).abs())
    return torch.stack(closed, dim=0).mean(dim=0)


def overshoot_share(
    m_id: torch.Tensor,
    m_mix: torch.Tensor,
    m_env: torch.Tensor,
) -> torch.Tensor:
    """Share of ``|g|`` by which the transplant passes ``m(x_e)``.

    Reported, not gated. Zero when the mix stays on the id side of ``m(x_e)``.
    """
    gap = m_id - m_env
    residual = (m_mix - m_env).abs()
    crossed = (m_mix - m_env) * gap < 0
    overshoot = torch.where(crossed, residual, torch.zeros_like(residual))
    return overshoot / gap.abs().clamp_min(1e-8)


def integrated_gap(
    model: nn.Module,
    x_from: torch.Tensor,
    x_to: torch.Tensor,
    y: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Path-integrated gradient of ``m`` from ``x_from`` to ``x_to``.

    On a model whose log-odds are linear in the pixels, the sum of this map
    equals ``m(x_to) - m(x_from)``.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    delta = x_to - x_from
    total = torch.zeros_like(x_to)
    for i in range(1, int(steps) + 1):
        alpha = float(i) / float(steps)
        point = (x_from + alpha * delta).detach().requires_grad_(True)
        value = log_odds(model(point), y).sum()
        grad, = torch.autograd.grad(value, point)
        total = total + grad.detach()
    return delta * (total / float(steps))


def gap_units(
    model: nn.Module,
    xs: list[torch.Tensor],
    y: torch.Tensor,
    grid_h: int,
    grid_w: int,
    steps: int,
) -> torch.Tensor:
    """Mean over environments of the unit-pooled path integral of ``g_e``.

    The sum over units equals the mean gap on a linear model.
    """
    pooled = []
    for x_e in xs[1:]:
        attr = integrated_gap(model, x_e, xs[0], y, steps)
        pooled.append(pool_sum(attr.sum(dim=1), grid_h, grid_w))
    return torch.stack(pooled, dim=0).mean(dim=0)


def shortcut_scores(units: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    """Sign the pooled integral by ``g_e`` so the ranking follows the gap."""
    sign = torch.where(gap >= 0, torch.ones_like(gap), -torch.ones_like(gap))
    return units * sign.unsqueeze(-1)


def allocate_shift(
    model: nn.Module,
    xs: list[torch.Tensor],
    grid_h: int,
    grid_w: int,
    area: float,
    cfg: Optional[ShiftConfig] = None,
) -> ShiftAllocation:
    """Solve one area. ``xs[0]`` is the in-distribution view. Every view is ``(B, 3, H, W)``."""
    cfg = cfg or ShiftConfig()
    if len(xs) < 2:
        raise ValueError("shift allocation needs the id view and one other environment")
    if not 0.0 < float(area) <= 0.5:
        raise ValueError("area must be in (0, 1/2]")
    if cfg.robust not in {"min", "mean"}:
        raise ValueError(f"unknown robust reduction {cfg.robust!r}")
    if cfg.shortcut not in {"transplant", "deletion"}:
        raise ValueError(f"unknown shortcut payoff {cfg.shortcut!r}")
    if cfg.kind not in {"allocate", "first_order"}:
        raise ValueError(f"unknown kind {cfg.kind!r}")
    x_id = xs[0]
    parameter = next(model.parameters())
    for view in xs:
        if view.shape != x_id.shape:
            raise ValueError("environment views must share a shape")
        if view.device != parameter.device:
            raise RuntimeError("views and the model are on different devices")

    units = grid_h * grid_w
    players = 2
    with eval_mode(model):
        with torch.no_grad():
            y = model(x_id).argmax(dim=-1)
            reference = [log_odds(model(view), y) for view in xs]
        # Evidence needs gradients through the classifier, so it runs before the freeze.
        evidence = _prior(model, xs, y, grid_h, grid_w, cfg) if cfg.init == "evidence" else None
        if cfg.kind == "first_order":
            return _first_order(model, xs, y, reference, grid_h, grid_w, float(area), cfg)
        flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            theta = _initial_theta(evidence, units, cfg.init, x_id)
            opt = torch.optim.Adam([theta], lr=float(cfg.lr))
            row_target = _row_targets(players, units, float(area), x_id.device, x_id.dtype)
            dual: list = [None]
            steps = 100 if cfg.steps is None else int(cfg.steps)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                robust_mask, shortcut_mask = _player_masks(theta, row_target, float(area), cfg, dual)
                loss = _loss(model, xs, y, robust_mask, shortcut_mask, grid_h, grid_w, cfg, reference)
                loss.backward()
                opt.step()
            with torch.no_grad():
                robust_mask, shortcut_mask = _player_masks(theta, row_target, float(area), cfg, dual)
                if cfg.independent or cfg.projection != "sinkhorn":
                    transport = torch.stack([robust_mask, shortcut_mask], dim=1)
                    target = row_target[:players]
                else:
                    transport = sinkhorn(theta.detach(), row_target, iters=SINKHORN_ITERS, dual=dual)
                    target = row_target
                pay_r = robust_payoff(
                    model, xs, y, robust_mask, grid_h, grid_w, cfg.robust, reference,
                )
                pay_s = _shortcut(model, xs, y, shortcut_mask, grid_h, grid_w, cfg, reference)
        finally:
            for p, flag in zip(model.parameters(), flags):
                p.requires_grad_(flag)
    return ShiftAllocation(
        robust=robust_mask.detach(),
        shortcut=shortcut_mask.detach(),
        transport=transport.detach(),
        row_target=target.detach(),
        robust_payoff=pay_r.detach(),
        shortcut_payoff=pay_s.detach(),
    )


def _shortcut(model, xs, y, mask, grid_h, grid_w, cfg: ShiftConfig, reference):
    if cfg.shortcut == "deletion":
        return deletion_shortcut_payoff(model, xs, y, mask, grid_h, grid_w, reference)
    return transplant_payoff(model, xs, y, mask, grid_h, grid_w, reference)


def _loss(model, xs, y, robust_mask, shortcut_mask, grid_h, grid_w, cfg, reference) -> torch.Tensor:
    robust = robust_payoff(model, xs, y, robust_mask, grid_h, grid_w, cfg.robust, reference)
    shortcut = _shortcut(model, xs, y, shortcut_mask, grid_h, grid_w, cfg, reference)
    return -(robust + shortcut).sum()


def _row_targets(players: int, units: int, area: float, device, dtype) -> torch.Tensor:
    budget = float(area) * float(units)
    if players * float(area) > 1.0 + 1e-6:
        raise ValueError("infeasible: players times area exceeds 1")
    empty = float(units) - players * budget
    values = [budget] * players + [max(empty, 0.0)]
    return torch.tensor(values, device=device, dtype=dtype)


def _player_masks(theta, row_target, area, cfg: ShiftConfig, dual) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.independent:
        masks = _independent(theta[:, :2], area, cfg.projection)
        return masks[:, 0], masks[:, 1]
    plan = _project(theta, row_target, cfg.projection, dual)
    return plan[:, 0], plan[:, 1]


def _project(theta, row_target, projection: str, dual) -> torch.Tensor:
    if projection == "sinkhorn":
        return sinkhorn(theta, row_target, iters=SINKHORN_ITERS, dual=dual)
    if projection != "hard_top_mass":
        raise ValueError(f"unknown projection {projection!r}")
    budgets = row_target if row_target.ndim == 1 else row_target[0]
    rows = [hard_top_mass(theta[:, index], float(budgets[index])) for index in range(theta.shape[1])]
    return torch.stack(rows, dim=1)


def _independent(theta: torch.Tensor, area: float, projection: str) -> torch.Tensor:
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


def _initial_theta(evidence, units: int, init: str, x: torch.Tensor) -> torch.Tensor:
    if init not in {"evidence", "uniform"}:
        raise ValueError(f"unknown init {init!r}")
    fill = math.log(1.0 / float(units))
    if init == "uniform" or evidence is None:
        theta = x.new_full((x.shape[0], 3, units), fill)
        return theta.detach().requires_grad_(True)
    rows = [torch.log(evidence[:, k] + EPS) for k in range(2)]
    rows.append(x.new_full((x.shape[0], units), fill))
    return torch.stack(rows, dim=1).detach().requires_grad_(True)


def _prior(model, xs, y, grid_h, grid_w, cfg: ShiftConfig) -> torch.Tensor:
    maps = [_class_evidence(model, view, y, grid_h, grid_w, cfg) for view in xs]
    robust = torch.stack(maps, dim=0).amin(dim=0)
    shortcut = torch.stack([(maps[0] - other).abs() for other in maps[1:]], dim=0).mean(dim=0)
    return torch.stack([_sum_norm(robust), _sum_norm(shortcut)], dim=1)


def _sum_norm(row: torch.Tensor) -> torch.Tensor:
    return row / row.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _class_evidence(model, x, y, grid_h, grid_w, cfg: ShiftConfig) -> torch.Tensor:
    from base_evidence.gradcam_regions import GradCAMRegionsProvider
    from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider

    hyp = HypothesisSet(
        ids=y.long().view(-1, 1),
        mask=torch.ones(y.shape[0], 1, dtype=torch.bool, device=y.device),
    )
    if cfg.backend == "gradcam":
        provider = GradCAMRegionsProvider(grid_h, grid_w)
    elif cfg.backend == "ig":
        provider = IntegratedGradientsRegionsProvider(
            grid_h, grid_w, steps=int(cfg.ig_steps), baseline="blur",
        )
    else:
        raise ValueError(f"unknown evidence backend {cfg.backend!r}")
    return provider.explain(x, model, hyp)[:, 0]


def _first_order(model, xs, y, reference, grid_h, grid_w, area, cfg: ShiftConfig) -> ShiftAllocation:
    batch = xs[0].shape[0]
    units = grid_h * grid_w
    mask_r = torch.zeros(batch, units, device=xs[0].device, dtype=xs[0].dtype, requires_grad=True)
    mask_s = torch.zeros(batch, units, device=xs[0].device, dtype=xs[0].dtype, requires_grad=True)
    value_r = robust_payoff(model, xs, y, mask_r, grid_h, grid_w, cfg.robust, reference).sum()
    grad_r, = torch.autograd.grad(value_r, mask_r)
    value_s = _shortcut(model, xs, y, mask_s, grid_h, grid_w, cfg, reference).sum()
    grad_s, = torch.autograd.grad(value_s, mask_s)
    budget = area * units
    robust, shortcut = _exclusive_top(grad_r.detach(), grad_s.detach(), budget)
    with torch.no_grad():
        pay_r = robust_payoff(model, xs, y, robust, grid_h, grid_w, cfg.robust, reference)
        pay_s = _shortcut(model, xs, y, shortcut, grid_h, grid_w, cfg, reference)
    target = _row_targets(2, units, area, xs[0].device, xs[0].dtype)[:2]
    return ShiftAllocation(
        robust=robust,
        shortcut=shortcut,
        transport=None,
        row_target=target,
        robust_payoff=pay_r.detach(),
        shortcut_payoff=pay_s.detach(),
    )


def _exclusive_top(grad_r: torch.Tensor, grad_s: torch.Tensor, budget: float) -> tuple[torch.Tensor, torch.Tensor]:
    out_r = torch.zeros_like(grad_r)
    out_s = torch.zeros_like(grad_s)
    for batch in range(grad_r.shape[0]):
        left_r = float(budget)
        left_s = float(budget)
        score = torch.maximum(grad_r[batch], grad_s[batch])
        order = torch.argsort(score, descending=True, stable=True)
        for unit in order.tolist():
            give_robust = float(grad_r[batch, unit]) >= float(grad_s[batch, unit])
            if give_robust and left_r > 0:
                take = min(1.0, left_r)
                out_r[batch, unit] = take
                left_r -= take
            elif left_s > 0:
                take = min(1.0, left_s)
                out_s[batch, unit] = take
                left_s -= take
            if left_r <= 0 and left_s <= 0:
                break
    return out_r, out_s


def mean_environment_gap(model: nn.Module, xs: list[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """``mean_{e != id} |m(x_id) - m(x_e)|``."""
    m_id = log_odds(model(xs[0]), y)
    gaps = [(m_id - log_odds(model(view), y)).abs() for view in xs[1:]]
    return torch.stack(gaps, dim=0).mean(dim=0)


def robust_deletion_units(
    model: nn.Module,
    xs: list[torch.Tensor],
    y: torch.Tensor,
    grid_h: int,
    grid_w: int,
    steps: int,
) -> torch.Tensor:
    """Min over environments of the blur-baseline path integral of ``m``, pooled to units."""
    pooled = []
    for view in xs:
        base = deletion_baseline(view, grid_h, grid_w)
        attr = integrated_gap(model, base, view, y, steps)
        pooled.append(pool_sum(attr.sum(dim=1), grid_h, grid_w).clamp_min(0))
    return torch.stack(pooled, dim=0).amin(dim=0)
