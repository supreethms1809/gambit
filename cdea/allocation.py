"""Allocation. FORMULATION.md sections 5 to 8.

Initialise from the sum-normalised evidence prior, then Adam on the
log-domain plan. The loss is the negated sum of player payoffs over images.
Each area is a separate call. Model parameters stay frozen and the module
stays in eval mode. The caller places ``x`` and the model on one device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.eval_mode import eval_mode
from core.grid import boundary_offset, delete, keep
from core.types import HypothesisSet
from cdea.payoffs import deletion_payoffs, shared_log_odds, unique_log_odds
from cdea.sinkhorn import hard_top_mass, sinkhorn

EPS = 1e-6
SINKHORN_ITERS = 20
# Offsets of the boundary-robust payoff come from this seed alone, never from the
# batch, so an image's allocation does not depend on n or on chunking.
OFFSET_SEED = 0


@dataclass
class Allocation:
    unique: torch.Tensor                 # (B, K, R)
    shared: Optional[torch.Tensor]       # (B, R) or None
    transport: torch.Tensor              # players, and the unallocated row when the plan is joint
    row_target: torch.Tensor


@dataclass
class AllocateConfig:
    steps: int = 100
    lr: float = 0.1
    projection: str = "sinkhorn"         # sinkhorn | hard_top_mass
    shared: bool = True
    independent: bool = False
    preserve: bool = False
    init: str = "evidence"               # evidence | uniform
    pair_only: bool = False
    backend: str = "gradcam"
    ig_steps: int = 16
    boundary_shift: bool = False         # FORMULATION.md section 4.1


def _prior(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    grid_h: int,
    grid_w: int,
    cfg: AllocateConfig,
) -> torch.Tensor:
    """Sum-normalised evidence, used only as a log-initialisation. ``(B, K, R)``."""
    backend = cfg.backend
    if backend == "gradcam":
        from base_evidence.gradcam_regions import GradCAMRegionsProvider

        raw = GradCAMRegionsProvider(grid_h, grid_w).explain(x, model, hypotheses)
    elif backend == "ig":
        from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider

        raw = IntegratedGradientsRegionsProvider(
            grid_h, grid_w, steps=cfg.ig_steps, baseline="blur",
        ).explain(x, model, hypotheses)
    elif backend.startswith("library:"):
        from base_evidence.library_adapters import CAM_METHODS, CamLibraryProvider, CaptumRegionsProvider

        method = backend.split(":", 1)[1]
        provider = CamLibraryProvider if method in CAM_METHODS else CaptumRegionsProvider
        raw = provider(method, grid_h, grid_w).explain(x, model, hypotheses)
    else:
        raise ValueError(f"unknown evidence backend {backend!r}")
    raw = raw.clamp_min(0)
    return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)


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


def _edited(x, masks, grid_h, grid_w, preserve: bool, offsets=None) -> torch.Tensor:
    edit = keep if preserve else delete
    players = range(masks.shape[1])
    if offsets is None:
        return torch.cat([edit(x, masks[:, p], grid_h, grid_w) for p in players], dim=0)
    return torch.cat([edit(x, masks[:, p], grid_h, grid_w, offset=offsets[p]) for p in players], dim=0)


def _player_logits(model, x, masks, grid_h, grid_w, preserve: bool, offsets=None) -> torch.Tensor:
    flat = model(_edited(x, masks, grid_h, grid_w, preserve, offsets))
    return flat.view(masks.shape[1], x.shape[0], -1).transpose(0, 1)


def _offsets(generator: torch.Generator, players: int, limit: tuple[int, int]) -> list[tuple[int, int]]:
    """One pixel offset per player, uniform on ``{-s, ..., s}`` per axis."""
    dy = torch.randint(-limit[0], limit[0] + 1, (players,), generator=generator)
    dx = torch.randint(-limit[1], limit[1] + 1, (players,), generator=generator)
    return [(int(a), int(b)) for a, b in zip(dy, dx)]


def _reference(model, x, grid_h, grid_w, preserve: bool) -> torch.Tensor:
    if not preserve:
        return model(x)
    blank = x.new_zeros(x.shape[0], grid_h * grid_w)
    return model(keep(x, blank, grid_h, grid_w))


def _loss(model, x, masks, hypotheses, grid_h, grid_w, preserve: bool, shared: bool, offsets=None) -> torch.Tensor:
    ids = hypotheses.ids.long()
    reference = _reference(model, x, grid_h, grid_w, preserve)
    edited = _player_logits(model, x, masks, grid_h, grid_w, preserve, offsets)
    if preserve:
        ref_unique = unique_log_odds(reference, ids)
        total = torch.zeros((), device=x.device, dtype=reference.dtype)
        for k in range(ids.shape[1]):
            total = total + (unique_log_odds(edited[:, k], ids)[:, k] - ref_unique[:, k]).sum()
        if shared:
            total = total + (shared_log_odds(edited[:, -1], ids) - shared_log_odds(reference, ids)).sum()
        return -total
    unique_payoff, shared_payoff = deletion_payoffs(reference, edited, hypotheses, shared=shared)
    total = unique_payoff.sum()
    if shared_payoff is not None:
        total = total + shared_payoff.sum()
    return -total


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


def _initial_theta(evidence, players: int, units: int, use_shared: bool, init: str, x: torch.Tensor) -> torch.Tensor:
    if init not in {"evidence", "uniform"}:
        raise ValueError(f"unknown init {init!r}")
    fill = math.log(1.0 / float(units))
    if init == "uniform" or evidence is None:
        theta = x.new_full((x.shape[0], players + 1, units), fill)
        return theta.detach().requires_grad_(True)
    rows = [torch.log(evidence[:, k] + EPS) for k in range(evidence.shape[1])]
    if use_shared:
        rows.append(torch.log(evidence.mean(dim=1) + EPS))
    while len(rows) < players:
        rows.append(x.new_full((x.shape[0], units), fill))
    rows.append(x.new_full((x.shape[0], units), fill))
    return torch.stack(rows, dim=1).detach().requires_grad_(True)


def _masks(theta, row_target, area: float, cfg: AllocateConfig, dual: list | None = None) -> torch.Tensor:
    if cfg.independent:
        return _independent_masks(theta[:, :-1], area, cfg.projection)
    return _project(theta, row_target, cfg.projection, dual)[:, :-1]


def allocate(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    grid_h: int,
    grid_w: int,
    area: float,
    cfg: Optional[AllocateConfig] = None,
) -> Allocation:
    """Solve one area. ``x`` is ``(B, 3, H, W)`` already on the model's device."""
    cfg = cfg or AllocateConfig()
    if not 0.0 < float(area) <= 1.0:
        raise ValueError("area must be in (0, 1]")
    parameter = next(model.parameters())
    if x.device != parameter.device:
        raise RuntimeError("x and the model are on different devices; pass both explicitly")
    if cfg.pair_only:
        hypotheses = HypothesisSet(ids=hypotheses.ids[:, :2], mask=hypotheses.mask[:, :2])
    if hypotheses.ids.shape[1] < 2:
        raise ValueError("allocation needs at least two hypotheses")

    units = grid_h * grid_w
    with eval_mode(model):
        with torch.no_grad():
            class_count = int(model(x[:1]).shape[-1])
        use_shared = bool(cfg.shared) and not cfg.independent and hypotheses.ids.shape[1] < class_count
        players = hypotheses.ids.shape[1] + (1 if use_shared else 0)
        evidence = _prior(model, x, hypotheses, grid_h, grid_w, cfg) if cfg.init == "evidence" else None
        flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            theta = _initial_theta(evidence, players, units, use_shared, cfg.init, x)
            opt = torch.optim.Adam([theta], lr=float(cfg.lr))
            row_target = _row_targets(players, units, float(area), x.device, x.dtype)
            dual: list = [None]
            generator = torch.Generator(device="cpu").manual_seed(OFFSET_SEED)
            limit = boundary_offset(x.shape[-2], x.shape[-1], grid_h, grid_w)
            for _ in range(int(cfg.steps)):
                opt.zero_grad(set_to_none=True)
                offsets = _offsets(generator, players, limit) if cfg.boundary_shift else None
                loss = _loss(
                    model, x.detach(), _masks(theta, row_target, float(area), cfg, dual),
                    hypotheses, grid_h, grid_w, cfg.preserve, use_shared, offsets,
                )
                loss.backward()
                opt.step()
            with torch.no_grad():
                if cfg.independent or cfg.projection != "sinkhorn":
                    player_masks = _masks(theta, row_target, float(area), cfg, dual)
                    transport = player_masks
                    target = row_target[:players]
                else:
                    transport = sinkhorn(theta.detach(), row_target, iters=SINKHORN_ITERS, dual=dual)
                    player_masks = transport[:, :-1]
                    target = row_target
        finally:
            for p, flag in zip(model.parameters(), flags):
                p.requires_grad_(flag)

    unique = player_masks[:, : hypotheses.ids.shape[1]]
    shared = player_masks[:, -1] if use_shared else None
    return Allocation(
        unique=unique.detach(),
        shared=None if shared is None else shared.detach(),
        transport=transport.detach(),
        row_target=target.detach(),
    )
