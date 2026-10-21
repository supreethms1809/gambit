"""Decomposition game.

Version 2 (``partition``, FORMULATION_DECOMPOSITION.md section 4) divides the
evidence region Omega of one pair (k = rank 0, l = rank 1) into four roles:
U_k, unique to k; U_l, unique to l; S, shared; and the empty role. Role sizes
are free. The plan is a softmax over the roles in every cell, so roles are
exclusive by construction and there is no Sinkhorn step. Each payoff subtracts
the null at the role's own size. Initialised at the subtraction rule.

Version 1 (``decompose``, tag ``decomp-declared``, ablation DA10): three
players own a·R units each, anywhere in the image, through the Sinkhorn plan
of ``allocate()``.

Both use binary draws with a straight-through gradient, boundary offsets,
Adam, and a per-image sum loss. Model parameters stay frozen and the module
stays in eval mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.eval_mode import eval_mode
from core.grid import boundary_offset
from core.types import HypothesisSet
from cdea._plan import (
    SINKHORN_ITERS,
    _draw,
    _independent_masks,
    _offsets,
    _player_logits,
    _project,
    _row_targets,
)
from cdea.allocation import DRAW_SEED, EPS, _prior
from cdea.null import N_NULL, NULL_SEED, null_at, null_drops, null_size_grid, size_null
from cdea.payoffs import class_drops, decomposition_payoffs
from cdea.region import OUTSIDE, RHO, evidence_region, roles_from_labels, rule_parts
from cdea.sinkhorn import sinkhorn

PLAYERS = 3   # U_k, U_l, S


@dataclass
class Decomposition:
    unique_k: torch.Tensor               # (B, R)
    unique_l: torch.Tensor               # (B, R)
    shared: torch.Tensor                 # (B, R)
    transport: torch.Tensor              # (B, 4, R) with the unallocated row; (B, 3, R) when independent
    row_target: torch.Tensor
    null: torch.Tensor                   # (B, 2) blur null of k and l; zeros when the null is off
    initial: torch.Tensor                # (B, 3, R) the projected plan at step 0
    prior_split: Optional[torch.Tensor]  # (B, 3, R) the min-split of the evidence, or None (uniform init)


@dataclass
class DecomposeConfig:
    steps: int = 100
    lr: float = 0.1
    init: str = "minsplit"               # minsplit | mean (DA5) | uniform (DA4)
    backend: str = "gradcam"             # gradcam on ResNet-50, ig on ViT
    ig_steps: int = 16
    projection: str = "sinkhorn"
    binary_masks: bool = True            # False is ablation DA6
    boundary_shift: bool = True          # False is ablation DA7
    draw_seed: int = DRAW_SEED
    null: bool = True                    # False is ablation DA8
    n_null: int = N_NULL
    null_seed: int = NULL_SEED
    independent: bool = False            # True is ablation DA1


def min_split_rows(evidence: torch.Tensor) -> torch.Tensor:
    """``(B, 2, R)`` sum-normalised evidence of k and l to ``(E_k - m, E_l - m, m)``, ``m = min``."""
    shared = torch.minimum(evidence[:, 0], evidence[:, 1])
    return torch.stack([evidence[:, 0] - shared, evidence[:, 1] - shared, shared], dim=1)


def _initial_theta(evidence: Optional[torch.Tensor], init: str, units: int, x: torch.Tensor):
    if init not in {"minsplit", "mean", "uniform"}:
        raise ValueError(f"unknown init {init!r}")
    fill = math.log(1.0 / float(units))
    empty = x.new_full((x.shape[0], 1, units), fill)
    if init == "uniform":
        theta = x.new_full((x.shape[0], PLAYERS + 1, units), fill)
        return theta.detach().requires_grad_(True), None
    if init == "minsplit":
        split = min_split_rows(evidence)
        rows = torch.log(split + EPS)
    else:
        split = None
        rows = torch.log(torch.cat([evidence, evidence.mean(dim=1, keepdim=True)], dim=1) + EPS)
    theta = torch.cat([rows, empty], dim=1)
    return theta.detach().requires_grad_(True), split


def _plan(theta, row_target, area: float, cfg: DecomposeConfig, dual: list | None = None) -> torch.Tensor:
    if cfg.independent:
        return _independent_masks(theta[:, :-1], area, cfg.projection)
    return _project(theta, row_target, cfg.projection, dual)[:, :-1]


def decompose(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    grid_h: int,
    grid_w: int,
    area: float,
    cfg: Optional[DecomposeConfig] = None,
) -> Decomposition:
    """Solve one area for the pair (rank 0, rank 1). ``x`` is on the model's device."""
    cfg = cfg or DecomposeConfig()
    if not 0.0 < float(area) <= 1.0:
        raise ValueError("area must be in (0, 1]")
    parameter = next(model.parameters())
    if x.device != parameter.device:
        raise RuntimeError("x and the model are on different devices; pass both explicitly")
    if hypotheses.ids.shape[1] < 2:
        raise ValueError("the decomposition needs a pair of hypotheses")
    pair = HypothesisSet(ids=hypotheses.ids[:, :2], mask=hypotheses.mask[:, :2])
    ids = pair.ids.long()
    units = grid_h * grid_w

    with eval_mode(model):
        evidence = _prior(model, x, pair, grid_h, grid_w, cfg) if cfg.init != "uniform" else None
        flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            with torch.no_grad():
                reference = model(x.detach())
            if cfg.null:
                null = null_drops(model, x.detach(), ids, grid_h, grid_w, float(area), cfg.n_null, cfg.null_seed)
            else:
                null = reference.new_zeros(x.shape[0], 2)
            theta, split = _initial_theta(evidence, cfg.init, units, x)
            opt = torch.optim.Adam([theta], lr=float(cfg.lr))
            row_target = _row_targets(PLAYERS, units, float(area), x.device, x.dtype)
            dual: list = [None]
            with torch.no_grad():
                initial = _plan(theta.detach(), row_target, float(area), cfg, [None])
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.draw_seed))
            limit = boundary_offset(x.shape[-2], x.shape[-1], grid_h, grid_w)
            joint = not cfg.independent and cfg.projection == "sinkhorn"
            for _ in range(int(cfg.steps)):
                opt.zero_grad(set_to_none=True)
                offsets = _offsets(generator, PLAYERS, limit) if cfg.boundary_shift else None
                masks = _plan(theta, row_target, float(area), cfg, dual)
                if cfg.binary_masks:
                    shape = (units,) if joint else (PLAYERS, units)
                    masks = _draw(masks, torch.rand(shape, generator=generator), joint)
                edited = _player_logits(model, x.detach(), masks, grid_h, grid_w, False, offsets)
                payoff = decomposition_payoffs(class_drops(reference, edited, ids), null)
                loss = -payoff.sum()
                loss.backward()
                opt.step()
            with torch.no_grad():
                if joint:
                    transport = sinkhorn(theta.detach(), row_target, iters=SINKHORN_ITERS, dual=dual)
                    player_masks = transport[:, :-1]
                    target = row_target
                else:
                    player_masks = _plan(theta.detach(), row_target, float(area), cfg, dual)
                    transport = player_masks
                    target = row_target[:PLAYERS]
        finally:
            for p, flag in zip(model.parameters(), flags):
                p.requires_grad_(flag)

    return Decomposition(
        unique_k=player_masks[:, 0].detach(),
        unique_l=player_masks[:, 1].detach(),
        shared=player_masks[:, 2].detach(),
        transport=transport.detach(),
        row_target=target.detach(),
        null=null.detach(),
        initial=initial.detach(),
        prior_split=None if split is None else split.detach(),
    )


# ---------------------------------------------------------------------------
# Version 2: a partition of Omega with free role sizes.
# ---------------------------------------------------------------------------

ROLES_V2 = 4   # U_k, U_l, S, empty


@dataclass
class PartitionConfig:
    steps: int = 100
    lr: float = 0.1
    init: str = "subtract"               # subtract | mean (DA5) | uniform (DA4)
    backend: str = "gradcam"             # gradcam on ResNet-50, ig on ViT
    ig_steps: int = 16
    rho: float = RHO
    binary_masks: bool = True            # False is ablation DA6
    boundary_shift: bool = True          # False is ablation DA7
    draw_seed: int = DRAW_SEED
    null: bool = True                    # False is ablation DA8
    n_null: int = N_NULL
    null_seed: int = NULL_SEED
    null_grid: bool = False              # True: measure the null on ``null_size_grid`` (ViT-B/16)
    independent: bool = False            # True is ablation DA1


@dataclass
class Partition:
    roles: torch.Tensor                  # (B, 4, R) bool: U_k, U_l, S, empty; exclusive unless independent
    labels: Optional[torch.Tensor]       # (B, R) long, OUTSIDE off Omega; None when roles may overlap
    plan: torch.Tensor                   # (B, 4, R) the final plan; independent: the three role probabilities and 1 - their max
    initial_roles: torch.Tensor          # (B, 4, R) bool: the step-0 hard plan
    init_parts: Optional[torch.Tensor]   # (B, 3, R) the parts the roles were initialised from; None for uniform
    omega: torch.Tensor                  # (B, R) bool
    evidence: torch.Tensor               # (B, 2, R) normalised reference evidence
    null: torch.Tensor                   # (B, 2, S + 1) blur null by size; zeros when the null is off


def reference_evidence(model: nn.Module, x: torch.Tensor, pair: HypothesisSet, grid_h: int, grid_w: int,
                       backend: str, ig_steps: int = 16) -> torch.Tensor:
    """``(B, 2, R)`` clamped, sum-normalised evidence of k and l: ``E_k`` and ``E_l`` of section 3.1."""
    from types import SimpleNamespace

    with eval_mode(model):
        return _prior(model, x, pair, grid_h, grid_w, SimpleNamespace(backend=backend, ig_steps=ig_steps)).detach()


def _initial_parts(evidence: torch.Tensor, init: str) -> Optional[torch.Tensor]:
    if init == "subtract":
        return rule_parts(evidence)
    if init == "mean":
        return torch.cat([evidence, evidence.mean(dim=1, keepdim=True)], dim=1)
    if init == "uniform":
        return None
    raise ValueError(f"unknown init {init!r}")


def _role_plan(theta: torch.Tensor, independent: bool) -> torch.Tensor:
    """``(B, 4, R)``. Joint: a softmax over the roles. Independent: one Bernoulli per role against the empty row."""
    if not independent:
        return torch.softmax(theta, dim=1)
    roles = torch.sigmoid(theta[:, :-1] - theta[:, -1:])
    return torch.cat([roles, 1.0 - roles.max(dim=1, keepdim=True).values], dim=1)


def _hard_roles(plan: torch.Tensor, omega: torch.Tensor, independent: bool) -> torch.Tensor:
    if not independent:
        labels = torch.where(omega, plan.argmax(dim=1), torch.full_like(omega, OUTSIDE, dtype=torch.long))
        return roles_from_labels(labels)
    taken = (plan[:, :-1] > 0.5) & omega.unsqueeze(1)
    empty = omega & ~taken.any(dim=1)
    return torch.cat([taken, empty.unsqueeze(1)], dim=1)


def partition(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    grid_h: int,
    grid_w: int,
    cfg: Optional[PartitionConfig] = None,
    evidence: Optional[torch.Tensor] = None,
) -> Partition:
    """Partition Omega for the pair (rank 0, rank 1). ``x`` is on the model's device.

    ``evidence`` is the ``(B, 2, R)`` reference evidence; pass it so that every
    method divides the same Omega. It defines Omega whatever ``cfg.init`` is.
    """
    cfg = cfg or PartitionConfig()
    parameter = next(model.parameters())
    if x.device != parameter.device:
        raise RuntimeError("x and the model are on different devices; pass both explicitly")
    if hypotheses.ids.shape[1] < 2:
        raise ValueError("the decomposition needs a pair of hypotheses")
    pair = HypothesisSet(ids=hypotheses.ids[:, :2], mask=hypotheses.mask[:, :2])
    ids = pair.ids.long()
    units = grid_h * grid_w
    if evidence is None:
        evidence = reference_evidence(model, x, pair, grid_h, grid_w, cfg.backend, cfg.ig_steps)
    evidence = evidence.detach().to(device=x.device, dtype=x.dtype)
    omega = evidence_region(evidence, cfg.rho)
    inside = omega.to(x.dtype).unsqueeze(1)
    top = int(omega.sum(dim=-1).max())

    with eval_mode(model):
        flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            with torch.no_grad():
                reference = model(x.detach())
            if cfg.null and top > 0:
                sizes = null_size_grid(top) if cfg.null_grid else None
                table = size_null(model, x.detach(), ids, grid_h, grid_w, top, cfg.n_null, cfg.null_seed,
                                  sizes=sizes)
            else:
                table = reference.new_zeros(x.shape[0], 2, top + 1)
            parts = _initial_parts(evidence, cfg.init)
            if parts is None:
                theta = x.new_zeros(x.shape[0], ROLES_V2, units)
            else:
                empty = x.new_full((x.shape[0], 1, units), math.log(EPS))
                theta = torch.cat([torch.log(parts + EPS), empty], dim=1)
            theta = theta.detach().requires_grad_(True)
            initial = _hard_roles(_role_plan(theta.detach(), cfg.independent), omega, cfg.independent)
            opt = torch.optim.Adam([theta], lr=float(cfg.lr))
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.draw_seed))
            limit = boundary_offset(x.shape[-2], x.shape[-1], grid_h, grid_w)
            for _ in range(int(cfg.steps)):
                opt.zero_grad(set_to_none=True)
                offsets = _offsets(generator, PLAYERS, limit) if cfg.boundary_shift else None
                masks = _role_plan(theta, cfg.independent)[:, :PLAYERS]
                if cfg.binary_masks:
                    shape = (PLAYERS, units) if cfg.independent else (units,)
                    masks = _draw(masks, torch.rand(shape, generator=generator), not cfg.independent)
                masks = masks * inside
                edited = _player_logits(model, x.detach(), masks, grid_h, grid_w, False, offsets)
                null = null_at(table, masks.sum(dim=-1)) if cfg.null else None
                payoff = decomposition_payoffs(class_drops(reference, edited, ids), null)
                loss = -payoff.sum()
                loss.backward()
                opt.step()
            with torch.no_grad():
                plan = _role_plan(theta.detach(), cfg.independent)
        finally:
            for p, flag in zip(model.parameters(), flags):
                p.requires_grad_(flag)

    roles = _hard_roles(plan, omega, cfg.independent)
    labels = None
    if not cfg.independent:
        labels = torch.where(omega, plan.argmax(dim=1), torch.full_like(omega, OUTSIDE, dtype=torch.long))
    return Partition(
        roles=roles,
        labels=labels,
        plan=(plan * omega.unsqueeze(1)).detach(),
        initial_roles=initial,
        init_parts=None if parts is None else parts.detach(),
        omega=omega,
        evidence=evidence,
        null=table.detach(),
    )


def v1_roles(out: Decomposition, omega: torch.Tensor, area: float) -> torch.Tensor:
    """``(B, 4, R)`` bool: a version-1 plan read as roles (ablation DA10).

    Each player takes its ``round(a R)`` cells of largest plan mass, as v1's
    hard masks did; a cell two players want goes to the larger mass, and the
    other player takes its next cell. The roles may lie anywhere in the image.
    Omega cells no player takes are empty.
    """
    plan = out.transport[:, :PLAYERS]
    batch, _, units = plan.shape
    budget = max(1, int(round(float(area) * units)))
    taken = torch.zeros(batch, PLAYERS, units, dtype=torch.bool, device=plan.device)
    for b in range(batch):
        order = plan[b].flatten().argsort(descending=True, stable=True)
        owned = torch.zeros(units, dtype=torch.bool, device=plan.device)
        counts = [0] * PLAYERS
        for flat in order.tolist():
            player, cell = divmod(flat, units)
            if counts[player] < budget and not bool(owned[cell]):
                taken[b, player, cell] = True
                owned[cell] = True
                counts[player] += 1
            if min(counts) == budget:
                break
    empty = omega & ~taken.any(dim=1)
    return torch.cat([taken, empty.unsqueeze(1)], dim=1)
