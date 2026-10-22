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
from typing import Callable, Optional

import torch
import torch.nn as nn

from core.eval_mode import eval_mode
from core.grid import boundary_offset, delete
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
from cdea.null import N_NULL, NULL_SEED, interpolate_sizes, null_at, null_drops, null_size_grid, size_masks, size_null
from cdea.payoffs import (
    cell_charges,
    class_drops,
    cross_penalty,
    decomposition_payoffs,
    partition_payoffs,
    shared_cells,
)
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
    polish: bool = True                  # False is ablation DA11: the straight-through search alone
    polish_moves: Optional[int] = None   # cap on accepted moves per image; None is 4 R
    payoff: str = "cell_positive"        # cell_positive | cell_abs (DA15) | set (DA14, version 2 as read on the GH200)
    shared_rule: bool = True             # False is ablation DA16: roles open to every cell, as the payoff alone
    shared_margin: float = 0.0           # g_k and g_l must both exceed this for the shared rule to bind


@dataclass
class Partition:
    roles: torch.Tensor                  # (B, 4, R) bool: U_k, U_l, S, empty; exclusive unless independent
    labels: Optional[torch.Tensor]       # (B, R) long, OUTSIDE off Omega; None when roles may overlap
    plan: torch.Tensor                   # (B, 4, R) the final plan; independent: the three role probabilities and 1 - their max
    initial_roles: torch.Tensor          # (B, 4, R) bool: the step-0 hard plan
    search_roles: torch.Tensor           # (B, 4, R) bool: the hard plan the straight-through search ends at
    welfare: torch.Tensor                # (B,) blur welfare of ``roles``, the polish's objective
    init_parts: Optional[torch.Tensor]   # (B, 3, R) the parts the roles were initialised from; None for uniform
    omega: torch.Tensor                  # (B, R) bool
    evidence: torch.Tensor               # (B, 2, R) normalised reference evidence
    null: torch.Tensor                   # (B, 2, S + 1) blur null by size; zeros when the null is off
    payoff: str = "set"                  # the payoff the partition maximised
    cell_excess: Optional[torch.Tensor] = None  # (B, 2, R) blur g_k, g_l of each Omega cell alone; None without a cell term
    shared: Optional[torch.Tensor] = None       # (B, R) bool: Omega cells that lower both; only they may be S


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


def closed_roles(shared: torch.Tensor) -> torch.Tensor:
    """``(B, 4, R)`` bool, the shared rule: U_k and U_l are closed on the cells of ``shared``, S on the rest.

    The empty role is always open.
    """
    closed = torch.zeros(shared.shape[0], ROLES_V2, shared.shape[1], dtype=torch.bool, device=shared.device)
    closed[:, :2] = shared.unsqueeze(1)
    closed[:, 2] = ~shared
    return closed


def _role_plan(theta: torch.Tensor, independent: bool, shared: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``(B, 4, R)``. Joint: a softmax over the roles. Independent: one Bernoulli per role against the empty row.

    With ``shared``, the roles ``closed_roles`` closes get logit -inf (the shared rule).
    """
    if shared is not None:
        theta = theta.masked_fill(closed_roles(shared), float("-inf"))
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
    polish_edit: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Partition:
    """Partition Omega for the pair (rank 0, rank 1). ``x`` is on the model's device.

    ``evidence`` is the ``(B, 2, R)`` reference evidence; pass it so that every
    method divides the same Omega. It defines Omega whatever ``cfg.init`` is.

    ``polish_edit(rows, cells)`` deletes the ``(n, R)`` whole-cell masks
    ``cells`` from the images ``x[rows]``. The polish, the choice of its start
    and the polish's own null then use it instead of blur (ablation DA13 passes
    ROAD's fill). The search always uses blur.
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
                sizes = null_size_grid(top, units) if cfg.null_grid else None
                table = size_null(model, x.detach(), ids, grid_h, grid_w, top, cfg.n_null, cfg.null_seed,
                                  sizes=sizes)
            else:
                table = reference.new_zeros(x.shape[0], 2, top + 1)
            cell = None
            if cfg.payoff != "set" or cfg.shared_rule:
                cell = single_cell_excess(model, ids, reference, blur_edit(x.detach(), grid_h, grid_w), omega,
                                          table, cfg.null)
            shared = shared_cells(cell, omega, cfg.shared_margin) if cfg.shared_rule else None
            charged = cell if cfg.payoff != "set" else None
            parts = _initial_parts(evidence, cfg.init)
            if parts is None:
                theta = x.new_zeros(x.shape[0], ROLES_V2, units)
            else:
                empty = x.new_full((x.shape[0], 1, units), math.log(EPS))
                theta = torch.cat([torch.log(parts + EPS), empty], dim=1)
            theta = theta.detach().requires_grad_(True)
            initial = _hard_roles(_role_plan(theta.detach(), cfg.independent, shared), omega, cfg.independent)
            opt = torch.optim.Adam([theta], lr=float(cfg.lr))
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.draw_seed))
            limit = boundary_offset(x.shape[-2], x.shape[-1], grid_h, grid_w)
            for _ in range(int(cfg.steps)):
                opt.zero_grad(set_to_none=True)
                offsets = _offsets(generator, PLAYERS, limit) if cfg.boundary_shift else None
                masks = _role_plan(theta, cfg.independent, shared)[:, :PLAYERS]
                if cfg.binary_masks:
                    shape = (PLAYERS, units) if cfg.independent else (units,)
                    masks = _draw(masks, torch.rand(shape, generator=generator), not cfg.independent)
                masks = masks * inside
                edited = _player_logits(model, x.detach(), masks, grid_h, grid_w, False, offsets)
                null = null_at(table, masks.sum(dim=-1)) if cfg.null else None
                penalty = None if charged is None else cross_penalty(masks, charged, cfg.payoff)
                payoff = partition_payoffs(class_drops(reference, edited, ids), null, cfg.payoff, penalty)
                loss = -payoff.sum()
                loss.backward()
                opt.step()
            with torch.no_grad():
                plan = _role_plan(theta.detach(), cfg.independent, shared)
            search = _hard_roles(plan, omega, cfg.independent)
            moves = 4 * units if cfg.polish_moves is None else int(cfg.polish_moves)
            with torch.no_grad():
                edit = polish_edit or blur_edit(x.detach(), grid_h, grid_w)
                polish_table, polish_cell = table, charged
                if polish_edit is not None:
                    polish_table = reference.new_zeros(x.shape[0], 2, top + 1)
                    if cfg.null and top > 0:
                        sizes = null_size_grid(top, units) if cfg.null_grid else None
                        polish_table = edit_null(model, ids, reference, edit, x.shape[0], units, top, cfg.n_null,
                                                 cfg.null_seed, sizes)
                    if charged is not None:
                        polish_cell = single_cell_excess(model, ids, reference, edit, omega, polish_table, cfg.null)
                terms = dict(payoff=cfg.payoff, cell_excess=polish_cell)
                welfare_search = role_welfare(model, x.detach(), ids, reference, search, polish_table, cfg.null,
                                              grid_h, grid_w, edit, **terms)
                if cfg.polish and not cfg.independent:
                    welfare_init = role_welfare(model, x.detach(), ids, reference, initial, polish_table, cfg.null,
                                                grid_h, grid_w, edit, **terms)
                    better = (welfare_search >= welfare_init).view(-1, 1, 1)
                    start = torch.where(better, search, initial)
                    roles, welfare = polish(model, x.detach(), ids, reference, start, omega, polish_table, cfg.null,
                                            grid_h, grid_w, moves, edit, shared=shared, **terms)
                else:
                    roles, welfare = search, welfare_search
        finally:
            for p, flag in zip(model.parameters(), flags):
                p.requires_grad_(flag)

    labels = None
    if not cfg.independent:
        labels = torch.where(omega, roles.to(x.dtype).argmax(dim=1), torch.full_like(omega, OUTSIDE, dtype=torch.long))
    return Partition(
        roles=roles,
        labels=labels,
        plan=(plan * omega.unsqueeze(1)).detach(),
        initial_roles=initial,
        search_roles=search,
        welfare=welfare.detach(),
        init_parts=None if parts is None else parts.detach(),
        omega=omega,
        evidence=evidence,
        null=table.detach(),
        payoff=cfg.payoff,
        cell_excess=None if cell is None else cell.detach(),
        shared=shared,
    )


# ---------------------------------------------------------------------------
# The discrete polish (section 4.4). The straight-through gradient is a first-
# order estimate: where a payoff has a kink, as |d~_l| does at d~_l = 0, a
# whole cell's move can change the payoff in the opposite direction to its
# gradient, and the search can end below its own step 0. The polish starts
# from the better of the search's end and step 0, and moves one cell to
# another role at a time, each move the one that most raises the exact
# welfare of the hard partition. The welfare is the search's objective without
# the boundary offsets: blur deletion of whole cells and each role's null at
# its size. It ends when no move raises the welfare, so the result is never
# below step 0. Ablation DA13 deletes with ROAD's fill instead, with a null
# measured the same way.
# ---------------------------------------------------------------------------

POLISH_BATCH = 64     # deleted inputs per forward pass; the result does not depend on it
POLISH_TOL = 1e-6     # smallest welfare gain a move must make


def blur_edit(x: torch.Tensor, grid_h: int, grid_w: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """CDEA's deletion as an ``edit(rows, cells)``: blur of the whole cells of ``x[rows]``."""
    return lambda rows, cells: delete(x[rows], cells.to(x.dtype), grid_h, grid_w)


def _deletion_drops(model, ids, reference, rows: torch.Tensor, masks: torch.Tensor, edit) -> torch.Tensor:
    """``(n, J)`` drop of the classes ``ids`` when image ``rows[i]`` loses the cells of ``masks[i]``."""
    out = []
    for start in range(0, rows.numel(), POLISH_BATCH):
        r = rows[start:start + POLISH_BATCH]
        edited = edit(r, masks[start:start + POLISH_BATCH])
        out.append(reference[r].gather(1, ids[r]) - model(edited).gather(1, ids[r]))
    return torch.cat(out, dim=0) if out else reference.new_zeros(0, ids.shape[1])


def edit_null(model, ids, reference, edit, batch: int, units: int, top: int, n: int, seed: int,
              sizes: Optional[list[int]] = None) -> torch.Tensor:
    """``(B, J, top + 1)``: ``size_null``'s table under any ``edit(rows, cells)``.

    The same nested random masks, every image. ``sizes`` measures only those
    and interpolates the rest, as ``size_null`` does.
    """
    measured = list(range(top + 1)) if sizes is None else sorted({int(s) for s in sizes} | {0})
    masks = size_masks(units, measured, n, seed).to(reference.device)
    values = reference.new_zeros(batch, ids.shape[1], len(measured))
    rows = torch.arange(batch, device=reference.device).repeat(n)
    for index, size in enumerate(measured):
        if size == 0:
            continue
        flat = masks[index].repeat_interleave(batch, dim=0)
        drops = _deletion_drops(model, ids, reference, rows, flat, edit)
        values[..., index] = drops.view(n, batch, -1).mean(dim=0)
    if measured == list(range(top + 1)):
        return values
    grid = torch.tensor(measured, dtype=values.dtype, device=values.device)
    return interpolate_sizes(values, grid, top)


def single_cell_excess(model, ids, reference, edit, omega: torch.Tensor, table: torch.Tensor,
                       use_null: bool) -> torch.Tensor:
    """``(B, 2, R)``: ``g_j(r)``, the drop of z_j when Omega cell r alone is deleted, less the null of one cell.

    Zero off Omega. One forward pass per Omega cell, without gradients.
    """
    batch, units = omega.shape
    out = reference.new_zeros(batch, ids.shape[1], units)
    pairs = torch.nonzero(omega)
    if pairs.numel() == 0:
        return out
    rows, cells = pairs[:, 0], pairs[:, 1]
    masks = torch.nn.functional.one_hot(cells, units).to(reference.dtype)
    drops = _deletion_drops(model, ids, reference, rows, masks, edit)
    if use_null:
        drops = drops - table[rows, :, 1]
    out[rows, :, cells] = drops
    return out


def _penalty(roles: torch.Tensor, cell_excess, payoff: str):
    if payoff == "set":
        return None
    if cell_excess is None:
        raise ValueError(f"payoff {payoff!r} needs the single-cell excess drops")
    return cross_penalty(roles[:, :PLAYERS].to(cell_excess.dtype), cell_excess, payoff)


def _welfare_terms(drops: torch.Tensor, sizes: torch.Tensor, table: torch.Tensor, use_null: bool,
                   payoff: str = "set", penalty=None) -> torch.Tensor:
    return partition_payoffs(drops, null_at(table, sizes) if use_null else None, payoff, penalty)


def role_welfare(model, x, ids, reference, roles, table, use_null: bool, grid_h: int, grid_w: int,
                 edit=None, payoff: str = "set", cell_excess=None) -> torch.Tensor:
    """``(B,)`` welfare ``u_k + u_l + u_S`` of a hard partition ``(B, 4, R)``, each role net of its null.

    Blur deletion unless ``edit`` is given. ``payoff`` and ``cell_excess`` are
    as in ``partition_payoffs`` and ``single_cell_excess``.
    """
    edit = edit or blur_edit(x, grid_h, grid_w)
    batch, _, units = roles.shape
    rows = torch.arange(batch, device=x.device).repeat_interleave(PLAYERS)
    masks = roles[:, :PLAYERS].reshape(batch * PLAYERS, units)
    drops = _deletion_drops(model, ids, reference, rows, masks, edit).view(batch, PLAYERS, -1)
    sizes = roles[:, :PLAYERS].sum(dim=-1).to(x.dtype)
    return _welfare_terms(drops, sizes, table, use_null, payoff, _penalty(roles, cell_excess, payoff)).sum(dim=-1)


def polish(model, x, ids, reference, roles, omega, table, use_null: bool, grid_h: int, grid_w: int,
           moves: int, edit=None, payoff: str = "set", cell_excess=None,
           shared: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-improvement single-cell moves on an exclusive partition ``(B, 4, R)``. Returns it and its welfare.

    Blur deletion unless ``edit`` is given. Each image moves on its own, so its
    result does not depend on the batch. With the ``(B, R)`` ``shared``, no move
    enters a role ``closed_roles`` closes; ``roles`` must already obey it.
    """
    edit = edit or blur_edit(x, grid_h, grid_w)
    batch, roles_count, units = roles.shape
    current = roles.clone()
    rows = torch.arange(batch, device=x.device).repeat_interleave(PLAYERS)
    drops = _deletion_drops(model, ids, reference, rows, current[:, :PLAYERS].reshape(-1, units),
                            edit).view(batch, PLAYERS, -1)
    sizes = current[:, :PLAYERS].sum(dim=-1).to(x.dtype)
    penalty = _penalty(current, cell_excess, payoff)
    charge = None if penalty is None else cell_charges(cell_excess, payoff)
    welfare = _welfare_terms(drops, sizes, table, use_null, payoff, penalty).sum(dim=-1)
    closed = None if shared is None else closed_roles(shared)
    active = omega.any(dim=-1).tolist()
    for _ in range(int(moves)):
        if not any(active):
            break
        candidates, del_rows, del_masks, owners = [], [], [], []
        for b in range(batch):
            if not active[b]:
                continue
            labels = current[b].to(x.dtype).argmax(dim=0)
            for r in torch.nonzero(omega[b]).flatten().tolist():
                source = int(labels[r])
                for target in range(roles_count):
                    if target == source or (closed is not None and bool(closed[b, target, r])):
                        continue
                    index = len(candidates)
                    candidates.append((b, r, source, target))
                    for p in (source, target):
                        if p < PLAYERS:
                            mask = current[b, p].clone()
                            mask[r] = p == target
                            del_rows.append(b)
                            del_masks.append(mask)
                            owners.append((index, p))
        owner_rows = torch.tensor([c[0] for c in candidates], device=x.device)
        new = _deletion_drops(model, ids, reference, torch.tensor(del_rows, device=x.device),
                              torch.stack(del_masks), edit)
        cand_drops = drops[owner_rows].clone()
        cand_sizes = sizes[owner_rows].clone()
        for (index, p), value in zip(owners, new):
            cand_drops[index, p] = value
        cand_penalty = None if penalty is None else penalty[owner_rows].clone()
        for index, (b, r, source, target) in enumerate(candidates):
            if source < PLAYERS:
                cand_sizes[index, source] -= 1
            if target < PLAYERS:
                cand_sizes[index, target] += 1
            if cand_penalty is not None:
                # Only the unique roles carry a cell charge: U_k pays phi(g_l), U_l pays phi(g_k).
                if source < 2:
                    cand_penalty[index, source] -= charge[b, source, r]
                if target < 2:
                    cand_penalty[index, target] += charge[b, target, r]
        cand_welfare = _welfare_terms(cand_drops, cand_sizes, table[owner_rows], use_null, payoff,
                                      cand_penalty).sum(dim=-1)
        gain = cand_welfare - welfare[owner_rows]
        for b in range(batch):
            if not active[b]:
                continue
            mine = [i for i, c in enumerate(candidates) if c[0] == b]
            best = max(mine, key=lambda i: float(gain[i]))
            if float(gain[best]) <= POLISH_TOL:
                active[b] = False
                continue
            _, r, source, target = candidates[best]
            current[b, source, r] = False
            current[b, target, r] = True
            drops[b] = cand_drops[best]
            sizes[b] = cand_sizes[best]
            welfare[b] = cand_welfare[best]
            if penalty is not None:
                penalty[b] = cand_penalty[best]
    return current, welfare


def budget_roles(out: Decomposition, omega: torch.Tensor, area: float) -> torch.Tensor:
    """``(B, 4, R)`` bool: a version-1 plan read as roles (ablation DA10).

    Each player takes its ``round(a R)`` cells of largest plan mass, as version 1's
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
