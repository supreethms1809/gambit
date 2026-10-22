"""Comparators for the decomposition game. FORMULATION_DECOMPOSITION.md section 4.

Every per-class method runs unmodified. One declared split rule turns its two
class maps into the three decomposition maps:

    min_split(E_k, E_l): clamp at 0, sum-normalise each map,
                         S = min(E_k, E_l), U_k = E_k - S, U_l = E_l - S.

The assignment comparators score each cell on its own and solve the exact
allocation of three a·R budgets to exclusive cells as a linear programme:

    occlusion_assign   g_{j,r} = excess drop of z_j when cell r alone is deleted
                       (CDEA's blur deletion, null at one cell)
    gradient_assign    g_{j,r} = <grad z_j, (x - b(x)) on cell r>, null = mean over cells

    s_k = g_k - |g_l|,  s_l = g_l - |g_k|,  s_S = min(g_k, g_l)

If occlusion_assign matches CDEA, the joint search adds nothing over additive
per-cell scores, and the paper's method is the assignment.

Version 2 (FORMULATION_DECOMPOSITION.md section 5) divides the evidence region
Omega instead, with free role sizes:

    rule rows          the two class maps sum-pooled to the grid, clamped,
                       sum-normalised; each Omega cell takes the largest of
                       E_k - m, E_l - m and m (``cdea.region.rule_labels``)
    occlusion_partition, gradient_partition
                       a cell takes its best of s_k, s_l, s_S if that score is
                       above 0, else the empty role. A cell with g_k > 0 and
                       g_l > 0 cannot be unique (the shared rule), so it is S.
                       With free sizes this is the exact optimum of the
                       additive objective under that rule.
    random_partition   each Omega cell takes a seeded uniform role
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.eval_mode import eval_mode
from core.grid import deletion_baseline, delete, pool_sum
from cdea.region import NEITHER, OUTSIDE, normalise, rule_labels, rule_parts


def _normalise(evidence: torch.Tensor) -> torch.Tensor:
    flat = evidence.clamp_min(0).reshape(evidence.shape[0], -1)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return flat.reshape(evidence.shape)


def min_split(e_k: torch.Tensor, e_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(U_k, U_l, S)`` from two class maps of the same shape ``(B, ...)``, at their own resolution."""
    if e_k.shape != e_l.shape:
        raise ValueError("the two class maps must share a shape")
    k = _normalise(e_k)
    l = _normalise(e_l)
    shared = torch.minimum(k, l)
    return k - shared, l - shared, shared


# ---------------------------------------------------------------------------
# Per-cell scores and the exact assignment
# ---------------------------------------------------------------------------

def cell_scores(g: torch.Tensor) -> torch.Tensor:
    """``(B, 2, R)`` per-cell excess drops of k and l to ``(B, 3, R)`` scores for U_k, U_l, S."""
    if g.ndim != 3 or g.shape[1] != 2:
        raise ValueError("g must be (batch, 2 classes, units)")
    g_k, g_l = g[:, 0], g[:, 1]
    return torch.stack([g_k - g_l.abs(), g_l - g_k.abs(), torch.minimum(g_k, g_l)], dim=1)


def assign(scores: torch.Tensor, budget: float) -> torch.Tensor:
    """Maximise ``sum A * scores`` with every row summing to ``budget``, columns at most 1, ``0 <= A <= 1``.

    ``scores`` is ``(B, P, R)``. Solved per image with HiGHS. A transportation
    polytope with integral budgets has integral vertices, so the solution is
    then a hard assignment; a fractional budget leaves at most a fractional cell
    per row.
    """
    from scipy.optimize import linprog

    batch, players, units = scores.shape
    if players * float(budget) > units + 1e-9:
        raise ValueError("infeasible: players times budget exceeds the units")
    a_eq = np.kron(np.eye(players), np.ones((1, units)))
    b_eq = np.full(players, float(budget))
    a_ub = np.tile(np.eye(units), (1, players))
    b_ub = np.ones(units)
    out = []
    # cpu() before double(): MPS has no float64.
    values = scores.detach().cpu().double().numpy()
    for i in range(batch):
        result = linprog(-values[i].reshape(-1), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                         bounds=(0.0, 1.0), method="highs")
        if not result.success:
            raise RuntimeError(f"assignment LP failed: {result.message}")
        out.append(torch.from_numpy(result.x.reshape(players, units)))
    return torch.stack(out, dim=0).to(device=scores.device, dtype=scores.dtype).clamp(0.0, 1.0)


def occlusion_drops(model: nn.Module, x: torch.Tensor, ids: torch.Tensor, grid_h: int, grid_w: int,
                    null: bool = True) -> torch.Tensor:
    """``(B, 2, R)`` excess drop of z_k and z_l when one cell alone is blur-deleted.

    R + 1 forward passes, plus the null's. Every class comes from the same forwards.
    """
    from cdea.null import null_drops

    units = grid_h * grid_w
    ids = ids[:, :2].long()
    with eval_mode(model), torch.no_grad():
        full = model(x).gather(1, ids)
        drops = []
        for r in range(units):
            mask = x.new_zeros(x.shape[0], units)
            mask[:, r] = 1.0
            drops.append(full - model(delete(x, mask, grid_h, grid_w)).gather(1, ids))
        g = torch.stack(drops, dim=-1)
        if null:
            g = g - null_drops(model, x, ids, grid_h, grid_w, 1.0 / units).unsqueeze(-1)
    return g


def gradient_drops(model: nn.Module, x: torch.Tensor, ids: torch.Tensor, grid_h: int, grid_w: int,
                   null: bool = True) -> torch.Tensor:
    """``(B, 2, R)`` first-order drop of z_k and z_l per cell, ``<grad z_j, (x - b(x)) on cell r>``.

    The null of a one-cell deletion is the mean over cells, its first-order expectation.
    """
    ids = ids[:, :2].long()
    x_in = x.detach().requires_grad_(True)
    with eval_mode(model):
        logits = model(x_in).gather(1, ids)
        delta = x.detach() - deletion_baseline(x.detach(), grid_h, grid_w)
        columns = []
        for j in range(2):
            grad = torch.autograd.grad(logits[:, j].sum(), x_in, retain_graph=j == 0)[0]
            columns.append(pool_sum((grad * delta).sum(dim=1), grid_h, grid_w))
    g = torch.stack(columns, dim=1).detach()
    if null:
        g = g - g.mean(dim=-1, keepdim=True)
    return g


def occlusion_assign(model, x, ids, grid_h: int, grid_w: int, area: float, g: torch.Tensor | None = None):
    """``(B, 3, R)`` exact assignment of the occlusion cell scores. Pass ``g`` to reuse it across areas."""
    g = occlusion_drops(model, x, ids, grid_h, grid_w) if g is None else g
    return assign(cell_scores(g), float(area) * grid_h * grid_w)


def gradient_assign(model, x, ids, grid_h: int, grid_w: int, area: float, g: torch.Tensor | None = None):
    """``(B, 3, R)`` exact assignment of the first-order cell scores (ablation DA3)."""
    g = gradient_drops(model, x, ids, grid_h, grid_w) if g is None else g
    return assign(cell_scores(g), float(area) * grid_h * grid_w)


# ---------------------------------------------------------------------------
# Per-class sources, unmodified
# ---------------------------------------------------------------------------

def captum_occlusion(model: nn.Module, x: torch.Tensor, class_idx: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """``(B, R)`` Captum ``Occlusion`` of z_j: window and stride one cell, author default baseline (0)."""
    from captum.attr import Occlusion

    height, width = x.shape[-2:]
    if height % grid_h or width % grid_w:
        raise ValueError("occlusion needs the image to divide into whole cells")
    window = (x.shape[1], height // grid_h, width // grid_w)
    with eval_mode(model):
        attr = Occlusion(model).attribute(x.detach(), target=class_idx.long(), sliding_window_shapes=window,
                                          strides=window)
    per_pixel = attr.detach().sum(dim=1)
    counts = pool_sum(torch.ones_like(per_pixel), grid_h, grid_w)
    return pool_sum(per_pixel, grid_h, grid_w) / counts


def captum_ig(model: nn.Module, x: torch.Tensor, class_idx: torch.Tensor, steps: int) -> torch.Tensor:
    """``(B, H, W)`` channel-sum IG of z_j, Captum, zero baseline, right Riemann sum."""
    from captum.attr import IntegratedGradients

    if steps < 2:
        raise ValueError("IG needs at least 2 steps")
    with eval_mode(model):
        attr = IntegratedGradients(model).attribute(
            x.detach(), baselines=torch.zeros_like(x), target=class_idx.long(), n_steps=steps,
            method="riemann_right", internal_batch_size=x.shape[0],
        )
    return attr.detach().sum(dim=1)


# ---------------------------------------------------------------------------
# Version 2: partitions of Omega
# ---------------------------------------------------------------------------

def grid_evidence(e: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """``(B, R)`` cell map: a pixel map ``(B, H, W)`` is sum-pooled; a cell map passes through."""
    if e.ndim == 3:
        return pool_sum(e, grid_h, grid_w)
    if e.ndim != 2 or e.shape[-1] != grid_h * grid_w:
        raise ValueError("evidence must be (batch, H, W) pixels or (batch, R) cells")
    return e


def rule_partition(e_k: torch.Tensor, e_l: torch.Tensor, omega: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """``(B, R)`` labels: the two maps pooled to the grid, clamped, normalised, and split by the rule on Omega."""
    evidence = normalise(torch.stack([grid_evidence(e_k, grid_h, grid_w), grid_evidence(e_l, grid_h, grid_w)], dim=1))
    return rule_labels(rule_parts(evidence.to(omega.device)), omega)


def cell_partition(g: torch.Tensor, omega: torch.Tensor, shared_rule: bool = True) -> torch.Tensor:
    """``(B, R)`` labels from ``(B, 2, R)`` per-cell excess drops: the best cell score if above 0, else empty.

    With ``shared_rule``, a cell that lowers both classes (``g_k > 0`` and
    ``g_l > 0``) cannot be unique, so it takes S, whose score ``min(g_k, g_l)``
    is then above 0.
    """
    g = g.to(omega.device)
    scores = cell_scores(g)
    if shared_rule:
        both = (g[:, 0] > 0) & (g[:, 1] > 0)
        scores[:, :2] = scores[:, :2].masked_fill(both.unsqueeze(1), float("-inf"))
    best, role = scores.max(dim=1)
    labels = torch.where(best > 0, role, torch.full_like(role, NEITHER))
    return torch.where(omega, labels, torch.full_like(labels, OUTSIDE))


def occlusion_partition(model, x, ids, grid_h: int, grid_w: int, omega: torch.Tensor,
                        g: torch.Tensor | None = None) -> torch.Tensor:
    """Single-cell blur drops, excess over the null of one cell, through ``cell_partition``."""
    g = occlusion_drops(model, x, ids, grid_h, grid_w) if g is None else g
    return cell_partition(g, omega)


def gradient_partition(model, x, ids, grid_h: int, grid_w: int, omega: torch.Tensor,
                       g: torch.Tensor | None = None) -> torch.Tensor:
    """First-order cell drops through ``cell_partition`` (ablation DA3)."""
    g = gradient_drops(model, x, ids, grid_h, grid_w) if g is None else g
    return cell_partition(g, omega)


def margin_partition(margin: torch.Tensor, omega: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """``(B, R)`` labels of the margin map on Omega: positive cells U_k, negative U_l, zero empty; no S."""
    cells = grid_evidence(margin, grid_h, grid_w).to(omega.device)
    labels = torch.full_like(cells, NEITHER, dtype=torch.long)
    labels = torch.where(cells > 0, torch.zeros_like(labels), labels)
    labels = torch.where(cells < 0, torch.ones_like(labels), labels)
    return torch.where(omega, labels, torch.full_like(labels, OUTSIDE))


def random_partition(omega: torch.Tensor, rows: list[int], generator: torch.Generator) -> torch.Tensor:
    """``(B, R)`` labels: a uniform role per Omega cell. Row ``i`` of the draw belongs to sample position ``i``,
    so an image's roles do not depend on the chunk."""
    table = torch.randint(0, 4, (max(rows) + 1, omega.shape[-1]), generator=generator)
    labels = table[torch.tensor(rows)].to(omega.device)
    return torch.where(omega, labels, torch.full_like(labels, OUTSIDE))
