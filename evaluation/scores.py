"""Primary scores. ROAD removal is the operator. Blur keep is not used here.

CD@a is the contrastive deletion score. The K×K matrix asks whether each
mask is specific to its class. ΔD asks whether the shortcut mask, rather than
a random mask of the same area, reduces cross-environment disagreement.
Two-patch recovery is the share of each class mask that lands on its patch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from evaluation.masks import mass_in
from evaluation.removal import noisy_linear_impute


def _restore(model: nn.Module, was_training: bool) -> None:
    model.train(was_training)


def _remove(x: torch.Tensor, mask: torch.Tensor, iters: Optional[int], noise: float, seed: int, offset: int = 0) -> torch.Tensor:
    """Impute pixels where ``mask`` is 1. ``mask`` is ``(B, H, W)`` or ``(B, 1, H, W)``.

    ``iters=None`` is ROAD's converged linear fill (``evaluation.removal``).

    ``offset`` is the first image's index in the full sample. The noise is then
    the rows a single call on the whole sample would have drawn for these images.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
        raise ValueError("removal mask does not match the image")
    keep = (1.0 - mask).clamp(0.0, 1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    from evaluation.masks import advance_generator

    advance_generator(generator, int(offset), tuple(x.shape[1:]), normal=True)
    return noisy_linear_impute(x, keep, iters, noise, generator)


def _margin(logits: torch.Tensor, kept: torch.Tensor, foil: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, kept[:, None]).squeeze(1) - logits.gather(1, foil[:, None]).squeeze(1)


def contrastive_deletion(
    model: nn.Module,
    x: torch.Tensor,
    mask_k: torch.Tensor,
    mask_l: torch.Tensor,
    class_k: torch.Tensor,
    class_l: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """CD = m(x without M_l) − m(x without M_k), with m = z_k − z_l.

    A positive score means removing the foil's evidence leaves k ahead of l,
    and removing k's evidence does not. The default removal is ROAD's
    converged linear fill with noise 0.01.
    """
    class_k = class_k.long()
    class_l = class_l.long()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            without_l = model(_remove(x, mask_l, iters, noise, seed, offset))
            without_k = model(_remove(x, mask_k, iters, noise, seed, offset))
            return _margin(without_l, class_k, class_l) - _margin(without_k, class_k, class_l)
    finally:
        _restore(model, was_training)


def drop_matrix(
    model: nn.Module,
    x: torch.Tensor,
    masks: torch.Tensor,
    class_ids: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
    remove=None,
) -> torch.Tensor:
    """``(B, P, K)`` drop in class j's logit when mask p is removed.

    Entry ``(p, j)`` is z_j(x) − z_j(x without M_p). ``masks`` is ``(B, P, H, W)``
    and ``class_ids`` is ``(B, K)``. ``remove(x, mask)`` replaces ROAD removal.
    """
    if masks.ndim != 4 or class_ids.shape[0] != masks.shape[0]:
        raise ValueError("masks must be (batch, players, H, W) and class ids (batch, classes)")
    class_ids = class_ids.long()
    if remove is None:
        def remove(images, mask):
            return _remove(images, mask, iters, noise, seed, offset)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = model(x)
            drops = []
            for p in range(masks.shape[1]):
                removed = model(remove(x, masks[:, p]))
                drops.append((base - removed).gather(1, class_ids))
            return torch.stack(drops, dim=1)
    finally:
        _restore(model, was_training)


def deletion_matrix(
    model: nn.Module,
    x: torch.Tensor,
    masks: torch.Tensor,
    class_ids: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """``(B, K, K)`` drop in class i's logit when mask j is removed.

    Entry ``(i, j)`` is z_i(x) − z_i(x without M_j). ``masks`` is ``(B, K, H, W)``
    and ``class_ids`` is ``(B, K)``. The transpose of ``drop_matrix`` with P = K.
    """
    if masks.ndim != 4 or class_ids.shape[:2] != masks.shape[:2]:
        raise ValueError("masks and class ids must be (batch, hypotheses, ...)")
    return drop_matrix(model, x, masks, class_ids, iters, noise, seed, offset).transpose(-1, -2)


def sufficiency_contrast(
    model: nn.Module,
    x: torch.Tensor,
    mask_k: torch.Tensor,
    mask_l: torch.Tensor,
    class_k: torch.Tensor,
    class_l: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """SC = m(x keep M_k) - m(x keep M_l). Kept pixels stay; the rest are ROAD-imputed."""
    class_k = class_k.long()
    class_l = class_l.long()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            keep_k = model(_remove(x, 1.0 - mask_k, iters, noise, seed, offset))
            keep_l = model(_remove(x, 1.0 - mask_l, iters, noise, seed + 1, offset))
            return _margin(keep_k, class_k, class_l) - _margin(keep_l, class_k, class_l)
    finally:
        _restore(model, was_training)


def shared_validity(
    model: nn.Module,
    x: torch.Tensor,
    mask_s: torch.Tensor,
    class_ids: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """Drop in ``s_H`` minus the mean absolute change in ``c_j``, under ROAD removal.

    Undefined when H covers every class. ``class_ids`` is ``(B, K)``.
    """
    from cdea.payoffs import shared_defined, shared_log_odds, unique_log_odds

    class_ids = class_ids.long()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full = model(x)
            if not shared_defined(full, class_ids):
                return torch.full((x.shape[0],), float("nan"), device=x.device)
            deleted = model(_remove(x, mask_s, iters, noise, seed, offset))
            shared_drop = shared_log_odds(full, class_ids) - shared_log_odds(deleted, class_ids)
            reorder = (unique_log_odds(full, class_ids) - unique_log_odds(deleted, class_ids)).abs().mean(dim=-1)
            return shared_drop - reorder
    finally:
        _restore(model, was_training)


def deletion_specificity(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean diagonal minus mean off-diagonal, and |diagonal| / |off-diagonal|.

    ``matrix`` is ``(B, K, K)``. Both returns are ``(B,)``.
    """
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2] or matrix.shape[-1] < 2:
        raise ValueError("deletion specificity needs a square matrix with at least two classes")
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    off_mean = (matrix.sum(dim=(-1, -2)) - diagonal.sum(dim=-1)) / (
        matrix.shape[-1] * (matrix.shape[-1] - 1)
    )
    diag_mean = diagonal.mean(dim=-1)
    ratio = diagonal.abs().mean(dim=-1) / off_mean.abs().clamp_min(1e-8)
    return diag_mean - off_mean, ratio


def _class_probability(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(model(x), dim=-1).gather(1, y.long()[:, None]).squeeze(1)


def _class_logit(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return model(x).gather(1, y.long()[:, None]).squeeze(1)


def environment_disagreement(
    model: nn.Module,
    x_id: torch.Tensor,
    x_ood: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """|p_y(x_id) − p_y(x_ood)|."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return (_class_probability(model, x_id, y) - _class_probability(model, x_ood, y)).abs()
    finally:
        _restore(model, was_training)


def environment_logit_gap(
    model: nn.Module,
    x_id: torch.Tensor,
    x_ood: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """|z_y(x_id) − z_y(x_ood)|, the unsaturated companion of ``environment_disagreement``.

    The probability gap saturates when the model is confident in both
    environments (p ≈ 1 twice even as the logits differ by a lot), which can
    hide real shortcut reliance. The logit gap stays sensitive there.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return (_class_logit(model, x_id, y) - _class_logit(model, x_ood, y)).abs()
    finally:
        _restore(model, was_training)


def disagreement_reduction(
    model: nn.Module,
    x_id: torch.Tensor,
    x_ood: Sequence[torch.Tensor],
    y: torch.Tensor,
    shortcut: torch.Tensor,
    random_mask: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """ΔD. The shortcut term minus the same term for a random mask of equal area.

    D is the mean of |p_y(id) − p_y(ood)| across the other environments. The
    same mask is removed in every environment.
    """
    if len(x_ood) < 1:
        raise ValueError("disagreement needs an out-of-distribution view")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            def gap(xid: torch.Tensor, oods: Sequence[torch.Tensor]) -> torch.Tensor:
                terms = [
                    (_class_probability(model, xid, y) - _class_probability(model, xo, y)).abs()
                    for xo in oods
                ]
                return torch.stack(terms, dim=0).mean(dim=0)

            def apply(mask: torch.Tensor) -> torch.Tensor:
                return gap(
                    _remove(x_id, mask, iters, noise, seed, offset),
                    [_remove(xo, mask, iters, noise, seed, offset) for xo in x_ood],
                )

            full = gap(x_id, x_ood)
            return (full - apply(shortcut)) - (full - apply(random_mask))
    finally:
        _restore(model, was_training)


def logit_disagreement_reduction(
    model: nn.Module,
    x_id: torch.Tensor,
    x_ood: Sequence[torch.Tensor],
    y: torch.Tensor,
    shortcut: torch.Tensor,
    random_mask: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """ΔD in logit space. Same construction as ``disagreement_reduction`` with
    D = mean |z_y(id) − z_y(ood)| across the other environments.

    Report beside the probability ΔD: when the model is confident everywhere,
    the probability gap is near zero whatever the masks do, while the logit
    gap still separates a shortcut mask from a random one.
    """
    if len(x_ood) < 1:
        raise ValueError("disagreement needs an out-of-distribution view")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            def gap(xid: torch.Tensor, oods: Sequence[torch.Tensor]) -> torch.Tensor:
                terms = [
                    (_class_logit(model, xid, y) - _class_logit(model, xo, y)).abs()
                    for xo in oods
                ]
                return torch.stack(terms, dim=0).mean(dim=0)

            def apply(mask: torch.Tensor) -> torch.Tensor:
                return gap(
                    _remove(x_id, mask, iters, noise, seed, offset),
                    [_remove(xo, mask, iters, noise, seed, offset) for xo in x_ood],
                )

            full = gap(x_id, x_ood)
            return (full - apply(shortcut)) - (full - apply(random_mask))
    finally:
        _restore(model, was_training)


def two_patch_recovery(
    mask_k: torch.Tensor,
    mask_l: torch.Tensor,
    patch_a: torch.Tensor,
    patch_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Share of M_k on patch A, and of M_l on patch B."""
    return mass_in(mask_k, patch_a), mass_in(mask_l, patch_b)


def _log_odds_batch(model: nn.Module, xs: Sequence[torch.Tensor], y: torch.Tensor) -> list[torch.Tensor]:
    from cdea.shift import log_odds

    return [log_odds(model(view), y) for view in xs]


def log_odds_disagreement(
    model: nn.Module,
    xs: Sequence[torch.Tensor],
    y: torch.Tensor,
) -> torch.Tensor:
    """``mean_{e != id} |m(x_id) - m(x_e)|``."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            margins = _log_odds_batch(model, xs, y)
            return torch.stack([(margins[0] - other).abs() for other in margins[1:]], dim=0).mean(0)
    finally:
        _restore(model, was_training)


def probability_disagreement(
    model: nn.Module,
    xs: Sequence[torch.Tensor],
    y: torch.Tensor,
) -> torch.Tensor:
    """Same disagreement in the probability of ``y``. Companion of the log-odds gap."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probs = [_class_probability(model, view, y) for view in xs]
            return torch.stack([(probs[0] - other).abs() for other in probs[1:]], dim=0).mean(0)
    finally:
        _restore(model, was_training)


def _road(model, xs, y, mask, iters, noise, seed, measure, offset=0):
    edited = [_remove(view, mask, iters, noise, seed + index, offset) for index, view in enumerate(xs)]
    return measure(model, edited, y)


def delta_d(
    model: nn.Module,
    xs: Sequence[torch.Tensor],
    y: torch.Tensor,
    shortcut: torch.Tensor,
    random_mask: torch.Tensor,
    iters: Optional[int],
    noise: float,
    seed: int,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop in log-odds disagreement minus the same drop for a random mask, and the probability companion."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            before = log_odds_disagreement(model, xs, y)
            before_p = probability_disagreement(model, xs, y)

            def odds(m, views, labels):
                return log_odds_disagreement(m, views, labels)

            def prob(m, views, labels):
                return probability_disagreement(m, views, labels)

            drop = before - _road(model, xs, y, shortcut, iters, noise, seed, odds, offset)
            drop_random = before - _road(model, xs, y, random_mask, iters, noise, seed + 1, odds, offset)
            drop_p = before_p - _road(model, xs, y, shortcut, iters, noise, seed, prob, offset)
            drop_p_random = before_p - _road(model, xs, y, random_mask, iters, noise, seed + 1, prob, offset)
            return drop - drop_random, drop_p - drop_p_random
    finally:
        _restore(model, was_training)


def robust_necessity(
    model: nn.Module,
    xs: Sequence[torch.Tensor],
    y: torch.Tensor,
    robust: torch.Tensor,
    random_mask: torch.Tensor,
    iters: Optional[int],
    noise: float,
    seed: int,
    offset: int = 0,
) -> torch.Tensor:
    """Min over environments of the log-odds drop under ROAD deletion, minus the random mask."""
    from cdea.shift import log_odds

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            def worst(mask: torch.Tensor) -> torch.Tensor:
                drops = []
                for index, view in enumerate(xs):
                    full = log_odds(model(view), y)
                    edited = log_odds(model(_remove(view, mask, iters, noise, seed + index, offset)), y)
                    drops.append(full - edited)
                return torch.stack(drops, dim=0).min(dim=0).values

            return worst(robust) - worst(random_mask)
    finally:
        _restore(model, was_training)


def transplant_closure(
    model: nn.Module,
    xs: Sequence[torch.Tensor],
    y: torch.Tensor,
    unit_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Shortcut payoff of the binary shortcut mask. Descriptive."""
    from cdea.shift import transplant_payoff

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return transplant_payoff(model, list(xs), y, unit_mask, grid_h, grid_w)
    finally:
        _restore(model, was_training)


def shortcut_patch_contrast(
    shortcut: torch.Tensor,
    patch_a: torch.Tensor,
    patch_b: torch.Tensor,
) -> torch.Tensor:
    """Mass of the shortcut mask on the class-tied patch minus its mass on the checker."""
    return mass_in(shortcut, patch_a) - mass_in(shortcut, patch_b)


# ---------------------------------------------------------------------------
# Decomposition game (FORMULATION_DECOMPOSITION.md section 5). Players are
# ordered U_k, U_l, S; classes k, l. Logit units.
# ---------------------------------------------------------------------------

def null_pixel_masks(
    grid_h: int,
    grid_w: int,
    area: float,
    height: int,
    width: int,
    n: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    """``(n, H, W)``: CDEA's seeded random cell scores (``cdea.null``) through the shared adapter at area a."""
    from cdea.null import null_unit_scores
    from evaluation.masks import to_budget_mask

    scores = null_unit_scores(grid_h * grid_w, n, seed)
    return to_budget_mask(scores, float(area), int(seed), grid_h=grid_h, grid_w=grid_w, height=height, width=width)


def null_drops_scored(model: nn.Module, x: torch.Tensor, class_ids: torch.Tensor, null_masks: torch.Tensor,
                      remove) -> torch.Tensor:
    """``(B, K)`` mean drop of each class over the null masks, with the scorer's removal."""
    masks = null_masks.to(device=x.device, dtype=x.dtype)
    total = None
    for mask in masks:
        drops = drop_matrix(model, x, mask.expand(x.shape[0], 1, -1, -1), class_ids, remove=remove)[:, 0]
        total = drops if total is None else total + drops
    return total / float(masks.shape[0])


def null_drops_road(model: nn.Module, x: torch.Tensor, class_ids: torch.Tensor, area: float,
                    grid_h: int, grid_w: int, n: int = 8, null_seed: int = 0, iters: Optional[int] = None,
                    noise: float = 0.01, seed: int = 0, offset: int = 0) -> torch.Tensor:
    """``(B, K)`` ROAD null: the null masks of section 3.2 removed with the method masks' ROAD noise."""
    masks = null_pixel_masks(grid_h, grid_w, area, x.shape[-2], x.shape[-1], n, null_seed)

    def remove(images, mask):
        return _remove(images, mask, iters, noise, seed, offset)

    return null_drops_scored(model, x, class_ids, masks, remove)


def decomposition_scores(drops: torch.Tensor, null: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image scores from the raw ``(B, 3, 2)`` drops and the ``(B, 2)`` null of the same operator.

    SPEC (raw) is the unique specificity, and SPEC_k + SPEC_l is CD. PUR, SV,
    BAL and SEP use the excess drop d − ν. Exact absolute values and minima.
    A NaN player (a method with no shared map) gives NaN scores for that player.
    """
    if drops.ndim != 3 or drops.shape[1:] != (3, 2):
        raise ValueError("drops must be (batch, 3 players, 2 classes)")
    e = drops - null.unsqueeze(1)
    spec_k = drops[:, 0, 0] - drops[:, 0, 1]
    spec_l = drops[:, 1, 1] - drops[:, 1, 0]
    sv = torch.minimum(e[:, 2, 0], e[:, 2, 1])
    return {
        "spec_k": spec_k,
        "spec_l": spec_l,
        "cd": spec_k + spec_l,
        "pur_k": e[:, 0, 0] - e[:, 0, 1].abs(),
        "pur_l": e[:, 1, 1] - e[:, 1, 0].abs(),
        "sv": sv,
        "bal": (e[:, 2, 0] - e[:, 2, 1]).abs(),
        "sep": sv - torch.maximum(e[:, 0].min(dim=-1).values, e[:, 1].min(dim=-1).values),
    }


def mask_patch_confusion(masks: torch.Tensor, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Precision ``(B, 3, 4)`` and recall ``(B, 3, 3)`` of the masks on the planted patches.

    ``masks`` are U_k, U_l, S and ``patches`` are A_k, A_l, C, all ``(B, 3, H, W)``.
    Precision ``[p, t]`` is the share of mask p's mass on patch t, with t = 3 the
    rest of the image. Recall ``[p, t]`` is the share of patch t that mask p covers.
    """
    if masks.shape != patches.shape or masks.shape[1] != 3:
        raise ValueError("masks and patches must both be (batch, 3, H, W)")
    m = masks.clamp_min(0.0)
    t = patches.to(dtype=m.dtype)
    inside = torch.einsum("bphw,bthw->bpt", m, t)
    total = m.sum(dim=(-1, -2))
    rest = (total - inside.sum(dim=-1)).clamp_min(0.0)
    mass = torch.cat([inside, rest.unsqueeze(-1)], dim=-1)
    precision = torch.where(total.unsqueeze(-1) > 0, mass / total.clamp_min(1e-8).unsqueeze(-1),
                            torch.zeros_like(mass))
    area = t.sum(dim=(-1, -2))
    recall = torch.where(area.unsqueeze(1) > 0, inside / area.clamp_min(1e-8).unsqueeze(1),
                         torch.zeros_like(inside))
    return precision, recall


# ---------------------------------------------------------------------------
# Shift 2x2 role matrix and null-subtracted scores (SHIFT_EVAL_PLAN.md).
# Players are ordered R (robust), S (shortcut); axes are rob (worst-env drop)
# and gap (cross-env disagreement drop).
# ---------------------------------------------------------------------------

def shift_drop_matrix(
    model: nn.Module,
    xs: Sequence[torch.Tensor],
    y: torch.Tensor,
    masks: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
    remove=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``((B, P, 2), (B, P))`` raw shift drops ``(rob_drop, gap_drop)`` and probability gap drop.

    ``masks`` is ``(B, P, H, W)``. For each mask p:
      - ``rob_drop`` is ``min_{e in E} [m(x_e) - m(x_e without M_p)]``;
      - ``gap_drop`` is ``D(x) - D(x without M_p)`` with ``D = mean_{e != id} |m(x_id) - m(x_e)|``;
      - ``prob_drop`` is the companion reduction in ``mean_{e != id} |p_y(x_id) - p_y(x_e)|``.
    """
    from cdea.shift import log_odds

    if len(xs) < 2:
        raise ValueError("shift_drop_matrix needs the id view and at least one other environment")
    if masks.ndim != 4 or masks.shape[0] != xs[0].shape[0]:
        raise ValueError("masks must be (batch, players, H, W)")
    y = y.long()
    if remove is None:
        def remove(view, mask, env_index=0):
            return _remove(view, mask, iters, noise, seed + env_index, offset)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full_logits = [model(view) for view in xs]
            m_full = [log_odds(logits, y) for logits in full_logits]
            p_full = [torch.softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1) for logits in full_logits]
            gap_before = torch.stack([(m_full[0] - other).abs() for other in m_full[1:]], dim=0).mean(dim=0)
            prob_before = torch.stack([(p_full[0] - other).abs() for other in p_full[1:]], dim=0).mean(dim=0)
            rows, prob_rows = [], []
            for p in range(masks.shape[1]):
                mask = masks[:, p]
                ed_logits = [model(remove(view, mask, index)) for index, view in enumerate(xs)]
                m_ed = [log_odds(logits, y) for logits in ed_logits]
                p_ed = [torch.softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1) for logits in ed_logits]
                rob_drop = torch.stack([m_full[i] - m_ed[i] for i in range(len(xs))], dim=0).min(dim=0).values
                gap_after = torch.stack([(m_ed[0] - other).abs() for other in m_ed[1:]], dim=0).mean(dim=0)
                prob_after = torch.stack([(p_ed[0] - other).abs() for other in p_ed[1:]], dim=0).mean(dim=0)
                rows.append(torch.stack([rob_drop, gap_before - gap_after], dim=-1))
                prob_rows.append(prob_before - prob_after)
            return torch.stack(rows, dim=1), torch.stack(prob_rows, dim=1)
    finally:
        _restore(model, was_training)


def null_shift_drops(
    model: nn.Module,
    xs: Sequence[torch.Tensor],
    y: torch.Tensor,
    null_masks: torch.Tensor,
    iters: Optional[int] = None,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
    remove=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``((B, 2), (B,))`` mean ``(null_rob, null_gap)`` and ``null_prob`` over the N seeded null masks."""
    masks = null_masks.to(device=xs[0].device, dtype=xs[0].dtype)
    batch = xs[0].shape[0]
    expanded = masks.unsqueeze(0).expand(batch, -1, -1, -1)
    drops, prob_drops = shift_drop_matrix(
        model, xs, y, expanded, iters=iters, noise=noise, seed=seed, offset=offset, remove=remove,
    )
    return drops.mean(dim=1), prob_drops.mean(dim=1)


def shift_scores(
    drops: torch.Tensor,
    null: torch.Tensor,
    prob_drops: torch.Tensor | None = None,
    null_prob: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-image shift scores from ``(B, 2, 2)`` raw drops ``[R, S] x [rob, gap]`` and ``(B, 2)`` null.

    ``rn`` is the excess worst-environment robust drop of M_R. ``delta_d`` is
    the excess cross-environment gap drop of M_S. ``pur_r`` penalises M_R for
    moving the cross-environment gap; ``pur_s`` penalises M_S for damaging
    worst-environment robust log-odds; ``role_sep`` is the 2x2 diagonal minus
    off-diagonal separation.
    """
    if drops.ndim != 3 or drops.shape[1:] != (2, 2):
        raise ValueError("drops must be (batch, 2 players, 2 axes)")
    e = drops - null.unsqueeze(1)
    out = {
        "rn": e[:, 0, 0],
        "delta_d": e[:, 1, 1],
        "pur_r": e[:, 0, 0] - e[:, 0, 1].abs(),
        "pur_s": e[:, 1, 1] - e[:, 1, 0].clamp_min(0.0),
        "role_sep": (e[:, 0, 0] - e[:, 1, 0]) + (e[:, 1, 1] - e[:, 0, 1]),
    }
    if prob_drops is not None and null_prob is not None:
        out["delta_d_prob"] = prob_drops[:, 1] - null_prob
    return out


# ---------------------------------------------------------------------------
# Decomposition version 2 (FORMULATION_DECOMPOSITION.md section 6). Roles are
# ordered U_k, U_l, S, empty; classes k, l and m (rank 2). A role's mask is its
# cells, whole. Drops are raw logit drops; the null and its 95th percentile are
# measured at each role's own size, from random cells of the whole grid.
# ---------------------------------------------------------------------------

SCORER_NULL = 32      # random n-cell masks per size (section 6.1)
SPLIT_SEED = 0        # the seeded halves of section 6.4
REMOVAL_BATCH = 64    # (image, mask) pairs per forward pass; scores do not depend on it
ROLE_NAMES = ("uk", "ul", "s", "none")
VERDICTS = ("", "mixture", "redundant", "shared", "unclear")


def noise_rows(x: torch.Tensor, seed: int, offset: int = 0) -> torch.Tensor:
    """``(B, C, H, W)``: the standard normal rows ``_remove`` draws for these images."""
    from evaluation.masks import advance_generator

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    advance_generator(generator, int(offset), tuple(x.shape[1:]), normal=True)
    return torch.randn(x.shape, generator=generator, device="cpu")


def removal_drops(model: nn.Module, x: torch.Tensor, cells: torch.Tensor, ids: torch.Tensor, operator: str,
                  grid: tuple[int, int], seed: int = 0, offset: int = 0, batch: int = REMOVAL_BATCH) -> torch.Tensor:
    """``(B, M, J)`` raw drop of each class when each whole-cell mask is removed.

    ``cells`` is ``(B, M, R)`` or ``(M, R)`` (the same masks for every image).
    ROAD gives every mask of an image that image's own noise rows, so a mask
    scores as ``_remove`` with the same ``seed`` and ``offset`` would score it.
    An empty mask removes nothing and drops 0 without a forward pass.
    """
    from core.grid import deletion_baseline, upsample_units
    from evaluation.removal import ROAD, noisy_linear_impute

    if cells.ndim == 2:
        cells = cells.unsqueeze(0).expand(x.shape[0], -1, -1)
    images, masks = x.shape[0], cells.shape[1]
    ids = ids.long()
    out = torch.zeros(images, masks, ids.shape[1], device=x.device, dtype=x.dtype)
    if masks == 0:
        return out
    height, width = x.shape[-2:]
    draws = noise_rows(x, seed, offset) if operator == "road" else None
    occupied = cells.reshape(images, masks, -1).sum(dim=-1) > 0
    pairs = [(b, m) for b in range(images) for m in range(masks) if bool(occupied[b, m])]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full = model(x).gather(1, ids)
            for start in range(0, len(pairs), int(batch)):
                part = pairs[start:start + int(batch)]
                rows = torch.tensor([b for b, _ in part], device=x.device)
                unit = torch.stack([cells[b, m] for b, m in part]).to(device=x.device, dtype=x.dtype)
                pixel = upsample_units(unit, grid[0], grid[1], height, width)
                source = x[rows]
                if operator == "road":
                    draw = draws[rows.cpu()]
                    edited = noisy_linear_impute(source, 1.0 - pixel, ROAD["iters"], ROAD["noise"], None, draw=draw)
                elif operator == "blur":
                    edited = source * (1 - pixel) + deletion_baseline(source, grid[0], grid[1]) * pixel
                else:
                    raise ValueError(f"unknown removal operator {operator!r}")
                drop = full[rows] - model(edited).gather(1, ids[rows])
                for (b, m), value in zip(part, drop):
                    out[b, m] = value
    finally:
        _restore(model, was_training)
    return out


def scored_size_null(model: nn.Module, x: torch.Tensor, ids: torch.Tensor, grid: tuple[int, int], sizes,
                     operator: str, seed: int = 0, offset: int = 0, n: int = SCORER_NULL,
                     null_seed: int = 0) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """``{s: (mean, q95)}``, each ``(B, J)``: the drop over ``n`` random s-cell masks of the whole grid.

    The masks are ``cdea.null.size_masks``: the same for every image and nested
    in s. ``sizes`` is one set for every image, or one set per image; an image
    is measured only at its own sizes and is NaN at the others.
    """
    from cdea.null import size_masks

    batch = x.shape[0]
    per_image = [set(sizes)] * batch if isinstance(sizes, (set, frozenset)) else [set(s) for s in sizes]
    per_image = [{int(s) for s in wanted if int(s) > 0} for wanted in per_image]
    union = sorted(set().union(*per_image))
    if not union:
        return {}
    shared = size_masks(grid[0] * grid[1], union, n, null_seed)
    masks = shared.unsqueeze(0).repeat(batch, 1, 1, 1)
    for b, wanted in enumerate(per_image):
        for i, s in enumerate(union):
            if s not in wanted:
                masks[b, i] = 0.0
    drops = removal_drops(model, x, masks.reshape(batch, len(union) * n, -1), ids, operator, grid, seed, offset)
    drops = drops.view(batch, len(union), n, -1)
    mean = drops.mean(dim=2)
    q95 = torch.quantile(drops, 0.95, dim=2)
    for b, wanted in enumerate(per_image):
        for i, s in enumerate(union):
            if s not in wanted:
                mean[b, i] = float("nan")
                q95[b, i] = float("nan")
    return {s: (mean[:, i], q95[:, i]) for i, s in enumerate(union)}


def split_halves(cells: torch.Tensor) -> torch.Tensor:
    """``(2, R)`` bool: the cells of one role split into two seeded halves, the first the larger.

    The permutation depends only on the role's size, so an image's halves do not depend on the batch.
    """
    index = torch.nonzero(cells, as_tuple=False).flatten()
    generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
    order = index[torch.randperm(index.numel(), generator=generator).to(index.device)]
    first = (index.numel() + 1) // 2
    halves = torch.zeros(2, cells.shape[-1], dtype=torch.bool, device=cells.device)
    halves[0, order[:first]] = True
    halves[1, order[first:]] = True
    return halves


def role_confirmed(count: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` bool from ``(..., 4, 2)`` counts on k and l: section 6.4, ``verify_roles``' rule.

    U_k: counts for k and not l. U_l: the mirror. S: both. Empty: neither.
    """
    k, l = count[..., 0], count[..., 1]
    return torch.stack([k[..., 0] & ~l[..., 0], l[..., 1] & ~k[..., 1], k[..., 2] & l[..., 2],
                        ~k[..., 3] & ~l[..., 3]], dim=-1)


def half_verdict(count: torch.Tensor) -> int:
    """Index into ``VERDICTS`` from the ``(2, 2)`` counts of two halves on k and l.

    A half that counts for one class only: mixture. Both halves count for
    neither: redundancy. Both count for both: shared. Otherwise unclear.
    """
    one = [bool(c[0]) != bool(c[1]) for c in count]
    if any(one):
        return 1
    if not count.any():
        return 2
    if count.all():
        return 3
    return 4


def resolution_shares(roles: torch.Tensor, confirmed: torch.Tensor, overlap: torch.Tensor,
                      omega: torch.Tensor) -> dict[str, torch.Tensor]:
    """Section 6.5. The share of the overlap inside Omega that ends in each confirmed verdict, and VR.

    A cell in several confirmed roles (only ablation DA1 has overlapping roles)
    counts once, in the first of S, U_k, U_l, empty. NaN when Omega holds no overlap.
    """
    total = (overlap * omega).sum(dim=-1)
    remaining = omega.clone()
    share = {}
    for role in (2, 0, 1, 3):
        take = roles[:, role] & confirmed[:, role].unsqueeze(-1) & remaining
        share[role] = (overlap * take).sum(dim=-1)
        remaining = remaining & ~take
    nan = torch.full_like(total, float("nan"))
    shared = torch.where(total > 0, share[2] / total.clamp_min(1e-12), nan)
    unique = torch.where(total > 0, (share[0] + share[1]) / total.clamp_min(1e-12), nan)
    neither = torch.where(total > 0, share[3] / total.clamp_min(1e-12), nan)
    unconfirmed = 1.0 - shared - unique - neither
    return {"res_shared": shared, "res_unique": unique, "res_neither": neither,
            "res_unconfirmed": unconfirmed, "vr": 1.0 - unconfirmed}


def partition_scores(drops: torch.Tensor, null: torch.Tensor, q95: torch.Tensor, sizes: torch.Tensor,
                     cell_excess: torch.Tensor, roles: torch.Tensor, overlap: torch.Tensor,
                     omega: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image version-2 scores (sections 6.3 to 6.6).

    ``drops``, ``null`` and ``q95`` are ``(B, 4, 3)``: each role's raw drop of
    k, l and m, and the scorer's null and its 95th percentile at the role's
    size. ``sizes`` is ``(B, 4)``. ``cell_excess`` is ``(B, 3, R)``: each cell
    deleted alone, minus the null of one cell. A score on an empty role is NaN.
    """
    e = drops - null
    empty = sizes <= 0
    nan = float("nan")

    def on(role: int, value: torch.Tensor) -> torch.Tensor:
        return torch.where(empty[:, role], torch.full_like(value, nan), value)

    g = torch.nan_to_num(cell_excess, nan=0.0)
    member = roles.to(g.dtype)
    pur_cell_k = (member[:, 0] * (g[:, 0] - g[:, 1].abs())).sum(dim=-1)
    pur_cell_l = (member[:, 1] * (g[:, 1] - g[:, 0].abs())).sum(dim=-1)
    sv_cell = (member[:, 2] * torch.minimum(g[:, 0], g[:, 1])).sum(dim=-1)
    sv = torch.minimum(e[:, 2, 0], e[:, 2, 1])
    count = drops > q95
    confirmed = role_confirmed(count[..., :2]) & ~empty
    out = {
        "pur_k": on(0, e[:, 0, 0] - e[:, 0, 1].abs()),
        "pur_l": on(1, e[:, 1, 1] - e[:, 1, 0].abs()),
        "sv": on(2, sv),
        "bal": on(2, (e[:, 2, 0] - e[:, 2, 1]).abs()),
        "pur_cell_k": on(0, pur_cell_k),
        "pur_cell_l": on(1, pur_cell_l),
        "sv_cell": on(2, sv_cell),
        "mix": on(2, sv - sv_cell),
        "gen_s": on(2, e[:, 2, 2]),
        "psv": on(2, sv - e[:, 2, 2].clamp_min(0)),
        "dm_uk": on(0, e[:, 0, 2]),
        "dm_ul": on(1, e[:, 1, 2]),
    }
    for role, name in enumerate(ROLE_NAMES):
        out[f"size_{name}"] = sizes[:, role].to(drops.dtype)
        out[f"confirmed_{name}"] = torch.where(empty[:, role], torch.full_like(sizes[:, role], nan, dtype=drops.dtype),
                                               confirmed[:, role].to(drops.dtype))
        for j, cls in enumerate(("k", "l", "m")):
            out[f"d_{name}_{cls}"] = on(role, drops[:, role, j])
            out[f"nu_{name}_{cls}"] = on(role, null[:, role, j])
            out[f"q_{name}_{cls}"] = on(role, q95[:, role, j])
    out.update(resolution_shares(roles, confirmed, overlap, omega))
    return out


def planted_role_scores(roles: torch.Tensor, patch_cells: torch.Tensor, omega: torch.Tensor) -> dict[str, torch.Tensor]:
    """Section 6.7. ``patch_cells`` is ``(B, 3, R)`` bool for A_k, A_l and C, whose roles are U_k, U_l and S.

    Omega coverage, RA over all patch cells (cells outside Omega count as
    wrong), RA in Omega, and the 3 x 5 confusion (patch x U_k, U_l, S, empty,
    outside Omega) as shares of each patch's cells.
    """
    patch = patch_cells.to(roles.device)
    size = patch.sum(dim=-1).to(torch.float32)
    inside = (patch & omega.unsqueeze(1)).sum(dim=-1).to(torch.float32)
    nan = torch.full_like(size, float("nan"))
    correct = torch.stack([(patch[:, t] & roles[:, t]).sum(dim=-1) for t in range(3)], dim=1).to(torch.float32)
    ra_all = torch.where(size > 0, correct / size.clamp_min(1), nan)
    ra_in = torch.where(inside > 0, correct / inside.clamp_min(1), nan)
    out = {
        "omega_coverage": inside.sum(dim=-1) / size.sum(dim=-1).clamp_min(1),
        "ra_all": ra_all.nanmean(dim=-1),
        "ra_in": ra_in.nanmean(dim=-1),
    }
    columns = [roles[:, r] for r in range(4)] + [~omega]
    for t, target in enumerate(("ak", "al", "c")):
        out[f"ra_all_{target}"] = ra_all[:, t]
        out[f"ra_in_{target}"] = ra_in[:, t]
        for c, column in enumerate(ROLE_NAMES + ("outside",)):
            hits = (patch[:, t] & columns[c]).sum(dim=-1).to(torch.float32)
            out[f"conf_{target}_{column}"] = torch.where(size[:, t] > 0, hits / size[:, t].clamp_min(1), nan[:, t])
    return out
