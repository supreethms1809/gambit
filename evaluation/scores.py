"""Primary scores. ROAD removal is the operator. Blur keep is not used here.

CD@a is the contrastive deletion score. The K×K matrix asks whether each
mask is specific to its class. ΔD asks whether the shortcut mask, rather than
a random mask of the same area, reduces cross-environment disagreement.
Two-patch recovery is the share of each class mask that lands on its patch.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from evaluation.masks import mass_in
from evaluation.removal import noisy_linear_impute


def _restore(model: nn.Module, was_training: bool) -> None:
    model.train(was_training)


def _remove(x: torch.Tensor, mask: torch.Tensor, iters: int, noise: float, seed: int, offset: int = 0) -> torch.Tensor:
    """Impute pixels where ``mask`` is 1. ``mask`` is ``(B, H, W)`` or ``(B, 1, H, W)``.

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
    iters: int = 24,
    noise: float = 0.01,
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """CD = m(x without M_l) − m(x without M_k), with m = z_k − z_l.

    A positive score means removing the foil's evidence leaves k ahead of l,
    and removing k's evidence does not. Defaults match the ROAD settings in
    ``scripts/eval_faithfulness.py``.
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
    iters: int = 24,
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
    iters: int = 24,
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
    iters: int = 24,
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
    iters: int = 24,
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
    iters: int = 24,
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
    iters: int = 24,
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
    iters: int,
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
    iters: int,
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
                    grid_h: int, grid_w: int, n: int = 8, null_seed: int = 0, iters: int = 24,
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
    iters: int = 24,
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
    iters: int = 24,
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
