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


def _remove(x: torch.Tensor, mask: torch.Tensor, iters: int, noise: float, seed: int) -> torch.Tensor:
    """Impute pixels where ``mask`` is 1. ``mask`` is ``(B, H, W)`` or ``(B, 1, H, W)``."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
        raise ValueError("removal mask does not match the image")
    keep = (1.0 - mask).clamp(0.0, 1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
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
            without_l = model(_remove(x, mask_l, iters, noise, seed))
            without_k = model(_remove(x, mask_k, iters, noise, seed))
            return _margin(without_l, class_k, class_l) - _margin(without_k, class_k, class_l)
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
) -> torch.Tensor:
    """``(B, K, K)`` drop in class i's logit when mask j is removed.

    Entry ``(i, j)`` is z_i(x) − z_i(x without M_j). ``masks`` is ``(B, K, H, W)``
    and ``class_ids`` is ``(B, K)``.
    """
    if masks.ndim != 4 or class_ids.shape[:2] != masks.shape[:2]:
        raise ValueError("masks and class ids must be (batch, hypotheses, ...)")
    class_ids = class_ids.long()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = model(x)
            drops = []
            for j in range(masks.shape[1]):
                removed = model(_remove(x, masks[:, j], iters, noise, seed))
                drop = base - removed
                drops.append(drop.gather(1, class_ids))
            # drops[j] is (B, K): drop in every class logit when mask j is removed.
            # Stack on the last axis so index j is the removed mask.
            return torch.stack(drops, dim=-1)
    finally:
        _restore(model, was_training)


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
) -> torch.Tensor:
    """SC = m(x keep M_k) - m(x keep M_l). Kept pixels stay; the rest are ROAD-imputed."""
    class_k = class_k.long()
    class_l = class_l.long()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            keep_k = model(_remove(x, 1.0 - mask_k, iters, noise, seed))
            keep_l = model(_remove(x, 1.0 - mask_l, iters, noise, seed + 1))
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
            deleted = model(_remove(x, mask_s, iters, noise, seed))
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
                    _remove(x_id, mask, iters, noise, seed),
                    [_remove(xo, mask, iters, noise, seed) for xo in x_ood],
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
                    _remove(x_id, mask, iters, noise, seed),
                    [_remove(xo, mask, iters, noise, seed) for xo in x_ood],
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


def _road(model, xs, y, mask, iters, noise, seed, measure):
    edited = [_remove(view, mask, iters, noise, seed + index) for index, view in enumerate(xs)]
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

            drop = before - _road(model, xs, y, shortcut, iters, noise, seed, odds)
            drop_random = before - _road(model, xs, y, random_mask, iters, noise, seed + 1, odds)
            drop_p = before_p - _road(model, xs, y, shortcut, iters, noise, seed, prob)
            drop_p_random = before_p - _road(model, xs, y, random_mask, iters, noise, seed + 1, prob)
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
                    edited = log_odds(model(_remove(view, mask, iters, noise, seed + index)), y)
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
