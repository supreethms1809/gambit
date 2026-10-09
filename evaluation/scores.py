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

    ``offset`` is the index of this chunk in the full sample. The noise stream
    is the same one a single call on the whole sample would have drawn.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
        raise ValueError("removal mask does not match the image")
    keep = (1.0 - mask).clamp(0.0, 1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if offset:
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
