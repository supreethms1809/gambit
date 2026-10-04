"""Extremal Perturbations, per class and on the logit margin.

The optimisation is TorchRay's ``extremal_perturbation``. This module chooses
the class, the area, and the reward. It does not change the mask update.
TorchRay turns ``requires_grad`` off on the classifier and leaves it off;
the wrapper restores the flags it found.

The function takes one image. A batch is a loop of those calls. The returned
mask is the native mask at ``area``, which is the method's own conversion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from core.types import HypothesisSet

from .hypotheses import foil_pair

_VENDOR = Path(__file__).resolve().parents[1] / "third_party" / "torchray"


def _import_torchray():
    """Import TorchRay, then drop its directory from ``sys.path``.

    That directory contains an ``examples`` package. Leaving it on the path
    hides this repository's ``examples`` modules.
    """
    added = str(_VENDOR) not in sys.path
    if added:
        sys.path.append(str(_VENDOR))
    try:
        from torchray.attribution.extremal_perturbation import (
            DELETE_VARIANT,
            DUAL_VARIANT,
            PRESERVE_VARIANT,
            Perturbation,
            extremal_perturbation,
            simple_reward,
        )
    finally:
        if added and str(_VENDOR) in sys.path:
            sys.path.remove(str(_VENDOR))
    return (
        extremal_perturbation,
        simple_reward,
        PRESERVE_VARIANT,
        DELETE_VARIANT,
        DUAL_VARIANT,
        Perturbation,
    )


def margin_reward(foil: int):
    """Reward ``z_target - z_foil``. ``target`` is the kept class TorchRay passes in."""
    _, _, preserve, delete, dual, _ = _import_torchray()
    foil = int(foil)

    def reward(activation, target, variant):
        def gap(pred):
            return pred[:, int(target)] - pred[:, foil]

        if variant == delete:
            return -gap(activation)
        if variant == preserve:
            return gap(activation)
        if variant == dual:
            half = activation.shape[0] // 2
            return gap(activation[:half]) - gap(activation[half:])
        raise ValueError(f"unknown variant {variant}")

    reward.__name__ = "margin_reward"
    return reward


def _one_image(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    area: float,
    reward_func,
    **kwargs,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("extremal_perturbation is called on one image")
    extremal_perturbation, _, _, _, _, _ = _import_torchray()
    params = list(model.parameters())
    flags = [p.requires_grad for p in params]
    was_training = model.training
    model.eval()
    try:
        mask, _hist = extremal_perturbation(
            model,
            image,
            int(target),
            areas=[float(area)],
            reward_func=reward_func,
            **kwargs,
        )
    finally:
        for param, flag in zip(params, flags):
            param.requires_grad_(flag)
        model.train(was_training)
    return mask[0, 0]


def class_masks(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: torch.Tensor,
    area: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """Native mask for each image's class. ``class_idx`` is ``(B,)``. Returns ``(B, H, W)``."""
    _, simple, _, _, _, _ = _import_torchray()
    ids = class_idx.detach().to(dtype=torch.long).view(-1)
    masks = [
        _one_image(model, x[i : i + 1], int(ids[i]), area, simple, **kwargs)
        for i in range(x.shape[0])
    ]
    return torch.stack(masks, dim=0)


def margin_masks(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    area: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """Native mask of ``z_k - z_l`` for the shared rank-0 / rank-1 pair. Returns ``(B, H, W)``."""
    kept, foil = foil_pair(hypotheses)
    masks = [
        _one_image(
            model,
            x[i : i + 1],
            int(kept[i]),
            area,
            margin_reward(int(foil[i])),
            **kwargs,
        )
        for i in range(x.shape[0])
    ]
    return torch.stack(masks, dim=0)
