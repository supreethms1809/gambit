"""Grad-CAM and integrated gradients of the logit margin z_k - z_l.

The class pair comes from ``foil_pair``. k is hypothesis rank 0 and l is rank 1.
The reference implementations are pytorch-grad-cam, with a target that returns
the margin, and Captum ``IntegratedGradients`` on a margin forward function.
Both use the right Riemann sum with the same step count as our IG provider.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.types import HypothesisSet

from core.eval_mode import eval_mode

from .hypotheses import foil_pair


class MarginOutputTarget:
    """pytorch-grad-cam target: one image's logit margin."""

    def __init__(self, kept: int, foil: int):
        self.kept = int(kept)
        self.foil = int(foil)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return model_output[self.kept] - model_output[self.foil]
        return model_output[:, self.kept] - model_output[:, self.foil]


def _layer(model: nn.Module, target_layer: nn.Module | None) -> nn.Module:
    if target_layer is not None:
        return target_layer
    from base_evidence.gradcam_regions import _find_target_layer

    return _find_target_layer(model)


def margin_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    target_layer: nn.Module | None = None,
) -> torch.Tensor:
    """``(B, H, W)`` Grad-CAM of z_k - z_l. k and l are the shared hypothesis pair."""
    import pytorch_grad_cam as pgc

    from base_evidence.gradcam_regions import cam_reshape_transform

    kept, foil = foil_pair(hypotheses)
    layer = _layer(model, target_layer)
    targets = [
        MarginOutputTarget(int(k), int(l)) for k, l in zip(kept.tolist(), foil.tolist())
    ]
    was_training = model.training
    try:
        # Eval mode: attribution must see the deterministic inference function.
        with eval_mode(model), pgc.GradCAM(
            model=model, target_layers=[layer], reshape_transform=cam_reshape_transform(model),
        ) as cam:
            heat = cam(input_tensor=x.detach(), targets=targets)
    finally:
        model.train(was_training)
    return torch.from_numpy(np.asarray(heat)).to(dtype=torch.float32, device=x.device)


def margin_integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    steps: int = 8,
) -> torch.Tensor:
    """``(B, H, W)`` signed channel-sum IG of z_k - z_l.

    Captum's right Riemann rule needs at least two steps. Positive entries
    support the kept class over the foil. The absolute value is not taken:
    a channel that argues against the margin stays negative and loses the
    top-a ranking.
    """
    from captum.attr import IntegratedGradients

    if steps < 2:
        raise ValueError("margin IG needs at least 2 steps")
    kept, foil = foil_pair(hypotheses)

    def margin_forward(inp: torch.Tensor) -> torch.Tensor:
        logits = model(inp)
        k = kept.to(device=logits.device)
        l = foil.to(device=logits.device)
        return (
            logits.gather(1, k.view(-1, 1)) - logits.gather(1, l.view(-1, 1))
        ).squeeze(1)

    was_training = model.training
    model.eval()
    try:
        attr = IntegratedGradients(margin_forward).attribute(
            x.detach(),
            baselines=torch.zeros_like(x),
            n_steps=steps,
            method="riemann_right",
        )
    finally:
        model.train(was_training)
    return attr.detach().sum(dim=1)
