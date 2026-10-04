"""Our Grad-CAM and IG against pytorch-grad-cam and Captum.

Grad-CAM is compared at the target layer, before pytorch-grad-cam resizes the
map to the image and min-max scales it for display. That resize is a viewing
step. Both sides then use the same adaptive average pool onto the region grid,
which is what ``GradCAMRegionsProvider`` returns.

IG uses the right Riemann sum on both sides, with the same step count.
Both sides sum channels and clamp at zero, matching
``IntegratedGradientsRegionsProvider``. Captum's default quadrature is
Gauss-Legendre, so the check passes ``method='riemann_right'``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_evidence.gradcam_regions import GradCAMRegionsProvider, _find_target_layer
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from core.types import HypothesisSet
from evaluation.metrics import spearman

SPEARMAN_MIN = 0.95


def _class_hypotheses(class_idx: torch.Tensor) -> HypothesisSet:
    ids = class_idx.detach().to(dtype=torch.long).view(-1, 1)
    mask = torch.ones(ids.shape[0], 1, dtype=torch.bool, device=ids.device)
    return HypothesisSet(ids=ids, mask=mask)


def library_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: torch.Tensor,
    grid_h: int,
    grid_w: int,
    target_layer: nn.Module | None = None,
) -> torch.Tensor:
    """``(B, R)`` library Grad-CAM pooled the same way as our provider."""
    import pytorch_grad_cam as pgc
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer = target_layer or _find_target_layer(model)
    ids = class_idx.detach().to(dtype=torch.long).view(-1)
    targets = [ClassifierOutputTarget(int(i)) for i in ids.tolist()]
    was_training = model.training
    try:
        with pgc.GradCAM(model=model, target_layers=[layer]) as cam:
            outputs = cam.activations_and_grads(x.detach())
            cam.model.zero_grad()
            loss = sum(target(output) for target, output in zip(targets, outputs))
            loss.backward()
            activations = cam.activations_and_grads.activations[0].numpy()
            gradients = cam.activations_and_grads.gradients[0].numpy()
            raw = cam.get_cam_image(x, layer, targets, activations, gradients)
            raw = np.maximum(raw, 0)
    finally:
        model.train(was_training)
    field = torch.from_numpy(np.asarray(raw, dtype=np.float32)).unsqueeze(1)
    return F.adaptive_avg_pool2d(field, (grid_h, grid_w)).flatten(1)


def gradcam_spearman(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Per-image Spearman of our Grad-CAM against pytorch-grad-cam."""
    model.eval()
    hypotheses = _class_hypotheses(class_idx)
    ours = GradCAMRegionsProvider(grid_h, grid_w).explain(x, model, hypotheses)[:, 0]
    lib = library_gradcam(model, x, class_idx, grid_h, grid_w).to(device=ours.device)
    return spearman(ours.detach().cpu(), lib.detach().cpu())


def ig_spearman(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: torch.Tensor,
    grid_h: int,
    grid_w: int,
    steps: int = 8,
) -> torch.Tensor:
    """Per-image Spearman of our IG against Captum's right Riemann sum."""
    from captum.attr import IntegratedGradients

    if steps < 2:
        raise ValueError("the IG cross-check needs at least 2 steps")
    model.eval()
    hypotheses = _class_hypotheses(class_idx)
    ours = IntegratedGradientsRegionsProvider(grid_h, grid_w, steps=steps).explain(
        x, model, hypotheses
    )[:, 0]
    attr = IntegratedGradients(model).attribute(
        x.detach(),
        baselines=torch.zeros_like(x),
        target=class_idx.detach().to(dtype=torch.long).view(-1),
        n_steps=steps,
        method="riemann_right",
    )
    summed = attr.detach().sum(dim=1, keepdim=True).clamp_min(0)
    lib = F.adaptive_avg_pool2d(summed, (grid_h, grid_w))
    lib = lib.flatten(1)
    return spearman(ours.detach().cpu(), lib.detach().cpu())
