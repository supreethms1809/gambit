"""Grad-CAM of the cross-entropy between the logits and the contrast class.

Prabhushankar et al. backpropagate J(P, Q), the recognition loss with the
contrast class Q as the target, and pool that gradient the way Grad-CAM pools
a class gradient. Q is hypothesis rank 1. This is a different target from the
logit margin z_k - z_l.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import HypothesisSet

from .hypotheses import foil_pair
from .margin import _layer


class CrossEntropyContrastTarget:
    """pytorch-grad-cam target: cross-entropy toward the contrast class."""

    def __init__(self, contrast: int):
        self.contrast = int(contrast)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        logits = model_output if model_output.ndim == 2 else model_output.unsqueeze(0)
        label = torch.tensor([self.contrast], device=logits.device)
        return F.cross_entropy(logits, label)


def contrastive_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    target_layer: nn.Module | None = None,
) -> torch.Tensor:
    """``(B, H, W)`` Grad-CAM of cross-entropy toward the foil class."""
    import pytorch_grad_cam as pgc

    _, foil = foil_pair(hypotheses)
    layer = _layer(model, target_layer)
    targets = [CrossEntropyContrastTarget(int(q)) for q in foil.tolist()]
    was_training = model.training
    try:
        with pgc.GradCAM(model=model, target_layers=[layer]) as cam:
            heat = cam(input_tensor=x.detach(), targets=targets)
    finally:
        model.train(was_training)
    return torch.from_numpy(np.asarray(heat)).to(dtype=torch.float32, device=x.device)
