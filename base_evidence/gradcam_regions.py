"""Grad-CAM pooled to grid regions: evidence (B, K, R), nonnegative.

Supports both CNN backbones (ResNet, MobileNet, EfficientNet) and Vision Transformers
(torchvision VisionTransformer). Target layer is auto-detected:
  - CNN: last Conv2d layer in the model
  - ViT: last encoder block (model.encoder.layers[-1])
Pass ``target_layer`` explicitly to override for any architecture.
"""
from __future__ import annotations
from typing import Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.types import Tensor, HypothesisSet


def _find_target_layer(model: nn.Module) -> nn.Module:
    """Return the appropriate GradCAM target layer for CNN or ViT models.

    - CNN (ResNet, MobileNet, EfficientNet): last Conv2d layer.
    - ViT (torchvision VisionTransformer): last transformer encoder block.
    Raises ValueError if neither is found and no target_layer is provided.
    """
    try:
        from torchvision.models import VisionTransformer
        if isinstance(model, VisionTransformer):
            return model.encoder.layers[-1]
    except (ImportError, AttributeError):
        pass
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError(
            "No Conv2d found in model and no target_layer provided. "
            "Pass target_layer explicitly for non-CNN/non-ViT architectures."
        )
    return last_conv


class GradCAMRegionsProvider:
    """BaseEvidenceProvider: Grad-CAM pooled to grid_h x grid_w regions. Returns (B, K, R) nonneg.

    Works for CNN backbones (activations are (B, C, H, W)) and ViT backbones (activations are
    (B, seq_len, dim) — CLS token is dropped and the remaining patch tokens are reshaped to 2D).
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        target_layer: Optional[Any] = None,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.R = grid_h * grid_w
        self._target_layer = target_layer
        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None

    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        if self._target_layer is not None:
            return self._target_layer
        return _find_target_layer(model)

    def _forward_hook(self, _module: Any, _input: Any, output: Tensor) -> None:
        self._activations = output.detach()

    def _backward_hook(self, _module: Any, _grad_input: Any, grad_output: Any) -> None:
        self._gradients = grad_output[0].detach()

    def _cam_for_class(
        self,
        model: nn.Module,
        x: Tensor,
        layer: nn.Module,
        class_idx: Tensor,
    ) -> Tensor:
        """Compute Grad-CAM for one class index per batch; return (B, R)."""
        self._activations = None
        self._gradients = None
        handle_fwd = layer.register_forward_hook(self._forward_hook)
        handle_bwd = layer.register_full_backward_hook(self._backward_hook)
        try:
            out = model(x)
            B = x.shape[0]
            one_hot = torch.zeros_like(out, device=out.device)
            one_hot.scatter_(1, class_idx.clamp_min(0).unsqueeze(1), 1.0)
            loss = (out * one_hot).sum()
            model.zero_grad()
            loss.backward()
        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        A = self._activations  # CNN: (B, C, h, w) | ViT: (B, seq_len, dim)
        G = self._gradients    # CNN: (B, C, h, w) | ViT: (B, seq_len, dim)
        B = x.shape[0]
        if A is None or G is None:
            return torch.zeros(B, self.R, device=x.device, dtype=x.dtype)

        if A.dim() == 3:
            # ViT: drop CLS token (index 0), reshape patch sequence to 2D spatial grid
            A = A[:, 1:, :]   # (B, num_patches, dim)
            G = G[:, 1:, :]
            num_patches = A.shape[1]
            ph = pw = int(math.isqrt(num_patches))
            if ph * pw != num_patches:
                # Non-square patch grid — fall back to 1D pooling
                ph, pw = 1, num_patches
            A = A.permute(0, 2, 1).reshape(B, -1, ph, pw)  # (B, dim, ph, pw)
            G = G.permute(0, 2, 1).reshape(B, -1, ph, pw)  # (B, dim, ph, pw)

        weights = G.mean(dim=(2, 3))  # (B, C)
        cam = (weights.unsqueeze(2).unsqueeze(3) * A).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)
        cam = F.adaptive_avg_pool2d(cam, (self.grid_h, self.grid_w))
        return cam.flatten(1)  # (B, R)

    def explain(self, x: Any, model: nn.Module, hypotheses: HypothesisSet) -> Tensor:
        """Return evidence (B, K, R), nonnegative."""
        layer = self._get_target_layer(model)
        ids = hypotheses.ids   # (B, K)
        mask = hypotheses.mask # (B, K)
        B, K = ids.shape
        device = x.device if hasattr(x, "device") else next(model.parameters()).device
        E = torch.zeros(B, K, self.R, device=device, dtype=torch.float32)
        for k in range(K):
            class_idx = ids[:, k]  # (B,)
            cam_r = self._cam_for_class(model, x, layer, class_idx)  # (B, R)
            E[:, k, :] = cam_r
        E = E.clamp_min(0.0)
        E = E * mask.unsqueeze(-1).float()
        return E
