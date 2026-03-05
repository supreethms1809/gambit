"""Grad-CAM pooled to grid regions: evidence (B, K, R), nonnegative."""
from __future__ import annotations
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.types import Tensor, HypothesisSet


def _find_last_conv2d(module: nn.Module) -> Optional[nn.Module]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


class GradCAMRegionsProvider:
    """BaseEvidenceProvider: Grad-CAM pooled to grid_h x grid_w regions. Returns (B, K, R) nonnegative."""

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

    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        if self._target_layer is not None:
            return self._target_layer
        found = _find_last_conv2d(model)
        if found is None:
            raise ValueError("No Conv2d in model and no target_layer given")
        return found

    def _cam_for_class(
        self,
        model: nn.Module,
        x: Tensor,
        layer: nn.Module,
        class_idx: Tensor,
    ) -> Tensor:
        """Compute Grad-CAM for one class index per batch; return (B, R)."""
        # Use local variables instead of instance state for thread safety
        activations = None
        gradients = None

        def forward_hook(_module: Any, _input: Any, output: Tensor) -> None:
            nonlocal activations
            activations = output.detach()

        def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
            nonlocal gradients
            gradients = grad_output[0].detach()

        handle_fwd = layer.register_forward_hook(forward_hook)
        handle_bwd = layer.register_full_backward_hook(backward_hook)
        try:
            out = model(x)
            B = x.shape[0]
            one_hot = torch.zeros_like(out, device=out.device)
            one_hot.scatter_(1, class_idx.clamp_min(0).unsqueeze(1), 1.0)
            loss = (out * one_hot).sum()
            model.zero_grad(set_to_none=True)
            loss.backward()
        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        A = activations  # (B, C, h, w)
        G = gradients    # (B, C, h, w)
        if A is None or G is None:
            return torch.zeros(B, self.R, device=x.device, dtype=x.dtype)
        weights = G.mean(dim=(2, 3))  # (B, C)
        cam = (weights.unsqueeze(2).unsqueeze(3) * A).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)
        # Pool to grid and flatten
        cam = F.adaptive_avg_pool2d(cam, (self.grid_h, self.grid_w))  # (B, 1, grid_h, grid_w)
        cam = cam.flatten(1)  # (B, R)
        return cam

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
        # Zero out invalid positions
        E = E * mask.unsqueeze(-1).float()
        return E
