"""Integrated Gradients pooled to grid regions: evidence (B, K, R), nonnegative."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.types import Tensor, HypothesisSet


class IntegratedGradientsRegionsProvider:
    """BaseEvidenceProvider: Integrated Gradients pooled to grid_h x grid_w regions."""

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        steps: int = 24,
        baseline: str = "zero",
    ):
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if baseline not in {"zero", "mean"}:
            raise ValueError("baseline must be 'zero' or 'mean'")
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.R = grid_h * grid_w
        self.steps = steps
        self.baseline = baseline

    def _baseline_input(self, x: Tensor) -> Tensor:
        if self.baseline == "mean":
            return x.mean(dim=(2, 3), keepdim=True).expand_as(x)
        return torch.zeros_like(x)

    def _ig_for_class(
        self,
        model: nn.Module,
        x: Tensor,
        class_idx: Tensor,
    ) -> Tensor:
        """Compute Integrated Gradients evidence for one class index per batch; return (B, R)."""
        x_base = self._baseline_input(x)
        delta = x - x_base
        total_grads = torch.zeros_like(x)

        for i in range(1, self.steps + 1):
            alpha = float(i) / float(self.steps)
            x_interp = (x_base + alpha * delta).detach().requires_grad_(True)
            logits = model(x_interp)
            scores = logits.gather(1, class_idx.clamp_min(0).unsqueeze(1)).sum()
            model.zero_grad(set_to_none=True)
            scores.backward()
            if x_interp.grad is not None:
                total_grads += x_interp.grad.detach()

        avg_grads = total_grads / float(self.steps)
        attr = delta * avg_grads
        ev = attr.abs().sum(dim=1, keepdim=True)  # nonnegative salience
        ev = F.adaptive_avg_pool2d(ev, (self.grid_h, self.grid_w))
        return ev.flatten(1)

    def explain(self, x: Any, model: nn.Module, hypotheses: HypothesisSet) -> Tensor:
        """Return evidence (B, K, R), nonnegative."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("IntegratedGradientsRegionsProvider expects x to be a torch.Tensor")
        ids = hypotheses.ids   # (B, K)
        mask = hypotheses.mask # (B, K)
        B, K = ids.shape
        device = x.device
        E = torch.zeros(B, K, self.R, device=device, dtype=torch.float32)

        was_training = model.training
        model.eval()
        try:
            for k in range(K):
                cls = ids[:, k]
                E[:, k, :] = self._ig_for_class(model, x, cls)
        finally:
            if was_training:
                model.train()

        E = E.clamp_min(0.0)
        E = E * mask.unsqueeze(-1).float()
        return E
