"""Paired summaries and rank correlation. These are descriptive unless a plan says otherwise."""

from __future__ import annotations

from typing import Dict, List

import torch


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation of two ``(N, R)`` tensors."""

    def _rank(values: torch.Tensor) -> torch.Tensor:
        order = values.argsort(dim=-1)
        ranks = torch.zeros_like(values)
        positions = torch.arange(values.shape[-1], dtype=values.dtype, device=values.device)
        ranks.scatter_(-1, order, positions.expand_as(values))
        return ranks

    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    numerator = (ra * rb).sum(dim=-1)
    denominator = (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-8)
    return numerator / denominator


def paired(a: List[float], b: List[float]) -> Dict[str, float]:
    """Paired difference of two per-image score lists. Empty input returns ``{}``."""
    if not a or len(a) != len(b):
        return {}
    delta = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    n = delta.numel()
    mean = delta.mean().item()
    se = (delta.std(unbiased=True) / (n ** 0.5)).item() if n > 1 else 0.0
    return {
        "delta": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "t": mean / se if se > 0 else float("nan"),
        "win_rate": (delta > 0).float().mean().item(),
        "n": float(n),
    }
