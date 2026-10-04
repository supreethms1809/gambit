"""Paired summaries and rank correlation. These are descriptive unless a plan says otherwise."""

from __future__ import annotations

from typing import Dict, List

import torch


def _average_rank_row(values: torch.Tensor) -> torch.Tensor:
    """Zero-based average ranks with ties sharing their mean rank."""
    flat = values.detach().double().reshape(-1)
    n = flat.numel()
    order = flat.argsort(stable=True)
    ranked = torch.empty(n, dtype=torch.float64)
    sorted_vals = flat[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranked[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranked


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation of two ``(N, R)`` tensors.

    Ties share their average rank (clamped attributions produce many exact
    zeros; arbitrary tie order would make the correlation depend on memory
    order). A row with no rank information — constant on either side — has
    undefined correlation and reports 0.0 rather than an arbitrary value.
    """
    if a.shape != b.shape:
        raise ValueError(f"spearman needs matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.ndim != 2:
        raise ValueError("spearman needs (N, R) tensors")
    out = torch.empty(a.shape[0], dtype=torch.float64)
    for i in range(a.shape[0]):
        ra = _average_rank_row(a[i]) - _average_rank_row(a[i]).mean()
        rb = _average_rank_row(b[i]) - _average_rank_row(b[i]).mean()
        denom = float(ra.norm() * rb.norm())
        out[i] = 0.0 if denom == 0.0 else float((ra * rb).sum() / denom)
    return out.to(dtype=a.dtype, device=a.device)


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
