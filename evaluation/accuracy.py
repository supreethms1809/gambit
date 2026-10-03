"""Top-1, balanced accuracy, and worst-group accuracy.

Checkpoint selection uses balanced accuracy. The model table reports top-1
beside it. Worst-group accuracy is a property of a shift model, not of an
explanation method.
"""

from __future__ import annotations

import torch
from torch import Tensor


def top1_and_balanced(
    pred: Tensor,
    target: Tensor,
    num_classes: int,
) -> tuple[float, float]:
    """Overall accuracy, then macro recall over classes that appear in ``target``."""
    pred = pred.detach().cpu().long().view(-1)
    target = target.detach().cpu().long().view(-1)
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    total = int(target.numel())
    if total == 0:
        return 0.0, 0.0
    top1 = float((pred == target).sum().item()) / total
    correct = torch.zeros(num_classes)
    seen = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = target == c
        seen[c] = int(mask.sum().item())
        correct[c] = int((pred[mask] == c).sum().item())
    present = seen > 0
    balanced = float((correct[present] / seen[present]).mean().item()) if bool(present.any()) else 0.0
    return top1, balanced


def worst_group_accuracy(pred: Tensor, target: Tensor, group: Tensor) -> float:
    """Lowest per-group accuracy. Groups with no examples are ignored."""
    pred = pred.detach().cpu().long().view(-1)
    target = target.detach().cpu().long().view(-1)
    group = group.detach().cpu().long().view(-1)
    if not (pred.shape == target.shape == group.shape):
        raise ValueError("pred, target, and group must have the same shape")
    if pred.numel() == 0:
        return 0.0
    scores = []
    for g in group.unique().tolist():
        mask = group == g
        scores.append(float((pred[mask] == target[mask]).float().mean().item()))
    return min(scores)
