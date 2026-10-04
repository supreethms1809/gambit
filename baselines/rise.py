"""RISE saliency, per class and as a margin between the shared hypothesis pair.

The mask generator and the per-class weighted sum are the official
implementation (Petsiuk, Das, and Saenko, BMVC 2018), vendored at
``third_party/rise``. NumPy's global random state draws the masks, so a repeat
seeds it before the call. The margin map uses those same masks and weights
each one by z_k - z_l.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.types import HypothesisSet

from .hypotheses import foil_pair

_RISE = None


def _rise_cls():
    global _RISE
    if _RISE is None:
        path = Path(__file__).resolve().parents[1] / "third_party" / "rise" / "explanations.py"
        spec = importlib.util.spec_from_file_location("rise_official", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RISE = module.RISE
    return _RISE


def _check(x: torch.Tensor, n_masks: int, p1: float) -> None:
    if x.ndim != 4:
        raise ValueError("RISE expects a batch of images")
    if n_masks < 1:
        raise ValueError("RISE needs at least one mask")
    if not 0.0 < p1 <= 1.0:
        raise ValueError("mask probability p1 must be in (0, 1]")


def _explainer(model: nn.Module, x: torch.Tensor, n_masks: int, s: int, p1: float, seed: int, batch: int):
    _, _, height, width = x.shape
    explainer = _rise_cls()(model, (height, width), gpu_batch=batch, device=x.device)
    np.random.seed(seed)
    explainer.generate_masks(n_masks, s, p1, savepath=None)
    return explainer


def class_maps(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    n_masks: int = 8000,
    s: int = 7,
    p1: float = 0.5,
    seed: int = 0,
    batch: int = 32,
) -> torch.Tensor:
    """``(B, H, W)`` official RISE map for one class.

    Defaults are the ResNet-50 ImageNet setting: 8000 masks, cell grid 7, and
    equal probability of keeping a cell.
    """
    _check(x, n_masks, p1)
    explainer = _explainer(model, x, n_masks, s, p1, seed, batch)
    was_training = model.training
    model.eval()
    try:
        maps = [explainer(x[i : i + 1])[int(class_idx)] for i in range(x.shape[0])]
    finally:
        model.train(was_training)
    out = torch.stack(maps, dim=0)
    if not torch.isfinite(out).all():
        raise ValueError("RISE produced a non-finite map")
    return out


def margin_maps(
    model: nn.Module,
    x: torch.Tensor,
    hypotheses: HypothesisSet,
    n_masks: int = 8000,
    s: int = 7,
    p1: float = 0.5,
    seed: int = 0,
    batch: int = 32,
) -> torch.Tensor:
    """``(B, H, W)`` RISE weighted by the logit margin z_k - z_l."""
    _check(x, n_masks, p1)
    kept, foil = foil_pair(hypotheses)
    if kept.shape[0] != x.shape[0]:
        raise ValueError("hypothesis batch does not match the image batch")
    explainer = _explainer(model, x, n_masks, s, p1, seed, batch)
    masks = explainer.masks
    count = masks.shape[0]
    _, _, height, width = x.shape
    flat = masks.view(count, height * width)
    was_training = model.training
    model.eval()
    try:
        rows = []
        for i in range(x.shape[0]):
            stack = masks * x[i : i + 1]
            weights = []
            k = int(kept[i])
            foil_i = int(foil[i])
            for start in range(0, count, batch):
                logits = model(stack[start : start + batch])
                weights.append(logits[:, k] - logits[:, foil_i])
            weight = torch.cat(weights, dim=0)
            rows.append((weight @ flat).view(height, width) / count / p1)
    finally:
        model.train(was_training)
    out = torch.stack(rows, dim=0)
    if not torch.isfinite(out).all():
        raise ValueError("RISE produced a non-finite map")
    return out
