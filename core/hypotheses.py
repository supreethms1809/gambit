from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Protocol
import torch
from .types import Tensor, HypothesisSet


class HypothesisSelector(Protocol):
    def select(self, logits: Tensor, probs: Tensor) -> HypothesisSet:
        pass


class TopMSelector:
    """Top hypotheses. K is min(m, C). There are no padded slots."""

    def __init__(self, m: int = 5):
        self.m = m

    def select(self, logits: Tensor, probs: Tensor) -> HypothesisSet:
        # logits/probs: (B, num_classes)
        num_classes = logits.shape[1]
        k_actual = min(self.m, num_classes)
        _top_probs, top_ids = logits.topk(k_actual, dim=-1)
        mask = torch.ones_like(top_ids, dtype=torch.bool)
        return HypothesisSet(ids=top_ids, mask=mask)