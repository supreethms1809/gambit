from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Protocol
import torch
from .types import Tensor, HypothesisSet


class HypothesisSelector(Protocol):
    def select(self, logits: Tensor, probs: Tensor) -> HypothesisSet:
        pass


class TopMSelector:
    """Select top-m hypotheses per batch; returns padded HypothesisSet with ids (B, K), mask (B, K)."""

    def __init__(self, m: int = 5):
        self.m = m

    def select(self, logits: Tensor, probs: Tensor) -> HypothesisSet:
        # logits/probs: (B, num_classes)
        B, num_classes = logits.shape[0], logits.shape[1]
        K = self.m
        k_actual = min(K, num_classes)
        top_probs, top_ids = logits.topk(k_actual, dim=-1)  # (B, k_actual)
        device = logits.device
        ids = torch.zeros(B, K, dtype=torch.long, device=device)
        mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        ids[:, :k_actual] = top_ids
        mask[:, :k_actual] = True
        return HypothesisSet(ids=ids, mask=mask)