from __future__ import annotations
from typing import Any, Dict, Optional, Protocol
import torch
from .types import Tensor, HypothesisSet, EnvBatch

class Allocator(Protocol):
    def allocate(
        self,
        x: Any,
        model: Any,
        unit_space: Any,
        hypotheses: HypothesisSet,
        evidence: Tensor,
        tokens: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
        env: Optional[EnvBatch] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        pass


class EvidenceAsMaskAllocator:
    """Minimal allocator: use evidence as the 'unique' mask (B, K, R)."""

    def allocate(
        self,
        x: Any,
        model: Any,
        unit_space: Any,
        hypotheses: HypothesisSet,
        evidence: Tensor,
        tokens: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
        env: Optional[EnvBatch] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        return {"unique": evidence}

