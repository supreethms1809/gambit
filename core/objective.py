from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Protocol
import torch
from .types import Tensor, HypothesisSet, EnvBatch
from .unit_space import EvidenceUnitSpace
from .base_evidence import BaseEvidenceProvider
from .hypotheses import HypothesisSelector

class AllocationObjective(Protocol):

    def compute(
        self,
        x: Any,
        model: Any,
        unit_space: Any,
        hypotheses: HypothesisSet,
        masks: Dict[str, Tensor],
        evidence: Tensor,
        tokens: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
        env: Optional[EnvBatch] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        pass