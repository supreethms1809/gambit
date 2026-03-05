from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Protocol
import torch
from .types import Tensor, HypothesisSet
from .unit_space import EvidenceUnitSpace

class BaseEvidenceProvider(Protocol):
    def explain(self, x: Any, model: Any, hypotheses: HypothesisSet) -> Tensor:
        pass