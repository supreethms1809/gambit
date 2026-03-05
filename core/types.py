from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol
import torch
from torch import Tensor

@dataclass
class HypothesisSet:
    ids: Tensor
    mask: Tensor

@dataclass
class Explanation:
    hypotheses: HypothesisSet
    masks: Dict[str, Tensor]
    metrics: Dict[str, Tensor]
    extras: Dict[str, Any]

@dataclass
class EnvBatch:
    xs: List[Any]
    env_ids: List[str]