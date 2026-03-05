from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Protocol
import torch
from .types import Tensor

class EvidenceUnitSpace(Protocol):

    def num_units(self) -> int:
        pass

    def embed_units(self, units: Any) -> Optional[Tensor]:
        pass
    
    def keep(self, x: Any, m: Tensor) -> Any:
        pass

    def remove(self, x: Any, m: Tensor) -> Any:
        pass