"""Central device selection: prefer CUDA, then MPS (Apple Silicon), then CPU."""
from __future__ import annotations
from typing import Optional
import torch


def get_device(prefer: Optional[torch.device] = None) -> torch.device:
    """
    Return the best available device: prefer > cuda > mps > cpu.
    Use this everywhere so training/eval use GPU when available (CUDA or MPS).
    """
    if prefer is not None:
        return prefer
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
