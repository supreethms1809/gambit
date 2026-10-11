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


def make_deterministic(device: torch.device) -> None:
    """Deterministic kernels on CUDA, so a repeated cell gives identical records.

    CDEA's Adam loop amplifies any run-to-run difference in a backward pass
    into a different allocation, so nondeterministic kernels show up as
    different scores. ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first
    cuBLAS call, so call this before the model is built. Ops without a
    deterministic CUDA kernel warn instead of failing.
    """
    if getattr(device, "type", str(device)) != "cuda":
        return
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
