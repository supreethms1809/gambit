"""Keep the classifier in eval mode while explaining it.

Allocation and attribution run dozens of forward passes. In train mode those
passes update BatchNorm running statistics and sample dropout masks, so the
model being explained drifts mid-explanation and the masks are not
reproducible. Every forward below must see the same deterministic function.

``eval_mode`` records the training flag, switches to eval, and restores the
flag afterwards — the same save/restore pattern as ``evaluation/scores.py``.
Callables without a ``training`` flag (e.g. CVE decision heads) pass through
unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def eval_mode(model: Any) -> Iterator[None]:
    """Yield with ``model`` in eval mode, then restore its training flag."""
    if not hasattr(model, "training") or not hasattr(model, "eval"):
        yield
        return
    was_training = bool(model.training)
    model.eval()
    try:
        yield
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
