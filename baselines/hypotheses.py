"""One hypothesis list for every method.

Top-K comes from the logits, matching ``TopMSelector``. The foil is rank 1,
the second entry of that list. A method receives this set. It does not call
``argmax`` itself.
"""

from __future__ import annotations

import torch

from core.hypotheses import TopMSelector
from core.types import HypothesisSet


def shared_hypotheses(logits: torch.Tensor, k: int) -> HypothesisSet:
    """Top-k hypotheses. When ``k`` exceeds the number of classes, the extra slots are masked off."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if logits.ndim != 2:
        raise ValueError("logits must be (batch, classes)")
    probs = torch.softmax(logits, dim=-1)
    return TopMSelector(k).select(logits, probs)


def foil_pair(hypotheses: HypothesisSet) -> tuple[torch.Tensor, torch.Tensor]:
    """Class k (rank 0) and foil l (rank 1) from a shared hypothesis set.

    Both columns have to be valid. A model with a single class has no foil,
    and a request for K=1 does not invent one.
    """
    if hypotheses.ids.ndim != 2 or hypotheses.ids.shape[1] < 2:
        raise ValueError("rank-1 foil needs two hypothesis columns")
    valid = hypotheses.mask[:, 0] & hypotheses.mask[:, 1]
    if not bool(valid.all()):
        raise ValueError("rank-1 foil is masked; fewer than two classes were available")
    return hypotheses.ids[:, 0].long(), hypotheses.ids[:, 1].long()
