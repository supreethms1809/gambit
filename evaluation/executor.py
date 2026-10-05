"""Score one contrastive pair on images the caller already chose.

Family C uses this for CD@5%. The hypotheses are the model's top-2, and the
foil mask is rank 1. The test split is not selected here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baselines.hypotheses import shared_hypotheses
from core.types import HypothesisSet
from evaluation.foil_masks import pair_budget_masks, pair_classes
from evaluation.scores import contrastive_deletion


class FamilyCExecutor:
    def __init__(
        self,
        model: nn.Module,
        fraction: float = 0.05,
        seed: int = 0,
        iters: int = 24,
        noise: float = 0.01,
        include_shared: bool = False,
    ):
        self.model = model
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.iters = int(iters)
        self.noise = float(noise)
        self.include_shared = bool(include_shared)

    def hypotheses(self, images: torch.Tensor) -> HypothesisSet:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits = self.model(images)
        finally:
            self.model.train(was_training)
        return shared_hypotheses(logits, 2)

    def score(
        self,
        images: torch.Tensor,
        unique: torch.Tensor,
        shared: torch.Tensor | None = None,
        hypotheses: HypothesisSet | None = None,
        **upsample,
    ) -> torch.Tensor:
        """CD for the rank-0 / rank-1 pair. ``unique`` is ``(B, K, R)``."""
        if hypotheses is None:
            hypotheses = self.hypotheses(images)
        class_k, class_l = pair_classes(hypotheses)
        mask_k, mask_l = pair_budget_masks(
            unique, shared, self.fraction, self.seed, self.include_shared, **upsample,
        )
        return contrastive_deletion(
            self.model,
            images,
            mask_k,
            mask_l,
            class_k,
            class_l,
            iters=self.iters,
            noise=self.noise,
            seed=self.seed,
        )
