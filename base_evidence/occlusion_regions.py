"""Occlusion evidence pooled to grid regions: evidence (B, K, R), nonnegative.

Perturbation-based rather than gradient-based. For a region ``r`` and hypothesis
``k``::

    evidence[k, r] = logit_k(x) - logit_k(x with region r removed)

A large drop means the region was carrying class ``k``. No gradients, no hooks, no
assumptions about architecture — so unlike Grad-CAM this works unchanged on
transformers (Grad-CAM's non-negative-activation assumption fails on LayerNorm'd
tokens and silently returns an all-zero field).

**Why this provider fits CDEA specifically.** The objective already lives in
intervention space: sufficiency is a logit under ``unit_space.keep``, and the
decomposition test deletes masks with ``unit_space.remove``. Grad-CAM and IG supply
evidence in *gradient* space, so masks are initialized from one currency and
optimized against another. Occlusion closes that gap by calling ``unit_space.remove``
itself — the same perturbation, the same baseline, the same code path. It is also
the marginal contribution that Shapley-style attribution is built from, which makes
it the natural evidence source for a game-theoretic allocator.

**Cost.** One forward pass over an occluded image yields every class's logit at once,
so a full evidence field is ``R`` forward passes per image, independent of ``K``.
At a 7x7 grid that is 49 passes — cheaper than Integrated Gradients, which needs
``steps x K`` forward *and* backward passes. At 28x28 it is 784, roughly 3x IG.

**The redundancy caveat.** Removing one region at a time under-estimates evidence
that is duplicated across regions: if two regions each independently suffice, taking
either alone changes nothing and both score ~0. That matters when the target is large
relative to a cell — a HAM10000 lesion covers ~13 cells of a 7x7 grid, so occluding
one leaves the diagnosis intact. Use ``mode="rise"`` there: it removes many regions at
once via random masks (Petsiuk et al., 2018), so redundant evidence still registers.
``mode="single"`` is the sharper choice when the target is small, as with brain tumor.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from core.types import Tensor, HypothesisSet


class OcclusionRegionsProvider:
    """BaseEvidenceProvider: occlusion evidence over grid_h x grid_w regions."""

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        unit_space: Any = None,
        mode: str = "single",
        n_masks: int = 512,
        keep_prob: float = 0.5,
        batch_regions: int = 64,
        seed: int = 0,
    ):
        """
        Args:
            grid_h, grid_w: region grid.
            unit_space:     used for the removal intervention, so the perturbation
                            matches the one the objective and the decomposition test
                            apply. If None, a VisionGridUnitSpace with a blur
                            baseline is constructed to match.
            mode:           "single" (one region at a time) or "rise" (random
                            multi-region masks; handles redundant evidence).
            n_masks:        number of random masks when mode="rise".
            keep_prob:      probability a region is *kept* in a rise mask.
            batch_regions:  how many perturbed copies to score per forward pass.
            seed:           reproducible rise masks.
        """
        if mode not in {"single", "rise"}:
            raise ValueError("mode must be 'single' or 'rise'")
        if not (0.0 < keep_prob < 1.0):
            raise ValueError("keep_prob must be in (0, 1)")
        if n_masks <= 0 or batch_regions <= 0:
            raise ValueError("n_masks and batch_regions must be > 0")
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.R = grid_h * grid_w
        self.mode = mode
        self.n_masks = n_masks
        self.keep_prob = keep_prob
        self.batch_regions = batch_regions
        self.seed = seed
        self._unit_space = unit_space

    def _space(self):
        if self._unit_space is None:
            from modality.grid_regions import VisionGridUnitSpace
            self._unit_space = VisionGridUnitSpace(self.grid_h, self.grid_w, baseline="blur")
        return self._unit_space

    @torch.no_grad()
    def explain(self, x: Any, model: Any, hypotheses: HypothesisSet) -> Tensor:
        """Return evidence (B, K, R), nonnegative."""
        space = self._space()
        ids = hypotheses.ids                      # (B, K)
        mask_valid = hypotheses.mask              # (B, K)
        B, K = ids.shape
        device = x.device if hasattr(x, "device") else next(model.parameters()).device
        cls = ids.clamp_min(0)

        base = model(x).gather(1, cls)            # (B, K) logits of each hypothesis
        E = torch.zeros(B, K, self.R, device=device, dtype=torch.float32)

        if self.mode == "single":
            # Score regions in chunks: one forward pass covers `chunk` occlusions.
            for start in range(0, self.R, self.batch_regions):
                idx = list(range(start, min(start + self.batch_regions, self.R)))
                for r in idx:
                    m = torch.zeros(B, self.R, device=device, dtype=x.dtype)
                    m[:, r] = 1.0
                    drop = base - model(space.remove(x, m)).gather(1, cls)  # (B, K)
                    E[:, :, r] = drop
        else:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            weight = torch.zeros(B, K, self.R, device=device, dtype=torch.float32)
            for _ in range(self.n_masks):
                keep = (torch.rand(self.R, generator=gen) < self.keep_prob).float()
                m = (1.0 - keep).to(device).unsqueeze(0).expand(B, -1)   # removed regions
                drop = base - model(space.remove(x, m)).gather(1, cls)   # (B, K)
                # Credit the drop to every region that was removed.
                E += drop.unsqueeze(-1) * m.unsqueeze(1)
                weight += m.unsqueeze(1)
            E = E / weight.clamp_min(1.0)

        # Only positive drops are evidence *for* a class. Negative drops (removal
        # helps the class) are real signal but the protocol requires non-negative
        # evidence, so they are clamped rather than folded in.
        E = E.clamp_min(0.0)
        E = E * mask_valid.unsqueeze(-1).float()
        return E
