"""
gambit — high-level API for the CDEA pipeline.

Usage (after `pip install -e .` from repo root):

    import gambit

    # Contrastive explanation
    explainer = gambit.ContrastiveExplainer(model)
    explanation = explainer.explain(x)          # x: (B, 3, H, W) tensor
    print(explanation.masks["unique"].shape)    # (B, K, H, W)
    print(explanation.masks["shared"].shape)    # (B, H, W)
    print(explanation.metrics)

    # Shift-aware robust/shortcut explanation
    from gambit import EnvBatch
    env = EnvBatch(xs=[x_id, x_ood], env_ids=[0, 1])
    explainer = gambit.ShiftExplainer(model)
    explanation = explainer.explain(x, env=env)
    print(explanation.masks["robust"].shape)    # (B, H, W)
    print(explanation.masks["shortcut"].shape)  # (B, H, W)
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import torch

from core.runner import CDEAExplainer
from core.types import EnvBatch, Explanation
from core.hypotheses import TopMSelector
from core.game_modes import resolve_contrastive_game, resolve_shift_game
from core.interaction import get_interaction
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from instantiations.shift.objective import RobustShortcutObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator

__all__ = ["ContrastiveExplainer", "ShiftExplainer", "EnvBatch", "Explanation"]

EvidenceMode = Literal["gradcam", "ig"]
GameMode = Literal["cooperative", "mixed", "competitive"]


def _build_evidence(mode: EvidenceMode, grid_h: int, grid_w: int, ig_steps: int):
    if mode == "gradcam":
        return GradCAMRegionsProvider(grid_h, grid_w)
    elif mode == "ig":
        return IntegratedGradientsRegionsProvider(grid_h, grid_w, steps=ig_steps)
    else:
        raise ValueError(f"evidence must be 'gradcam' or 'ig', got {mode!r}")


class ContrastiveExplainer:
    """
    High-level contrastive CDEA explainer.

    Parameters
    ----------
    model : nn.Module
        Trained classifier. Must accept (B, C, H, W) input and return (B, num_classes) logits.
    game_mode : str
        'cooperative' | 'mixed' (default) | 'competitive'
    evidence : str
        'gradcam' (default) | 'ig'
    grid_size : int or (int, int)
        Spatial grid resolution. 7 for CNNs with 224-px input, 14 for ViT.
    top_k : int
        Number of top classes to explain per image.
    num_steps : int
        Allocator optimisation steps.
    ig_steps : int
        Integrated Gradients steps (ignored when evidence='gradcam').
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: Any,
        game_mode: GameMode = "mixed",
        evidence: EvidenceMode = "gradcam",
        grid_size: int | tuple[int, int] = 7,
        top_k: int = 5,
        num_steps: int = 25,
        ig_steps: int = 24,
        device: Optional[torch.device] = None,
    ):
        grid_h, grid_w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        cfg = resolve_contrastive_game(game_mode)

        unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur")
        selector = TopMSelector(m=top_k)
        base_evidence = _build_evidence(evidence, grid_h, grid_w, ig_steps)
        objective = ContrastiveObjective(
            lambda_suff=1.0,
            lambda_margin=cfg.lambda_margin,
            lambda_sparse=0.05,
            lambda_overlap=cfg.lambda_overlap,
        )
        allocator = OptimizationAllocator(
            objective,
            num_steps=num_steps,
            lr=0.2,
            use_shared=cfg.use_shared,
            lambda_disjoint=cfg.lambda_disjoint,
            lambda_partition=cfg.lambda_partition,
        )
        self._explainer = CDEAExplainer(
            model=model,
            unit_space=unit_space,
            selector=selector,
            base_evidence=base_evidence,
            allocator=allocator,
            objective=objective,
            device=device,
        )
        self.unit_space = unit_space
        self.model = model

    def explain(self, x: torch.Tensor) -> Explanation:
        """
        Run contrastive explanation on a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, H, W). Single image: use x.unsqueeze(0).

        Returns
        -------
        Explanation
            .hypotheses     — top-K class indices, shape (B, K)
            .masks["unique"] — unique evidence per class, shape (B, K, H, W)
            .masks["shared"] — shared evidence, shape (B, H, W)  [mixed/cooperative only]
            .metrics        — dict of suff, margin, overlap, sparse scalars
            .extras["probs"] — softmax probabilities, shape (B, num_classes)
        """
        return self._explainer.explain(x)


class ShiftExplainer:
    """
    High-level shift-aware robust/shortcut CDEA explainer.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    game_mode : str
        'cooperative' | 'mixed' (default) | 'competitive'
    evidence : str
        'gradcam' (default) | 'ig'
    grid_size : int or (int, int)
        Spatial grid resolution.
    top_k : int
        Number of top classes to explain per image.
    num_steps : int
        Allocator optimisation steps.
    ig_steps : int
        Integrated Gradients steps (ignored when evidence='gradcam').
    device : torch.device, optional
    """

    def __init__(
        self,
        model: Any,
        game_mode: GameMode = "mixed",
        evidence: EvidenceMode = "gradcam",
        grid_size: int | tuple[int, int] = 7,
        top_k: int = 5,
        num_steps: int = 50,
        ig_steps: int = 24,
        device: Optional[torch.device] = None,
    ):
        grid_h, grid_w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        cfg = resolve_shift_game(game_mode)

        unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur")
        selector = TopMSelector(m=top_k)
        base_evidence = _build_evidence(evidence, grid_h, grid_w, ig_steps)
        objective = RobustShortcutObjective(
            lambda_mean=cfg.lambda_mean,
            lambda_var=cfg.lambda_var,
            lambda_gap=cfg.lambda_gap,
            lambda_shortcut=cfg.lambda_shortcut,
            lambda_disjoint=cfg.lambda_disjoint,
            lambda_sparse=cfg.lambda_sparse,
        )
        allocator = RobustShortcutOptimizationAllocator(
            objective,
            num_steps=num_steps,
            lr=0.5,
            lambda_disjoint=cfg.lambda_disjoint,
        )
        self._explainer = CDEAExplainer(
            model=model,
            unit_space=unit_space,
            selector=selector,
            base_evidence=base_evidence,
            allocator=allocator,
            objective=objective,
            device=device,
        )
        self.unit_space = unit_space
        self.model = model

    def explain(self, x: torch.Tensor, env: EnvBatch) -> Explanation:
        """
        Run shift-aware explanation on a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, H, W).
        env : EnvBatch
            In-distribution and out-of-distribution views.
            e.g. EnvBatch(xs=[x_id, x_ood], env_ids=[0, 1])

        Returns
        -------
        Explanation
            .masks["robust"]   — robust evidence mask, shape (B, H, W)
            .masks["shortcut"] — shortcut evidence mask, shape (B, H, W)
            .metrics           — dict of rob_mean, rob_var, sho_gap, disjoint, sparse
        """
        return self._explainer.explain(x, env=env)
