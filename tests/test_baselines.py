"""Baseline harness: shared hypotheses, one conversion, toy recovery, library agreement."""

from __future__ import annotations

import importlib.metadata as metadata

import pytest
import torch
import torch.nn as nn

import numpy as np

from baselines.adapter import adapt_scores, budget_or_floor, random_floor, row_failed
from baselines.crosscheck import SPEARMAN_MIN, gradcam_spearman, ig_spearman
from baselines.hypotheses import foil_pair, shared_hypotheses
from baselines.margin import margin_gradcam, margin_integrated_gradients
from baselines.toy import (
    BOX,
    SIZE,
    make_both_cues,
    make_class_batch,
    make_region_batch,
    region_accuracy,
    region_boxes,
    train_region_model,
)
from baselines.versions import CAPTUM_VERSION, GRAD_CAM_VERSION
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from base_evidence.library_adapters import CaptumRegionsProvider
from core.types import HypothesisSet
from evaluation.masks import mass_in, to_budget_mask
from instantiations.contrastive.allocator import OptimizationAllocator
from instantiations.contrastive.objective import ContrastiveObjective
from modality.grid_regions import VisionGridUnitSpace


def _recall(attr: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Share of the known square that lands in the top-a mask. ``a`` is the square's area."""
    if attr.ndim == 2:
        attr = attr.unsqueeze(0)
    if box.ndim == 2:
        box = box.unsqueeze(0)
    if box.shape[0] == 1 and attr.shape[0] != 1:
        box = box.expand(attr.shape[0], -1, -1)
    fraction = float(BOX * BOX) / float(SIZE * SIZE)
    chosen, failed = budget_or_floor(attr, fraction, seed=0)
    assert not bool(failed.any())
    hit = (chosen * box).flatten(1).sum(dim=-1)
    return hit / box.flatten(1).sum(dim=-1).clamp_min(1)


def test_library_versions_are_pinned() -> None:
    assert metadata.version("grad-cam") == GRAD_CAM_VERSION
    assert metadata.version("captum") == CAPTUM_VERSION


def test_shared_hypotheses_use_rank_order_and_mask_the_extra_slots() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0], [5.0, 0.0, 4.0, 1.0]])
    hypotheses = shared_hypotheses(logits, k=3)
    kept, foil = foil_pair(hypotheses)
    assert kept.tolist() == [1, 0]
    assert foil.tolist() == [3, 2]
    short = shared_hypotheses(torch.randn(2, 3), k=5)
    assert short.ids.shape == (2, 5)
    assert bool(short.mask[:, :3].all())
    assert not bool(short.mask[:, 3:].any())
    with pytest.raises(ValueError):
        foil_pair(shared_hypotheses(torch.randn(2, 4), k=1))
    with pytest.raises(ValueError):
        foil_pair(shared_hypotheses(torch.randn(2, 1), k=2))


def test_constant_and_tied_maps_keep_area_and_follow_the_seed() -> None:
    scores = torch.ones(2, 20)
    first, failed = budget_or_floor(scores, 0.2, seed=0)
    second, _ = budget_or_floor(scores, 0.2, seed=1)
    assert not bool(failed.any())
    assert torch.equal(first.sum(dim=-1), torch.full((2,), 4.0))
    assert torch.equal(first.sum(dim=-1), second.sum(dim=-1))
    assert not torch.equal(first, second)
    assert torch.equal(first, budget_or_floor(scores, 0.2, seed=0)[0])


def test_all_negative_maps_rank_without_clamping() -> None:
    scores = torch.tensor([[-5.0, -0.1, -4.0, -3.0]])
    masks, failed = budget_or_floor(scores, 0.25, seed=1)
    assert not bool(failed.any())
    assert masks.view(-1)[1] == 1
    assert masks.sum() == 1


def test_nan_and_empty_rows_become_the_floor_and_stay_in_the_batch() -> None:
    scores = torch.tensor([[1.0, 0.0, 0.4, 0.2], [float("nan"), 1.0, 0.0, 0.3]])
    masks, failed = budget_or_floor(scores, 0.5, seed=0)
    assert failed.tolist() == [False, True]
    assert masks.shape[0] == scores.shape[0]
    finite = scores.clone()
    finite[1] = 0
    assert torch.equal(masks[0], adapt_scores(finite, 0.5, seed=0)[0])
    assert torch.equal(masks[1], random_floor(scores, 0.5, seed=0)[1])
    empty = torch.zeros(2, 0)
    empty_masks, empty_failed = budget_or_floor(empty, 0.5, seed=0)
    assert empty_masks.shape == (2, 0)
    assert bool(empty_failed.all())
    assert bool(row_failed(torch.tensor([[1.0, float("inf")]])).all())


def test_cdea_masks_score_the_same_through_the_adapter() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 4),
    )
    model.eval()
    unit_space = VisionGridUnitSpace(4, 4)
    allocator = OptimizationAllocator(
        ContrastiveObjective(
            lambda_suff=1.0, lambda_margin=1.0, lambda_sparse=0.05, lambda_overlap=0.2
        ),
        num_steps=8,
        lr=0.4,
    )
    x = torch.rand(2, 3, 16, 16)
    hypotheses = HypothesisSet(
        ids=torch.tensor([[0, 1], [2, 0]]),
        mask=torch.ones(2, 2, dtype=torch.bool),
    )
    evidence = torch.rand(2, 2, 16).abs()
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    region = allocator.allocate(
        x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, evidence=evidence
    )["unique"][:, 0].detach()
    kwargs = dict(grid_h=4, grid_w=4, height=16, width=16)
    direct = to_budget_mask(region, 0.25, seed=0, **kwargs)
    via = adapt_scores(region, 0.25, seed=0, **kwargs)
    assert torch.equal(direct, via)
    target = torch.zeros(2, 16, 16)
    target[:, :8, :8] = 1
    assert torch.equal(mass_in(direct, target), mass_in(via, target))


def test_gradientshap_follows_the_torch_seed() -> None:
    torch.manual_seed(0)
    model = train_region_model(steps=1, seed=0)
    x, _ = make_region_batch(2, seed=4)
    hypotheses = HypothesisSet(ids=torch.zeros(2, 1, dtype=torch.long), mask=torch.ones(2, 1, dtype=torch.bool))
    provider = CaptumRegionsProvider("gradientshap", 4, 4, n_samples=2)

    def once(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return provider.explain(x, model, hypotheses).detach()

    assert torch.allclose(once(0), once(0))
    assert not torch.allclose(once(0), once(1))


def test_toy_model_recovers_the_known_regions() -> None:
    model = train_region_model(steps=40, seed=0)
    assert region_accuracy(model) >= 0.95
    red, blue = region_boxes()
    x_red = make_class_batch(4, label=0, seed=50)
    hypotheses = shared_hypotheses(model(x_red), k=2)
    assert torch.equal(hypotheses.ids[:, 0], torch.zeros(4, dtype=torch.long))

    grad = GradCAMRegionsProvider(SIZE, SIZE).explain(x_red, model, hypotheses)[:, 0]
    grad = grad.view(x_red.shape[0], SIZE, SIZE)
    ig = IntegratedGradientsRegionsProvider(SIZE, SIZE, steps=4).explain(x_red, model, hypotheses)[:, 0]
    ig = ig.view(x_red.shape[0], SIZE, SIZE)
    for attr in (grad, ig):
        on_red = _recall(attr, red)
        on_blue = _recall(attr, blue)
        assert bool((on_red > 0.5).all()), on_red.tolist()
        assert bool((on_red > on_blue).all()), (on_red - on_blue).tolist()

    both = make_both_cues(4, seed=7)
    pair = HypothesisSet(
        ids=torch.tensor([[0, 1]]).expand(4, 2).contiguous(),
        mask=torch.ones(4, 2, dtype=torch.bool),
    )
    for attr in (
        margin_gradcam(model, both, pair),
        margin_integrated_gradients(model, both, pair, steps=4),
    ):
        on_red = _recall(attr, red)
        on_blue = _recall(attr, blue)
        assert bool((on_red > on_blue).all()), (on_red - on_blue).tolist()
        assert bool((on_red > 0.5).all()), on_red.tolist()


def test_gradcam_and_ig_match_the_reference_libraries() -> None:
    model = train_region_model(steps=5, seed=1)
    x, y = make_region_batch(4, seed=9)
    gc = gradcam_spearman(model, x, y, grid_h=8, grid_w=8)
    ig = ig_spearman(model, x, y, grid_h=8, grid_w=8, steps=4)
    assert bool((gc >= SPEARMAN_MIN).all())
    assert bool((ig >= SPEARMAN_MIN).all())
