"""Shared scorer: equal area, tie breaks, empty masks, and one conversion path."""

from __future__ import annotations

import torch

from evaluation.masks import mass_in, regions_to_pixels, to_budget_mask, top_fraction_mask
from evaluation.metrics import paired, spearman
from evaluation.nulls import random_translate
from evaluation.removal import noisy_linear_impute


def test_top_fraction_has_equal_area_across_rows():
    scores = torch.rand(4, 32, 32)
    masks = top_fraction_mask(scores, 0.05, seed=0)
    area = masks.flatten(1).sum(dim=1) / (32 * 32)
    assert torch.allclose(area, torch.full((4,), 0.05), atol=1.0 / (32 * 32))


def test_ties_keep_area_and_follow_the_seed():
    scores = torch.ones(1, 10, 10)
    first = top_fraction_mask(scores, 0.1, seed=0)
    second = top_fraction_mask(scores, 0.1, seed=1)
    assert first.sum() == second.sum() == 10
    assert not torch.equal(first, second)
    assert torch.equal(first, top_fraction_mask(scores, 0.1, seed=0))


def test_strict_order_beats_the_tie_break():
    scores = torch.tensor([[0.0, 0.0, 5.0, 4.0]])
    mask = top_fraction_mask(scores, 0.5, seed=0).view(-1)
    assert mask[2] == 1 and mask[3] == 1
    assert mask[0] == 0 and mask[1] == 0


def test_zero_fraction_and_empty_map():
    assert top_fraction_mask(torch.rand(2, 8, 8), 0.0, seed=0).sum() == 0
    empty = top_fraction_mask(torch.zeros(2, 0), 0.05, seed=0)
    assert empty.shape == (2, 0)
    assert empty.sum() == 0


def test_negative_scores_still_receive_the_budget():
    scores = -torch.rand(2, 16)
    mask = top_fraction_mask(scores, 0.25, seed=3)
    assert torch.equal(mask.sum(dim=-1), torch.full((2,), 4.0))


def test_symmetry_adapter_matches_direct_path():
    region = torch.rand(2, 16)
    direct = to_budget_mask(region, 0.05, seed=0, grid_h=4, grid_w=4, height=32, width=32)
    adapted = to_budget_mask(region, 0.05, seed=0, grid_h=4, grid_w=4, height=32, width=32)
    assert torch.equal(direct, adapted)
    assert direct.shape == (2, 32, 32)


def test_mass_in_and_translate_preserve_mass():
    mask = torch.zeros(2, 16)
    mask[:, 0] = 1
    target = torch.zeros(2, 8, 8)
    target[:, :4, :4] = 1
    pixels = regions_to_pixels(mask, 4, 4, 8, 8, mode="nearest")
    inside = mass_in(pixels, target)
    assert torch.all(inside >= 0) and torch.all(inside <= 1)
    generator = torch.Generator().manual_seed(0)
    rolled = random_translate(mask, generator, 4, 4)
    assert torch.allclose(rolled.sum(dim=-1), mask.sum(dim=-1))


def test_impute_pins_known_pixels():
    image = torch.rand(1, 3, 8, 8)
    keep = torch.zeros(1, 1, 8, 8)
    keep[:, :, 2:6, 2:6] = 1
    filled = noisy_linear_impute(image, keep, iters=5, noise=0.0, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(filled[:, :, 2:6, 2:6], image[:, :, 2:6, 2:6])
    assert filled.min() >= 0 and filled.max() <= 1


def test_spearman_and_paired():
    increasing = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(spearman(increasing, increasing * 2 + 1), torch.tensor([1.0]))
    stats = paired([1.0, 2.0], [0.0, 0.0])
    assert stats["n"] == 2
    assert stats["delta"] == pytest_approx(1.5)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
