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


def test_translate_always_moves_the_mask():
    mask = torch.zeros(4, 16)
    mask[:, 0] = 1
    for seed in range(20):
        rolled = random_translate(mask, torch.Generator().manual_seed(seed), 4, 4)
        assert not torch.equal(rolled, mask), f"null equals the mask at seed {seed}"
        assert torch.allclose(rolled.sum(dim=-1), mask.sum(dim=-1))


def test_impute_pins_known_pixels():
    image = torch.rand(1, 3, 8, 8)
    keep = torch.zeros(1, 1, 8, 8)
    keep[:, :, 2:6, 2:6] = 1
    for iters in (5, None):
        filled = noisy_linear_impute(image, keep, iters=iters, noise=0.0, gen=torch.Generator().manual_seed(0))
        assert torch.allclose(filled[:, :, 2:6, 2:6], image[:, :, 2:6, 2:6])
        assert filled.min() >= 0 and filled.max() <= 1


def _road_reference(image, keep):
    """ROAD's NoisyLinearImputer without noise, road_evaluation e6b9987, road/imputations.py.

    ``setup_sparse_system`` transcribed (one channel-stacked right-hand side,
    weights renormalised at the border), solved with scipy's ``spsolve``.
    """
    import numpy as np
    from scipy.sparse import csc_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve

    weights = [((1, 1), 1 / 12), ((0, 1), 1 / 6), ((-1, 1), 1 / 12), ((1, -1), 1 / 12),
               ((0, -1), 1 / 6), ((-1, -1), 1 / 12), ((1, 0), 1 / 6), ((-1, 0), 1 / 6)]
    img = image.double().numpy()
    mask = keep.numpy()
    height, width = mask.shape
    flat_mask = mask.flatten()
    flat_img = img.reshape(img.shape[0], -1)
    indices = np.argwhere(flat_mask == 0).flatten()
    to_var = np.zeros(len(flat_mask), dtype=int)
    to_var[indices] = np.arange(len(indices))
    a = lil_matrix((len(indices), len(indices)))
    b = np.zeros((len(indices), img.shape[0]))
    total = np.ones(len(indices))
    for (dy, dx), w in weights:
        row, col = indices // width + dy, indices % width + dx
        valid = (row >= 0) & (col >= 0) & (row < height) & (col < width)
        coords = (indices + dy * width + dx)[valid]
        ids = np.argwhere(valid).flatten()
        has = flat_mask[coords] > 0.5
        b[ids[has], :] -= w * flat_img[:, coords[has]].T
        a[ids[~has], to_var[coords[~has]]] = w
        total[~valid] -= w
    a[np.arange(len(indices)), np.arange(len(indices))] = -total
    out = flat_img.copy()
    out[:, indices] = spsolve(csc_matrix(a), b).T
    return torch.from_numpy(out.reshape(img.shape))


def test_converged_impute_matches_roads_direct_solve():
    import pytest

    pytest.importorskip("scipy")
    torch.manual_seed(0)
    image = torch.nn.functional.avg_pool2d(torch.rand(1, 3, 224, 224), 9, 1, 4)
    holes = []
    for r0, r1, c0, c1 in ((96, 128, 96, 128), (56, 168, 56, 168), (0, 64, 0, 64)):
        keep = torch.ones(224, 224)
        keep[r0:r1, c0:c1] = 0
        holes.append(keep)
    cells = torch.ones(7, 7)
    cells[3, 3] = 0
    holes.append(1 - torch.nn.functional.interpolate(cells.view(1, 1, 7, 7), size=(224, 224))[0, 0])
    for keep in holes:
        filled = noisy_linear_impute(image, keep.view(1, 1, 224, 224), iters=None, noise=0.0,
                                     gen=torch.Generator().manual_seed(0))
        assert float((filled[0].double() - _road_reference(image[0], keep)).abs().max()) < 1e-3


def test_converged_impute_fills_a_flat_image_with_its_value():
    # 24 Jacobi sweeps from black left a 32 px cell of a flat 0.5 image at 0.17.
    image = torch.full((1, 3, 224, 224), 0.5)
    keep = torch.ones(1, 1, 224, 224)
    keep[..., 56:168, 56:168] = 0
    filled = noisy_linear_impute(image, keep, iters=None, noise=0.0, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(filled, image, atol=1e-3)


def test_impute_stays_finite_inside_a_solid_mask():
    # An earlier fill divided the neighbour mean by the kept-neighbour mass,
    # which amplified interior fill ~1e6x per step to inf/NaN.
    image = torch.rand(2, 3, 32, 32)
    keep = torch.ones(2, 1, 32, 32)
    keep[:, :, 8:24, 8:24] = 0
    filled = noisy_linear_impute(image, keep, iters=None, noise=0.01,
                                 gen=torch.Generator().manual_seed(0))
    assert torch.isfinite(filled).all()
    assert filled.min() >= 0 and filled.max() <= 1
    assert torch.allclose(filled[keep.expand(2, 3, 32, 32) == 1],
                          image[keep.expand(2, 3, 32, 32) == 1])


def test_impute_is_identity_when_nothing_is_masked():
    image = torch.rand(2, 3, 16, 16)
    keep = torch.ones(2, 1, 16, 16)
    filled = noisy_linear_impute(image, keep, iters=None, noise=0.01,
                                 gen=torch.Generator().manual_seed(0))
    assert torch.allclose(filled, image)


def test_spearman_and_paired():
    increasing = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(spearman(increasing, increasing * 2 + 1), torch.tensor([1.0]))
    stats = paired([1.0, 2.0], [0.0, 0.0])
    assert stats["n"] == 2
    assert stats["delta"] == pytest_approx(1.5)


def test_spearman_ties_share_rank_and_constants_report_zero():
    tied_a = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
    tied_b = torch.tensor([[1.0, 1.0, 9.0, 9.0]])
    assert torch.allclose(spearman(tied_a, tied_b), torch.tensor([1.0]))
    assert torch.allclose(spearman(tied_a, tied_a), torch.tensor([1.0]))
    const = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    varied = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(spearman(const, varied), torch.tensor([0.0]))
    assert torch.allclose(spearman(const, const), torch.tensor([0.0]))


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
