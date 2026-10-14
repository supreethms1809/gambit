"""The decomposition game: toys, payoff units, determinism, and the assignment LP.

The toy classifier reads one number per cell: the amplitude of a pixel
checkerboard, demodulated. Blur deletion removes the checkerboard, so deleting
a cell sets its feature to (almost exactly) 0 and nothing else moves. Per-cell
features go to location-specific channels of the last convolution, then a
global average pool and a head, so Grad-CAM on it is CAM: an exact per-cell
class contribution.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional

import torch
import torch.nn as nn

from baselines.decomposition import (
    assign,
    cell_scores,
    gradient_assign,
    min_split,
    occlusion_assign,
    occlusion_drops,
)
from cdea.decompose import DecomposeConfig, decompose
from cdea.null import null_drops, null_unit_masks
from cdea.payoffs import class_drops, decomposition_payoffs, smooth_abs, smooth_min
from core.grid import delete, pool_sum
from core.types import HypothesisSet

SIZE = 32
GRID = (4, 4)
UNITS = GRID[0] * GRID[1]


def _checker(size: int = SIZE) -> torch.Tensor:
    i = torch.arange(size)
    return ((i.view(-1, 1) + i.view(1, -1)) % 2 * 2 - 1).float()


class CellToy(nn.Module):
    """Logits ``W f + b``, or ``head(f)``, where ``f_r`` is cell r's checkerboard amplitude."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor,
                 head: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.register_buffer("weight", weight.float())
        self.register_buffer("bias", bias.float())
        self.register_buffer("checker", _checker())
        self.register_buffer("location", torch.eye(UNITS).view(UNITS, UNITS, 1).reshape(UNITS, *GRID))
        self.conv = nn.Conv2d(UNITS, UNITS, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(UNITS).view(UNITS, UNITS, 1, 1))
        self.head = head

    def features(self, x: torch.Tensor) -> torch.Tensor:
        demod = 2.0 * (x[:, 0] - 0.5) * self.checker
        counts = pool_sum(torch.ones_like(demod), *GRID)
        return pool_sum(demod, *GRID) / counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        channels = self.conv(f.view(-1, UNITS, 1, 1) * self.location.unsqueeze(0))
        pooled = channels.mean(dim=(-1, -2)) * UNITS
        if self.head is not None:
            return self.head(pooled)
        return pooled @ self.weight.t() + self.bias


def toy_image(amplitude: torch.Tensor, batch: int = 1) -> torch.Tensor:
    """Channel 0 is ``0.5 + 0.5 * a_r * checker`` in cell r; the other channels are 0.5."""
    cell = SIZE // GRID[0]
    amp = amplitude.view(*GRID).repeat_interleave(cell, 0).repeat_interleave(cell, 1)
    x = torch.full((batch, 3, SIZE, SIZE), 0.5)
    x[:, 0] = 0.5 + 0.5 * amp * _checker()
    return x


def _pair(model: nn.Module, x: torch.Tensor) -> HypothesisSet:
    with torch.no_grad():
        ids = model(x).topk(2, dim=-1).indices
    return HypothesisSet(ids=ids, mask=torch.ones_like(ids, dtype=torch.bool))


def _cells(*pairs) -> torch.Tensor:
    w = torch.zeros(3, UNITS)
    for cls, cell, value in pairs:
        w[cls, cell] = value
    return w


K_ONLY, L_ONLY, SHARED, IDLE = 0, 3, 12, 15


def additive_toy():
    weight = _cells((0, K_ONLY, 1.0), (1, L_ONLY, 1.0), (0, SHARED, 1.0), (1, SHARED, 1.0))
    model = CellToy(weight, torch.tensor([0.5, 0.0, -5.0])).eval()
    return model, toy_image(torch.ones(UNITS))


def _argmax_cells(maps: torch.Tensor) -> list[int]:
    return [int(maps[0, p].argmax()) for p in range(maps.shape[1])]


def test_toy_deletion_removes_exactly_one_cell() -> None:
    model, x = additive_toy()
    mask = torch.zeros(1, UNITS)
    mask[0, K_ONLY] = 1.0
    f = model.features(delete(x, mask, *GRID))
    assert abs(float(f[0, K_ONLY])) < 1e-4
    others = torch.cat([f[0, :K_ONLY], f[0, K_ONLY + 1:]])
    assert torch.allclose(others, torch.ones_like(others), atol=1e-4)


# 1 -------------------------------------------------------------------------

def test_additive_toy_every_method_recovers_the_three_roles() -> None:
    model, x = additive_toy()
    h = _pair(model, x)
    assert h.ids[0].tolist() == [0, 1]
    area = 1.0 / UNITS
    out = decompose(model, x, h, *GRID, area, DecomposeConfig(steps=60))
    cdea_maps = torch.stack([out.unique_k, out.unique_l, out.shared], dim=1)
    assert _argmax_cells(cdea_maps) == [K_ONLY, L_ONLY, SHARED]

    occ = occlusion_assign(model, x, h.ids, *GRID, area)
    assert _argmax_cells(occ) == [K_ONLY, L_ONLY, SHARED]

    from base_evidence.gradcam_regions import GradCAMRegionsProvider

    evidence = GradCAMRegionsProvider(*GRID).explain(x, model, h)
    split = torch.stack(min_split(evidence[:, 0], evidence[:, 1]), dim=1)
    assert _argmax_cells(split) == [K_ONLY, L_ONLY, SHARED]


# 2 -------------------------------------------------------------------------

SUPPRESSOR, WEAK_K, ONLY_L, BOTH = 0, 3, 12, 15


def suppression_toy():
    weight = _cells((0, SUPPRESSOR, 1.0), (1, SUPPRESSOR, -1.0), (0, WEAK_K, 0.6),
                    (1, ONLY_L, 1.0), (0, BOTH, 0.5), (1, BOTH, 0.5))
    model = CellToy(weight, torch.tensor([0.0, 1.0, -5.0])).eval()
    return model, toy_image(torch.ones(UNITS))


def test_suppressing_the_rival_is_not_unique_evidence() -> None:
    """D3: a cell that raises z_k and lowers z_l. Margin attribution ranks it first; U_k does not take it."""
    model, x = suppression_toy()
    h = _pair(model, x)
    assert h.ids[0].tolist() == [0, 1]
    out = decompose(model, x, h, *GRID, 1.0 / UNITS, DecomposeConfig(steps=100))
    assert int(out.unique_k[0].argmax()) == WEAK_K
    assert float(out.unique_k[0, SUPPRESSOR]) < 0.5

    from baselines.margin import margin_gradcam

    margin = margin_gradcam(model, x, h)
    per_cell = pool_sum(margin, *GRID)
    assert int(per_cell[0].argmax()) == SUPPRESSOR


# 3 -------------------------------------------------------------------------

EITHER_A, EITHER_B, DECOY, L_CELL, S_CELL = 0, 3, 12, 15, 6


def interaction_toy():
    """z_k needs *either* a or b present, so only deleting both moves it; a decoy cell is additive."""
    tau = 0.1

    def head(f: torch.Tensor) -> torch.Tensor:
        either = tau * torch.logsumexp(torch.stack([f[:, EITHER_A], f[:, EITHER_B]], dim=-1) / tau, dim=-1)
        z_k = 2.0 * either + 0.8 * f[:, DECOY] + 0.5 * f[:, S_CELL]
        z_l = 1.0 * f[:, L_CELL] + 0.5 * f[:, S_CELL] + 1.0
        z_o = torch.full_like(z_k, -5.0)
        return torch.stack([z_k, z_l, z_o], dim=-1)

    model = CellToy(torch.zeros(3, UNITS), torch.zeros(3), head=head).eval()
    return model, toy_image(torch.ones(UNITS))


def test_joint_search_finds_redundant_evidence_that_single_cells_miss() -> None:
    model, x = interaction_toy()
    h = _pair(model, x)
    assert h.ids[0].tolist() == [0, 1]
    area = 2.0 / UNITS
    occ = occlusion_assign(model, x, h.ids, *GRID, area)
    occ_k = set(torch.nonzero(occ[0, 0] > 0.5).flatten().tolist())
    assert occ_k != {EITHER_A, EITHER_B}
    out = decompose(model, x, h, *GRID, area, DecomposeConfig(steps=100))
    top_k = set(out.unique_k[0].topk(2).indices.tolist())
    assert top_k == {EITHER_A, EITHER_B}


# 4 -------------------------------------------------------------------------

def test_smooth_abs_and_min_stay_within_eps() -> None:
    t = torch.linspace(-3, 3, 101, dtype=torch.float64)
    assert bool((smooth_abs(t) - t.abs()).max() <= 1e-3 + 1e-12)
    a, b = torch.randn(200, dtype=torch.float64), torch.randn(200, dtype=torch.float64)
    gap = torch.minimum(a, b) - smooth_min(a, b)
    assert bool((gap >= 0).all() and (gap <= 0.5e-3 + 1e-12).all())


def test_null_is_subtracted_and_an_empty_mask_scores_minus_the_null() -> None:
    model, x = additive_toy()
    h = _pair(model, x)
    null = null_drops(model, x, h.ids, *GRID, 1.0 / UNITS)
    full = model(x)
    empty = full.unsqueeze(1).expand(-1, 3, -1)
    drops = class_drops(full, empty, h.ids)
    assert torch.equal(drops, torch.zeros_like(drops))
    excess = drops - null.unsqueeze(1)
    assert torch.allclose(excess[:, 0], -null)
    payoff = decomposition_payoffs(drops, null)
    assert torch.allclose(payoff[:, 0], -null[:, 0] - smooth_abs(-null[:, 1]))
    assert torch.allclose(payoff[:, 2], smooth_min(-null[:, 0], -null[:, 1]))


def test_null_masks_have_the_area_and_do_not_depend_on_the_batch() -> None:
    masks = null_unit_masks(49, 0.05)
    assert masks.shape == (8, 49)
    assert torch.allclose(masks.sum(-1), torch.full((8,), 0.05 * 49))
    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0, 1, UNITS))])
    h2 = _pair(model, x2)
    both = null_drops(model, x2, h2.ids, *GRID, 0.125)
    one = null_drops(model, x2[1:], h2.ids[1:], *GRID, 0.125)
    assert torch.allclose(both[1:], one, atol=1e-6)


# 6 -------------------------------------------------------------------------

def test_decomposition_is_chunk_invariant_and_repeats() -> None:
    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0.2, 1, UNITS))])
    h2 = _pair(model, x2)
    cfg = DecomposeConfig(steps=8)
    whole = decompose(model, x2, h2, *GRID, 0.125, cfg)
    again = decompose(model, x2, h2, *GRID, 0.125, cfg)
    part = decompose(model, x2[1:], HypothesisSet(ids=h2.ids[1:], mask=h2.mask[1:]), *GRID, 0.125, cfg)
    assert torch.equal(whole.transport, again.transport)
    assert torch.allclose(whole.transport[1:], part.transport, atol=1e-5)


def test_decomposition_marginals_and_exclusivity() -> None:
    model, x = additive_toy()
    out = decompose(model, x, _pair(model, x), *GRID, 0.125, DecomposeConfig(steps=8))
    assert torch.allclose(out.transport.sum(-2), torch.ones(1, UNITS), atol=1e-3)
    assert torch.allclose(out.transport.sum(-1)[0, :3], torch.full((3,), 0.125 * UNITS), atol=1e-3)
    assert out.prior_split is not None and out.initial.shape == (1, 3, UNITS)


def test_ablation_switches_run() -> None:
    model, x = additive_toy()
    h = _pair(model, x)
    for cfg in (DecomposeConfig(steps=3, independent=True), DecomposeConfig(steps=3, init="uniform"),
                DecomposeConfig(steps=3, init="mean"), DecomposeConfig(steps=3, binary_masks=False),
                DecomposeConfig(steps=3, boundary_shift=False), DecomposeConfig(steps=3, null=False)):
        out = decompose(model, x, h, *GRID, 0.125, cfg)
        assert bool(torch.isfinite(out.transport).all())
    assert decompose(model, x, h, *GRID, 0.125, DecomposeConfig(steps=1, null=False)).null.abs().max() == 0


def test_decomposition_repeats_exactly_on_an_accelerator() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from core.device import make_deterministic

    device = torch.device("cuda")
    make_deterministic(device)
    model, x = additive_toy()
    model = model.to(device)
    x = torch.cat([x, toy_image(torch.linspace(0.2, 1, UNITS))]).to(device)
    h = _pair(model, x)
    first = decompose(model, x, h, *GRID, 0.125, DecomposeConfig(steps=10))
    second = decompose(model, x, h, *GRID, 0.125, DecomposeConfig(steps=10))
    assert torch.equal(first.transport, second.transport)


# 7 -------------------------------------------------------------------------

def _brute_force(scores: torch.Tensor, budget: int) -> float:
    """Best total over hard assignments: each player takes ``budget`` distinct cells."""
    players, units = scores.shape
    best = float("-inf")
    for owner in itertools.product(range(players + 1), repeat=units):
        if all(owner.count(p) == budget for p in range(players)):
            total = sum(float(scores[owner[r], r]) for r in range(units) if owner[r] < players)
            best = max(best, total)
    return best


def test_assignment_keeps_budgets_and_matches_brute_force() -> None:
    torch.manual_seed(0)
    for _ in range(5):
        scores = torch.randn(1, 3, 6)
        plan = assign(scores, 2.0)
        assert torch.allclose(plan.sum(-1), torch.full((1, 3), 2.0), atol=1e-6)
        assert bool((plan.sum(-2) <= 1 + 1e-6).all())
        assert abs(float((plan * scores).sum()) - _brute_force(scores[0], 2)) < 1e-6
    fractional = assign(torch.randn(2, 3, 49), 2.45)
    assert torch.allclose(fractional.sum(-1), torch.full((2, 3), 2.45), atol=1e-6)


def test_cell_scores_follow_the_definitions() -> None:
    g = torch.tensor([[[1.0, 0.0, 1.0, -0.5], [0.0, 1.0, 1.0, 0.5]]])
    s = cell_scores(g)
    assert s[0, 0].tolist() == [1.0, -1.0, 0.0, -1.0]
    assert s[0, 1].tolist() == [-1.0, 1.0, 0.0, 0.0]
    assert s[0, 2].tolist() == [0.0, 0.0, 1.0, -0.5]


def test_occlusion_drops_read_the_toy_exactly() -> None:
    model, x = additive_toy()
    h = _pair(model, x)
    g = occlusion_drops(model, x, h.ids, *GRID, null=False)
    expected = torch.zeros(1, 2, UNITS)
    expected[0, 0, K_ONLY] = expected[0, 0, SHARED] = 1.0
    expected[0, 1, L_ONLY] = expected[0, 1, SHARED] = 1.0
    assert torch.allclose(g, expected, atol=1e-4)
    plan = gradient_assign(model, x, h.ids, *GRID, 1.0 / UNITS)
    assert _argmax_cells(plan) == [K_ONLY, L_ONLY, SHARED]


def test_min_split_shares_the_overlap() -> None:
    k = torch.tensor([[2.0, 1.0, 1.0, 0.0]])
    l = torch.tensor([[0.0, 1.0, 1.0, -3.0]])
    u_k, u_l, s = min_split(k, l)
    assert torch.allclose(s, torch.tensor([[0.0, 0.25, 0.25, 0.0]]))
    assert torch.allclose(u_k, torch.tensor([[0.5, 0.0, 0.0, 0.0]]))
    assert torch.allclose(u_l, torch.tensor([[0.0, 0.25, 0.25, 0.0]]))
