"""FORMULATION.md section 13."""

from __future__ import annotations

import ast
from pathlib import Path

import torch
import torch.nn as nn

from core.grid import deletion_baseline, pool_sum
from core.types import HypothesisSet
from cdea.allocation import AllocateConfig, allocate
from cdea.first_order import first_order_masks, unit_gradient
from cdea.payoffs import shared_log_odds, unique_log_odds
from cdea.sinkhorn import hard_top_mass, sinkhorn

ROOT = Path(__file__).resolve().parents[1]


class CellLinear(nn.Module):
    """Logits are a linear map of per-cell channel means."""

    def __init__(self, weight: torch.Tensor, grid: tuple[int, int]):
        super().__init__()
        self.weight = nn.Parameter(weight.float(), requires_grad=False)
        self.grid_h, self.grid_w = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.mean(dim=1)
        totals = pool_sum(spatial, self.grid_h, self.grid_w)
        counts = pool_sum(torch.ones_like(spatial), self.grid_h, self.grid_w).clamp_min(1)
        return (totals / counts) @ self.weight.t()


def _hypotheses(ids: torch.Tensor) -> HypothesisSet:
    return HypothesisSet(ids=ids, mask=torch.ones_like(ids, dtype=torch.bool))


def test_sinkhorn_hits_the_marginals() -> None:
    torch.manual_seed(0)
    theta = torch.randn(4, 6, 49)
    budget = 0.05 * 49
    empty = 49 - 5 * budget
    target = torch.tensor([budget] * 5 + [empty])
    plan = sinkhorn(theta, target, iters=20)
    assert torch.allclose(plan.sum(-1), target.expand(4, -1), atol=1e-3)
    assert torch.allclose(plan.sum(-2), torch.ones(4, 49), atol=1e-3)


def test_fractional_budget_is_kept() -> None:
    scores = torch.randn(2, 49)
    projected = hard_top_mass(scores, 2.45)
    assert torch.allclose(projected.sum(-1), torch.full((2,), 2.45), atol=1e-3)
    assert bool((projected >= 0).all() and (projected <= 1).all())


def test_first_order_matches_grad_times_input_gap() -> None:
    torch.manual_seed(1)
    grid = (4, 4)
    weight = torch.randn(3, 16)
    model = CellLinear(weight, grid).eval()
    x = torch.rand(2, 3, 32, 32)
    ids = torch.tensor([[0, 1, 2], [1, 0, 2]])
    hypotheses = _hypotheses(ids)
    got = unit_gradient(model, x, hypotheses, *grid)
    x_in = x.detach().requires_grad_(True)
    contrast = unique_log_odds(model(x_in), ids)
    grad = torch.autograd.grad(contrast[:, 0].sum(), x_in)[0]
    base = deletion_baseline(x, *grid)
    manual = pool_sum((grad * (x - base)).sum(1), *grid)
    assert torch.allclose(got[:, 0], manual, atol=1e-5)
    masks = first_order_masks(model, x, hypotheses, *grid, area=2 / 16)
    top = manual.topk(2, dim=-1).indices
    for image in range(2):
        assert float(masks[image, 0, top[image]].min()) > 0.5


def test_covering_every_class_has_no_shared_player() -> None:
    torch.manual_seed(2)
    weight = torch.randn(2, 4)
    model = CellLinear(weight, (2, 2)).eval()
    x = torch.rand(2, 3, 16, 16)
    hypotheses = _hypotheses(torch.tensor([[0, 1], [1, 0]]))
    out = allocate(
        model, x, hypotheses, 2, 2, area=0.25,
        cfg=AllocateConfig(steps=2, lr=0.1, init="uniform", shared=True),
    )
    assert out.shared is None
    assert out.unique.shape == (2, 2, 4)
    logits = model(x)
    try:
        shared_log_odds(logits, hypotheses.ids)
    except ValueError:
        return
    raise AssertionError("shared log-odds must be undefined when H is every class")


def test_a_unit_useful_to_both_classes_goes_to_the_shared_player() -> None:
    grid = (4, 4)
    units = 16
    weight = torch.zeros(3, units)
    weight[0, 0] = 8
    weight[1, 1] = 8
    weight[0, 5] = 8
    weight[1, 5] = 8
    model = CellLinear(weight, grid).eval()
    spatial = torch.full((1, 32, 32), 0.15)
    def paint(cell, value):
        y, x = divmod(cell, 4)
        spatial[:, y * 8:(y + 1) * 8, x * 8:(x + 1) * 8] = value
    paint(0, 1.0)
    paint(1, 1.0)
    paint(5, 1.0)
    image = spatial.unsqueeze(1).expand(1, 3, 32, 32).contiguous()
    hypotheses = _hypotheses(torch.tensor([[0, 1]]))
    out = allocate(
        model, image, hypotheses, *grid, area=1 / 16,
        cfg=AllocateConfig(steps=40, lr=0.3, init="uniform"),
    )
    assert out.shared is not None
    assert int(out.unique[0, 0].argmax()) == 0
    assert int(out.unique[0, 1].argmax()) == 1
    assert int(out.shared[0].argmax()) == 5


def test_red_and_blue_squares_separate() -> None:
    """A red square and a blue square are evidence for different classes."""
    grid = (4, 4)
    weight = torch.zeros(3, 16)
    weight[0, 0] = 6   # red channel is not separate; the cell mean carries the cue
    weight[1, 3] = 6
    model = CellLinear(weight, grid).eval()
    image = torch.full((1, 3, 32, 32), 0.2)
    image[:, 0, :8, :8] = 1
    image[:, 2, :8, 24:32] = 1
    hypotheses = _hypotheses(torch.tensor([[0, 1]]))
    out = allocate(
        model, image, hypotheses, *grid, area=1 / 16,
        cfg=AllocateConfig(steps=30, lr=0.3, init="uniform", shared=False),
    )
    assert int(out.unique[0, 0].argmax()) == 0
    assert int(out.unique[0, 1].argmax()) == 3


def test_chunks_match_the_whole_sample() -> None:
    torch.manual_seed(3)
    weight = torch.randn(4, 4)
    model = CellLinear(weight, (2, 2)).eval()
    x = torch.rand(4, 3, 16, 16)
    hypotheses = _hypotheses(torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]]))
    cfg = AllocateConfig(steps=3, lr=0.1, init="uniform")
    whole = allocate(model, x, hypotheses, 2, 2, 0.25, cfg)
    parts = [
        allocate(model, x[i:i + 2], HypothesisSet(ids=hypotheses.ids[i:i + 2], mask=hypotheses.mask[i:i + 2]),
                 2, 2, 0.25, cfg)
        for i in (0, 2)
    ]
    stacked = torch.cat([part.unique for part in parts], dim=0)
    assert torch.allclose(whole.unique, stacked, atol=1e-5)


def test_cdea_does_not_import_the_scorer() -> None:
    for path in (ROOT / "cdea").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in {"evaluation", "baselines", "analysis"}
