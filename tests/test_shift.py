"""Shift payoffs and the allocation. SHIFT_FORMULATION.md."""

from __future__ import annotations

import ast
from pathlib import Path

import torch
import torch.nn as nn

from cdea.shift import (
    ShiftConfig,
    allocate_shift,
    gap_units,
    log_odds,
    transplant_payoff,
)
from scripts.check_degenerate import d5_marginals


class _Cell(nn.Module):
    """Log-odds rise with cell 0 in every view and with cell 3 only where it is painted."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cell0 = x[:, :, :8, :8].mean(dim=(1, 2, 3))
        cell3 = x[:, :, :8, 24:32].mean(dim=(1, 2, 3))
        score = self.scale * (4.0 * cell0 + 4.0 * cell3)
        return torch.stack([score, torch.zeros_like(score)], dim=1)


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        weight = torch.zeros(3, 8, 8)
        weight[:, :4, :4] = 1.0
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = (x * self.weight).sum(dim=(1, 2, 3)) + self.anchor * 0
        return torch.stack([score, torch.zeros_like(score)], dim=1)


def _views():
    x_id = torch.zeros(2, 3, 32, 32)
    x_ood = torch.zeros(2, 3, 32, 32)
    x_id[:, :, :8, :8] = 1
    x_ood[:, :, :8, :8] = 1
    x_id[:, :, :8, 24:32] = 1
    return x_id, x_ood


def test_identical_environments_have_zero_shortcut_payoff():
    model = _Cell().eval()
    x, _ = _views()
    y = model(x).argmax(-1)
    mask = torch.zeros(2, 16)
    mask[:, 3] = 1
    payoff = transplant_payoff(model, [x, x], y, mask, 4, 4)
    assert torch.allclose(payoff, torch.zeros_like(payoff), atol=1e-6)


def test_a_patch_that_differs_goes_to_the_shortcut_player():
    model = _Cell().eval()
    x_id, x_ood = _views()
    out = allocate_shift(
        model, [x_id, x_ood], 4, 4, 0.25,
        ShiftConfig(steps=40, lr=0.2, init="uniform"),
    )
    assert int(out.shortcut.mean(0).argmax()) == 3
    assert int(out.robust.mean(0).argmax()) == 0


def test_marginals_and_chunks():
    model = _Cell().eval()
    x_id, x_ood = _views()
    x_id = x_id.repeat(2, 1, 1, 1)
    x_ood = x_ood.repeat(2, 1, 1, 1)
    cfg = ShiftConfig(steps=3, lr=0.1, init="uniform")
    whole = allocate_shift(model, [x_id, x_ood], 4, 4, 0.25, cfg)
    assert d5_marginals(whole.transport, whole.row_target)
    left = allocate_shift(model, [x_id[:2], x_ood[:2]], 4, 4, 0.25, cfg)
    right = allocate_shift(model, [x_id[2:], x_ood[2:]], 4, 4, 0.25, cfg)
    joined = torch.cat([left.shortcut, right.shortcut], dim=0)
    assert torch.allclose(joined, whole.shortcut, atol=1e-5)


def test_gap_attribution_is_complete_on_a_linear_model():
    model = _Linear().eval()
    x_e = torch.zeros(1, 3, 8, 8)
    x_id = torch.zeros(1, 3, 8, 8)
    x_id[:, :, :4, :4] = 0.5
    x_e[:, :, :4, :4] = 0.1
    y = model(x_id).argmax(-1)
    units = gap_units(model, [x_id, x_e], y, 2, 2, steps=4)
    gap = (log_odds(model(x_id), y) - log_odds(model(x_e), y))
    assert torch.allclose(units.sum(-1), gap, atol=1e-4)


def test_shift_checkpoint_names_stay():
    from models.build import paper_checkpoint_name

    assert paper_checkpoint_name(
        "waterbirds", "resnet50", True, False, 15, 1e-4, 0, "imagenet",
    ) == "waterbirds_resnet50_pt_ft_ep15_lr0.0001_imagenet_seed0.pt"
    assert paper_checkpoint_name(
        "colored_mnist", "vit_b_16", True, False, 15, 1e-4, 1, "imagenet",
    ) == "colored_mnist_vit_b_16_pt_ft_ep15_lr0.0001_imagenet_seed1.pt"


def test_shift_degenerate_routes():
    from scripts.check_degenerate import d2_soft_hard_gap, d3_overshoot_share, d4_beats_translation

    assert d2_soft_hard_gap(torch.tensor(1.0), torch.tensor(1.2))
    assert not d2_soft_hard_gap(torch.tensor(1.0), torch.tensor(2.0))
    assert d4_beats_translation(torch.tensor([0.4, 0.2]), torch.tensor([0.1, 0.0]))
    assert d3_overshoot_share(torch.tensor([0.0, 0.5])) == 0.25


def test_shift_import_boundary():
    root = Path(__file__).resolve().parents[1] / "cdea" / "shift.py"
    tree = ast.parse(root.read_text())
    banned = {"evaluation", "baselines", "analysis"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        else:
            continue
        assert banned.isdisjoint(names)
