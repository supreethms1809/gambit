"""Synthetic checks for the D1–D6 thresholds. No dataset and no checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.degenerate import (
    d1_shared_blanket,
    d2_soft_versus_hard,
    d3_foil_suppression,
    d4_arbitrary_cells,
    d5_budget,
    d6_shift_masks,
    evidence_capture_ratio,
    same_area_hard,
)


def test_shared_blanket_is_open_and_a_sparse_mask_is_closed() -> None:
    blanket = torch.ones(2, 49) * 0.8
    sparse = torch.zeros(2, 49)
    sparse[:, :2] = 1.0
    assert d1_shared_blanket(blanket)["open"]
    assert not d1_shared_blanket(sparse)["open"]


def test_same_area_hard_matches_mass() -> None:
    soft = torch.tensor([[0.9, 0.4, 0.4, 0.1]])
    hard = same_area_hard(soft)
    assert int(hard.sum().item()) == 2
    assert hard[0, 0].item() == 1.0
    assert hard[0, 1].item() == 1.0


def test_soft_logit_gap_and_foil_suppression() -> None:
    assert d2_soft_versus_hard(torch.tensor([2.0]), torch.tensor([1.0]))["open"]
    assert not d2_soft_versus_hard(torch.tensor([1.1]), torch.tensor([1.0]))["open"]
    suppressed = d3_foil_suppression(
        torch.tensor([1.0]), torch.tensor([-2.0]), torch.tensor([1.0]), torch.tensor([1.0])
    )
    supported = d3_foil_suppression(
        torch.tensor([2.0]), torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([0.4])
    )
    assert suppressed["open"]
    assert not supported["open"]


def test_capture_budget_and_shift_complement() -> None:
    evidence = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    on_evidence = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    ratio = evidence_capture_ratio(on_evidence, evidence)
    assert d4_arbitrary_cells(ratio, torch.tensor([1.0]))["open"] is False
    assert d4_arbitrary_cells(torch.tensor([1.0]), torch.tensor([1.0]))["open"]
    assert d5_budget(torch.tensor([1.4]), torch.tensor([1.0]))["open"]
    assert not d5_budget(torch.tensor([1.0]), torch.tensor([1.0]))["open"]
    robust = torch.zeros(1, 49)
    robust[0, 0] = 1.0
    complement = 1.0 - robust
    assert d6_shift_masks(robust, complement)["open"]
    assert not d6_shift_masks(robust, robust)["open"]
