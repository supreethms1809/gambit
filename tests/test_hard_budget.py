"""Hard budget on the allocators, and the family-C foil pair."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.types import HypothesisSet
from evaluation.executor import FamilyCExecutor
from evaluation.foil_masks import pair_scores
from instantiations.contrastive.allocator import OptimizationAllocator
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.hard_budget import budget_mass, budgeted_mask
from modality.grid_regions import VisionGridUnitSpace


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        pooled = torch.relu(self.conv(x)).mean(dim=(2, 3))
        return self.fc(pooled)


def test_hard_budget_matches_the_grid_fraction_and_passes_gradients():
    logits = torch.randn(2, 16, requires_grad=True)
    projected = budgeted_mask(logits, 49)
    assert budget_mass(16, 49) == 1.0
    assert torch.allclose(projected.sum(dim=-1), torch.ones(2), atol=1e-4)
    assert float(projected.detach().min()) >= 0.0
    assert float(projected.detach().max()) <= 1.0
    (projected * torch.linspace(0.1, 1.0, 16)).sum().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0


def test_allocator_returns_the_budget_and_drops_invalid_rows():
    regions = 16
    objective = ContrastiveObjective()
    allocator = OptimizationAllocator(objective, num_steps=2, lr=0.1, use_shared=True)
    valid = torch.tensor([[True, True, False]])
    hypotheses = HypothesisSet(ids=torch.tensor([[1, 2, 3]]), mask=valid)
    evidence = torch.rand(1, 3, regions)
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    masks = allocator.allocate(
        x=torch.rand(1, 3, 16, 16),
        model=_Tiny(),
        unit_space=VisionGridUnitSpace(4, 4),
        hypotheses=hypotheses,
        evidence=evidence,
    )
    unique = masks["unique"]
    assert torch.allclose(unique[0, 0].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(unique[0, 1].sum(), torch.tensor(1.0), atol=1e-4)
    assert float(unique[0, 2].abs().sum()) == 0.0
    assert torch.allclose(masks["shared"].sum(), torch.tensor(1.0), atol=1e-4)


def test_foil_scores_use_unique_masks_by_default():
    unique = torch.zeros(1, 3, 4)
    unique[0, 0, 0] = 1.0
    unique[0, 1, 1] = 1.0
    unique[0, 2, 2] = 5.0
    shared = torch.tensor([[0.2, 0.0, 0.0, 0.3]])
    kept, foil = pair_scores(unique, shared)
    assert torch.allclose(kept, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(foil, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_foil_scores_add_shared_to_rank_0_and_rank_1_in_the_sensitivity_check():
    unique = torch.zeros(1, 3, 4)
    unique[0, 0, 0] = 1.0
    unique[0, 1, 1] = 1.0
    unique[0, 2, 2] = 5.0
    shared = torch.tensor([[0.2, 0.0, 0.0, 0.3]])
    kept, foil = pair_scores(unique, shared, include_shared=True)
    assert torch.allclose(kept, torch.tensor([[1.2, 0.0, 0.0, 0.3]]))
    assert torch.allclose(foil, torch.tensor([[0.2, 1.0, 0.0, 0.3]]))


def test_executor_scores_the_given_foil_pair(monkeypatch):
    seen = {}

    def _capture(model, images, mask_k, mask_l, class_k, class_l, **kwargs):
        seen["class_k"] = class_k.detach().cpu()
        seen["class_l"] = class_l.detach().cpu()
        seen["mask_k"] = mask_k.detach().cpu()
        return torch.zeros(images.shape[0])

    monkeypatch.setattr("evaluation.executor.contrastive_deletion", _capture)
    hypotheses = HypothesisSet(
        ids=torch.tensor([[4, 2]]),
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    unique = torch.zeros(1, 2, 4)
    unique[0, 0, 0] = 3.0
    unique[0, 1, 3] = 3.0
    score = FamilyCExecutor(_Tiny(), fraction=0.25, seed=0, iters=0).score(
        torch.rand(1, 3, 8, 8),
        unique,
        hypotheses=hypotheses,
    )
    assert score.shape == (1,)
    assert int(seen["class_k"]) == 4
    assert int(seen["class_l"]) == 2
    assert float(seen["mask_k"].sum()) == 1.0
