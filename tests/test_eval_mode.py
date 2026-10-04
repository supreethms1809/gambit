"""Explaining a model must not change it: eval mode during forwards, flag restored after."""

from __future__ import annotations

import torch
import torch.nn as nn

from base_evidence.gradcam_regions import GradCAMRegionsProvider
from core.eval_mode import eval_mode
from core.types import EnvBatch, HypothesisSet
from instantiations.contrastive.allocator import OptimizationAllocator
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from instantiations.shift.objective import RobustShortcutObjective
from modality.grid_regions import VisionGridUnitSpace


class _DropoutNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.drop(x)
        return self.fc(self.pool(x).flatten(1))


def _hypotheses(b: int, k: int, num_classes: int) -> HypothesisSet:
    ids = torch.randint(0, num_classes, (b, k))
    return HypothesisSet(ids=ids, mask=torch.ones(b, k, dtype=torch.bool))


def test_eval_mode_helper_restores_flag():
    model = _DropoutNet().train()
    assert model.training
    with eval_mode(model):
        assert not model.training
    assert model.training
    model.eval()
    with eval_mode(model):
        assert not model.training
    assert not model.training


def test_eval_mode_passes_through_plain_callables():
    called = []

    def decision(x):
        called.append(True)
        return x

    with eval_mode(decision):
        decision(torch.zeros(1))
    assert called


def test_contrastive_allocator_keeps_train_mode_and_bn_stats():
    torch.manual_seed(0)
    b, k, r = 2, 2, 16
    space = VisionGridUnitSpace(4, 4)
    model = _DropoutNet().train()
    objective = ContrastiveObjective()
    allocator = OptimizationAllocator(objective, num_steps=4, lr=0.5)
    x = torch.rand(b, 3, 32, 32)
    hypotheses = _hypotheses(b, k, 4)
    evidence = torch.rand(b, k, r)
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    before = model.bn.running_mean.clone()
    masks = allocator.allocate(
        x=x, model=model, unit_space=space, hypotheses=hypotheses, evidence=evidence
    )
    assert model.training, "allocator must restore the training flag"
    assert torch.equal(model.bn.running_mean, before), "BN stats must not move during allocation"
    assert torch.isfinite(masks["unique"]).all()


def test_contrastive_allocation_deterministic_in_train_mode():
    torch.manual_seed(1)
    b, k, r = 2, 2, 16
    space = VisionGridUnitSpace(4, 4)
    objective = ContrastiveObjective()
    allocator = OptimizationAllocator(objective, num_steps=4, lr=0.5)
    x = torch.rand(b, 3, 32, 32)
    hypotheses = _hypotheses(b, k, 4)
    evidence = torch.rand(b, k, r)
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    model = _DropoutNet().train()
    first = allocator.allocate(
        x=x, model=model, unit_space=space, hypotheses=hypotheses, evidence=evidence
    )["unique"]
    second = allocator.allocate(
        x=x, model=model, unit_space=space, hypotheses=hypotheses, evidence=evidence
    )["unique"]
    assert torch.equal(first, second), "dropout must be off during allocation"


def test_shift_allocator_keeps_train_mode():
    torch.manual_seed(2)
    b, r = 2, 16
    space = VisionGridUnitSpace(4, 4)
    model = _DropoutNet().train()
    objective = RobustShortcutObjective()
    allocator = RobustShortcutOptimizationAllocator(objective, num_steps=3, lr=0.5)
    x = torch.rand(b, 3, 32, 32)
    env = EnvBatch(xs=[x, torch.rand_like(x)], env_ids=["id", "ood"])
    hypotheses = _hypotheses(b, 2, 4)
    evidence = torch.rand(b, 2, r)
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    before = model.bn.running_mean.clone()
    masks = allocator.allocate(
        x=x, model=model, unit_space=space, hypotheses=hypotheses, evidence=evidence, env=env
    )
    assert model.training, "allocator must restore the training flag"
    assert torch.equal(model.bn.running_mean, before), "BN stats must not move during allocation"
    assert torch.isfinite(masks["robust"]).all()


def test_gradcam_provider_restores_train_mode():
    torch.manual_seed(3)
    model = _DropoutNet().train()
    provider = GradCAMRegionsProvider(4, 4)
    x = torch.rand(2, 3, 32, 32)
    out = provider.explain(x, model, _hypotheses(2, 2, 4))
    assert model.training, "provider must restore the training flag"
    assert out.shape == (2, 2, 16)
