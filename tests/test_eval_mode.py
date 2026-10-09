"""Explaining a model must not change it: eval mode during forwards, flag restored after."""

from __future__ import annotations

import torch
import torch.nn as nn

from base_evidence.gradcam_regions import GradCAMRegionsProvider
from core.eval_mode import eval_mode
from core.types import HypothesisSet
from cdea.allocation import AllocateConfig, allocate


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


def test_allocation_restores_train_mode_and_bn_stats():
    torch.manual_seed(0)
    model = _DropoutNet().train()
    before = model.bn.running_mean.clone()
    x = torch.rand(2, 3, 32, 32)
    allocate(
        model, x, _hypotheses(2, 2, 4), 4, 4, 0.2,
        AllocateConfig(steps=2, lr=0.1, init="uniform", shared=False),
    )
    assert model.training
    assert torch.equal(model.bn.running_mean, before)


def test_gradcam_provider_restores_train_mode():
    torch.manual_seed(3)
    model = _DropoutNet().train()
    provider = GradCAMRegionsProvider(4, 4)
    x = torch.rand(2, 3, 32, 32)
    out = provider.explain(x, model, _hypotheses(2, 2, 4))
    assert model.training
    assert out.shape == (2, 2, 16)
