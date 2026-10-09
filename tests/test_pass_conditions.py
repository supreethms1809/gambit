"""Grid deletion, hypothesis selection, and the evidence providers."""

from __future__ import annotations

import torch
import torch.nn as nn

from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from core.grid import delete, keep
from core.hypotheses import TopMSelector
from core.types import HypothesisSet


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))


def test_hard_delete_and_keep_cover_the_cell():
    image = torch.rand(2, 3, 28, 28)
    mask = torch.zeros(2, 16)
    mask[:, 0] = 1
    removed = delete(image, mask, 4, 4)
    kept = keep(image, mask, 4, 4)
    assert removed.shape == image.shape
    assert kept.shape == image.shape
    assert not torch.allclose(removed, image)
    assert not torch.allclose(kept, removed)


def test_top_m_selector_does_not_pad():
    logits = torch.tensor([[0.1, 0.5, 0.2, 0.9]])
    selected = TopMSelector(m=5).select(logits, torch.softmax(logits, dim=-1))
    assert selected.ids.shape == (1, 4)
    assert bool(selected.mask.all())
    assert int(selected.ids[0, 0]) == 3


def test_gradcam_and_integrated_gradients_return_region_evidence():
    torch.manual_seed(0)
    model = TinyCNN().eval()
    x = torch.rand(2, 3, 32, 32)
    hypotheses = HypothesisSet(ids=torch.tensor([[0, 1], [2, 0]]), mask=torch.ones(2, 2, dtype=torch.bool))
    gradcam = GradCAMRegionsProvider(4, 4).explain(x, model, hypotheses)
    integrated = IntegratedGradientsRegionsProvider(4, 4, steps=2).explain(x, model, hypotheses)
    assert gradcam.shape == (2, 2, 16)
    assert integrated.shape == (2, 2, 16)
    assert torch.isfinite(gradcam).all() and torch.isfinite(integrated).all()
    assert bool((gradcam >= 0).all() and (integrated >= 0).all())
