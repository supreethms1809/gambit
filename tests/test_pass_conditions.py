"""Pass-condition tests: grid_regions keep/remove, TopMSelector, GradCAM regions, CDEAExplainer.explain(x)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Run from repo root with PYTHONPATH=. so core, modality, base_evidence are importable
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.types import HypothesisSet, Explanation
from core.runner import CDEAExplainer
from core.hypotheses import TopMSelector
from core.allocator import EvidenceAsMaskAllocator
from core.objective import AllocationObjective
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider


# ---- Minimal CNN for testing ----
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def test_grid_regions_keep_remove():
    """Pass: visualize keep/remove and it behaves correctly (shape + semantic check)."""
    B, C, H, W = 2, 3, 14, 14
    grid_h, grid_w = 7, 7
    R = grid_h * grid_w
    unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="mean")
    assert unit_space.num_units() == R

    x = torch.rand(B, C, H, W)
    m = torch.zeros(B, R)
    m[:, 0] = 1.0  # keep/remove first region only

    x_keep = unit_space.keep(x, m)
    x_remove = unit_space.remove(x, m)
    assert x_keep.shape == x.shape and x_remove.shape == x.shape
    # Where mask=1: keep keeps x, remove uses baseline. So x_keep != x_remove in general.
    diff_keep_remove = (x_keep - x_remove).abs().sum()
    assert diff_keep_remove > 0, "keep and remove should differ"
    # Full mask: keep preserves x, remove is baseline
    m_all = torch.ones(B, R)
    x_keep_all = unit_space.keep(x, m_all)
    assert torch.allclose(x_keep_all, x), "keep with full mask should equal x"
    print("PASS: grid_regions keep/remove (shapes and behavior)")


def test_grid_regions_embed_units_default_none():
    """Pass: embed_units is disabled by default for backward compatibility."""
    unit_space = VisionGridUnitSpace(7, 7, baseline="mean")
    x = torch.rand(2, 3, 14, 14)
    emb = unit_space.embed_units(x)
    assert emb is None
    print("PASS: grid_regions embed_units default None")


def test_grid_regions_embed_units_shape_dtype_device():
    """Pass: embed_units returns deterministic (B,R,D) embeddings with matching dtype/device."""
    B, C, H, W = 3, 3, 14, 14
    grid_h, grid_w = 7, 7
    R = grid_h * grid_w
    D = 32
    unit_space = VisionGridUnitSpace(grid_h, grid_w, embed_dim=D)
    x = torch.rand(B, C, H, W)
    emb1 = unit_space.embed_units(x)
    emb2 = unit_space.embed_units(x)
    assert emb1 is not None
    assert emb2 is not None
    assert emb1.shape == (B, R, D)
    assert emb1.dtype == x.dtype
    assert emb1.device == x.device
    assert torch.allclose(emb1, emb2), "embeddings should be deterministic across calls"
    assert torch.allclose(emb1[0], emb1[1]), "region embedding table should be shared across batch"
    print("PASS: grid_regions embed_units (B,R,D), deterministic, dtype/device aligned")


def test_top_m_selector():
    """Pass: ids.shape==(B,K), mask matches valid positions."""
    B, num_classes = 4, 10
    K = 5
    selector = TopMSelector(m=K)
    logits = torch.randn(B, num_classes)
    probs = torch.softmax(logits, dim=-1)
    out = selector.select(logits, probs)
    assert isinstance(out, HypothesisSet)
    assert out.ids.shape == (B, K), f"ids.shape {out.ids.shape}"
    assert out.mask.shape == (B, K), f"mask.shape {out.mask.shape}"
    assert out.mask.dtype == torch.bool
    assert out.mask.sum() == B * min(K, num_classes), "mask should be True for valid top-k only"
    print("PASS: TopMSelector ids (B,K), mask valid")


def test_gradcam_regions_provider():
    """Pass: returns E shape (B,K,R), nonnegative."""
    B, C, H, W = 2, 3, 28, 28
    num_classes = 10
    grid_h, grid_w = 7, 7
    R = grid_h * grid_w
    K = 3
    model = TinyCNN(num_classes=num_classes)
    model.eval()
    provider = GradCAMRegionsProvider(grid_h, grid_w)
    x = torch.rand(B, C, H, W)
    ids = torch.randint(0, num_classes, (B, K))
    mask = torch.ones(B, K, dtype=torch.bool)
    hypotheses = HypothesisSet(ids=ids, mask=mask)
    E = provider.explain(x, model, hypotheses)
    assert E.shape == (B, K, R), f"E.shape {E.shape}"
    assert (E >= 0).all(), "evidence should be nonnegative"
    print("PASS: GradCAMRegionsProvider (B,K,R) nonnegative")


def test_integrated_gradients_regions_provider():
    """Pass: returns E shape (B,K,R), nonnegative."""
    B, C, H, W = 2, 3, 28, 28
    num_classes = 10
    grid_h, grid_w = 7, 7
    R = grid_h * grid_w
    K = 3
    model = TinyCNN(num_classes=num_classes)
    model.eval()
    provider = IntegratedGradientsRegionsProvider(grid_h, grid_w, steps=4, baseline="zero")
    x = torch.rand(B, C, H, W)
    ids = torch.randint(0, num_classes, (B, K))
    mask = torch.ones(B, K, dtype=torch.bool)
    hypotheses = HypothesisSet(ids=ids, mask=mask)
    E = provider.explain(x, model, hypotheses)
    assert E.shape == (B, K, R), f"E.shape {E.shape}"
    assert (E >= 0).all(), "IG evidence should be nonnegative"
    print("PASS: IntegratedGradientsRegionsProvider (B,K,R) nonnegative")


class _MinimalObjective:
    """Minimal objective that returns metrics so explain() completes."""

    def compute(self, x, model, unit_space, hypotheses, masks, evidence, tokens=None, attn=None, env=None, **kwargs):
        m = masks["unique"]
        return {"loss": m.abs().sum() * 0.0}  # dummy scalar


def test_explain_returns_explanation():
    """Pass: calling explain(x) returns an Explanation without errors."""
    B, C, H, W = 2, 3, 28, 28
    num_classes = 10
    grid_h, grid_w = 7, 7
    K = 5
    model = TinyCNN(num_classes=num_classes)
    model.eval()
    unit_space = VisionGridUnitSpace(grid_h, grid_w)
    selector = TopMSelector(m=K)
    base_evidence = GradCAMRegionsProvider(grid_h, grid_w)
    allocator = EvidenceAsMaskAllocator()
    objective: AllocationObjective = _MinimalObjective()
    explainer = CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=selector,
        base_evidence=base_evidence,
        allocator=allocator,
        objective=objective,
        interaction=None,
        normalize_evidence=True,
    )
    x = torch.rand(B, C, H, W)
    expl = explainer.explain(x)
    assert isinstance(expl, Explanation)
    assert expl.hypotheses.ids.shape == (B, K)
    assert expl.masks["unique"].shape == (B, K, grid_h * grid_w)
    assert "loss" in expl.metrics
    print("PASS: explain(x) returns Explanation without errors")


if __name__ == "__main__":
    test_grid_regions_keep_remove()
    test_grid_regions_embed_units_default_none()
    test_grid_regions_embed_units_shape_dtype_device()
    test_top_m_selector()
    test_gradcam_regions_provider()
    test_integrated_gradients_regions_provider()
    test_explain_returns_explanation()
    print("All pass conditions OK.")
