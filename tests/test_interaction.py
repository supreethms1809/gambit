"""Pass condition: swap interaction flag (none / attention / transformer) and rerun without changing anything else."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.types import HypothesisSet, Explanation
from core.runner import CDEAExplainer
from core.hypotheses import TopMSelector
from core.allocator import EvidenceAsMaskAllocator
from core.interaction import get_interaction, NoOpInteraction
from core.objective import AllocationObjective
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator


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


class _MinimalObjective:
    def compute(self, x, model, unit_space, hypotheses, masks, evidence, tokens=None, attn=None, env=None, **kwargs):
        return {"loss": masks["unique"].abs().sum() * 0.0}


def _make_explainer(interaction_flag: str, d_model: int = 32):
    B, C, H, W = 2, 3, 28, 28
    grid_h, grid_w = 7, 7
    K = 5
    model = TinyCNN(num_classes=10)
    model.eval()
    unit_space = VisionGridUnitSpace(grid_h, grid_w, embed_dim=d_model)
    selector = TopMSelector(m=K)
    base_evidence = GradCAMRegionsProvider(grid_h, grid_w)
    allocator = EvidenceAsMaskAllocator()
    objective: AllocationObjective = _MinimalObjective()
    interaction = get_interaction(interaction_flag, d_model=d_model) if interaction_flag != "none" else None
    return CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=selector,
        base_evidence=base_evidence,
        allocator=allocator,
        objective=objective,
        interaction=interaction,
        normalize_evidence=True,
    )


def test_swap_flag_none():
    explainer = _make_explainer("none")
    x = torch.rand(2, 3, 28, 28)
    expl = explainer.explain(x)
    assert isinstance(expl, Explanation)
    assert expl.extras["tokens"] is not None
    assert expl.extras["attn"] is None
    print("PASS: interaction=none")


def test_swap_flag_attention():
    explainer = _make_explainer("attention")
    x = torch.rand(2, 3, 28, 28)
    expl = explainer.explain(x)
    assert isinstance(expl, Explanation)
    assert expl.extras["tokens"] is not None
    assert expl.extras["attn"] is not None
    assert expl.extras["attn"].shape[-2:] == (5, 5)
    print("PASS: interaction=attention")


def test_swap_flag_transformer():
    explainer = _make_explainer("transformer")
    x = torch.rand(2, 3, 28, 28)
    expl = explainer.explain(x)
    assert isinstance(expl, Explanation)
    assert expl.extras["tokens"] is not None
    assert expl.extras["attn"] is not None
    print("PASS: interaction=transformer")


def test_explicit_noop():
    explainer = _make_explainer("none")
    explainer.interaction = NoOpInteraction()
    x = torch.rand(2, 3, 28, 28)
    expl = explainer.explain(x)
    assert isinstance(expl, Explanation)
    print("PASS: explicit NoOpInteraction")


def test_real_vision_grid_with_attention():
    """Pass: actual VisionGridUnitSpace embeddings drive attention without test-only subclasses."""
    B, K, d_model = 2, 5, 32
    model = TinyCNN(num_classes=10).eval()
    unit_space = VisionGridUnitSpace(7, 7, embed_dim=d_model)
    explainer = CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=TopMSelector(m=K),
        base_evidence=GradCAMRegionsProvider(7, 7),
        allocator=EvidenceAsMaskAllocator(),
        objective=_MinimalObjective(),
        interaction=get_interaction("attention", d_model=d_model),
        normalize_evidence=True,
    )
    x = torch.rand(B, 3, 28, 28)
    expl = explainer.explain(x)
    assert isinstance(expl, Explanation)
    assert expl.extras["tokens"] is not None
    assert expl.extras["tokens"].shape == (B, K, d_model)
    assert expl.extras["attn"] is not None
    assert expl.extras["attn"].shape[-2:] == (K, K)
    print("PASS: real VisionGridUnitSpace + attention")


def test_interaction_changes_contrastive_masks():
    """Pass: attention-conditioned path in allocator changes masks."""
    torch.manual_seed(7)
    B, K, R = 2, 4, 16
    x = torch.rand(B, 3, 28, 28)
    model = TinyCNN(num_classes=10).eval()
    unit_space = VisionGridUnitSpace(4, 4)
    ids = torch.randint(0, 10, (B, K))
    hypotheses = HypothesisSet(ids=ids, mask=torch.ones(B, K, dtype=torch.bool))
    evidence = torch.rand(B, K, R)
    evidence = evidence / evidence.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    objective = ContrastiveObjective(
        lambda_suff=1.0,
        lambda_margin=1.0,
        lambda_sparse=0.05,
        lambda_overlap=0.2,
        attn_weight_blend=1.0,
    )
    allocator = OptimizationAllocator(
        objective=objective,
        num_steps=8,
        lr=0.25,
        use_shared=True,
        lambda_disjoint=0.1,
        lambda_partition=0.1,
        attn_mix=1.0,
    )

    attn = torch.tensor(
        [
            [0.0, 3.0, 1.0, 0.5],
            [2.0, 0.0, 0.5, 0.5],
            [0.2, 1.5, 0.0, 2.0],
            [1.0, 0.3, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    attn = attn.unsqueeze(0).expand(B, -1, -1).contiguous()

    masks_none = allocator.allocate(
        x=x,
        model=model,
        unit_space=unit_space,
        hypotheses=hypotheses,
        evidence=evidence,
        attn=None,
    )
    masks_attn = allocator.allocate(
        x=x,
        model=model,
        unit_space=unit_space,
        hypotheses=hypotheses,
        evidence=evidence,
        attn=attn,
    )

    delta_unique = (masks_none["unique"] - masks_attn["unique"]).abs().mean().item()
    delta_shared = (masks_none["shared"] - masks_attn["shared"]).abs().mean().item()
    assert delta_unique > 1e-5 or delta_shared > 1e-5, "attention-conditioned path should change allocation"
    print("PASS: attention-conditioned allocator path changes masks")


if __name__ == "__main__":
    test_swap_flag_none()
    test_swap_flag_attention()
    test_swap_flag_transformer()
    test_explicit_noop()
    test_real_vision_grid_with_attention()
    test_interaction_changes_contrastive_masks()
    print("All interaction flag swaps OK.")
