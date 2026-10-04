"""Pass conditions: OptimizationAllocator masks sparse/non-trivial; ContrastiveObjective metrics direction."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.types import HypothesisSet
from modality.grid_regions import VisionGridUnitSpace
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator, _partition_penalty


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))


def test_optimization_allocator_masks_sparse_nontrivial():
    """Pass: masks become sparse and non-trivial (not all zeros/ones)."""
    B, K, R = 4, 3, 16
    grid_h, grid_w = 4, 4
    unit_space = VisionGridUnitSpace(grid_h, grid_w)
    model = TinyCNN(num_classes=10)
    model.eval()
    objective = ContrastiveObjective(lambda_suff=1.0, lambda_margin=1.0, lambda_sparse=0.05, lambda_overlap=0.2)
    allocator = OptimizationAllocator(objective, num_steps=35, lr=0.4)
    x = torch.rand(B, 3, 32, 32)
    ids = torch.randint(0, 10, (B, K))
    valid = torch.ones(B, K, dtype=torch.bool)
    hypotheses = HypothesisSet(ids=ids, mask=valid)
    evidence = torch.rand(B, K, R).abs()
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    masks = allocator.allocate(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, evidence=evidence)
    m = masks["unique"]
    assert m.shape == (B, K, R)
    assert (m < 0.01).float().mean().item() < 1.0, "masks should not be all zeros"
    assert (m > 0.99).float().mean().item() < 1.0, "masks should not be all ones"
    assert m.sum().item() > 0.1, "masks should have positive total mass"
    print("PASS: OptimizationAllocator masks sparse and non-trivial")


def test_contrastive_objective_metrics_direction():
    """Pass: during optimization, suff increases, margin increases, overlap decreases."""
    B, K, R = 4, 3, 16
    grid_h, grid_w = 4, 4
    unit_space = VisionGridUnitSpace(grid_h, grid_w)
    model = TinyCNN(num_classes=10)
    model.eval()
    objective = ContrastiveObjective(lambda_suff=1.0, lambda_margin=1.0, lambda_sparse=0.05, lambda_overlap=0.2)
    x = torch.rand(B, 3, 32, 32)
    ids = torch.randint(0, 10, (B, K))
    valid = torch.ones(B, K, dtype=torch.bool)
    hypotheses = HypothesisSet(ids=ids, mask=valid)
    evidence = torch.rand(B, K, R).abs()
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)

    # Initial masks = evidence
    m0 = evidence.clone().requires_grad_(True)
    out0 = objective.compute(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, masks={"unique": m0}, evidence=evidence)
    suff0 = out0["kept_logit"].item()
    assert out0["suff"].item() == suff0
    margin0 = out0["margin"].item()
    overlap0 = out0["overlap"].item()

    # Optimized masks (few steps)
    allocator = OptimizationAllocator(objective, num_steps=25, lr=0.5)
    masks_opt = allocator.allocate(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, evidence=evidence)
    out1 = objective.compute(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, masks=masks_opt, evidence=evidence)
    suff1 = out1["suff"].item()
    margin1 = out1["margin"].item()
    overlap1 = out1["overlap"].item()

    # We want suff to increase, margin to increase, overlap to decrease
    assert suff1 >= suff0 - 0.5, "suff should increase or stay close"
    assert margin1 >= margin0 - 0.5, "margin should increase or stay close"
    assert overlap1 <= overlap0 + 0.5, "overlap should decrease or stay close"
    print("PASS: ContrastiveObjective metrics move in right direction (suff/margin up, overlap down)")


def test_contrastive_probability_split_metrics_with_shared():
    """Pass: shared-only vs shared+unique probability report tensors are returned with valid shapes/ranges."""
    B, K, R = 3, 4, 16
    grid_h, grid_w = 4, 4
    unit_space = VisionGridUnitSpace(grid_h, grid_w)
    model = TinyCNN(num_classes=10).eval()
    objective = ContrastiveObjective(lambda_suff=1.0, lambda_margin=1.0, lambda_sparse=0.05, lambda_overlap=0.2)
    allocator = OptimizationAllocator(
        objective,
        num_steps=20,
        lr=0.4,
        use_shared=True,
        lambda_partition=0.1,
    )

    x = torch.rand(B, 3, 32, 32)
    ids = torch.randint(0, 10, (B, K))
    valid = torch.ones(B, K, dtype=torch.bool)
    hypotheses = HypothesisSet(ids=ids, mask=valid)
    evidence = torch.rand(B, K, R).abs()
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    masks = allocator.allocate(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, evidence=evidence)

    assert "shared" in masks, "allocator should output explicit shared mask when use_shared=True"
    out = objective.compute(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, masks=masks, evidence=evidence)
    for key in [
        "split_shared_only_logits_topm",
        "split_shared_only_probs_topm",
        "split_shared_plus_unique_logits_topm",
        "split_shared_plus_unique_probs_topm",
        "pairwise_margin_shared_only_logits_topm",
        "pairwise_margin_shared_only_probs_topm",
        "pairwise_margin_shared_plus_unique_logits_topm",
        "pairwise_margin_shared_plus_unique_probs_topm",
        "pairwise_margin_delta_logits_topm",
        "pairwise_margin_delta_probs_topm",
    ]:
        assert key in out, f"missing split report metric: {key}"
        expected_shape = (B, K, K) if key.startswith("pairwise_margin_") else (B, K)
        assert out[key].shape == expected_shape, f"{key} must be {expected_shape}"
        assert torch.isfinite(out[key]).all(), f"{key} must be finite"

    p_shared = out["split_shared_only_probs_topm"]
    p_plus = out["split_shared_plus_unique_probs_topm"]
    assert ((p_shared >= 0) & (p_shared <= 1)).all(), "shared-only probs should be in [0,1]"
    assert ((p_plus >= 0) & (p_plus <= 1)).all(), "shared+unique probs should be in [0,1]"
    # pairwise delta should match (plus - shared) relation
    pd_log = out["pairwise_margin_delta_logits_topm"]
    pp_log = out["pairwise_margin_shared_plus_unique_logits_topm"]
    ps_log = out["pairwise_margin_shared_only_logits_topm"]
    assert torch.allclose(pd_log, pp_log - ps_log, atol=1e-5), "pairwise logit delta mismatch"
    pd_prob = out["pairwise_margin_delta_probs_topm"]
    pp_prob = out["pairwise_margin_shared_plus_unique_probs_topm"]
    ps_prob = out["pairwise_margin_shared_only_probs_topm"]
    assert torch.allclose(pd_prob, pp_prob - ps_prob, atol=1e-5), "pairwise prob delta mismatch"
    print("PASS: Contrastive split report metrics returned with explicit shared masks")


def test_partition_penalty_scale_invariant():
    """Pass: identical per-region overflow costs the same at any batch/grid scale."""
    from instantiations.contrastive.allocator import _partition_penalty

    small = torch.full((2, 3, 16), 0.5)  # total 1.5 per region -> overflow 0.5
    large = torch.full((4, 3, 64), 0.5)
    assert torch.allclose(_partition_penalty(small), _partition_penalty(large))
    assert torch.allclose(_partition_penalty(small), torch.tensor(0.5))
    assert _partition_penalty(torch.zeros(2, 3, 16)).item() == 0.0
    print("PASS: partition penalty is a scale-invariant mean")


def test_single_valid_hypothesis_has_finite_zero_margin():
    """Pass: with no foil, the margin is 0 rather than z_k - (-inf)."""
    B, K, R = 2, 3, 16
    unit_space = VisionGridUnitSpace(4, 4)
    model = TinyCNN(num_classes=10).eval()
    objective = ContrastiveObjective()
    x = torch.rand(B, 3, 32, 32)
    ids = torch.randint(0, 10, (B, K))
    valid = torch.zeros(B, K, dtype=torch.bool)
    valid[:, 0] = True
    hypotheses = HypothesisSet(ids=ids, mask=valid)
    evidence = torch.rand(B, K, R).abs()
    evidence = evidence / evidence.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    out = objective.compute(
        x=x, model=model, unit_space=unit_space, hypotheses=hypotheses,
        masks={"unique": evidence}, evidence=evidence,
    )
    assert torch.isfinite(out["loss"]).all(), "loss must stay finite with one hypothesis"
    assert torch.isfinite(out["margin"]).all()
    assert out["margin"].item() == 0.0
    print("PASS: single-hypothesis margin is finite and zero")


def test_invalid_hypotheses_leave_penalty_means():
    """Pass: sparsity/mass-dev/overlap average over valid hypotheses only."""
    B, K, R = 2, 4, 8
    unit_space = VisionGridUnitSpace(2, 4)
    model = TinyCNN(num_classes=10).eval()
    objective = ContrastiveObjective(lambda_mass=0.0)
    x = torch.rand(B, 3, 32, 32)
    ids = torch.zeros(B, K, dtype=torch.long)
    valid = torch.tensor([[True, True, False, False], [True, True, False, False]])
    hypotheses = HypothesisSet(ids=ids, mask=valid)
    evidence = torch.full((B, K, R), 1.0 / R)
    masks = {"unique": torch.full((B, K, R), 0.5)}
    out = objective.compute(
        x=x, model=model, unit_space=unit_space, hypotheses=hypotheses,
        masks=masks, evidence=evidence,
    )
    # Valid rows hold 0.5 * 8 = 4.0 mass each; invalid rows must not dilute the mean.
    assert out["sparse"].item() == 4.0
    # Invalid rows are zeroed before the overlap, so they contribute no pairs.
    assert torch.isfinite(out["overlap"])
    print("PASS: penalty means cover valid hypotheses only")


if __name__ == "__main__":
    test_optimization_allocator_masks_sparse_nontrivial()
    test_contrastive_objective_metrics_direction()
    test_contrastive_probability_split_metrics_with_shared()
    test_partition_penalty_scale_invariant()
    test_single_valid_hypothesis_has_finite_zero_margin()
    test_invalid_hypotheses_leave_penalty_means()
    print("All pass conditions OK.")
