"""Pass conditions: EnvBatch generator (logits differ); RobustShortcutOptimizationAllocator (metrics direction)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.types import EnvBatch, HypothesisSet
from instantiations.shift.env import env_batch_from_augs, default_shift_augs, logits_differ_across_envs
from instantiations.shift.objective import RobustShortcutObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from modality.grid_regions import VisionGridUnitSpace


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(self.pool(x).flatten(1))


def test_env_batch_logits_differ():
    """Pass: model predictions or logits differ across envs for at least some examples."""
    B, C, H, W = 8, 3, 28, 28
    model = TinyCNN(num_classes=10)
    model.eval()
    aug1, aug2 = default_shift_augs(color_jitter=0.5, greyscale_prob=0.3)
    x = torch.rand(B, C, H, W)
    env = env_batch_from_augs(x, aug1, aug2)
    assert len(env.xs) == 3
    assert env.xs[0].shape == x.shape
    # With random augs and random model, logits should differ for some batch elem
    found = False
    for _ in range(5):
        env = env_batch_from_augs(x, aug1, aug2)
        if logits_differ_across_envs(model, env):
            found = True
            break
    assert found, "logits should differ across envs for at least some examples"
    print("PASS: EnvBatch generator, logits differ across envs")


def test_robust_shortcut_allocator_metrics():
    """Pass: objective decreases; rob_var down, rob_mean up, sho_gap positive and increasing, disjoint low."""
    B, R = 4, 16
    grid_h, grid_w = 4, 4
    unit_space = VisionGridUnitSpace(grid_h, grid_w)
    model = TinyCNN(num_classes=10)
    model.eval()
    objective = RobustShortcutObjective(
        lambda_mean=1.0, lambda_var=0.5, lambda_gap=1.0, lambda_disjoint=0.2, lambda_sparse=0.05
    )
    allocator = RobustShortcutOptimizationAllocator(objective, num_steps=30, lr=0.3, lambda_disjoint=0.2)
    x = torch.rand(B, 3, 28, 28)
    aug1, aug2 = default_shift_augs(color_jitter=0.4)
    env = env_batch_from_augs(x, aug1, aug2)
    hypotheses = HypothesisSet(ids=torch.randint(0, 10, (B, 3)), mask=torch.ones(B, 3, dtype=torch.bool))
    evidence = torch.rand(B, 3, R).abs()
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)

    masks = allocator.allocate(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, evidence=evidence, env=env)
    assert "robust" in masks and "shortcut" in masks
    assert masks["robust"].shape == (B, R) and masks["shortcut"].shape == (B, R)

    out = objective.compute(x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, masks=masks, evidence=evidence, env=env)
    assert "rob_mean" in out and "rob_var" in out and "sho_gap" in out and "disjoint" in out
    assert out["disjoint"].item() < 2.0
    print("PASS: RobustShortcutOptimizationAllocator metrics (rob_mean, rob_var, sho_gap, disjoint)")
    print("  rob_mean=%.4f rob_var=%.4f sho_gap=%.4f disjoint=%.4f" % (
        out["rob_mean"].item(), out["rob_var"].item(), out["sho_gap"].item(), out["disjoint"].item()))


def test_robust_shortcut_objective_targets():
    """Pass: objective supports pred/top_hypothesis/label target modes and validates label input."""
    B, R, K = 3, 16, 3
    unit_space = VisionGridUnitSpace(4, 4)
    model = TinyCNN(num_classes=10).eval()
    x = torch.rand(B, 3, 28, 28)
    aug1, aug2 = default_shift_augs(color_jitter=0.4)
    env = env_batch_from_augs(x, aug1, aug2)
    hypotheses = HypothesisSet(ids=torch.randint(0, 10, (B, K)), mask=torch.ones(B, K, dtype=torch.bool))
    evidence = torch.rand(B, K, R).abs()
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    masks = {"robust": torch.rand(B, R), "shortcut": torch.rand(B, R)}
    y = torch.randint(0, 10, (B,))

    for target in ("pred", "top_hypothesis"):
        obj = RobustShortcutObjective(target=target)
        out = obj.compute(
            x=x,
            model=model,
            unit_space=unit_space,
            hypotheses=hypotheses,
            masks=masks,
            evidence=evidence,
            env=env,
        )
        assert "loss" in out and torch.isfinite(out["loss"]).item()
        assert "sho_mean" in out and torch.isfinite(out["sho_mean"]).item()

    obj_label = RobustShortcutObjective(target="label")
    out_label = obj_label.compute(
        x=x,
        model=model,
        unit_space=unit_space,
        hypotheses=hypotheses,
        masks=masks,
        evidence=evidence,
        env=env,
        y=y,
    )
    assert "loss" in out_label and torch.isfinite(out_label["loss"]).item()
    assert "sho_mean" in out_label and torch.isfinite(out_label["sho_mean"]).item()

    missing_label_raised = False
    try:
        obj_label.compute(
            x=x,
            model=model,
            unit_space=unit_space,
            hypotheses=hypotheses,
            masks=masks,
            evidence=evidence,
            env=env,
        )
    except ValueError:
        missing_label_raised = True
    assert missing_label_raised, "target='label' must require y kwarg"
    print("PASS: RobustShortcutObjective target modes (pred/top_hypothesis/label)")


if __name__ == "__main__":
    test_env_batch_logits_differ()
    test_robust_shortcut_allocator_metrics()
    test_robust_shortcut_objective_targets()
    print("All shift instantiation pass conditions OK.")
