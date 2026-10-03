"""Locks the S05 formulation decisions: one overlap weight, one name per quantity."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.game_modes import resolve_contrastive_game
from core.types import EnvBatch, HypothesisSet
from instantiations.contrastive.allocator import OptimizationAllocator
from instantiations.contrastive.objective import (
    DEFAULT_LAMBDA_SHARED_SPARSE,
    ContrastiveObjective,
    pairwise_overlap,
)
from instantiations.shift.objective import RobustShortcutObjective
from modality.grid_regions import VisionGridUnitSpace


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def test_pairwise_overlap_counts_each_pair_once() -> None:
    masks = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    assert pairwise_overlap(masks).item() == pytest.approx(1.0)


def test_retired_disjoint_weight_is_rejected() -> None:
    objective = ContrastiveObjective()
    with pytest.raises(ValueError, match="lambda_overlap"):
        OptimizationAllocator(objective, lambda_disjoint=0.2)
    with pytest.raises(ValueError, match="lambda_overlap"):
        resolve_contrastive_game(
            "manual",
            use_shared=True,
            lambda_margin=1.0,
            lambda_overlap=0.2,
            lambda_disjoint=0.34,
            lambda_partition=0.0,
        )


def test_shared_sparse_default_closes_the_blanket() -> None:
    assert ContrastiveObjective().lambda_shared_sparse == pytest.approx(DEFAULT_LAMBDA_SHARED_SPARSE)
    assert DEFAULT_LAMBDA_SHARED_SPARSE == pytest.approx(0.25)


def test_kept_logit_is_the_raw_logit_and_suff_is_its_alias() -> None:
    batch, classes, regions = 2, 2, 4
    unit_space = VisionGridUnitSpace(2, 2)
    model = TinyCNN().eval()
    objective = ContrastiveObjective(lambda_overlap=0.2)
    x = torch.rand(batch, 3, 8, 8)
    hypotheses = HypothesisSet(
        ids=torch.zeros(batch, classes, dtype=torch.long),
        mask=torch.ones(batch, classes, dtype=torch.bool),
    )
    evidence = torch.full((batch, classes, regions), 1.0 / regions)
    masks = {"unique": torch.full((batch, classes, regions), 0.5)}
    out = objective.compute(
        x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, masks=masks, evidence=evidence
    )
    assert out["kept_logit"].item() == pytest.approx(out["suff"].item())
    assert out["overlap"].item() == pytest.approx(pairwise_overlap(masks["unique"]).mean().item())


def test_shift_mass_target_penalizes_a_blanket_and_an_empty_mask() -> None:
    batch, regions = 2, 49
    unit_space = VisionGridUnitSpace(7, 7)
    model = TinyCNN().eval()
    x = torch.rand(batch, 3, 14, 14)
    env = EnvBatch(xs=[x, x.clone()], env_ids=["id", "ood"])
    hypotheses = HypothesisSet(
        ids=torch.zeros(batch, 2, dtype=torch.long),
        mask=torch.ones(batch, 2, dtype=torch.bool),
    )
    evidence = torch.full((batch, 2, regions), 1.0 / regions)
    objective = RobustShortcutObjective(
        lambda_mean=0.0,
        lambda_var=0.0,
        lambda_gap=0.0,
        lambda_shortcut=0.0,
        lambda_disjoint=0.0,
        lambda_sparse=0.0,
        lambda_mass=0.1,
    )
    assert objective.lambda_mass == pytest.approx(0.1)

    def mass_dev(mask: torch.Tensor) -> float:
        out = objective.compute(
            x=x,
            model=model,
            unit_space=unit_space,
            hypotheses=hypotheses,
            masks={"robust": mask, "shortcut": mask},
            evidence=evidence,
            env=env,
        )
        return out["mass_dev"].item()

    on_target = torch.zeros(batch, regions)
    on_target[:, 0] = 1.0
    assert mass_dev(torch.ones(batch, regions)) > mass_dev(on_target)
    assert mass_dev(torch.zeros(batch, regions)) > mass_dev(on_target)


def test_method_modules_describe_joint_gradient_descent() -> None:
    banned = ("equilibrium", "best-response", "best response", "provably")
    paths = list((REPO / "instantiations").rglob("*.py"))
    paths.append(REPO / "core" / "game_modes.py")
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            if word in text:
                hits.append(f"{path.relative_to(REPO)}: {word}")
    assert hits == []
