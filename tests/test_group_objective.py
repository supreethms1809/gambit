"""Unpaired group-statistics objective.

The shared red square is the robust cue. The bottom-right square is present
only in the second group, so it is the shortcut cue. Waterbirds val is the
smoke run and is skipped when those images are not on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from core.types import EnvBatch, HypothesisSet
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from instantiations.shift.objective import GroupStatisticsObjective
from modality.grid_regions import VisionGridUnitSpace

REPO = Path(__file__).resolve().parents[1]


class _RedMean(nn.Module):
    """Class 0 is the mean red channel. Class 1 is unused."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        red = x[:, 0].mean(dim=(1, 2))
        return torch.stack([red, torch.zeros_like(red)], dim=1)


def _images() -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.full((4, 3, 8, 8), 0.2)
    images[:, 0, 0:4, 0:4] = 1.0
    images[2:, 0, 4:8, 4:8] = 1.0
    group = torch.tensor([0, 0, 1, 1])
    return images, group


def _masks(region: int) -> dict[str, torch.Tensor]:
    robust = torch.zeros(4, 4)
    shortcut = torch.zeros(4, 4)
    robust[:, 0] = 1.0
    shortcut[:, region] = 1.0
    return {"robust": robust, "shortcut": shortcut}


def _objective() -> GroupStatisticsObjective:
    return GroupStatisticsObjective(
        lambda_mean=1.0,
        lambda_var=0.5,
        lambda_gap=1.0,
        lambda_disjoint=0.0,
        lambda_sparse=0.0,
        lambda_mass=0.0,
        target="label",
    )


def test_shared_square_is_stable_and_the_group_square_is_not() -> None:
    images, group = _images()
    space = VisionGridUnitSpace(2, 2, baseline="mean")
    hypotheses = HypothesisSet(
        ids=torch.zeros(4, 1, dtype=torch.long),
        mask=torch.ones(4, 1, dtype=torch.bool),
    )
    y = torch.zeros(4, dtype=torch.long)
    objective = _objective()
    shared = objective.compute(
        x=images,
        model=_RedMean(),
        unit_space=space,
        hypotheses=hypotheses,
        masks=_masks(0),
        evidence=torch.ones(4, 4),
        group=group,
        y=y,
    )
    unique = objective.compute(
        x=images,
        model=_RedMean(),
        unit_space=space,
        hypotheses=hypotheses,
        masks=_masks(3),
        evidence=torch.ones(4, 4),
        group=group,
        y=y,
    )
    assert float(shared["rob_var"]) < float(unique["sho_var"])
    assert float(shared["sho_var"]) < float(unique["sho_var"])
    assert torch.isfinite(shared["loss"])


def test_one_group_and_a_bad_group_tensor_are_rejected() -> None:
    images, group = _images()
    space = VisionGridUnitSpace(2, 2, baseline="mean")
    hypotheses = HypothesisSet(
        ids=torch.zeros(4, 1, dtype=torch.long),
        mask=torch.ones(4, 1, dtype=torch.bool),
    )
    objective = _objective()
    masks = _masks(3)
    model = _RedMean()
    with pytest.raises(ValueError, match="two groups"):
        objective.compute(
            x=images,
            model=model,
            unit_space=space,
            hypotheses=hypotheses,
            masks=masks,
            evidence=torch.ones(4, 4),
            group=torch.zeros(4, dtype=torch.long),
            y=torch.zeros(4, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="one id"):
        objective.compute(
            x=images,
            model=model,
            unit_space=space,
            hypotheses=hypotheses,
            masks=masks,
            evidence=torch.ones(4, 4),
            group=group[:3],
            y=torch.zeros(4, dtype=torch.long),
        )
    with pytest.raises(TypeError, match="group"):
        objective.compute(
            x=images,
            model=model,
            unit_space=space,
            hypotheses=hypotheses,
            masks=masks,
            evidence=torch.ones(4, 4),
            y=torch.zeros(4, dtype=torch.long),
        )


def test_allocator_puts_the_robust_mask_on_the_shared_square() -> None:
    images, group = _images()
    space = VisionGridUnitSpace(2, 2, baseline="mean")
    hypotheses = HypothesisSet(
        ids=torch.zeros(4, 1, dtype=torch.long),
        mask=torch.ones(4, 1, dtype=torch.bool),
    )
    y = torch.zeros(4, dtype=torch.long)
    objective = GroupStatisticsObjective(
        lambda_mean=1.0,
        lambda_var=1.0,
        lambda_gap=1.0,
        lambda_disjoint=0.2,
        lambda_sparse=0.01,
        lambda_mass=0.0,
        target="label",
    )
    allocator = RobustShortcutOptimizationAllocator(
        objective, num_steps=80, lr=0.5, lambda_disjoint=0.2, init_from_evidence=False
    )
    evidence = torch.zeros(4, 4)
    masks = allocator.allocate(
        x=images,
        model=_RedMean(),
        unit_space=space,
        hypotheses=hypotheses,
        evidence=evidence,
        env=EnvBatch(xs=[images], env_ids=["unpaired"]),
        group=group,
        y=y,
    )
    assert float(masks["robust"][:, 0].min()) > 0.9
    again = allocator.allocate(
        x=images,
        model=_RedMean(),
        unit_space=space,
        hypotheses=hypotheses,
        evidence=evidence,
        env=EnvBatch(xs=[images], env_ids=["unpaired"]),
        group=group,
        y=y,
    )
    assert torch.allclose(masks["robust"], again["robust"])
    assert torch.allclose(masks["shortcut"], again["shortcut"])


@pytest.mark.skipif(
    not (REPO / "data" / "places365" / "backgrounds" / "val_256").is_dir()
    or not (REPO / "data" / "CUB_200_2011" / "segmentations").is_dir(),
    reason="Places backgrounds or CUB masks are not on disk",
)
def test_group_objective_runs_on_waterbirds_val() -> None:
    from instantiations.shift.waterbirds import WaterbirdsClassifier

    dataset = WaterbirdsClassifier(split="val", image_size=32)
    land = [i for i, water in enumerate(dataset.use_water) if not water][:2]
    water = [i for i, flag in enumerate(dataset.use_water) if flag][:2]
    assert len(land) == 2 and len(water) == 2
    chosen = land + water
    images = torch.stack([dataset[i][0] for i in chosen])
    labels = torch.tensor([dataset[i][1] for i in chosen], dtype=torch.long)
    group = torch.tensor([0, 0, 1, 1])
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    model.eval()
    space = VisionGridUnitSpace(4, 4, baseline="mean")
    hypotheses = HypothesisSet(
        ids=labels.view(-1, 1),
        mask=torch.ones(4, 1, dtype=torch.bool),
    )
    objective = GroupStatisticsObjective(lambda_disjoint=0.2, target="label")
    allocator = RobustShortcutOptimizationAllocator(
        objective, num_steps=2, lr=0.1, lambda_disjoint=0.2, init_from_evidence=False
    )
    masks = allocator.allocate(
        x=images,
        model=model,
        unit_space=space,
        hypotheses=hypotheses,
        evidence=torch.zeros(4, 16),
        env=EnvBatch(xs=[images], env_ids=["unpaired"]),
        group=group,
        y=labels,
    )
    out = objective.compute(
        x=images,
        model=model,
        unit_space=space,
        hypotheses=hypotheses,
        masks=masks,
        evidence=torch.zeros(4, 16),
        group=group,
        y=labels,
    )
    assert masks["robust"].shape == (4, 16)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["rob_var"])
    assert torch.isfinite(out["sho_var"])
