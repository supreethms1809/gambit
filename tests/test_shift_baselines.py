"""Shift baselines: the patch that exists in only one environment, and matched Extremal masks."""

from __future__ import annotations

import pytest
import torch

from baselines.extremal import class_masks
from baselines.shift_maps import environment_maps, per_environment_extremal
from baselines.toy import make_class_batch, region_boxes, train_region_model
from evaluation.masks import mass_in


def test_the_id_only_patch_is_the_shortcut_and_the_shared_square_is_robust() -> None:
    red, _blue = region_boxes()
    corner = torch.zeros_like(red)
    corner[:, 2:10, 22:30] = 1
    robust, shortcut = environment_maps([red + corner, red.clone(), red + corner])
    assert float(mass_in(shortcut, corner)) == pytest.approx(1.0)
    assert float(mass_in(robust, red)) == pytest.approx(1.0)
    assert float(shortcut.sum()) == pytest.approx(float(corner.sum()))
    assert float(robust.sum()) == pytest.approx(float(red.sum()))


def test_a_single_environment_and_a_non_finite_map_are_rejected() -> None:
    red, _blue = region_boxes()
    with pytest.raises(ValueError, match="at least one"):
        environment_maps([red])
    bad = red.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        environment_maps([red, bad])


def test_identical_environments_leave_the_extremal_mask_as_robust() -> None:
    model = train_region_model(steps=2, seed=0)
    image = make_class_batch(1, label=0, seed=1)
    labels = torch.zeros(1, dtype=torch.long)
    kwargs = dict(max_iter=2, jitter=False, step=2, sigma=4, area=0.1)
    robust, shortcut = per_environment_extremal(model, [image, image.clone()], labels, **kwargs)
    direct = class_masks(model, image, labels, **kwargs)
    assert torch.allclose(robust, direct)
    assert torch.equal(shortcut, torch.zeros_like(shortcut))
