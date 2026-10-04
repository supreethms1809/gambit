"""Extremal Perturbations: device patch, same call as TorchRay, toy square."""

from __future__ import annotations

import torch

from baselines.extremal import _import_torchray, class_masks, margin_masks
from baselines.toy import BOX, SIZE, make_both_cues, make_class_batch, region_boxes, train_region_model
from core.types import HypothesisSet
from evaluation.masks import mass_in

_, _, _, _, _, Perturbation = _import_torchray()
extremal_perturbation, *_ = _import_torchray()


def test_perturbation_to_keeps_the_moved_pyramid() -> None:
    image = torch.rand(1, 3, 8, 8)
    pyramid = Perturbation(image, num_levels=2)
    if torch.backends.mps.is_available():
        moved = pyramid.to("mps")
        assert moved.pyramid.device.type == "mps"
        return
    moved = pyramid.to(torch.device("cpu"))
    assert moved.pyramid.device.type == "cpu"
    assert moved.pyramid.data_ptr() == pyramid.pyramid.data_ptr()


def test_wrapper_matches_torchray_and_restores_the_classifier() -> None:
    torch.manual_seed(0)
    model = train_region_model(steps=2, seed=0)
    for param in model.parameters():
        param.requires_grad_(True)
    image = make_class_batch(1, label=0, seed=1)
    kwargs = dict(max_iter=3, jitter=False, step=4, sigma=4)
    direct, _ = extremal_perturbation(model, image, 0, areas=[0.1], **kwargs)
    for param in model.parameters():
        param.requires_grad_(True)
    wrapped = class_masks(model, image, torch.zeros(1, dtype=torch.long), area=0.1, **kwargs)
    assert torch.allclose(wrapped, direct[:, 0])
    assert all(param.requires_grad for param in model.parameters())


def test_known_square_is_recovered_for_the_class_and_the_margin() -> None:
    model = train_region_model(steps=30, seed=0)
    red, blue = region_boxes()
    image = make_class_batch(1, label=0, seed=3)
    area = (BOX * BOX) / (SIZE * SIZE)
    kwargs = dict(max_iter=200, jitter=False, step=2, sigma=4)
    mask = class_masks(model, image, torch.zeros(1, dtype=torch.long), area=area, **kwargs)
    on_red = mass_in(mask, red)
    on_blue = mass_in(mask, blue)
    assert float(on_red) > 0.5
    assert float(on_red) > float(on_blue)

    both = make_both_cues(1, seed=7)
    pair = HypothesisSet(
        ids=torch.tensor([[0, 1]]),
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    margin = margin_masks(model, both, pair, area=area, **kwargs)
    on_red = mass_in(margin, red)
    on_blue = mass_in(margin, blue)
    assert float(on_red) > 0.4
    assert float(on_red) > float(on_blue)


def test_two_jittered_runs_match() -> None:
    model = train_region_model(steps=1, seed=2)
    image = make_class_batch(1, label=0, seed=2)
    kwargs = dict(max_iter=2, jitter=True, step=2, sigma=4)
    first = class_masks(model, image, torch.zeros(1, dtype=torch.long), **kwargs)
    second = class_masks(model, image, torch.zeros(1, dtype=torch.long), **kwargs)
    assert torch.equal(first, second)
