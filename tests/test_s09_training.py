"""S09 training grid and val metrics. No checkpoint is trained here."""

from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.accuracy import top1_and_balanced, worst_group_accuracy
from instantiations.shift.biased_data import ColoredMNIST
from instantiations.shift.waterbirds import confounder_uses_water
from scripts.launch_paper_training import CONTRASTIVE, SHIFT, paper_cells
from scripts.train_backbone import _build_model


def test_scores_match_a_known_prediction():
    pred = torch.tensor([0, 0, 0, 1])
    target = torch.tensor([0, 0, 1, 1])
    top1, balanced = top1_and_balanced(pred, target, num_classes=3)
    assert top1 == 0.75
    assert balanced == 0.75  # class 2 is absent; class 0 recall 1, class 1 recall 0.5


def test_worst_group_is_the_minimum():
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 0, 1, 0])
    group = torch.tensor([0, 0, 1, 1])
    assert worst_group_accuracy(pred, target, group) == 0.5


def test_paper_grid_is_five_seeds_and_skips_imagenet_s():
    cells = paper_cells()
    assert len(cells) == (len(CONTRASTIVE) + len(SHIFT)) * 2 * 5
    assert {c["seed"] for c in cells} == {0, 1, 2, 3, 4}
    names = {c["dataset"] for c in cells}
    assert "mnist" not in names
    assert "pets" not in names
    assert names == set(CONTRASTIVE) | set(SHIFT)
    probes = [c for c in cells if c["dataset"] == "cifar10"]
    assert all(c["freeze_backbone"] and c["lr"] == 1e-3 for c in probes)
    shifts = [c for c in cells if c["dataset"] == "colored_mnist"]
    assert all(not c["freeze_backbone"] and c["lr"] == 1e-4 for c in shifts)


def test_resnet50_head_matches_the_dataset():
    model = _build_model("resnet50", 17, pretrained=False)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 17
    assert model.fc.in_features == 2048


def test_waterbird_confounder_rate():
    labels = [1] * 2000 + [0] * 2000
    train = confounder_uses_water(labels, "train", seed=43)
    assert abs(sum(train[:2000]) / 2000 - 0.95) < 0.03
    assert abs(sum(train[2000:]) / 2000 - 0.05) < 0.03
    val = confounder_uses_water(labels, "val", seed=43)
    assert abs(sum(val[:2000]) / 2000 - 0.5) < 0.05
    assert confounder_uses_water(labels, "train", seed=43) == train


def test_colored_mnist_seed_fixes_the_colors():
    first = ColoredMNIST(root="data", train=True, download=False, correlation=0.9, seed=43)
    second = ColoredMNIST(root="data", train=True, download=False, correlation=0.9, seed=43)
    other = ColoredMNIST(root="data", train=True, download=False, correlation=0.9, seed=44)
    assert torch.equal(first._colors, second._colors)
    assert not torch.equal(first._colors, other._colors)
    image_a, _label_a = first[0]
    image_b, _label_b = second[0]
    assert torch.equal(image_a, image_b)
