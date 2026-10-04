"""Input-convention choice and the checkpoint cache key."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.wrapper import (
    CONVENTION_IMAGENET,
    CONVENTION_RAW,
    NormalizedModel,
    choose_input_convention,
    maybe_wrap,
    read_input_convention,
    write_input_convention,
)
from scripts.train_backbone import paper_checkpoint_name


def test_tie_keeps_raw_and_a_higher_imagenet_score_wins():
    assert choose_input_convention(0.5, 0.5) == CONVENTION_RAW
    assert choose_input_convention(0.5, 0.49) == CONVENTION_RAW
    assert choose_input_convention(0.5, 0.51) == CONVENTION_IMAGENET


def test_convention_file_roundtrip(tmp_path):
    path = tmp_path / "input_convention.txt"
    assert read_input_convention(path) == CONVENTION_RAW
    write_input_convention(CONVENTION_IMAGENET, path)
    assert read_input_convention(path) == CONVENTION_IMAGENET


def test_raw_checkpoint_name_is_unchanged_and_imagenet_is_separate():
    raw = paper_checkpoint_name(
        "cifar10", "resnet50", True, True, 15, 1e-3, 0, CONVENTION_RAW,
    )
    imagenet = paper_checkpoint_name(
        "cifar10", "resnet50", True, True, 15, 1e-3, 0, CONVENTION_IMAGENET,
    )
    assert raw == "cifar10_resnet50_pt_lp_ep15_lr0.001_seed0.pt"
    assert imagenet == "cifar10_resnet50_pt_lp_ep15_lr0.001_imagenet_seed0.pt"
    assert raw != imagenet


def test_maybe_wrap_is_the_identity_for_raw_and_a_layer_for_imagenet():
    inner = nn.Linear(3, 2)
    assert maybe_wrap(inner, CONVENTION_RAW) is inner
    wrapped = maybe_wrap(nn.Conv2d(3, 1, 1), CONVENTION_IMAGENET)
    assert isinstance(wrapped, NormalizedModel)
    raw = torch.rand(1, 3, 4, 4)
    wrapped(raw)
