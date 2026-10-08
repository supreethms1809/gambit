"""Chefer bridge (B6 + the plan's logit-equivalence test)."""

from __future__ import annotations

import pytest
import torch
import torchvision

from baselines.chefer import _import_chefer, class_relprop, from_torchvision


def test_converted_logits_match_torchvision_on_random_input() -> None:
    torch.manual_seed(0)
    src = torchvision.models.vit_b_16(weights=None)
    src.eval()
    converted = from_torchvision(src)
    x = torch.rand(1, 3, 224, 224)
    # No torch.no_grad: the vendored forward registers hooks, so it needs
    # grad enabled (no backward is called here).
    want = src(x)
    got = converted(x)
    assert torch.max(torch.abs(want - got)).item() <= 1e-4


def test_converted_logits_match_real_weights_on_a_real_photo() -> None:
    from PIL import Image

    from torchvision.models import ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    src = torchvision.models.vit_b_16(weights=weights)
    src.eval()
    converted = from_torchvision(src)
    img = Image.open("data/oxford-iiit-pet/images/Abyssinian_1.jpg").convert("RGB")
    x = weights.transforms()(img).unsqueeze(0)
    want = src(x)
    got = converted(x)
    assert torch.max(torch.abs(want - got)).item() <= 1e-4


def test_class_relprop_is_finite_and_deterministic() -> None:
    torch.manual_seed(0)
    src = torchvision.models.vit_b_16(weights=None)
    src.eval()
    converted = from_torchvision(src)
    x = torch.rand(1, 3, 224, 224)
    first = class_relprop(converted, x, 3)
    second = class_relprop(converted, x, 3)
    assert first.shape == (14, 14)
    assert bool(torch.isfinite(first).all())
    assert torch.equal(first, second)


def test_chefer_import_leaves_our_baselines_intact() -> None:
    import baselines.cve as ours_before

    _import_chefer()
    import baselines.cve as ours_after

    assert ours_before is ours_after
    assert hasattr(ours_after, "greedy_edits")


def test_non_vit_state_dict_raises() -> None:
    model = torchvision.models.resnet18()
    with pytest.raises(ValueError, match="vit_b_16"):
        from_torchvision(model)
