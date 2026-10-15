"""Input-convention choice and the checkpoint cache key."""

from __future__ import annotations

import torch
import torch.nn as nn

import models.wrapper
from models.wrapper import (
    CONVENTION_IMAGENET,
    CONVENTION_RAW,
    INPUT_CONVENTION,
    NormalizedModel,
    choose_input_convention,
    maybe_wrap,
)
from models.build import paper_checkpoint_name


def test_tie_keeps_raw_and_a_higher_imagenet_score_wins():
    assert choose_input_convention(0.5, 0.5) == CONVENTION_RAW
    assert choose_input_convention(0.5, 0.49) == CONVENTION_RAW
    assert choose_input_convention(0.5, 0.51) == CONVENTION_IMAGENET


def test_the_convention_is_built_in_and_names_the_paper_checkpoints():
    from evaluation.run_models import paper_checkpoint_path

    assert INPUT_CONVENTION == CONVENTION_IMAGENET
    assert not hasattr(models.wrapper, "read_input_convention")
    expected = paper_checkpoint_name("cifar10", "resnet50", True, True, 15, 1e-3, 0, CONVENTION_IMAGENET)
    assert paper_checkpoint_path("cifar10", "resnet50", 0).name == expected


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


def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Flatten(), nn.Linear(4 * 8 * 8, 3))


def _saved_like_get_or_train(model: nn.Module, convention: str) -> dict:
    inner = model.model if isinstance(model, NormalizedModel) else model
    return {"state_dict": inner.state_dict(), "input_convention": convention}


def test_imagenet_checkpoint_reloads_with_its_normalisation():
    from models.wrapper import load_checkpoint_into

    trained = maybe_wrap(_tiny(), CONVENTION_IMAGENET).eval()
    blob = _saved_like_get_or_train(trained, CONVENTION_IMAGENET)
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        expected = trained(x)
        torch.manual_seed(1)
        loaded = load_checkpoint_into(_tiny(), blob).eval()
        assert isinstance(loaded, NormalizedModel)
        assert torch.allclose(loaded(x), expected, atol=1e-6)
        # The bug this guards: the bare network on raw input is a different function.
        bare = _tiny()
        bare.load_state_dict(blob["state_dict"])
        assert not torch.allclose(bare.eval()(x), expected, atol=1e-3)


def test_raw_and_legacy_checkpoints_stay_unwrapped():
    from models.wrapper import checkpoint_input_convention, load_checkpoint_into

    model = _tiny()
    legacy_metadata = {"state_dict": model.state_dict()}
    bare_state = model.state_dict()
    assert checkpoint_input_convention(legacy_metadata) == CONVENTION_RAW
    assert checkpoint_input_convention(bare_state) == CONVENTION_RAW
    assert not isinstance(load_checkpoint_into(_tiny(), legacy_metadata), NormalizedModel)
    assert not isinstance(load_checkpoint_into(_tiny(), bare_state), NormalizedModel)


def test_unknown_convention_is_refused():
    import pytest

    from models.wrapper import checkpoint_input_convention

    with pytest.raises(ValueError):
        checkpoint_input_convention({"state_dict": {}, "input_convention": "zscore"})


def test_eval_model_builder_restores_the_checkpoint_convention(tmp_path):
    from evaluation.run_models import load_cell_model
    from models.build import build_backbone

    trained = maybe_wrap(build_backbone("resnet50", 10, pretrained=False), CONVENTION_IMAGENET).eval()
    path = tmp_path / "ck.pt"
    torch.save(_saved_like_get_or_train(trained, CONVENTION_IMAGENET), path)
    loaded = load_cell_model(
        "cifar10", "resnet50", 0, source="checkpoint", checkpoint=str(path),
    ).model.eval()
    assert isinstance(loaded, NormalizedModel)
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(loaded(x), trained(x), atol=1e-5)


def test_gradcam_target_layer_sees_through_the_wrapper():
    from torchvision.models import vit_b_16

    from base_evidence.gradcam_regions import _find_target_layer

    vit = vit_b_16(weights=None)
    wrapped = maybe_wrap(vit, CONVENTION_IMAGENET)
    assert _find_target_layer(wrapped) is vit.encoder.layers[-1].ln_1
    assert _find_target_layer(wrapped) is not vit.conv_proj


def _vit_with_live_head():
    """torchvision zero-initialises the ViT head, which zeroes every gradient."""
    from torchvision.models import vit_b_16

    torch.manual_seed(0)
    vit = vit_b_16(weights=None)
    torch.nn.init.normal_(vit.heads.head.weight, std=0.02)
    return vit


def test_vit_last_block_output_has_no_patch_gradient():
    """Why the target layer is not the last block: its patch tokens never reach the head."""
    vit = _vit_with_live_head().eval()
    seen = {}

    def keep(_module, _inputs, output):
        output.retain_grad()
        seen["tokens"] = output

    handle = vit.encoder.layers[-1].register_forward_hook(keep)
    logits = vit(torch.rand(1, 3, 224, 224))
    handle.remove()
    (logits[:, 3] - logits[:, 5]).sum().backward()
    grad = seen["tokens"].grad
    assert grad[:, 0].abs().max() > 0
    assert grad[:, 1:].abs().max() == 0


def test_margin_gradcam_runs_on_a_wrapped_vit():
    from torchvision.models import vit_b_16

    from baselines.hypotheses import shared_hypotheses
    from baselines.margin import margin_gradcam

    model = maybe_wrap(_vit_with_live_head(), CONVENTION_IMAGENET).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        hypotheses = shared_hypotheses(model(x), 2)
    heat = margin_gradcam(model, x, hypotheses)
    assert heat.shape == (2, 224, 224)
    assert torch.isfinite(heat).all()
    # An all-zero map ranks every pixel equally, so its top-a mask is a tie-break.
    for image in heat:
        assert image.std() > 0


def test_gradcam_regions_provider_is_not_zero_on_a_vit():
    from base_evidence.gradcam_regions import GradCAMRegionsProvider
    from baselines.hypotheses import shared_hypotheses

    model = maybe_wrap(_vit_with_live_head(), CONVENTION_IMAGENET).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        hypotheses = shared_hypotheses(model(x), 2)
    field = GradCAMRegionsProvider(14, 14).explain(x, model, hypotheses)
    assert field.shape == (2, 2, 196)
    assert (field.flatten(2).std(dim=-1) > 0).all()


def test_gradientshap_noise_does_not_trip_the_wrapper_range_check():
    from base_evidence.library_adapters import CaptumRegionsProvider
    from baselines.hypotheses import shared_hypotheses

    torch.manual_seed(0)
    model = maybe_wrap(_tiny(), CONVENTION_IMAGENET).eval()
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        hypotheses = shared_hypotheses(model(x), 2)
    field = CaptumRegionsProvider("gradientshap", 2, 2).explain(x, model, hypotheses)
    assert field.shape == (2, 2, 4)
    assert torch.isfinite(field).all()


def test_library_cam_provider_releases_its_hooks():
    from torchvision.models import resnet18

    from base_evidence.library_adapters import CamLibraryProvider, _find_target_layer
    from baselines.hypotheses import shared_hypotheses

    model = resnet18(weights=None, num_classes=4).eval()
    layer = _find_target_layer(model)
    before = (len(layer._forward_hooks), len(layer._backward_hooks))
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        hypotheses = shared_hypotheses(model(x), 2)
    CamLibraryProvider("layercam", 2, 2).explain(x, model, hypotheses)
    assert (len(layer._forward_hooks), len(layer._backward_hooks)) == before


def test_rise_maps_carry_no_autograd_graph():
    from baselines.hypotheses import shared_hypotheses
    from baselines.rise import class_maps, margin_maps

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Linear(4, 3)).eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        hypotheses = shared_hypotheses(model(x), 2)
    assert not margin_maps(model, x, hypotheses, n_masks=8, s=4, batch=4).requires_grad
    assert not class_maps(model, x, 0, n_masks=8, s=4, batch=4).requires_grad
