"""S14. Contrastive Grad-CAM and RISE on the toy squares.

The published qualitative figures and the ImageNet deletion/insertion table
are not this check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines.adapter import budget_or_floor
from baselines.contrastive_gradcam import CrossEntropyContrastTarget, contrastive_gradcam
from baselines.hypotheses import shared_hypotheses
from baselines.rise import class_maps, margin_maps
from baselines.toy import BOX, SIZE, make_both_cues, make_class_batch, region_boxes, train_region_model
from core.types import HypothesisSet


def _recall(attr: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    fraction = float(BOX * BOX) / float(SIZE * SIZE)
    chosen, failed = budget_or_floor(attr, fraction, seed=0)
    assert not bool(failed.any())
    if box.shape[0] == 1 and attr.shape[0] != 1:
        box = box.expand(attr.shape[0], -1, -1)
    hit = (chosen * box).flatten(1).sum(dim=-1)
    return hit / box[0].sum()


def _pair(batch: int) -> HypothesisSet:
    return HypothesisSet(
        ids=torch.tensor([[0, 1]]).expand(batch, 2).contiguous(),
        mask=torch.ones(batch, 2, dtype=torch.bool),
    )


def test_contrast_target_is_cross_entropy_toward_the_foil() -> None:
    logits = torch.tensor([[1.5, -0.5]])
    target = CrossEntropyContrastTarget(1)
    assert torch.allclose(target(logits), F.cross_entropy(logits, torch.tensor([1])))
    assert torch.allclose(target(logits[0]), F.cross_entropy(logits, torch.tensor([1])))
    margin = logits[0, 0] - logits[0, 1]
    assert not torch.allclose(target(logits), margin)


def test_contrastive_gradcam_recovers_the_kept_square() -> None:
    model = train_region_model(steps=40, seed=0)
    red, blue = region_boxes()
    x = make_class_batch(2, label=0, seed=3)
    hypotheses = shared_hypotheses(model(x), k=2)
    heat = contrastive_gradcam(model, x, hypotheses)
    on_red = _recall(heat, red)
    on_blue = _recall(heat, blue)
    assert bool((on_red > 0.5).all()), on_red.tolist()
    assert bool((on_red > on_blue).all())

    both = make_both_cues(2, seed=7)
    heat = contrastive_gradcam(model, both, _pair(2))
    assert bool((_recall(heat, red) > _recall(heat, blue)).all())


def test_rise_class_and_margin_recover_the_known_square() -> None:
    model = train_region_model(steps=40, seed=0)
    red, blue = region_boxes()
    x = make_class_batch(2, label=0, seed=3)
    heat = class_maps(model, x, class_idx=0, n_masks=256, s=8, p1=0.5, seed=0, batch=64)
    assert bool((_recall(heat, red) > 0.5).all())
    assert bool((_recall(heat, red) > _recall(heat, blue)).all())

    both = make_both_cues(2, seed=7)
    margin = margin_maps(model, both, _pair(2), n_masks=64, s=8, p1=0.5, seed=0, batch=32)
    assert bool((_recall(margin, red) > 0.5).all())
    assert bool((_recall(margin, red) > _recall(margin, blue)).all())


def test_rise_class_maps_match_the_vendored_forward() -> None:
    model = train_region_model(steps=5, seed=1)
    x = make_class_batch(1, label=0, seed=4)
    via = class_maps(model, x, class_idx=0, n_masks=16, s=4, p1=0.5, seed=2, batch=8)

    path = Path(__file__).resolve().parents[1] / "third_party" / "rise" / "explanations.py"
    spec = importlib.util.spec_from_file_location("rise_official_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    explainer = module.RISE(model, (SIZE, SIZE), gpu_batch=8, device=x.device)
    np.random.seed(2)
    explainer.generate_masks(16, 4, 0.5, savepath=None)
    direct = explainer(x)[0]
    assert torch.allclose(via[0], direct)


def test_rise_repeat_matches_and_bad_arguments_raise() -> None:
    model = train_region_model(steps=2, seed=2)
    x = make_class_batch(1, label=0, seed=5)
    kwargs = dict(class_idx=0, n_masks=8, s=4, p1=0.5, seed=1, batch=4)
    first = class_maps(model, x, **kwargs)
    assert torch.equal(first, class_maps(model, x, **kwargs))
    model.train()
    class_maps(model, x, **kwargs)
    assert model.training

    try:
        class_maps(model, x, class_idx=0, n_masks=0, s=4, p1=0.5)
        raise AssertionError("zero masks should raise")
    except ValueError:
        pass
    try:
        class_maps(model, x, class_idx=0, n_masks=4, s=4, p1=0.0)
        raise AssertionError("p1 of zero should raise")
    except ValueError:
        pass
    one = HypothesisSet(ids=torch.zeros(1, 1, dtype=torch.long), mask=torch.ones(1, 1, dtype=torch.bool))
    try:
        margin_maps(model, x, one, n_masks=4, s=4)
        raise AssertionError("a single hypothesis has no foil")
    except ValueError:
        pass
