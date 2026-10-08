"""SC-CVE (B6): toy square, identity against the vendored function, edges, seeding."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.sc_cve import (
    NoFlipError,
    edits_to_rank,
    edits_to_scores,
    run_sc_cve_edits,
)
from baselines.toy import make_class_batch, train_region_model
from evaluation.masks import mass_in


def _toy_head(model):
    def decision(feat: torch.Tensor) -> torch.Tensor:
        return model.fc(feat.mean(dim=(2, 3)))

    return decision


def test_known_square_is_the_region_that_gets_replaced() -> None:
    # The toy grid is 32x32, but the vendored search scores every
    # query-x-distractor pair per step, so it only ever runs on small maps
    # (upstream: 7x7). Pool the ReLU features to 4x4 first: average pooling
    # preserves the spatial mean, so the fc head decides exactly as on the
    # full grid, and the red/blue squares still sit in known pooled cells.
    model = train_region_model(steps=30, seed=0)
    query_image = make_class_batch(1, label=0, seed=3)
    distractor_image = make_class_batch(1, label=1, seed=4)
    assert int(model(query_image).argmax(dim=-1).item()) == 0
    assert int(model(distractor_image).argmax(dim=-1).item()) == 1

    def features(image: torch.Tensor) -> torch.Tensor:
        full = F.relu(model.conv(image))[0]
        return F.adaptive_avg_pool2d(full.unsqueeze(0), (4, 4))[0]

    query = features(query_image)
    distractor = features(distractor_image).unsqueeze(0)
    edits = run_sc_cve_edits(
        query, distractor, _toy_head(model), 1,
        lambd=0.0, temperature=None, topk=None, device="cpu",
    )
    assert len(edits) > 0
    # The flip actually happened: replay the edits through the head.
    current = query.clone()
    height, width = query.shape[-2:]
    for query_cell, source_cell in edits:
        y, x = divmod(query_cell, width)
        sy, sx = divmod(source_cell, width)
        current[:, y, x] = distractor[0][:, sy, sx]
    assert int(_toy_head(model)(current.unsqueeze(0)).argmax(-1).item()) == 1
    rank = edits_to_rank(edits, height, width, "cpu")
    red = torch.zeros(1, height, width)
    blue = torch.zeros(1, height, width)
    red[:, :2, :2] = 1    # rows/cols 2-10 of 32 px pool into cells 0-1
    blue[:, 2:, 2:] = 1   # rows/cols 22-30 pool into cells 2-3
    on_red = mass_in((rank > 0).to(torch.float32).unsqueeze(0), red)
    on_blue = mass_in((rank > 0).to(torch.float32).unsqueeze(0), blue)
    assert float(on_red) > 0.5
    assert float(on_red) > float(on_blue)
    scores = edits_to_scores(rank)
    assert float(scores.max().item()) == float(len(edits))
    first = int(torch.argmax(scores).item())
    y, x = divmod(first, width)
    assert int(rank[y, x].item()) == 1


def test_wrapper_matches_the_vendored_function() -> None:
    sys.path.insert(0, "third_party/sc_cve")
    try:
        from counterfactuals.explainer.counterfactuals import compute_counterfactual
    finally:
        sys.path.remove("third_party/sc_cve")
    torch.manual_seed(0)
    channels, height, width, classes = 4, 4, 4, 3
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(channels, classes))
    query = torch.rand(channels, height, width)
    distractors = torch.rand(2, channels, height, width)
    expected = [
        (int(q), int(s))
        for q, s in compute_counterfactual(
            query, distractors, head, 2, None, None,
            lambd=0.0, temperature=None, topk=None, device="cpu",
        )
    ]
    got = run_sc_cve_edits(
        query, distractors, head, 2,
        lambd=0.0, temperature=None, topk=None, device="cpu",
    )
    assert got == expected
    assert all(isinstance(q, int) and isinstance(s, int) for q, s in got)


def test_aux_path_encodes_consistently() -> None:
    torch.manual_seed(0)
    channels, height, width, classes, aux = 4, 4, 4, 3, 8
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(channels, classes))
    query = torch.rand(channels, height, width)
    distractors = torch.rand(2, channels, height, width)
    query_aux = torch.rand(aux, height, width)
    distractor_aux = torch.rand(2, aux, height, width)
    try:
        edits = run_sc_cve_edits(
            query, distractors, head, 2,
            lambd=0.4, temperature=0.1, topk=0.5,
            query_aux=query_aux, distractor_aux=distractor_aux, device="cpu",
        )
    except NoFlipError:
        edits = []
    rank = edits_to_rank(edits, height, width, "cpu")
    assert int(rank.max().item()) == len(edits)
    scores = edits_to_scores(rank)
    assert int((scores > 0).sum().item()) == len(edits)


def test_a_non_finite_feature_map_raises() -> None:
    bad = torch.zeros(4, 4, 4)
    bad[0, 0, 0] = float("nan")
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(4, 3))
    with pytest.raises(ValueError, match="non-finite"):
        run_sc_cve_edits(
            bad, torch.zeros(1, 4, 4, 4), head, 1,
            lambd=0.0, temperature=None, topk=None, device="cpu",
        )


def test_exhaustion_without_a_flip_is_noflip() -> None:
    torch.manual_seed(1)
    channels, height, width, classes = 4, 4, 4, 3
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(channels, classes))
    with pytest.raises(NoFlipError):
        run_sc_cve_edits(
            torch.rand(channels, height, width), torch.rand(2, channels, height, width),
            head, 2, lambd=0.0, temperature=None, topk=None, device="cpu",
        )


def test_same_inputs_give_same_edits() -> None:
    torch.manual_seed(0)
    channels, height, width, classes = 4, 4, 4, 3
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(channels, classes))
    query = torch.rand(channels, height, width)
    distractors = torch.rand(2, channels, height, width)
    kwargs = dict(lambd=0.0, temperature=None, topk=None, device="cpu")
    first = run_sc_cve_edits(query, distractors, head, 2, **kwargs)
    second = run_sc_cve_edits(query, distractors, head, 2, **kwargs)
    assert first == second


def test_distractor_sampling_is_seeded() -> None:
    import torchvision.models

    from evaluation.run_methods import find_distractors

    model = torchvision.models.resnet18()
    model.eval()
    first = [(i, bool((img > 0).any())) for i, img in
             find_distractors(model, "cifar10", 0, seed=7, tries=4, device="cpu", n=2)]
    second = [(i, bool((img > 0).any())) for i, img in
              find_distractors(model, "cifar10", 0, seed=7, tries=4, device="cpu", n=2)]
    assert [i for i, _ in first] == [i for i, _ in second]
