"""CVE greedy search: closed form matches cell replacement, and the toy square is edited."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from baselines.cve import earliest_edits, gap_linear_log_probs, greedy_edits, replacement_log_probs
from baselines.toy import BOX, SIZE, make_class_batch, region_boxes, train_region_model
from evaluation.masks import mass_in


def test_pool_then_linear_matches_editing_the_cells() -> None:
    torch.manual_seed(0)
    channels, height, width, classes = 4, 4, 4, 3
    current = torch.rand(channels, height, width)
    distractor = torch.rand(channels, height, width)
    linear = torch.nn.Linear(channels, classes)
    target = 2

    def decision(feat: torch.Tensor) -> torch.Tensor:
        return linear(feat.mean(dim=(2, 3)))

    free = torch.arange(height * width)
    by_edit = replacement_log_probs(distractor, decision, target)(current, free)
    by_pool = gap_linear_log_probs(distractor, linear, target)(current, free)
    assert torch.allclose(by_edit, by_pool, atol=1e-5)
    assert int(torch.argmax(by_edit).item()) == int(torch.argmax(by_pool).item())


def test_known_square_is_the_region_that_gets_replaced() -> None:
    model = train_region_model(steps=30, seed=0)
    query_image = make_class_batch(1, label=0, seed=3)
    distractor_image = make_class_batch(1, label=1, seed=4)
    assert int(model(query_image).argmax(dim=-1).item()) == 0
    assert int(model(distractor_image).argmax(dim=-1).item()) == 1

    def features(image: torch.Tensor) -> torch.Tensor:
        return F.relu(model.conv(image))[0]

    def decision(feat: torch.Tensor) -> torch.Tensor:
        return model.fc(feat.mean(dim=(2, 3)))

    query = features(query_image)
    distractor = features(distractor_image)
    edits = greedy_edits(
        query,
        distractor,
        decision,
        distractor_class=1,
        log_probs=gap_linear_log_probs(distractor, model.fc, 1),
    )
    assert edits.flipped
    red, blue = region_boxes()
    query_mask = (edits.rank > 0).to(dtype=torch.float32).unsqueeze(0)
    on_red = mass_in(query_mask, red)
    on_blue = mass_in(query_mask, blue)
    assert float(on_red) > 0.5
    assert float(on_red) > float(on_blue)

    source = torch.zeros(1, SIZE, SIZE)
    for index in edits.source_index.tolist():
        y, x = divmod(int(index), SIZE)
        source[0, y, x] = 1
    assert float(mass_in(source, blue)) > 0.5

    first = earliest_edits(edits.rank, max_cells=1)
    assert int(first.sum().item()) == 1
    y, x = divmod(int(edits.query_index[0].item()), SIZE)
    assert first[y, x] == 1
    assert int(edits.query_index.numel()) <= BOX * BOX


def test_a_map_that_already_predicts_the_distractor_is_not_edited() -> None:
    model = train_region_model(steps=5, seed=1)
    image = make_class_batch(1, label=1, seed=5)
    features = F.relu(model.conv(image))[0]

    def decision(feat: torch.Tensor) -> torch.Tensor:
        return model.fc(feat.mean(dim=(2, 3)))

    assert int(decision(features.unsqueeze(0)).argmax(dim=-1).item()) == 1
    edits = greedy_edits(
        features,
        features,
        decision,
        distractor_class=1,
        log_probs=gap_linear_log_probs(features, model.fc, 1),
    )
    assert edits.flipped
    assert int(edits.query_index.numel()) == 0
    assert int(edits.rank.sum().item()) == 0


def test_equal_scores_keep_the_earliest_cell() -> None:
    features = torch.zeros(1, 1, 2)
    distractor = torch.ones(1, 1, 2)
    linear = torch.nn.Linear(1, 2)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.copy_(torch.tensor([1.0, 0.0]))

    def decision(feat: torch.Tensor) -> torch.Tensor:
        return linear(feat.mean(dim=(2, 3)))

    edits = greedy_edits(
        features,
        distractor,
        decision,
        distractor_class=1,
        log_probs=gap_linear_log_probs(distractor, linear, 1),
        max_edits=1,
    )
    assert int(edits.query_index[0].item()) == 0
    assert int(edits.source_index[0].item()) == 0


def test_a_non_finite_feature_map_raises() -> None:
    bad = torch.zeros(1, 1, 1)
    bad[0, 0, 0] = float("nan")
    linear = torch.nn.Linear(1, 2)

    def decision(feat: torch.Tensor) -> torch.Tensor:
        return linear(feat.mean(dim=(2, 3)))

    with pytest.raises(ValueError, match="non-finite"):
        greedy_edits(
            bad,
            torch.zeros(1, 1, 1),
            decision,
            distractor_class=1,
            log_probs=lambda _current, _free: torch.zeros(1, 1),
        )
