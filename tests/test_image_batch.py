"""A chunked cell scores every image as one call on the whole sample would.

ROAD noise and tie-breaks walk the sample in order, so a chunk skips the
prefix before it. A chunked method and a whole-sample method then see the
same draws for the same image.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from evaluation.masks import top_fraction_mask
from evaluation.scores import _remove


class TinyCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def test_tie_break_and_road_noise_match_the_full_sample():
    scores = torch.tensor([[1.0, 1.0, 0.2, 0.2], [0.0, 3.0, 3.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    full = top_fraction_mask(scores, 0.5, seed=7)
    parts = [top_fraction_mask(scores[i:i + 1], 0.5, seed=7, offset=i) for i in range(3)]
    assert torch.equal(full, torch.cat(parts, dim=0))

    images = torch.rand(4, 3, 8, 8)
    mask = torch.zeros(4, 8, 8)
    mask[:, 2:, :] = 1
    for iters in (2, None):
        whole = _remove(images, mask, iters=iters, noise=0.01, seed=11)
        chunks = torch.cat([_remove(images[s:e], mask[s:e], iters=iters, noise=0.01, seed=11, offset=s)
                            for s, e in ((0, 2), (2, 4))])
        assert torch.equal(whole, chunks)


def test_a_chunked_cell_scores_like_the_whole_sample():
    from baselines.adapter import budget_or_floor
    from evaluation.run_cell import CellSpec, _score_pair
    from evaluation.run_methods import PairMaps, hypotheses_for

    torch.manual_seed(0)
    model = TinyCNN().eval()
    x = torch.rand(4, 3, 224, 224)
    h = hypotheses_for(model, x, 5)
    k, l = h.ids[:, 0], h.ids[:, 1]
    full = model(x).detach()
    maps = PairMaps(k=torch.rand(4, 49).round(decimals=1), l=torch.rand(4, 49).round(decimals=1), grid=(7, 7))
    spec = CellSpec(dataset="cifar10", backbone="resnet50", operators=("road",))
    sample = SimpleNamespace(index=list(range(4)), labels=[0, 1, 2, 3])

    def score(start, end):
        part = PairMaps(k=maps.k[start:end], l=maps.l[start:end], grid=maps.grid)
        return _score_pair(model, x[start:end], part, k[start:end], l[start:end], full[start:end], 0.05,
                           spec, "base_evidence", sample, budget_or_floor, start, "code", "knobs")

    whole = score(0, 4)
    chunked = score(0, 2) + score(2, 4)
    for key in ("cd", "cd1", "sc", "z_k_without_k", "z_l_without_l"):
        assert [row[key] for row in whole] == [row[key] for row in chunked], key


def test_margin_ig_internal_batch_matches_one_batch():
    from captum.attr import IntegratedGradients

    torch.manual_seed(2)
    model = TinyCNN().eval()
    x = torch.rand(4, 3, 16, 16)

    def forward(inp):
        return model(inp)[:, 0]

    whole = IntegratedGradients(forward).attribute(x, baselines=torch.zeros_like(x), n_steps=4, method="riemann_right")
    chunked = IntegratedGradients(forward).attribute(
        x, baselines=torch.zeros_like(x), n_steps=4, method="riemann_right", internal_batch_size=4)
    assert torch.allclose(whole, chunked, atol=1e-5)
