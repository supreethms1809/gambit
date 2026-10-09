"""Chunked cells match a full-sample call.

The mask step is Adam on a mean loss. A chunk's mean is larger than the
full-sample mean by N/C, and Adam's epsilon makes that scale visible, so the
chunk multiplies the loss by C/N. ROAD noise and tie-breaks walk the sample
in order, so a chunk skips that prefix.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.types import EnvBatch, HypothesisSet
from evaluation.masks import advance_generator, top_fraction_mask
from evaluation.run_cell import _ranges, _whole_sample
from evaluation.scores import _remove
from instantiations.contrastive.allocator import OptimizationAllocator
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from instantiations.shift.objective import RobustShortcutObjective
from modality.grid_regions import VisionGridUnitSpace


class TinyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def test_ranges_and_which_methods_stay_whole():
    assert _ranges(64, 4) == [(i, i + 4) for i in range(0, 64, 4)]
    assert _ranges(5, 4) == [(0, 4), (4, 5)]
    assert _ranges(5, 0) == [(0, 5)]
    assert _whole_sample("cve", None)
    assert _whole_sample("random_floor", None)
    assert _whole_sample("spray", None)
    assert _whole_sample("cdea_shift", "AS1_unpaired_mass")
    assert not _whole_sample("cdea", None)
    assert not _whole_sample("cdea_shift", None)
    assert not _whole_sample("extremal", None)


def test_tie_break_and_road_noise_match_the_full_sample():
    scores = torch.tensor([[1.0, 1.0, 0.2, 0.2], [0.0, 3.0, 3.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    full = top_fraction_mask(scores, 0.5, seed=7)
    parts = [top_fraction_mask(scores[i:i + 1], 0.5, seed=7, offset=i) for i in range(3)]
    assert torch.equal(full, torch.cat(parts, dim=0))

    generator = torch.Generator().manual_seed(0)
    images = torch.rand(4, 3, 8, 8)
    keep = torch.zeros(4, 1, 8, 8)
    keep[:, :, :2, :] = 1
    advance_generator(generator, 0, (3, 8, 8), normal=True)
    whole = _remove(images, keep.squeeze(1), iters=2, noise=0.01, seed=11)
    chunks = torch.cat([
        _remove(images[s:e], keep[s:e].squeeze(1), iters=2, noise=0.01, seed=11, offset=s)
        for s, e in ((0, 2), (2, 4))
    ])
    assert torch.equal(whole, chunks)


def test_contrastive_chunks_match_the_full_sample_mask_step():
    torch.manual_seed(0)
    model = TinyCNN()
    model.eval()
    unit = VisionGridUnitSpace(4, 4)
    objective = ContrastiveObjective()
    x = torch.rand(4, 3, 16, 16)
    hypotheses = HypothesisSet(ids=torch.tensor([[0, 1, 2]]).expand(4, 3).contiguous(),
                               mask=torch.ones(4, 3, dtype=torch.bool))
    evidence = torch.rand(4, 3, 16)
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)
    common = dict(objective=objective, num_steps=6, lr=0.4, use_shared=True, lambda_partition=0.1)
    full = OptimizationAllocator(**common).allocate(
        x=x, model=model, unit_space=unit, hypotheses=hypotheses, evidence=evidence)
    parts = []
    for start in (0, 2):
        sl = slice(start, start + 2)
        h = HypothesisSet(ids=hypotheses.ids[sl], mask=hypotheses.mask[sl])
        part = OptimizationAllocator(**common, loss_scale=0.5).allocate(
            x=x[sl], model=model, unit_space=unit, hypotheses=h, evidence=evidence[sl])
        parts.append(part)
    for key in ("unique", "shared"):
        assert torch.allclose(full[key], torch.cat([p[key] for p in parts], dim=0), atol=1e-5)


def test_shift_chunks_match_the_full_sample_mask_step():
    torch.manual_seed(1)
    model = TinyCNN()
    model.eval()
    unit = VisionGridUnitSpace(4, 4)
    objective = RobustShortcutObjective()
    x = torch.rand(4, 3, 16, 16)
    hypotheses = HypothesisSet(ids=torch.zeros(4, 1, dtype=torch.long), mask=torch.ones(4, 1, dtype=torch.bool))
    evidence = torch.rand(4, 16)
    evidence = evidence / evidence.sum(dim=-1, keepdim=True)

    def env_of(batch):
        return EnvBatch(xs=[batch, batch.flip(-1)], env_ids=["id", "ood"])

    common = dict(objective=objective, num_steps=4, lr=0.3, lambda_disjoint=0.2, init_seed=3)
    full = RobustShortcutOptimizationAllocator(**common).allocate(
        x=x, model=model, unit_space=unit, hypotheses=hypotheses, evidence=evidence, env=env_of(x))
    parts = []
    for start in (0, 2):
        sl = slice(start, start + 2)
        h = HypothesisSet(ids=hypotheses.ids[sl], mask=hypotheses.mask[sl])
        part = RobustShortcutOptimizationAllocator(**common, loss_scale=0.5, init_offset=start).allocate(
            x=x[sl], model=model, unit_space=unit, hypotheses=h, evidence=evidence[sl], env=env_of(x[sl]))
        parts.append(part)
    for key in ("robust", "shortcut"):
        assert torch.allclose(full[key], torch.cat([p[key] for p in parts], dim=0), atol=1e-5)


def test_margin_ig_internal_batch_matches_one_batch():
    from captum.attr import IntegratedGradients

    torch.manual_seed(2)
    model = TinyCNN()
    model.eval()
    x = torch.rand(4, 3, 16, 16)

    def forward(inp):
        return model(inp)[:, 0]

    whole = IntegratedGradients(forward).attribute(x, baselines=torch.zeros_like(x), n_steps=4, method="riemann_right")
    chunked = IntegratedGradients(forward).attribute(
        x, baselines=torch.zeros_like(x), n_steps=4, method="riemann_right", internal_batch_size=2)
    assert torch.allclose(whole, chunked, atol=1e-5)


class Tiny224(nn.Module):
    """A small CNN on 224 px input, so the cell runs on CPU in a test."""

    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv2(torch.relu(self.conv1(x))))).flatten(1))


def _cell_records(tmp_path, monkeypatch, image_batch):
    import csv
    import gzip
    from types import SimpleNamespace

    import evaluation.run_data as run_data
    import evaluation.run_models as run_models
    from evaluation.run_cell import CellSpec, run_contrastive
    from evaluation.run_data import ContrastiveSample
    from evaluation.run_methods import Knobs

    torch.manual_seed(5)
    model = Tiny224().eval()
    x = torch.rand(6, 3, 224, 224)
    monkeypatch.setattr(run_models, "load_cell_model", lambda *a, **k: SimpleNamespace(
        model=model, num_classes=6, path="tiny", source="smoke"))
    monkeypatch.setattr(run_data, "contrastive_sample", lambda *a, **k: ContrastiveSample(
        x=x.clone(), labels=torch.zeros(6, dtype=torch.long), index=list(range(6))))
    spec = CellSpec(game="contrastive", dataset="cifar10", backbone="resnet50", seed=0, split="val", n=6,
                    methods=("cdea",), areas=(0.05,), operators=("road",),
                    knobs=Knobs(cdea_steps=20, ig_steps=2), out_dir=str(tmp_path / f"ib{image_batch}"),
                    image_batch=image_batch)
    out = run_contrastive(spec, torch.device("cpu"))
    with gzip.open(out / "records.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: (r["method"], int(r["image_index"])))


def test_cdea_cell_chunks_match_the_full_sample(tmp_path, monkeypatch):
    """The primary CDEA row must get the chunk's loss scale, like ablations and candidates."""
    whole = _cell_records(tmp_path, monkeypatch, 0)
    chunked = _cell_records(tmp_path, monkeypatch, 4)  # 4 + 2: uneven chunks
    assert [r["method"] for r in whole] == [r["method"] for r in chunked]
    assert {r["method"] for r in whole} == {"cdea", "cdea+shared"}
    for a, b in zip(whole, chunked):
        for key in ("cd", "cd1", "z_k_without_k", "z_l_without_k", "z_k_without_l", "z_l_without_l"):
            assert abs(float(a[key]) - float(b[key])) <= 1e-5, (a["method"], a["image_index"], key, a[key], b[key])
