"""Sampler is not a class prefix, and the test split stays locked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

from evaluation.sampling import seeded_indices, seeded_subset
from evaluation.splits import (
    SplitLockedError,
    VAL_CARVE_SEED,
    assert_split_allowed,
    carve_grouped,
    carve_val_from_train_pool,
    holdout_complement_indices,
    imagefolder_three_way,
    load_indices,
    write_spec,
)


class _Indexed:
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return index


def test_seeded_subset_is_not_a_class_prefix(tmp_path: Path):
    for name in ("cats", "dogs"):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(30):
            (folder / f"{i:03d}.jpg").write_bytes(b"")
    dataset = ImageFolder(str(tmp_path))
    prefix = [dataset.targets[i] for i in range(10)]
    assert len(set(prefix)) == 1

    subset = seeded_subset(dataset, 10, seed=0)
    assert subset.indices != list(range(10))
    labels = [dataset.targets[i] for i in subset.indices]
    assert len(set(labels)) == 2


def test_seeded_indices_cover_requested_count_without_replacement():
    indices = seeded_indices(50, 20, seed=1)
    assert len(indices) == 20
    assert len(set(indices)) == 20
    assert seeded_indices(50, 20, seed=1) == indices
    assert seeded_indices(50, 20, seed=2) != indices


def test_holdout_matches_random_split():
    n = 100
    n_train = int(0.8 * n)
    train_ds, holdout = random_split(
        _Indexed(n),
        [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42),
    )
    complement = holdout_complement_indices(n, seed=42)
    assert sorted(complement) == sorted(holdout.indices)
    assert set(complement).isdisjoint(train_ds.indices)


def test_val_is_carved_from_the_train_pool_only():
    n = 100
    splits = imagefolder_three_way(n, seed=42)
    complement = set(holdout_complement_indices(n, seed=42))
    assert set(splits["test"]) == complement
    assert set(splits["train"]).isdisjoint(complement)
    assert set(splits["val"]).isdisjoint(complement)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == n
    generator = torch.Generator().manual_seed(42)
    pool = torch.randperm(n, generator=generator).tolist()[: int(0.8 * n)]
    train, val = carve_val_from_train_pool(pool, seed=VAL_CARVE_SEED)
    assert set(train) == set(splits["train"])
    assert set(val) == set(splits["val"])


def test_grouped_carve_does_not_split_a_group():
    indices = list(range(20))
    group_of = {i: f"g{i // 2}" for i in indices}
    class_of = {i: "a" if i < 10 else "b" for i in indices}
    train, val = carve_grouped(indices, group_of, val_fraction=0.2, seed=0, by_class=class_of)
    train_groups = {group_of[i] for i in train}
    val_groups = {group_of[i] for i in val}
    assert train_groups.isdisjoint(val_groups)
    assert set(train).isdisjoint(val)
    assert {class_of[i] for i in train} == {"a", "b"}
    assert {class_of[i] for i in val} == {"a", "b"}


def test_test_split_refused_without_final(tmp_path: Path):
    plan = tmp_path / "EVAL_PLAN.md"
    with pytest.raises(SplitLockedError, match="--final"):
        assert_split_allowed("test", final=False, config_hash="abc", eval_plan_path=plan)
    assert_split_allowed("val", final=False, config_hash=None, eval_plan_path=plan)


def test_test_split_requires_hash_in_frozen_plan(tmp_path: Path):
    plan = tmp_path / "EVAL_PLAN.md"
    plan.write_text("status: eval-plan-frozen\nconfig deadbeef\n", encoding="utf-8")
    assert_split_allowed("test", final=True, config_hash="deadbeef", eval_plan_path=plan)
    with pytest.raises(SplitLockedError, match="not listed"):
        assert_split_allowed("test", final=True, config_hash="nope", eval_plan_path=plan)
    plain = tmp_path / "plain.md"
    plain.write_text("draft, not frozen\n", encoding="utf-8")
    with pytest.raises(SplitLockedError, match="not frozen"):
        assert_split_allowed("test", final=True, config_hash="deadbeef", eval_plan_path=plain)
    with pytest.raises(SplitLockedError, match="not found"):
        assert_split_allowed(
            "test",
            final=True,
            config_hash="deadbeef",
            eval_plan_path=tmp_path / "missing.md",
        )


def test_recorded_splits_keep_train_and_val_disjoint():
    root = Path(__file__).resolve().parents[1] / "data" / "splits"
    for name in ("mnist", "cifar10", "pets", "stanford_dogs", "ham10000", "brain_tumor"):
        spec = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        train, val, test = (set(spec["indices"][s]) for s in ("train", "val", "test"))
        assert train.isdisjoint(val)
        assert train and val and test
        if spec["index_length"]["train"] == spec["index_length"]["test"]:
            assert train.isdisjoint(test) and val.isdisjoint(test)
            assert len(train | val | test) == spec["index_length"]["train"]
    ham = json.loads((root / "ham10000.json").read_text(encoding="utf-8"))
    brain = json.loads((root / "brain_tumor.json").read_text(encoding="utf-8"))
    assert ham["roots"]["val"] == "ham10000/train"
    assert ham["roots"]["test"] == "ham10000/val"
    assert brain["roots"]["val"] == "brain_tumor/Training"
    assert brain["roots"]["test"] == "brain_tumor/Testing"


def test_load_indices_checks_length_and_lock(tmp_path: Path):
    spec = {
        "dataset": "toy",
        "indices": {"train": [0, 1], "val": [2], "test": [3, 4]},
        "index_length": {"train": 5, "val": 5, "test": 5},
    }
    write_spec("toy", spec, splits_dir=tmp_path)
    assert load_indices("toy", "val", splits_dir=tmp_path, n_items=5) == [2]
    with pytest.raises(RuntimeError, match="length"):
        load_indices("toy", "val", splits_dir=tmp_path, n_items=4)
    with pytest.raises(SplitLockedError):
        load_indices(
            "toy",
            "test",
            splits_dir=tmp_path,
            eval_plan_path=tmp_path / "EVAL_PLAN.md",
            n_items=5,
        )
    # The spec on disk is the one just written, not a hand-edited number.
    saved = json.loads((tmp_path / "toy.json").read_text(encoding="utf-8"))
    assert saved["indices"]["test"] == [3, 4]
