"""S07 split rules. The on-disk check runs only when the datasets have been downloaded."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.paper_datasets import (  # noqa: E402
    S07_DATASETS,
    label_list,
    missing_classes,
    open_unsplit,
    require_every_class,
)
from evaluation.splits import load_spec, per_class_val_carve  # noqa: E402

DATA = REPO / "data"


def test_per_class_carve_keeps_every_class_on_both_sides() -> None:
    indices = list(range(50))
    label_of = {i: i // 10 for i in indices}
    train, val = per_class_val_carve(indices, label_of, seed=43)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(indices)
    for cls in range(5):
        members = {i for i in indices if label_of[i] == cls}
        assert members & set(train)
        assert members & set(val)
        assert len(members & set(val)) == 1


def test_missing_classes_reports_the_gap() -> None:
    labels = [0, 0, 1, 1]
    assert missing_classes(labels, [0, 1], 3) == [1, 2]
    assert require_every_class(labels, [0, 2], 2, "train")["0"] == 1
    with pytest.raises(RuntimeError, match="val"):
        require_every_class(labels, [0, 1], 2, "val")


@pytest.mark.parametrize("name", S07_DATASETS)
def test_s07_split_file_covers_every_class(name: str) -> None:
    path = DATA / "splits" / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"{name} split file is not written yet")
    spec = load_spec(name)
    train, val, test = (set(spec["indices"][s]) for s in ("train", "val", "test"))
    assert train.isdisjoint(val)
    if spec["index_length"]["train"] == spec["index_length"]["test"]:
        assert train.isdisjoint(test) and val.isdisjoint(test)
        assert len(train | val | test) == spec["index_length"]["train"]
    n_classes = {"cifar100": 100, "oxford_pets": 37, "cub200": 200}[name]
    for split in ("train", "val", "test"):
        ds = open_unsplit(name, split, DATA)
        assert len(ds) == spec["index_length"][split]
        counts = require_every_class(label_list(ds), spec["indices"][split], n_classes, split)
        assert min(counts.values()) >= 1
