"""ImageNet unit: val is a per-class 20% carve of ImageNet val, test is the rest."""

from __future__ import annotations

from PIL import Image

import scripts.write_paper_splits as wps


def test_imagenet_split_carves_two_of_ten_per_class(tmp_path, monkeypatch):
    for wnid in ("n01", "n02", "n03"):
        folder = tmp_path / "imagenet" / "val" / wnid
        folder.mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (8, 8)).save(folder / f"{i}.JPEG")
    monkeypatch.setattr(wps, "DATA", tmp_path)
    spec = wps._imagenet()
    val, test = spec["indices"]["val"], spec["indices"]["test"]
    assert spec["indices"]["train"] == []
    assert len(val) == 6 and len(test) == 24
    assert not set(val) & set(test)
    assert sorted(val + test) == list(range(30))
    assert {v // 10 for v in val} == {0, 1, 2}
