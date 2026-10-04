"""S22. The final grids are not complete, and the tag is not created."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.completion_check import (
    CONTRASTIVE_DATASETS,
    CONTRASTIVE_METHODS,
    assert_run_record,
    completion_report,
    contrastive_cells,
    main,
    runs_complete,
)

ROOT = Path(__file__).resolve().parents[1]


def test_failed_cells_are_rerun_and_done_cells_stay_finished(tmp_path: Path) -> None:
    cells = [
        {"id": "keep"},
        {"id": "again"},
        {"id": "wait"},
    ]
    (tmp_path / "keep.done").write_text("keep\n", encoding="utf-8")
    (tmp_path / "again.failed").write_text("boom\n", encoding="utf-8")
    report = completion_report(cells, tmp_path)
    assert report["complete"] is False
    assert report["done"] == 1
    assert [cell["id"] for cell in report["rerun"]] == ["again"]
    assert [cell["id"] for cell in report["waiting"]] == ["wait"]


def test_every_done_marker_completes_a_manifest(tmp_path: Path) -> None:
    cells = [{"id": "one"}, {"id": "two"}]
    for cell in cells:
        (tmp_path / f"{cell['id']}.done").write_text("ok\n", encoding="utf-8")
    assert completion_report(cells, tmp_path)["complete"] is True


def test_contrastive_manifest_names_the_eight_datasets() -> None:
    cells = contrastive_cells()
    assert len(cells) == 8 * 2 * 7 * 5
    assert {cell["dataset"] for cell in cells} == set(CONTRASTIVE_DATASETS)
    assert "imagenet_s" in CONTRASTIVE_DATASETS
    assert {cell["method"] for cell in cells} == set(CONTRASTIVE_METHODS)
    assert all("split" not in cell for cell in cells)


def test_run_record_assertions_accept_a_matching_record_and_reject_a_bad_one() -> None:
    record = {
        "masks": [[1.0, 0.0, 0.0, 0.0]],
        "probabilities": [0.25, 0.75],
        "n": 4,
    }
    assert_run_record(record, area=0.25, tolerance=1e-6, n=4)
    with pytest.raises(AssertionError, match="outside"):
        assert_run_record({**record, "masks": [[1.1, 0.0, 0.0, 0.0]]}, area=0.25, tolerance=1.0, n=4)
    with pytest.raises(AssertionError, match="not finite"):
        assert_run_record({**record, "probabilities": [float("nan")]}, area=0.25, tolerance=1.0, n=4)
    with pytest.raises(AssertionError, match="area"):
        assert_run_record(record, area=0.05, tolerance=1e-3, n=4)
    with pytest.raises(AssertionError, match="n does not match"):
        assert_run_record(record, area=0.25, tolerance=1e-6, n=8)


def test_the_real_grids_are_not_complete_and_the_tag_is_absent() -> None:
    report = runs_complete()
    assert report["complete"] is False
    assert report["pending"] > 0
    text = (ROOT / "docs" / "paper" / "FINAL_RUNS.md").read_text(encoding="utf-8")
    assert "Final runs are not complete" in text
    assert "final-runs-v1" in text
    assert "was not created" in text
    assert "ImageNet-S" in text
    tags = subprocess.check_output(
        ["git", "tag", "--list", "final-runs-v1"],
        cwd=ROOT,
        text=True,
    )
    assert tags.strip() == ""
    with pytest.raises(SystemExit, match="not complete") as caught:
        main()
    message = str(caught.value)
    assert "was not created" in message
    assert "pending=" in message
