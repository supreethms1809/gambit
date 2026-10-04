"""The shift grid and the ablations are listed and are not launched."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.splits import SplitLockedError
from scripts.final_grid import run_grid
from scripts.shift_grid import (
    SHIFT_DATASETS,
    SHIFT_METHODS,
    ablation_cells,
    main,
    marker_progress,
    shift_cells,
)


def test_a_missing_log_means_the_contrastive_grid_was_not_launched(tmp_path: Path) -> None:
    progress = marker_progress(tmp_path / "absent")
    assert progress == {"done": 0, "failed": 0, "launched": False}
    (tmp_path / "only.failed").write_text("x\n", encoding="utf-8")
    assert marker_progress(tmp_path)["launched"] is True
    assert marker_progress(tmp_path)["failed"] == 1


def test_shift_cells_cover_the_six_datasets_and_five_methods() -> None:
    cells = shift_cells()
    assert len(cells) == 6 * 2 * 5 * 5
    assert {cell["dataset"] for cell in cells} == set(SHIFT_DATASETS)
    assert {cell["method"] for cell in cells} == set(SHIFT_METHODS)
    unpaired = [cell for cell in cells if cell["dataset"] == "waterbirds_groups"]
    assert unpaired and all(cell["objective"] == "unpaired" for cell in unpaired)
    assert all(cell["objective"] == "paired" for cell in cells if cell["dataset"] != "waterbirds_groups")
    assert all("split" not in cell for cell in cells)
    assert len({cell["id"] for cell in cells}) == len(cells)


def test_ablation_rows_remove_one_component_each() -> None:
    cells = ablation_cells()
    by_id = {cell["id"]: cell for cell in cells}
    assert by_id["abl_A1_independent"]["removes"] == "joint_allocation"
    assert by_id["abl_A2_no_margin"]["removes"] == "margin"
    assert by_id["abl_A3_no_overlap"]["removes"] == "overlap"
    assert by_id["abl_A4_no_shared"]["removes"] == "shared_mask"
    assert {by_id[f"abl_A5_init_{name}"]["init"] for name in ("zero", "evidence")} == {"zero", "evidence"}
    assert {by_id[f"abl_A6_interaction_{name}"]["interaction"] for name in ("none", "attention", "transformer")} == {
        "none",
        "attention",
        "transformer",
    }
    assert [by_id[f"abl_A7_steps_{n}"]["steps"] for n in (10, 25, 50, 100)] == [10, 25, 50, 100]
    assert {by_id[f"abl_A8_preset_{name}"]["preset"] for name in ("mixed", "cooperative", "competitive")} == {
        "mixed",
        "cooperative",
        "competitive",
    }
    shift = [cell for cell in cells if cell["ablation"] == "S"]
    assert {(cell["objective"], cell["mass_target"]) for cell in shift} == {
        ("paired", True),
        ("paired", False),
        ("unpaired", True),
        ("unpaired", False),
    }
    assert len({cell["id"] for cell in cells}) == len(cells)


def test_final_is_refused_and_a_dry_cell_can_be_marked(tmp_path: Path) -> None:
    called = {"n": 0}

    def execute(_cell: dict) -> None:
        called["n"] += 1

    with pytest.raises(SplitLockedError, match="not frozen"):
        run_grid(shift_cells()[:1], tmp_path, execute, final=True)
    assert called["n"] == 0
    row = run_grid(ablation_cells()[:1], tmp_path, execute, final=False)[0]
    assert row["status"] == "done"
    assert called["n"] == 1


def test_the_module_refuses_to_launch() -> None:
    with pytest.raises(SystemExit, match="not frozen") as caught:
        main()
    assert "launched=false" in str(caught.value)
