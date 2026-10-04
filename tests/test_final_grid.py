"""The final grid skips finished cells and refuses to launch while the plan is a draft."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.splits import EVAL_PLAN_PATH, SplitLockedError, assert_split_allowed
from scripts.final_grid import main, pending_cells, run_grid


def test_finished_cells_are_skipped_and_failures_are_marked(tmp_path: Path) -> None:
    cells = [
        {"dataset": "cifar10", "model": "resnet50", "method": "cdea", "seed": 0},
        {"dataset": "cifar10", "model": "resnet50", "method": "cdea", "seed": 1},
    ]
    done = tmp_path / "cifar10_resnet50_cdea_seed0.done"
    done.write_text("already\n", encoding="utf-8")
    seen: list[str] = []

    def execute(cell: dict) -> None:
        seen.append(cell["id"] if "id" in cell else f"seed{cell['seed']}")
        if cell["seed"] == 1:
            raise RuntimeError("boom")

    rows = run_grid(cells, tmp_path, execute, final=False)
    assert [row["status"] for row in rows] == ["skipped", "failed"]
    assert seen == ["seed1"]
    assert (tmp_path / "cifar10_resnet50_cdea_seed1.failed").is_file()
    assert not (tmp_path / "cifar10_resnet50_cdea_seed1.done").is_file()
    assert pending_cells(cells, tmp_path)[0]["seed"] == 1


def test_a_rerun_skips_a_cell_that_just_finished(tmp_path: Path) -> None:
    cells = [{"id": "only", "dataset": "cifar10"}]
    calls = {"n": 0}

    def execute(_cell: dict) -> None:
        calls["n"] += 1

    assert run_grid(cells, tmp_path, execute, final=False)[0]["status"] == "done"
    assert run_grid(cells, tmp_path, execute, final=False)[0]["status"] == "skipped"
    assert calls["n"] == 1


def test_final_is_refused_while_the_draft_says_it_is_not_frozen() -> None:
    text = EVAL_PLAN_PATH.read_text(encoding="utf-8")
    assert "not frozen" in text
    with pytest.raises(SplitLockedError, match="not frozen"):
        assert_split_allowed(
            "test",
            final=True,
            config_hash="CD@5%",
            eval_plan_path=EVAL_PLAN_PATH,
        )
    with pytest.raises(SplitLockedError, match="not frozen"):
        run_grid([], Path("/tmp/gambit-final-grid-unused"), lambda _cell: None, final=True)


def test_the_module_refuses_to_launch() -> None:
    with pytest.raises(SystemExit, match="not frozen"):
        main()
