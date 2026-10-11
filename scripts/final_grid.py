"""Resumable grid runner. It does not launch the contrastive final runs.

A cell is skipped when its done marker exists and carries the current stamp
(the method-code and knobs hashes), so a cell finished under older code runs
again instead of leaving stale records in place. ``--final`` is refused
while ``EVAL_PLAN.md`` says it is not frozen, and a dirty tree is refused by
the same check as the other final entry points. This module does not read the
test split.

The contrastive grid stays unlaunched: the plan is not frozen, gate G0 is
closed, and the training queue has not finished.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.reporting import refuse_final_if_dirty
from evaluation.splits import EVAL_PLAN_PATH, SplitLockedError, plan_is_frozen


def cell_id(cell: Mapping) -> str:
    if "id" in cell and str(cell["id"]):
        return str(cell["id"])
    return f"{cell['dataset']}_{cell['model']}_{cell['method']}_seed{cell['seed']}"


def is_done(marker: Path, stamp: str | None = None) -> bool:
    """A marker counts when it exists and, given a stamp, its second line is that stamp."""
    if not marker.is_file():
        return False
    if stamp is None:
        return True
    lines = marker.read_text(encoding="utf-8").splitlines()
    return len(lines) > 1 and lines[1].strip() == stamp


def pending_cells(cells: Sequence[Mapping], log_dir: Path, stamp: str | None = None) -> list[Mapping]:
    """Cells without a current done marker. Order is preserved."""
    return [cell for cell in cells if not is_done(Path(log_dir) / f"{cell_id(cell)}.done", stamp)]


def run_grid(
    cells: Sequence[Mapping],
    log_dir: Path,
    execute: Callable[[Mapping], None],
    *,
    final: bool,
    eval_plan_path: Path = EVAL_PLAN_PATH,
    stamp: str | None = None,
) -> list[dict[str, str]]:
    """Run cells that are not already done. Returns one status row per cell."""
    if final:
        if not eval_plan_path.is_file() or not plan_is_frozen(
            eval_plan_path.read_text(encoding="utf-8")
        ):
            raise SplitLockedError("EVAL_PLAN.md is not frozen")
        refuse_final_if_dirty(True)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for cell in cells:
        name = cell_id(cell)
        marker = log_dir / f"{name}.done"
        if is_done(marker, stamp):
            rows.append({"id": name, "status": "skipped"})
            continue
        try:
            execute(cell)
        except Exception as exc:
            (log_dir / f"{name}.failed").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            rows.append({"id": name, "status": "failed"})
            continue
        marker.write_text(name + "\n" + (stamp or "") + "\n", encoding="utf-8")
        rows.append({"id": name, "status": "done"})
    return rows


def main() -> None:
    """Do not start the grid. The plan is not frozen."""
    raise SystemExit(
        "refusing to launch the contrastive grid: EVAL_PLAN.md is not frozen"
    )


if __name__ == "__main__":
    main()
