"""Completion check for the final grids. It does not create the final-runs tag.

A cell is complete only when its done marker exists. A failed marker is a
rerun, and a missing marker is still pending. Run-record assertions check
mask range, area, probabilities, finiteness, and the planned image count.
This module does not read the test split, does not launch a grid, and does
not create a git tag.

The contrastive, shift, and ablation logs are absent, so the check does not
pass.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.final_grid import cell_id

CONTRASTIVE_DATASETS = (
    "cifar10",
    "cifar100",
    "oxford_pets",
    "stanford_dogs",
    "cub200",
    "imagenet",
    "ham10000",
    "brain_tumor",
)
CONTRASTIVE_METHODS = (
    "cdea",
    "margin_gradcam",
    "margin_ig",
    "extremal",
    "cve",
    "base_evidence",
    "random_floor",
)
MODELS = ("resnet50", "vit_b_16")
SEEDS = (0, 1, 2, 3, 4)

CONTRASTIVE_LOG = REPO / "results" / "paper" / "logs" / "contrastive_grid"
SHIFT_LOG = REPO / "results" / "paper" / "logs" / "shift_grid"
ABLATION_LOG = REPO / "results" / "paper" / "logs" / "ablations"


def contrastive_cells() -> list[dict]:
    """Job list for the eight contrastive datasets. ImageNet runs on Spark.

    CVE is ResNet-50 only: a ViT's CLS head does not read patch tokens after
    the last block, so there is no spatial decision network to edit
    (EVAL_PLAN.md section 4.2).
    """
    cells = []
    for dataset in CONTRASTIVE_DATASETS:
        for model in MODELS:
            for method in CONTRASTIVE_METHODS:
                if method == "cve" and model == "vit_b_16":
                    continue
                for seed in SEEDS:
                    cells.append({
                        "id": f"contrastive_{dataset}_{model}_{method}_seed{seed}",
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": seed,
                        "game": "contrastive",
                    })
    return cells


def completion_report(cells: Sequence[Mapping], log_dir: Path) -> dict:
    """Done markers finish a cell. Failed markers are the rerun list."""
    log_dir = Path(log_dir)
    done = 0
    rerun: list[Mapping] = []
    pending: list[Mapping] = []
    for cell in cells:
        name = cell_id(cell)
        if log_dir.is_dir() and (log_dir / f"{name}.done").is_file():
            done += 1
        elif log_dir.is_dir() and (log_dir / f"{name}.failed").is_file():
            rerun.append(cell)
        else:
            pending.append(cell)
    return {
        "cells": len(cells),
        "done": done,
        "failed": len(rerun),
        "pending": len(pending),
        "complete": len(cells) > 0 and done == len(cells),
        "rerun": rerun,
        "waiting": pending,
    }


def _flat(values) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise AssertionError("record value is not numeric")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        out: list[float] = []
        for item in values:
            out.extend(_flat(item))
        return out
    number = float(values)
    return [number]


def _finite_unit_interval(values, label: str) -> list[float]:
    flat = _flat(values)
    if not flat:
        raise AssertionError(f"{label} is empty")
    for number in flat:
        if not math.isfinite(number):
            raise AssertionError(f"{label} is not finite")
        if number < 0 or number > 1:
            raise AssertionError(f"{label} outside [0, 1]")
    return flat


def assert_run_record(record: Mapping, *, area: float, tolerance: float, n: int) -> None:
    """Masks in [0, 1], area within tolerance, probabilities in [0, 1], finite, n matches."""
    masks = record["masks"]
    rows = masks if isinstance(masks, Sequence) and masks and isinstance(masks[0], Sequence) else [masks]
    for row in rows:
        flat = _finite_unit_interval(row, "mask")
        got = sum(flat) / len(flat)
        if abs(got - area) > tolerance:
            raise AssertionError("area does not match the plan")
    _finite_unit_interval(record["probabilities"], "probability")
    if int(record["n"]) != int(n):
        raise AssertionError("n does not match the plan")


def record_matches(summary: dict, *, code_hash: str, knob_hash: str, gate: bool = False) -> bool:
    """Old records and fast knobs do not count. A gate cell also rejects ``fast``."""
    if summary.get("method_code_hash") != code_hash:
        return False
    if summary.get("knobs_hash") != knob_hash:
        return False
    if gate and summary.get("fast"):
        return False
    return True


def shift_cells() -> list[dict]:
    """Job list for the shift units. Hashes are checked by ``record_matches``."""
    from evaluation.environments import SHIFT_DATASETS
    from evaluation.run_methods import SHIFT_CORE

    cells = []
    for dataset in SHIFT_DATASETS:
        for model in MODELS:
            for method in SHIFT_CORE:
                for seed in SEEDS:
                    cells.append({
                        "id": f"shift_{dataset}_{model}_{method}_seed{seed}",
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": seed,
                        "game": "shift",
                    })
    return cells


def runs_complete(contrastive_log: Path = CONTRASTIVE_LOG) -> dict:
    """The contrastive manifest has a done marker for every cell."""
    part = completion_report(contrastive_cells(), contrastive_log)
    return {
        "complete": part["complete"],
        "pending": part["pending"] + part["failed"],
        "parts": {"contrastive": part},
    }


def main() -> None:
    """Do not create the tag. The grids have not finished."""
    report = runs_complete()
    raise SystemExit(
        "final runs are not complete; "
        "the final-runs tag was not created; "
        f"pending={report['pending']}"
    )


if __name__ == "__main__":
    main()
