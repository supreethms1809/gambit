"""S24. Generated numbers match a fresh render. The paper audit does not pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.audit_results import (
    audit_file,
    main,
    masks_checked,
    paper_audit,
    required_datasets,
)
from analysis.build_results import write_results

ROOT = Path(__file__).resolve().parents[1]


def _record(dataset: str, method: str, value: float, seed: int = 0) -> dict:
    return {
        "dataset": dataset,
        "model": "resnet50",
        "method": method,
        "seed": seed,
        "game": "contrastive",
        "value": value,
        "masks": [[1.0, 0.0, 0.0, 0.0]],
        "probabilities": [0.25, 0.75],
        "n": 4,
    }


def _toy_records() -> list[dict]:
    values = {
        "toy_a": {"cdea": 0.30, "margin": 0.10, "extremal": 0.20, "cve": 0.15},
        "toy_b": {"cdea": 0.40, "margin": 0.10, "extremal": 0.10, "cve": 0.20},
        "toy_c": {"cdea": 0.50, "margin": 0.20, "extremal": 0.25, "cve": 0.10},
    }
    records = []
    for dataset, methods in values.items():
        for method, value in methods.items():
            records.append(_record(dataset, method, value))
    return records


def test_a_generated_file_matches_and_a_hand_edit_does_not(tmp_path: Path) -> None:
    records = _toy_records()
    kwargs = {"area": 0.25, "tolerance": 1e-6, "n": 4, "n_boot": 20}
    write_results(records, tmp_path, **kwargs)
    path = tmp_path / "RESULTS.md"
    original = path.read_text(encoding="utf-8")
    assert audit_file(path, records, **kwargs)["passed"] is True
    path.write_text(original.replace("0.300000", "9.999999", 1), encoding="utf-8")
    report = audit_file(path, records, **kwargs)
    assert report["passed"] is False
    assert report["numbers_match"] is False
    assert "9.999999" in path.read_text(encoding="utf-8")


def test_masks_need_every_dataset_and_the_paper_file_is_absent() -> None:
    assert masks_checked([]) is False
    assert masks_checked(required_datasets()) is True
    assert "imagenet" in required_datasets()
    assert "sixth_shift_tbd" in required_datasets()
    report = paper_audit()
    assert report["passed"] is False
    assert report["results_present"] is False
    assert report["masks_checked"] is False
    text = (ROOT / "docs" / "paper" / "AUDIT.md").read_text(encoding="utf-8")
    assert "The audit does not pass" in text
    assert "RESULTS.md is absent" in text
    assert "Masks were not checked" in text
    assert "fresh session" in text
    assert "fixed in code" in text
    with pytest.raises(SystemExit, match="does not pass") as caught:
        main()
    message = str(caught.value)
    assert "RESULTS.md is absent" in message
    assert "masks were not checked" in message
    assert not (ROOT / "results" / "paper" / "RESULTS.md").exists()
