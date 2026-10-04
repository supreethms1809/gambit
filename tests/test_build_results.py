"""S23. Results text is generated from records, and the paper file is not written."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.build_results import fmt, main, render_results, write_results
from analysis.stats import family_summary

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
            for seed in (0, 1):
                records.append(_record(dataset, method, value, seed))
    return records


def test_family_numbers_come_from_the_summary(tmp_path: Path) -> None:
    records = _toy_records()
    markdown, latex, summary = render_results(
        records, area=0.25, tolerance=1e-6, n=4, n_boot=50
    )
    expected = family_summary(
        {
            "CDEA versus margin attribution": [0.20, 0.30, 0.30],
            "CDEA versus contrastive Extremal Perturbations": [0.10, 0.30, 0.25],
            "CDEA versus CVE": [0.15, 0.20, 0.40],
        },
        n_boot=50,
    )
    assert summary["family_c"][0]["mean"] == expected[0]["mean"]
    assert fmt(expected[0]["mean"]) in markdown
    assert fmt(expected[0]["p_holm"]) in markdown
    assert fmt(expected[0]["mean"]) in latex
    written = write_results(records, tmp_path, area=0.25, tolerance=1e-6, n=4, n_boot=50)
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert fmt(written["family_c"][0]["mean"]) in text
    assert "Family S has no selected baseline." in text
    assert "The second backbone was not measured." in text
    assert "The blur operator was not measured." in text
    assert "The 2.5% and 10% budgets were not measured." in text
    assert "The repeat run was not measured." in text
    assert "Ablation records are not in the input." in text
    assert "Cost records are not in the input." in text
    assert "The model table is not in the input." in text
    assert "drop toy_a:" in text


def test_a_bad_mask_writes_nothing(tmp_path: Path) -> None:
    records = _toy_records()
    records[0]["masks"] = [[1.1, 0.0, 0.0, 0.0]]
    with pytest.raises(AssertionError, match="outside"):
        write_results(records, tmp_path, area=0.25, tolerance=1.0, n=4, n_boot=20)
    assert not (tmp_path / "RESULTS.md").exists()


def test_the_paper_results_file_is_not_written() -> None:
    with pytest.raises(SystemExit, match="not complete") as caught:
        main()
    assert "was not created" in str(caught.value)
    assert not (ROOT / "results" / "paper" / "RESULTS.md").exists()
