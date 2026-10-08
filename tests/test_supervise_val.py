"""Audit logic of the val-game supervisor: clean, errored, NaN, missing."""

import csv
import gzip
import json

from scripts.supervise_val import audit_cell, cell_id, cells


def _write_cell(tmp_path, monkeypatch, game, dataset, backbone, methods, rows):
    import scripts.supervise_val as sup

    runs = tmp_path / "runs"
    markers = runs / "_markers" / "val"
    cell_dir = runs / "val" / game / dataset / backbone / "seed0"
    cell_dir.mkdir(parents=True)
    markers.mkdir(parents=True)
    (markers / "val_contrastive_cifar10_resnet50_seed0.done").write_text("x\n")
    (cell_dir / "summary.json").write_text(json.dumps(
        {"methods": methods, "errors": [m for m, s in methods.items() if s["status"] == "error"]}))
    if rows is not None:
        with gzip.open(cell_dir / "records.csv.gz", "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "cd", "cd1"])
            writer.writeheader()
            writer.writerows(rows)
    monkeypatch.setattr(sup, "RUNS", runs)
    monkeypatch.setattr(sup, "MARKERS", runs / "_markers")


def test_audit_accepts_a_clean_cell(tmp_path, monkeypatch):
    _write_cell(tmp_path, monkeypatch, "contrastive", "cifar10", "resnet50",
                {"cdea": {"status": "ok"}},
                [{"method": "cdea", "cd": "0.5", "cd1": "0.2"}])
    assert audit_cell({"game": "contrastive", "dataset": "cifar10",
                       "backbone": "resnet50", "seed": 0}) == (True, "clean")


def test_audit_rejects_method_errors_and_nan_scores(tmp_path, monkeypatch):
    _write_cell(tmp_path, monkeypatch, "contrastive", "cifar10", "resnet50",
                {"cdea": {"status": "error", "error": "boom"}},
                [{"method": "cdea", "cd": "nan", "cd1": "nan"}])
    clean, detail = audit_cell({"game": "contrastive", "dataset": "cifar10",
                                "backbone": "resnet50", "seed": 0})
    assert not clean and "cdea" in detail


def test_audit_rejects_nonfinite_scores_without_method_errors(tmp_path, monkeypatch):
    _write_cell(tmp_path, monkeypatch, "contrastive", "cifar10", "resnet50",
                {"cdea": {"status": "ok"}},
                [{"method": "cdea", "cd": "inf", "cd1": "0.2"}])
    clean, detail = audit_cell({"game": "contrastive", "dataset": "cifar10",
                                "backbone": "resnet50", "seed": 0})
    assert not clean and "non-finite cd" in detail


def test_audit_tolerates_scores_absent_by_design(tmp_path, monkeypatch):
    # CVE has no foil map, so CD is legitimately "nan" while CD1 is scored.
    _write_cell(tmp_path, monkeypatch, "contrastive", "cifar10", "resnet50",
                {"cve": {"status": "ok"}},
                [{"method": "cve", "cd": "nan", "cd1": "0.3"}])
    assert audit_cell({"game": "contrastive", "dataset": "cifar10",
                       "backbone": "resnet50", "seed": 0}) == (True, "clean")


def test_audit_rejects_a_missing_cell(tmp_path, monkeypatch):
    import scripts.supervise_val as sup

    monkeypatch.setattr(sup, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(sup, "MARKERS", tmp_path / "runs" / "_markers")
    clean, detail = audit_cell({"game": "shift", "dataset": "waterbirds",
                                "backbone": "resnet50", "seed": 0})
    assert not clean and "no done marker" in detail


def test_cell_inventory_is_14_seed0_cells():
    wanted = cells(0)
    assert len(wanted) == 14
    assert cell_id(wanted[0]) == "val_contrastive_cifar10_resnet50_seed0"


def test_main_reports_clean_without_relaunching(tmp_path, monkeypatch):
    import scripts.supervise_val as sup

    _write_cell(tmp_path, monkeypatch, "contrastive", "cifar10", "resnet50",
                {"cdea": {"status": "ok"}},
                [{"method": "cdea", "cd": "0.5", "cd1": "0.2"}])
    calls = []
    monkeypatch.setattr(sup, "cells", lambda seed: [
        {"game": "contrastive", "dataset": "cifar10", "backbone": "resnet50", "seed": 0}])
    monkeypatch.setattr(sup, "relaunch", lambda seed, jobs: calls.append(jobs) or (0, 0))
    assert sup.main(["--seeds", "0", "--rounds", "0"]) == 0
    assert calls == []


def test_main_relaunches_dirty_cells_then_reports_dirty(tmp_path, monkeypatch, capsys):
    import scripts.supervise_val as sup

    runs = tmp_path / "runs"
    (runs / "_markers" / "val").mkdir(parents=True)
    monkeypatch.setattr(sup, "RUNS", runs)
    monkeypatch.setattr(sup, "MARKERS", runs / "_markers")
    monkeypatch.setattr(sup, "cells", lambda seed: [
        {"game": "shift", "dataset": "waterbirds", "backbone": "resnet50", "seed": 0}])
    monkeypatch.setattr(sup, "game_processes", lambda: [])
    monkeypatch.setattr(sup, "QUIET_SECONDS", 0)
    calls = []
    monkeypatch.setattr(sup, "relaunch", lambda seed, jobs: calls.append(jobs) or (0, 0))
    assert sup.main(["--seeds", "0", "--rounds", "1", "--poll", "0"]) == 1
    assert calls == [3]
    assert "GAMES_DIRTY" in capsys.readouterr().out
