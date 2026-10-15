"""The phase launcher builds the planned cells, keeps two large and two small running, and fails errored cells."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.launch_runs as lr
from evaluation.run_methods import DECOMP_ABLATIONS, DECOMP_CANDIDATES


def _cells(tmp_path, *extra):
    return lr.build_cells(lr.parse(["--out-root", str(tmp_path), *extra]))


def by_phase(cells, phase):
    return [c for c in cells if c.phase == phase]


def test_val_phases_match_the_plan(tmp_path):
    cells = _cells(tmp_path, "--phases", "dev,planted-val,natural-val,shift-gate,shift-val")
    dev = by_phase(cells, "dev")
    assert {c.dataset for c in dev} == {"cifar10", "ham10000"}
    assert all(c.backbone == "resnet50" and c.seed == 0 and c.n == 64 for c in dev)
    assert all(c.ablations == tuple(DECOMP_ABLATIONS) and c.candidates == tuple(DECOMP_CANDIDATES) for c in dev)
    assert all("extremal_minsplit" in c.methods for c in dev)
    assert all(c.areas == (0.025, 0.05, 0.10) for c in dev)

    planted = by_phase(cells, "planted-val")
    assert len(planted) == 10 and all(c.n == 200 for c in planted)
    assert all(bool(c.ablations) == (c.backbone == "resnet50") for c in planted)

    natural = by_phase(cells, "natural-val")
    assert len(natural) == 6 + 8   # ResNet-50 without the dev pair, ViT on all 8
    assert not any(c.backbone == "resnet50" and c.dataset in ("cifar10", "ham10000") for c in natural)

    gate = by_phase(cells, "shift-gate")
    assert len(gate) == 4 and all(c.out.endswith("gate_shift") and c.areas == (0.05,) for c in gate)
    assert {c.seed for c in gate} == {0, 1}

    shift = by_phase(cells, "shift-val")
    assert len(shift) == 10
    assert all(c.n == (200 if c.backbone == "resnet50" else 64) for c in shift)
    assert all("S6_precomputed" not in c.ablations for c in shift)
    assert all(bool(c.ablations) == (c.backbone == "resnet50" and c.dataset in ("waterbirds", "planted_patch"))
               for c in shift)
    assert all(not any("gradcam" in k for k in c.candidates) for c in shift if c.backbone == "vit_b_16")


def test_chosen_cdea_and_the_test_phase(tmp_path):
    natural = _cells(tmp_path, "--phases", "natural-val", "--cdea", "cdea_decomp@lr0.2_t100")
    assert all(c.candidates == ("cdea_decomp@lr0.2_t100",) for c in natural)
    with pytest.raises(SystemExit, match="needs --cdea, --final, --config-hash"):
        _cells(tmp_path, "--phases", "test")
    test = _cells(tmp_path, "--phases", "test", "--cdea", "cdea_decomp@lr0.2_t100", "--final",
                  "--config-hash", "abc", "--n-test", "resnet50:300")
    assert len(test) == 5 * 2 * 9
    assert all("cdea_decomp" not in c.methods and c.split == "test" for c in test)
    assert {c.n for c in test if c.dataset != "planted_pair" and c.backbone == "resnet50"} == {300}
    with_ablations = {(c.dataset, c.seed) for c in test if c.ablations}
    assert with_ablations == {(d, s) for d in ("cifar100", "cub200", "brain_tumor", "imagenet") for s in (0, 1, 2)}


def test_two_cells_never_write_one_directory(tmp_path):
    with pytest.raises(SystemExit, match="would write the same records"):
        _cells(tmp_path, "--phases", "dev,dev")


def test_smoke_records_never_land_in_the_paper_results():
    with pytest.raises(SystemExit, match="--out-root"):
        lr.parse(["--phases", "dev", "--fast"])


def test_size_classes_and_recorded_peaks(tmp_path):
    cells = {c.id: c for c in _cells(tmp_path, "--phases", "dev,planted-val,shift-gate")}
    planted_vit = cells["planted-val__val_decomposition_planted_pair_vit_b_16_seed0"]
    planted_r50 = cells["planted-val__val_decomposition_planted_pair_resnet50_seed0"]
    gate = cells["shift-gate__val_shift_waterbirds_resnet50_seed0"]
    assert (lr.size_of(planted_vit), lr.size_of(planted_r50), lr.size_of(gate)) == ("large", "small", "large")
    planted_r50.cell_dir.mkdir(parents=True)
    (planted_r50.cell_dir / "summary.json").write_text(json.dumps({"device_peak_mb": 30_000}))
    assert lr.size_of(planted_r50) == "large"


def test_admission_keeps_two_large_and_lets_small_fill_idle_large_slots():
    kind = {"L1": "large", "L2": "large", "L3": "large", "S1": "small", "S2": "small", "S3": "small", "S4": "small"}
    admit = lr.make_admit(kind, large=2, small=2)
    job = lambda name: SimpleNamespace(name=name)  # noqa: E731
    running = []
    for name in ("L1", "L2", "S1", "S2"):
        assert admit(running, job(name))
        running.append(name)
    assert not admit(running, job("L3"))   # two large already
    assert not admit(running, job("S3"))   # two small, and L3 still waits
    running.remove("L1")
    assert admit(running, job("L3"))
    running.append("L3")
    running.remove("L2")
    running.remove("L3")
    assert admit(running, job("S3"))       # no large waits, so small takes a large slot
    running.append("S3")
    assert admit(running, job("S4"))
    strict = lr.make_admit({"S1": "small", "S2": "small", "S3": "small"}, 2, 2, fill=False)
    assert not strict(["S1", "S2"], job("S3"))


def test_training_writes_the_checkpoint_the_cell_loads(tmp_path):
    from evaluation.run_models import paper_checkpoint_path
    from models.build import paper_checkpoint_name
    from models.wrapper import INPUT_CONVENTION

    cells = _cells(tmp_path, "--phases", "dev,planted-val,natural-val,shift-gate,shift-val")
    for cell in cells:
        if cell.dataset == "imagenet":
            continue
        t = lr.training_cell(cell)
        trained = paper_checkpoint_name(t["dataset"], t["model_name"], True, t["freeze_backbone"], lr.EPOCHS,
                                        t["lr"], t["seed"], INPUT_CONVENTION)
        assert trained == paper_checkpoint_path(cell.dataset, cell.backbone, cell.seed).name, cell.id


def test_a_method_error_fails_the_cell_but_known_gaps_do_not(tmp_path, monkeypatch):
    args = lr.parse(["--phases", "shift-gate", "--out-root", str(tmp_path), "--model-source", "smoke"])
    cell = lr.build_cells(args)[0]
    statuses = {"cdea_shift": {"status": "ok"}, "cdea_shift:S6_precomputed": {"status": "error"}}

    def fake_run(spec, device):
        cell.cell_dir.mkdir(parents=True, exist_ok=True)
        (cell.cell_dir / "summary.json").write_text(json.dumps({"methods": statuses}))
        return cell.cell_dir

    monkeypatch.setattr("scripts.paper_run.run", fake_run)
    monkeypatch.setattr(lr, "stamp", lambda args: "h k")
    assert lr.run_one(cell, args)
    assert lr.is_done(cell, "h k")
    statuses["spray"] = {"status": "error", "error": "OOM"}
    (lr.marker_dir(cell) / f"{cell.id}.done").unlink()
    assert not lr.run_one(cell, args)
    assert (lr.marker_dir(cell) / f"{cell.id}.failed").is_file()


def test_child_argv_drops_the_parallel_flags():
    argv = ["--phases", "dev", "--large", "3", "--small=1", "--keep-going", "--no-fill", "--mem-gate-gb", "8"]
    assert lr.child_argv(argv, "X") == ["--phases", "dev", "--one", "X"]
