"""Parallel launchers: concurrency cap, exit codes, cell order, child arguments."""

import sys
from pathlib import Path

import pytest

from scripts.parallel_cells import Job, mem_class, order_balanced, run_parallel, thread_env


def _job(tmp_path, name, code=0, seconds=0.4):
    script = (
        "import sys, time, pathlib;"
        f"p = pathlib.Path({str(tmp_path)!r});"
        f"(p / '{name}.start').write_text(str(time.time()));"
        f"time.sleep({seconds});"
        f"(p / '{name}.end').write_text(str(time.time()));"
        f"print('hello {name}');"
        f"sys.exit({code})"
    )
    return Job(name=name, command=[sys.executable, "-c", script], log=tmp_path / "logs" / f"{name}.log")


def test_run_parallel_caps_concurrency_and_returns_exit_codes(tmp_path):
    jobs = [_job(tmp_path, "a"), _job(tmp_path, "b", code=3), _job(tmp_path, "c")]
    codes = run_parallel(jobs, 2, poll_seconds=0.05)
    assert codes == {"a": 0, "b": 3, "c": 0}
    spans = {n: (float((tmp_path / f"{n}.start").read_text()), float((tmp_path / f"{n}.end").read_text()))
             for n in "abc"}
    events = sorted([(s, 1) for s, _ in spans.values()] + [(e, -1) for _, e in spans.values()])
    live = peak = 0
    for _, step in events:
        live += step
        peak = max(peak, live)
    assert peak == 2
    assert "hello b" in (tmp_path / "logs" / "b.log").read_text()


def test_run_parallel_refuses_zero_jobs(tmp_path):
    with pytest.raises(ValueError):
        run_parallel([_job(tmp_path, "a")], 0)


def test_thread_env_respects_an_existing_cap(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    env = thread_env(4)
    assert "OMP_NUM_THREADS" not in env
    assert int(env["MKL_NUM_THREADS"]) >= 1


def test_training_cells_keep_seed_order_and_put_long_cells_first():
    from scripts.launch_paper_training import cell_id, paper_cells, select_cells

    cells = select_cells(paper_cells(), seeds=[1, 0])
    assert len(cells) == 2 * 22
    assert [c["seed"] for c in cells] == [1] * 22 + [0] * 22
    assert cell_id(cells[0]) == "planted_patch_vit_b_16_ft_lr0.0001_seed1"
    assert cells[21]["dataset"] == "brain_tumor" and cells[21]["model_name"] == "resnet50"
    only = select_cells(paper_cells(), seeds=[0], datasets=["cifar10"], models=["resnet50"])
    assert [cell_id(c) for c in only] == ["cifar10_resnet50_lp_lr0.001_seed0"]


def test_training_one_refuses_an_unknown_cell(tmp_path):
    from scripts.launch_paper_training import main

    with pytest.raises(SystemExit, match="unknown cell"):
        main(["--seeds", "0", "--one", "nope", "--log-dir", str(tmp_path)])


def test_eval_child_argv_drops_jobs_and_names_the_cell():
    from scripts.launch_paper_eval import child_argv

    argv = ["--split", "val", "--jobs", "4", "--seeds", "0", "--jobs=2",
            "--mem-gate-gb", "12"]
    assert child_argv(argv, "val_shift_waterbirds_resnet50_seed0") == [
        "--split", "val", "--seeds", "0", "--one", "val_shift_waterbirds_resnet50_seed0"]


def test_mem_class_marks_the_hog_cells_high():
    assert mem_class("shift", "planted_patch", "resnet50") == "high"
    assert mem_class("shift", "waterbirds", "vit_b_16") == "high"
    assert mem_class("contrastive", "planted_patch", "vit_b_16") == "high"
    assert mem_class("contrastive", "planted_patch", "resnet50") == "low"
    assert mem_class("contrastive", "cifar10", "resnet50") == "low"
    assert mem_class("contrastive", "ham10000", "vit_b_16") == "high"
    assert mem_class("mystery", "future_dataset", "future_net") == "low"


def test_order_balanced_interleaves_two_high_with_two_low():
    jobs = [Job(name=n, command=["true"], log=Path(n)) for n in
            ["h1", "h2", "h3", "l1", "l2", "l3"]]
    key = lambda job: "high" if job.name.startswith("h") else "low"  # noqa: E731
    first_four = [j.name for j in order_balanced(jobs, key)[:4]]
    assert sorted(first_four) == ["h1", "h2", "l1", "l2"]
    assert [j.name for j in order_balanced([], key)] == []


def test_eval_parallel_path_matches_run_parallel_signature(tmp_path, monkeypatch):
    """The jobs>1 path must call run_parallel with only its real parameters
    (regression: a stray env= kwarg crashed both relaunch rounds instantly)."""
    import types

    import scripts.parallel_cells as pc
    from scripts import launch_paper_eval as ev

    def fake_run_parallel(jobs, max_jobs, *, poll_seconds=2.0, cwd=None,
                          min_free_gb=0.0, mem_timeout_s=1800.0):
        return {j.name: 0 for j in jobs}

    monkeypatch.setattr(pc, "run_parallel", fake_run_parallel)
    args = types.SimpleNamespace(jobs=3, out=str(tmp_path), split="val", mem_gate_gb=12.0)
    cells = [{"id": f"val_contrastive_cifar10_{bb}_seed0", "game": "contrastive",
              "dataset": "cifar10", "backbone": bb, "seed": 0}
             for bb in ("resnet50", "vit_b_16")]
    ev._run_cells_in_parallel(cells, tmp_path, args)  # must not raise TypeError


def test_mem_key_prefers_a_recorded_peak_over_the_seed_table(tmp_path):
    import json
    import types

    from scripts.launch_paper_eval import mem_key

    args = types.SimpleNamespace(out=str(tmp_path), split="val")
    seed_high = {"game": "shift", "dataset": "waterbirds", "backbone": "vit_b_16", "seed": 0}
    assert mem_key(args)(seed_high) == "high"  # seed table, no summary yet
    cell_dir = tmp_path / "val" / "contrastive" / "cifar10" / "resnet50" / "seed0"
    cell_dir.mkdir(parents=True)
    (cell_dir / "summary.json").write_text(json.dumps({"device_peak_mb": 45000.0}))
    assert mem_key(args)({"game": "contrastive", "dataset": "cifar10",
                          "backbone": "resnet50", "seed": 0}) == "high"  # peak wins
    (cell_dir / "summary.json").write_text(json.dumps({"device_peak_mb": 3000.0}))
    assert mem_key(args)({"game": "contrastive", "dataset": "cifar10",
                          "backbone": "resnet50", "seed": 0}) == "low"


def test_smoke_checkpoint_path_is_separate_from_the_paper_path(tmp_path):
    from evaluation.run_models import PAPER_CKPT_DIR, paper_checkpoint_path

    paper = paper_checkpoint_path("cifar10", "resnet50", 0)
    smoke = paper_checkpoint_path("cifar10", "resnet50", 0, ckpt_dir=tmp_path, num_epochs=1)
    assert paper.parent == PAPER_CKPT_DIR and "_ep15_" in paper.name
    assert smoke.parent == tmp_path and "_ep1_" in smoke.name
    assert paper.name.replace("_ep15_", "_ep1_") == smoke.name


def test_smoke_source_uses_a_present_smoke_checkpoint(tmp_path):
    from evaluation.run_models import paper_checkpoint_path
    from scripts.smoke_e2e import _source

    assert _source("imagenet", "resnet50", 0, tmp_path, 1) == ("imagenet", None)
    assert _source("cifar10", "resnet50", 0, tmp_path, 1) == ("smoke", None)
    path = paper_checkpoint_path("cifar10", "resnet50", 0, ckpt_dir=tmp_path, num_epochs=1)
    Path(path).write_bytes(b"")
    assert _source("cifar10", "resnet50", 0, tmp_path, 1) == ("checkpoint", str(path))
