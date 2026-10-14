"""Train-then-play order: one checkpoint, then its games, seed by seed."""

import pytest

from scripts.launch_paper_chain import (
    chain_units,
    child_argv,
    games_for,
    mem_of,
    needs_training,
    order_seed,
    size_label,
    unit_id,
)


def test_games_follow_the_dataset():
    assert games_for("cifar10") == ("contrastive",)
    assert games_for("waterbirds") == ("shift",)
    assert games_for("stanford_dogs") == ("contrastive", "shift")
    assert games_for("imagenet") == ("contrastive",)
    assert needs_training("cifar10")
    assert not needs_training("imagenet")


def test_memory_class_marks_shift_and_vit_big():
    def unit(dataset, backbone, games):
        return {"dataset": dataset, "backbone": backbone, "games": games}

    assert mem_of(unit("cifar10", "resnet50", ("contrastive",))) == "low"
    assert mem_of(unit("cifar10", "vit_b_16", ("contrastive",))) == "high"
    assert mem_of(unit("waterbirds", "resnet50", ("shift",))) == "high"
    assert mem_of(unit("stanford_dogs", "resnet50", ("contrastive", "shift"))) == "high"
    assert size_label("high") == "big"
    assert size_label("low") == "small"


def test_seed_blocks_stay_apart_and_the_first_wave_is_two_and_two():
    units = chain_units([0, 1])
    assert [u["seed"] for u in units] == [0] * (len(units) // 2) + [1] * (len(units) // 2)
    seed0 = order_seed([u for u in units if u["seed"] == 0])
    assert {u["seed"] for u in seed0} == {0}
    first = [mem_of(u) for u in seed0[:4]]
    assert first.count("high") == 2
    assert first.count("low") == 2
    assert unit_id(seed0[0]).endswith("_seed0")


def test_child_argv_drops_the_parallel_flags():
    argv = ["--seeds", "0", "--jobs", "4", "--mem-gate-gb", "12", "--jobs=2"]
    assert child_argv(argv, "cifar10_resnet50_seed0") == [
        "--seeds", "0", "--one", "cifar10_resnet50_seed0"]


def test_run_unit_trains_then_plays_every_game(tmp_path, monkeypatch):
    import scripts.launch_paper_chain as chain

    events = []

    def fake_train(*args, **kwargs):
        events.append(("train", kwargs.get("force")))
        return True

    def fake_grid(cells, log_dir, execute, *, final, stamp):
        events.append(("game", cells[0]["game"]))
        return [{"id": cells[0]["id"], "status": "done"}]

    monkeypatch.setattr(chain, "train_cell", fake_train)
    monkeypatch.setattr(chain, "_release_device", lambda: events.append(("release",)))
    monkeypatch.setattr("scripts.final_grid.run_grid", fake_grid)
    args = chain.parse([
        "--seeds", "0", "--datasets", "stanford_dogs", "--backbones", "resnet50",
        "--jobs", "1", "--out", str(tmp_path), "--train-log-dir", str(tmp_path / "train"),
    ])
    unit = chain.chain_units([0], ["stanford_dogs"], ["resnet50"])[0]
    assert chain.run_unit(unit, args)
    assert events == [("train", False), ("release",), ("game", "contrastive"), ("game", "shift")]


def test_imagenet_skips_training(tmp_path, monkeypatch):
    import scripts.launch_paper_chain as chain

    events = []
    monkeypatch.setattr(chain, "train_cell", lambda *a, **k: events.append("train") or True)

    def fake_grid(cells, log_dir, execute, *, final, stamp):
        events.append(cells[0]["game"])
        return [{"id": cells[0]["id"], "status": "done"}]

    monkeypatch.setattr("scripts.final_grid.run_grid", fake_grid)
    args = chain.parse([
        "--seeds", "0", "--datasets", "imagenet", "--backbones", "resnet50",
        "--jobs", "1", "--out", str(tmp_path),
    ])
    unit = chain.chain_units([0], ["imagenet"], ["resnet50"])[0]
    assert chain.run_unit(unit, args)
    assert events == ["contrastive"]


def test_a_failed_cell_does_not_start_the_next_seed(tmp_path, monkeypatch):
    import scripts.launch_paper_chain as chain

    calls = []

    def fake_run(unit, args):
        calls.append(unit_id(unit))
        return unit["dataset"] != "cifar10"

    monkeypatch.setattr(chain, "run_unit", fake_run)
    monkeypatch.setattr(chain, "unit_done", lambda *a, **k: False)
    with pytest.raises(SystemExit) as caught:
        chain.main([
            "--seeds", "0,1", "--jobs", "1", "--datasets", "cifar10,ham10000",
            "--backbones", "resnet50", "--out", str(tmp_path),
            "--train-log-dir", str(tmp_path / "train"),
        ])
    assert caught.value.code == 1
    assert calls == ["cifar10_resnet50_seed0"]


def test_unknown_cell_is_refused(tmp_path):
    from scripts.launch_paper_chain import main

    with pytest.raises(SystemExit, match="unknown cell"):
        main(["--seeds", "0", "--one", "nope", "--out", str(tmp_path)])
