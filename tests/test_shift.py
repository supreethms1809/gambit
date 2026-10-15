"""Shift payoffs, allocation, additive assignment, 2x2 scores, and degenerate checks. SHIFT_FORMULATION.md."""

from __future__ import annotations

import ast
import csv
import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from baselines.shift_maps import shift_assign
from cdea.shift import (
    ShiftConfig,
    allocate_shift,
    gap_units,
    log_odds,
    shift_cell_drops,
    transplant_payoff,
)
from core.grid import shift_pixels, transplant, upsample_units
from evaluation.scores import null_pixel_masks, null_shift_drops, shift_drop_matrix, shift_scores
from scripts.check_degenerate import d5_marginals, shift_gaps


class _Cell(nn.Module):
    """Log-odds rise with cell 0 in every view and with cell 3 only where it is painted."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cell0 = x[:, :, :8, :8].mean(dim=(1, 2, 3))
        cell3 = x[:, :, :8, 24:32].mean(dim=(1, 2, 3))
        score = self.scale * (4.0 * cell0 + 4.0 * cell3)
        return torch.stack([score, torch.zeros_like(score)], dim=1)


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        weight = torch.zeros(3, 8, 8)
        weight[:, :4, :4] = 1.0
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = (x * self.weight).sum(dim=(1, 2, 3)) + self.anchor * 0
        return torch.stack([score, torch.zeros_like(score)], dim=1)


def _views():
    x_id = torch.zeros(2, 3, 32, 32)
    x_ood = torch.zeros(2, 3, 32, 32)
    x_id[:, :, :8, :8] = 1
    x_ood[:, :, :8, :8] = 1
    x_id[:, :, :8, 24:32] = 1
    return x_id, x_ood


def test_identical_environments_have_zero_shortcut_payoff():
    model = _Cell().eval()
    x, _ = _views()
    y = model(x).argmax(-1)
    mask = torch.zeros(2, 16)
    mask[:, 3] = 1
    payoff = transplant_payoff(model, [x, x], y, mask, 4, 4)
    assert torch.allclose(payoff, torch.zeros_like(payoff), atol=1e-6)


def test_transplant_with_offset_moves_the_copied_cells():
    torch.manual_seed(0)
    x_id = torch.zeros(1, 3, 28, 28)
    x_env = torch.ones(1, 3, 28, 28)
    mask = torch.zeros(1, 49)
    mask[0, 24] = 1.0
    pixel = shift_pixels(upsample_units(mask, 7, 7, 28, 28), 2, -1)
    expected = (1.0 - pixel) * x_id + pixel * x_env
    assert torch.allclose(transplant(x_id, x_env, mask, 7, 7, offset=(2, -1)), expected)
    assert torch.equal(transplant(x_id, x_env, mask, 7, 7, offset=(0, 0)), transplant(x_id, x_env, mask, 7, 7))


def test_a_patch_that_differs_goes_to_the_shortcut_player():
    model = _Cell().eval()
    x_id, x_ood = _views()
    out = allocate_shift(
        model, [x_id, x_ood], 4, 4, 0.25,
        ShiftConfig(steps=40, lr=0.2, init="uniform"),
    )
    assert int(out.shortcut.mean(0).argmax()) == 3
    assert int(out.robust.mean(0).argmax()) == 0
    assert out.initial is not None and out.initial.shape == (2, 2, 16)
    assert out.overshoot is not None and out.overshoot.shape == (2,)


def test_shift_assign_recovers_robust_and_shortcut_cells_and_keeps_budgets():
    model = _Cell().eval()
    x_id, x_ood = _views()
    y = model(x_id).argmax(-1)
    g = shift_cell_drops(model, [x_id, x_ood], y, 4, 4)
    assert g.shape == (2, 2, 16)
    # Cell 0 is the only robust cell; cell 3 is the only shortcut cell under transplant.
    assert int(g[0, 0].argmax()) == 0
    assert int(g[0, 1].argmax()) == 3
    robust, shortcut = shift_assign(model, [x_id, x_ood], 4, 4, 1.0 / 16.0, g=g)
    assert int(robust[0].argmax()) == 0
    assert int(shortcut[0].argmax()) == 3
    assert torch.allclose(robust.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(shortcut.sum(-1), torch.ones(2), atol=1e-6)
    assert bool(((robust + shortcut) <= 1.0 + 1e-6).all())


def test_marginals_and_chunks():
    model = _Cell().eval()
    x_id, x_ood = _views()
    x_id = x_id.repeat(2, 1, 1, 1)
    x_ood = x_ood.repeat(2, 1, 1, 1)
    cfg = ShiftConfig(steps=3, lr=0.1, init="uniform")
    whole = allocate_shift(model, [x_id, x_ood], 4, 4, 0.25, cfg)
    assert d5_marginals(whole.transport, whole.row_target)
    left = allocate_shift(model, [x_id[:2], x_ood[:2]], 4, 4, 0.25, cfg)
    right = allocate_shift(model, [x_id[2:], x_ood[2:]], 4, 4, 0.25, cfg)
    joined = torch.cat([left.shortcut, right.shortcut], dim=0)
    assert torch.allclose(joined, whole.shortcut, atol=1e-5)


def test_views_on_another_device_raise_runtime_error():
    model = _Cell().eval()
    x_id, x_ood = _views()
    with pytest.raises(RuntimeError, match="different devices"):
        allocate_shift(model, [x_id, x_ood.to("meta")], 4, 4, 0.25, ShiftConfig(steps=1, init="uniform"))


def test_shift_ablation_switches_run():
    from evaluation.run_methods import SHIFT_ABLATIONS, shift_ablation

    model = _Cell().eval()
    x_id, x_ood = _views()
    for name in SHIFT_ABLATIONS:
        if name == "S6_precomputed":
            continue
        cfg = shift_ablation(name)
        cfg.steps = 3
        cfg.init = "uniform"
        out = allocate_shift(model, [x_id, x_ood], 4, 4, 0.25, cfg)
        assert bool(torch.isfinite(out.robust).all() and torch.isfinite(out.shortcut).all()), name


def test_gap_attribution_is_complete_on_a_linear_model():
    model = _Linear().eval()
    x_e = torch.zeros(1, 3, 8, 8)
    x_id = torch.zeros(1, 3, 8, 8)
    x_id[:, :, :4, :4] = 0.5
    x_e[:, :, :4, :4] = 0.1
    y = model(x_id).argmax(-1)
    units = gap_units(model, [x_id, x_e], y, 2, 2, steps=4)
    gap = (log_odds(model(x_id), y) - log_odds(model(x_e), y))
    assert torch.allclose(units.sum(-1), gap, atol=1e-4)


def test_shift_2x2_drop_matrix_and_purity_scores():
    model = _Cell().eval()
    x_id, x_ood = _views()
    y = model(x_id).argmax(-1)
    mask_r = torch.zeros(2, 32, 32)
    mask_r[:, :8, :8] = 1.0
    mask_s = torch.zeros(2, 32, 32)
    mask_s[:, :8, 24:32] = 1.0
    masks = torch.stack([mask_r, mask_s], dim=1)
    drops, prob_drops = shift_drop_matrix(model, [x_id, x_ood], y, masks, iters=0, noise=0.0, seed=0)
    assert drops.shape == (2, 2, 2) and prob_drops.shape == (2, 2)
    # R drops worst-case log-odds (col 0) and leaves the cross-env gap alone (col 1 = 0).
    assert bool((drops[:, 0, 0] > 3.0).all())
    assert torch.allclose(drops[:, 0, 1], torch.zeros(2), atol=1e-5)
    # S closes the cross-env gap (col 1 > 3.0) and has 0 worst-case robust drop (col 0 = 0).
    assert torch.allclose(drops[:, 1, 0], torch.zeros(2), atol=1e-5)
    assert bool((drops[:, 1, 1] > 3.0).all())
    scores = shift_scores(drops, torch.zeros(2, 2), prob_drops, torch.zeros(2))
    assert float(scores["rn"][0]) == pytest.approx(4.0, abs=1e-4)
    assert float(scores["delta_d"][0]) == pytest.approx(4.0, abs=1e-4)
    assert float(scores["pur_r"][0]) == pytest.approx(4.0, abs=1e-4)
    assert float(scores["pur_s"][0]) == pytest.approx(4.0, abs=1e-4)
    assert float(scores["role_sep"][0]) == pytest.approx(8.0, abs=1e-4)


def test_score_shift_writes_matrix_purity_and_stage_payoffs(tmp_path):
    from baselines.adapter import budget_or_floor, random_floor
    from evaluation.masks import mass_in, top_fraction_mask
    from evaluation.run_cell import CellSpec, _score_shift
    from evaluation.run_methods import Knobs, cdea_shift_maps
    from evaluation.scores import shortcut_patch_contrast, transplant_closure
    from scripts.gate_shift import load_shift_records, main as gate_main

    model = _Cell().eval()
    x_id, x_ood = _views()
    y = model(x_id).argmax(-1)
    spec = CellSpec(game="shift", dataset="waterbirds", backbone="resnet50", seed=0, split="val")
    sample = SimpleNamespace(index=[10, 11], labels=torch.tensor([0, 0]), foreground=None, mask_a=None, mask_b=None)
    maps = cdea_shift_maps(model, [x_id, x_ood], "resnet50", Knobs(cdea_steps=2), torch.device("cpu"),
                           ShiftConfig(init="uniform", steps=2), area=0.05)
    rows = _score_shift(
        model, [x_id, x_ood], maps, y, 0.05, spec, "cdea_shift", sample, 0,
        "hash1", "knob1", budget_or_floor, random_floor,
        null_pixel_masks, null_shift_drops, shift_drop_matrix, shift_scores,
        transplant_closure, shortcut_patch_contrast, mass_in, top_fraction_mask, 7, 7,
    )
    assert len(rows) == 2
    expected_keys = {
        "d_r_rob", "d_r_gap", "d_s_rob", "d_s_gap", "null_rob", "null_gap",
        "delta_d", "delta_d_prob", "rn", "pur_r", "pur_s", "role_sep", "tc",
        "delta_d_translated", "payoff_soft_r", "payoff_hard_r", "payoff_soft_s",
        "payoff_hard_s", "overshoot_share", "same_as_init",
    }
    assert expected_keys <= set(rows[0])

    # Write a shift cell and test shift_gaps and gate_shift readout.
    cell_dir = tmp_path / "val" / "shift" / "waterbirds" / "resnet50" / "seed0"
    cell_dir.mkdir(parents=True)
    with gzip.open(cell_dir / "shift.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    gaps = shift_gaps(cell_dir)
    assert 0.05 in gaps and gaps[0.05]["n"] == 2
    assert "d2_pass" in gaps[0.05] and "d4_pass" in gaps[0.05]
    loaded = load_shift_records(tmp_path, "val", "hash1")
    assert ("waterbirds", "resnet50", 0, 0.05) in loaded
    gate_main(["--records", str(tmp_path), "--split", "val"])


def test_shift_checkpoint_names_stay():
    from models.build import paper_checkpoint_name

    assert paper_checkpoint_name(
        "waterbirds", "resnet50", True, False, 15, 1e-4, 0, "imagenet",
    ) == "waterbirds_resnet50_pt_ft_ep15_lr0.0001_imagenet_seed0.pt"
    assert paper_checkpoint_name(
        "colored_mnist", "vit_b_16", True, False, 15, 1e-4, 1, "imagenet",
    ) == "colored_mnist_vit_b_16_pt_ft_ep15_lr0.0001_imagenet_seed1.pt"


def test_shift_degenerate_routes():
    from scripts.check_degenerate import d2_soft_hard_gap, d3_overshoot_share, d4_beats_translation

    assert d2_soft_hard_gap(torch.tensor(1.0), torch.tensor(1.2))
    assert not d2_soft_hard_gap(torch.tensor(1.0), torch.tensor(2.0))
    assert d4_beats_translation(torch.tensor([0.4, 0.2]), torch.tensor([0.1, 0.0]))
    assert d3_overshoot_share(torch.tensor([0.0, 0.5])) == 0.25


def _cifar_available() -> bool:
    return (Path(__file__).resolve().parents[1] / "data" / "cifar-10-batches-py").exists()


@pytest.mark.skipif(not _cifar_available(), reason="CIFAR-10 is not on this machine")
def test_planted_patch_counterfactual_changes_both_patches_in_place():
    from evaluation.planted_cues import PlantedPatchCIFAR

    ds = PlantedPatchCIFAR(split="val")
    item = ds[0]
    present, moved, removed = item["present"], item["moved"], item["removed"]
    mask_a, mask_b = item["mask_a"].bool(), item["mask_b"].bool()
    outside = ~(mask_a | mask_b)
    # Outside (a, b), present, moved, and removed are identical.
    assert torch.allclose(present[:, outside], moved[:, outside])
    assert torch.allclose(present[:, outside], removed[:, outside])
    # Inside mask_a and mask_b, moved changes both patches in-place at (a, b).
    assert not torch.allclose(present[:, mask_a], moved[:, mask_a])
    assert not torch.allclose(present[:, mask_b], moved[:, mask_b])


def test_shift_import_boundary():
    root = Path(__file__).resolve().parents[1] / "cdea" / "shift.py"
    tree = ast.parse(root.read_text())
    banned = {"evaluation", "baselines", "analysis"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        else:
            continue
        assert banned.isdisjoint(names)
