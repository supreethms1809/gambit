from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.game_modes import resolve_contrastive_game, resolve_shift_game


def test_contrastive_presets_are_separated():
    coop = resolve_contrastive_game("cooperative")
    comp = resolve_contrastive_game("competitive")
    mixed = resolve_contrastive_game("mixed")

    assert coop.use_shared is True
    assert coop.lambda_overlap == 0.0
    assert coop.lambda_disjoint == 0.0
    assert comp.use_shared is False
    assert comp.lambda_overlap > mixed.lambda_overlap
    assert comp.lambda_disjoint > mixed.lambda_disjoint
    assert mixed.use_shared is True
    print("PASS: contrastive cooperative/competitive/mixed presets are separated")


def test_contrastive_manual_mode():
    cfg = resolve_contrastive_game(
        "manual",
        use_shared=False,
        lambda_margin=0.9,
        lambda_overlap=0.12,
        lambda_disjoint=0.34,
        lambda_partition=0.0,
    )
    assert cfg.mode == "manual"
    assert cfg.use_shared is False
    assert abs(cfg.lambda_margin - 0.9) < 1e-8
    assert abs(cfg.lambda_overlap - 0.12) < 1e-8
    assert abs(cfg.lambda_disjoint - 0.34) < 1e-8
    assert abs(cfg.lambda_partition - 0.0) < 1e-8
    print("PASS: contrastive manual mode")


def test_shift_presets_are_separated():
    coop = resolve_shift_game("cooperative")
    comp = resolve_shift_game("competitive")
    mixed = resolve_shift_game("mixed")

    assert coop.lambda_gap == 0.0
    assert coop.lambda_shortcut > 0.0
    assert coop.lambda_disjoint == 0.0
    assert comp.lambda_gap > mixed.lambda_gap
    assert comp.lambda_shortcut == 0.0
    assert comp.lambda_disjoint > mixed.lambda_disjoint
    print("PASS: shift cooperative/competitive/mixed presets are separated")


def test_shift_manual_mode():
    cfg = resolve_shift_game(
        "manual",
        lambda_mean=1.0,
        lambda_var=0.3,
        lambda_gap=0.7,
        lambda_shortcut=0.6,
        lambda_disjoint=0.2,
        lambda_sparse=0.05,
    )
    assert cfg.mode == "manual"
    assert abs(cfg.lambda_mean - 1.0) < 1e-8
    assert abs(cfg.lambda_var - 0.3) < 1e-8
    assert abs(cfg.lambda_gap - 0.7) < 1e-8
    assert abs(cfg.lambda_shortcut - 0.6) < 1e-8
    assert abs(cfg.lambda_disjoint - 0.2) < 1e-8
    assert abs(cfg.lambda_sparse - 0.05) < 1e-8
    print("PASS: shift manual mode")


def test_manual_missing_values_raise():
    try:
        resolve_shift_game("manual", lambda_mean=1.0, lambda_var=0.5, lambda_gap=1.0)
        assert False, "manual shift mode should raise on missing values"
    except ValueError:
        pass

    try:
        resolve_contrastive_game("manual", use_shared=True, lambda_margin=1.0, lambda_overlap=0.1)
        assert False, "manual contrastive mode should raise on missing values"
    except ValueError:
        pass
    print("PASS: manual mode validates required values")


if __name__ == "__main__":
    test_contrastive_presets_are_separated()
    test_contrastive_manual_mode()
    test_shift_presets_are_separated()
    test_shift_manual_mode()
    test_manual_missing_values_raise()
    print("All game mode tests OK.")
