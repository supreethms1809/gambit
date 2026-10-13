"""``allocate()`` is unchanged by moving its plan helpers into ``cdea/_plan.py``.

The reference outputs were written by the code before the move
(``GAMBIT_WRITE_FIXTURE=1`` regenerates them; do not regenerate to make the
test pass). CPU, float32, fixed seeds, so the comparison is exact.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from cdea.allocation import AllocateConfig, allocate
from core.types import HypothesisSet

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "allocate_reference.pt"

CONFIGS = {
    "default": AllocateConfig(steps=6),
    "fractional": AllocateConfig(steps=6, binary_masks=False),
    "fixed_boundary": AllocateConfig(steps=6, boundary_shift=False),
    "uniform_pair": AllocateConfig(steps=6, init="uniform", pair_only=True),
    "independent": AllocateConfig(steps=6, independent=True, shared=False),
    "hard_top_mass": AllocateConfig(steps=6, projection="hard_top_mass"),
}


def small_cnn() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 7),
    )
    return model.eval()


def _inputs():
    torch.manual_seed(1)
    x = torch.rand(2, 3, 56, 56)
    model = small_cnn()
    with torch.no_grad():
        ids = model(x).topk(5, dim=-1).indices
    return model, x, HypothesisSet(ids=ids, mask=torch.ones_like(ids, dtype=torch.bool))


def run_all() -> dict:
    model, x, h = _inputs()
    out = {}
    for name, cfg in CONFIGS.items():
        result = allocate(model, x, h, 7, 7, 0.05, cfg)
        out[name] = {"unique": result.unique, "transport": result.transport,
                     "shared": result.shared if result.shared is not None else torch.empty(0)}
    return out


def test_allocate_matches_the_reference_from_before_the_move() -> None:
    if os.environ.get("GAMBIT_WRITE_FIXTURE"):
        torch.save(run_all(), FIXTURE)
    reference = torch.load(FIXTURE)
    current = run_all()
    for name, parts in reference.items():
        for key, value in parts.items():
            assert torch.equal(current[name][key], value), f"{name}.{key} changed"
