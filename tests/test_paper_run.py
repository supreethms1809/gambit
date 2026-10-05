"""The per-cell runner: method lists, foil swapping, ablation configs, pass counts."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.types import HypothesisSet
from evaluation.run_cell import PassCounter, _method_runs
from evaluation.run_methods import (
    ABLATIONS,
    CONTRASTIVE_CORE,
    SHIFT_ABLATIONS,
    CdeaConfig,
    ShiftConfig,
    default_backend,
    swapped,
)
from scripts.paper_run import parse, spec_from_args


def test_swapped_exchanges_rank_0_and_rank_1_only():
    h = HypothesisSet(ids=torch.tensor([[4, 2, 7]]), mask=torch.tensor([[True, True, False]]))
    s = swapped(h)
    assert s.ids.tolist() == [[2, 4, 7]]
    assert s.mask.tolist() == [[True, True, False]]
    assert h.ids.tolist() == [[4, 2, 7]]


def test_every_ablation_changes_the_config_it_names():
    base = CdeaConfig()
    assert ABLATIONS["A1_independent"](base).independent is True
    assert ABLATIONS["A2_no_margin"](base).lambda_margin == 0.0
    assert ABLATIONS["A3_no_overlap"](base).lambda_overlap == 0.0
    assert ABLATIONS["A4_no_shared"](base).use_shared is False
    assert ABLATIONS["A5_init_zero"](base).init_from_evidence is False
    assert ABLATIONS["A6_interaction_attention"](base).attn_mix > 0
    assert ABLATIONS["A7_steps_100"](base).steps == 100
    assert ABLATIONS["A8_preset_competitive"](base).preset == "competitive"
    assert SHIFT_ABLATIONS["AS1_unpaired_nomass"](ShiftConfig()).objective == "unpaired"
    assert SHIFT_ABLATIONS["AS1_unpaired_nomass"](ShiftConfig()).lambda_mass == 0.0


def test_vit_uses_ig_and_skips_cve():
    assert default_backend("vit_b_16") == "ig"
    assert default_backend("resnet50") == "gradcam"
    spec = spec_from_args(parse(["--game", "contrastive", "--dataset", "cifar10", "--backbone", "vit_b_16",
                                 "--methods", "core"]))
    assert "cve" not in spec.methods
    spec = spec_from_args(parse(["--game", "contrastive", "--dataset", "cifar10", "--methods", "core"]))
    assert list(spec.methods) == list(CONTRASTIVE_CORE)


def test_ablations_run_as_cdea_variants():
    spec = spec_from_args(parse(["--game", "contrastive", "--dataset", "cifar10", "--methods", "none",
                                 "--ablations", "A2_no_margin,A3_no_overlap"]))
    assert _method_runs(spec) == [("cdea", "A2_no_margin", None), ("cdea", "A3_no_overlap", None)]
    spec = spec_from_args(parse(["--game", "shift", "--dataset", "waterbirds", "--methods", "none",
                                 "--ablations", "all"]))
    assert {m for m, _a, _c in _method_runs(spec)} == {"cdea_shift"}


def test_pass_counter_counts_forward_and_backward_images():
    model = nn.Linear(3, 2)
    counter = PassCounter(model)
    x = torch.rand(4, 3, requires_grad=True)
    model(x).sum().backward()
    with torch.no_grad():
        model(torch.rand(2, 3))
    assert counter.forward == 6
    assert counter.backward == 4
    counter.close()


def test_selection_candidates_cover_the_plan_grid():
    from evaluation.run_methods import CONTRASTIVE_CANDIDATES, SHIFT_CANDIDATES, candidate_applies

    cdea = [c for c in CONTRASTIVE_CANDIDATES if c.startswith("cdea@")]
    assert len(cdea) == 18
    assert len([c for c in cdea if candidate_applies(c, "vit_b_16")]) == 9
    assert len([c for c in CONTRASTIVE_CANDIDATES if c.startswith("extremal@")]) == 4
    assert len([c for c in CONTRASTIVE_CANDIDATES if c.startswith("rise_margin@")]) == 4
    assert len([c for c in SHIFT_CANDIDATES if c.startswith("cdea_shift@")]) == 12
    spec = spec_from_args(parse(["--game", "contrastive", "--dataset", "cifar10", "--methods", "none",
                                 "--candidates", "all", "--backbone", "vit_b_16"]))
    assert "margin_gradcam@default" not in spec.candidates
    assert all("@gradcam" not in c for c in spec.candidates)
