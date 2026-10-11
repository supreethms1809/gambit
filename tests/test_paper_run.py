"""The per-cell runner: method lists, foil swapping, ablation configs, pass counts."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.types import HypothesisSet
from evaluation.run_cell import PassCounter, _method_runs
from evaluation.run_methods import (
    ABLATIONS,
    CONTRASTIVE_CORE,
    CdeaConfig,
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
    assert ABLATIONS["A2_no_shared"](base).shared is False
    assert ABLATIONS["A3_preserve"](base).preserve is True
    assert ABLATIONS["A4_pair"](base).pair_only is True
    assert ABLATIONS["A5_uniform"](base).init == "uniform"
    assert ABLATIONS["A6_hard_top_mass"](base).projection == "hard_top_mass"
    assert ABLATIONS["A7_steps_100"](base).steps == 100
    assert ABLATIONS["A9_first_order"](base).kind == "first_order"
    assert ABLATIONS["A10_boundary_shift"](base).boundary_shift is True


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
                                 "--ablations", "A2_no_shared,A3_preserve"]))
    assert _method_runs(spec) == [("cdea", "A2_no_shared", None), ("cdea", "A3_preserve", None)]


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
    from evaluation.run_methods import CONTRASTIVE_CANDIDATES, candidate_applies

    cdea = [c for c in CONTRASTIVE_CANDIDATES if c.startswith("cdea@")]
    assert len(cdea) == 6
    assert len([c for c in cdea if candidate_applies(c, "vit_b_16")]) == 4
    assert len([c for c in cdea if candidate_applies(c, "resnet50")]) == 4
    assert len([c for c in CONTRASTIVE_CANDIDATES if c.startswith("extremal@")]) == 4
    assert len([c for c in CONTRASTIVE_CANDIDATES if c.startswith("rise_margin@")]) == 4
    spec = spec_from_args(parse(["--game", "contrastive", "--dataset", "cifar10", "--methods", "none",
                                 "--candidates", "all", "--backbone", "vit_b_16"]))
    assert "margin_gradcam@default" not in spec.candidates
    assert all("@gradcam" not in c for c in spec.candidates)


def test_runner_outputs_do_not_dirty_the_tree():
    """A --final grid writes here between cells; a visible file would refuse the next cell."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # Paths a run creates between cells. (The committed smoke REPORT.md is tracked,
    # so check-ignore does not apply to it.)
    for path in ("results/paper/runs/val/contrastive/x/records.csv.gz",
                 "results/paper/cells/val/contrastive/x/records.csv.gz",
                 "results/paper/smoke/fast_knobs/val/contrastive/x/summary.json"):
        out = subprocess.run(["git", "check-ignore", "-q", path], cwd=root)
        assert out.returncode == 0, f"{path} is not ignored"


def test_records_path_separates_val_and_test(tmp_path):
    from evaluation.run_cell import CellSpec, _write
    from evaluation.run_models import LoadedModel

    for split in ("val", "test"):
        spec = CellSpec(game="contrastive", dataset="cifar10", backbone="resnet50", seed=0,
                        split=split, out_dir=str(tmp_path))
        out = _write(spec, [{"split": split}], {"rows": 1}, device=None)
        assert out.parts[-5] == split
    assert (tmp_path / "val").is_dir() and (tmp_path / "test").is_dir()


def test_n_is_set_per_backbone():
    import pytest

    from scripts.launch_paper_eval import parse_n

    assert parse_n("resnet50:200,vit_b_16:64") == {"resnet50": 200, "vit_b_16": 64}
    assert parse_n(None) == {}
    with pytest.raises(SystemExit):
        parse_n("200")


def test_record_index_names_the_dataset_image():
    from torch.utils.data import Subset

    from evaluation.run_data import root_indices

    base = list(range(100))
    split = Subset(base, [10, 20, 30, 40, 50])
    sample = Subset(split, [4, 1])
    assert root_indices(sample) == [50, 20]


def test_stage_payoffs_are_recorded_for_an_allocation_only():
    """D2 inputs ride on CDEA rows. Baseline rows keep the columns they had."""
    from types import SimpleNamespace

    from baselines.adapter import budget_or_floor
    from core.grid import pool_sum
    from evaluation.run_cell import CellSpec, _score_pair
    from evaluation.run_methods import Knobs, PairMaps, cdea_pair, hypotheses_for

    class CellLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(6, 49), requires_grad=False)

        def forward(self, x):
            spatial = x.mean(dim=1)
            counts = pool_sum(torch.ones_like(spatial), 7, 7)
            return (pool_sum(spatial, 7, 7) / counts) @ self.weight.t()

    torch.manual_seed(0)
    model = CellLinear().eval()
    x = torch.rand(2, 3, 224, 224)
    h = hypotheses_for(model, x, 5)
    k, l = h.ids[:, 0], h.ids[:, 1]
    full = model(x).detach()
    spec = CellSpec(dataset="cifar10", backbone="resnet50", operators=("blur",))
    sample = SimpleNamespace(index=[10, 11], labels=[0, 1])
    args = (k, l, full, 0.05, spec, None, sample, budget_or_floor, 0, "code", "knobs")

    maps = cdea_pair(model, x, h, "resnet50", Knobs(cdea_steps=2), torch.device("cpu"),
                     CdeaConfig(init="uniform"), area=0.05)
    rows = _score_pair(model, x, maps, *args[:4], args[4], "cdea", *args[6:])
    stage_keys = {f"payoff_{s}_{r}" for s in ("soft", "hard", "scored_blur") for r in ("k", "l")}
    unique_rows = [row for row in rows if row["method"] == "cdea"]
    assert unique_rows and stage_keys <= set(unique_rows[0])
    assert all(torch.isfinite(torch.tensor([row[key] for row in unique_rows for key in stage_keys])))
    # The shared sensitivity rows delete two masks; D2 is about the unique players.
    assert not stage_keys & {key for row in rows if row["method"] == "cdea+shared" for key in row}

    baseline = PairMaps(k=torch.rand(2, 49), l=torch.rand(2, 49), grid=(7, 7))
    base_rows = _score_pair(model, x, baseline, *args[:4], args[4], "base_evidence", *args[6:])
    assert not stage_keys & set(base_rows[0])
