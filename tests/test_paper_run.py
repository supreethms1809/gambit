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
