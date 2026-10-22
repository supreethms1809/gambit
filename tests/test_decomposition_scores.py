"""Decomposition scores, the runner rows, and the planted-pair dataset (handoff tests 8 and 9)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.grid import pool_sum
from evaluation.scores import (
    contrastive_deletion,
    decomposition_scores,
    deletion_matrix,
    drop_matrix,
    mask_patch_confusion,
    null_drops_road,
    null_pixel_masks,
)


class CellLinear(nn.Module):
    def __init__(self, classes: int = 6):
        super().__init__()
        torch.manual_seed(0)
        self.weight = nn.Parameter(torch.randn(classes, 49), requires_grad=False)

    def forward(self, x):
        spatial = x.mean(dim=1)
        counts = pool_sum(torch.ones_like(spatial), 7, 7)
        return (pool_sum(spatial, 7, 7) / counts) @ self.weight.t()


def _masks(batch: int, players: int, size: int = 224) -> torch.Tensor:
    torch.manual_seed(1)
    return (torch.rand(batch, players, size, size) < 0.05).float()


# 8 -------------------------------------------------------------------------

def test_spec_k_plus_spec_l_is_cd_on_the_same_masks_and_seeds() -> None:
    model = CellLinear().eval()
    torch.manual_seed(2)
    x = torch.rand(3, 3, 224, 224)
    ids = model(x).topk(2, dim=-1).indices
    masks = _masks(3, 3)
    drops = drop_matrix(model, x, masks, ids, seed=7, offset=5)
    scores = decomposition_scores(drops, torch.zeros(3, 2))
    cd = contrastive_deletion(model, x, masks[:, 0], masks[:, 1], ids[:, 0], ids[:, 1], seed=7, offset=5)
    assert torch.allclose(scores["spec_k"] + scores["spec_l"], cd, atol=1e-5)
    assert torch.allclose(scores["cd"], cd, atol=1e-5)


def test_drop_matrix_is_the_deletion_matrix_when_players_are_classes() -> None:
    model = CellLinear().eval()
    x = torch.rand(2, 3, 224, 224)
    ids = model(x).topk(3, dim=-1).indices
    masks = _masks(2, 3)
    assert torch.equal(deletion_matrix(model, x, masks, ids, seed=3),
                       drop_matrix(model, x, masks, ids, seed=3).transpose(-1, -2))


def test_scores_follow_their_definitions() -> None:
    # Players U_k, U_l, S; classes k, l.
    drops = torch.tensor([[[3.0, 1.0], [0.5, 2.0], [2.0, 1.5]]])
    null = torch.tensor([[1.0, 0.5]])
    s = decomposition_scores(drops, null)
    assert float(s["spec_k"]) == 2.0 and float(s["spec_l"]) == 1.5
    assert float(s["pur_k"]) == pytest.approx((3 - 1) - abs(1 - 0.5))
    assert float(s["pur_l"]) == pytest.approx((2 - 0.5) - abs(0.5 - 1))
    assert float(s["sv"]) == pytest.approx(min(1.0, 1.0))
    assert float(s["bal"]) == pytest.approx(0.0)
    # min excess over classes: U_k (2, 0.5) -> 0.5; U_l (-0.5, 1.5) -> -0.5.
    assert float(s["sep"]) == pytest.approx(1.0 - 0.5)
    missing = drops.clone()
    missing[:, 2] = float("nan")
    assert torch.isnan(decomposition_scores(missing, null)["sv"]).all()


def test_patch_confusion_precision_and_recall() -> None:
    patches = torch.zeros(1, 3, 8, 8)
    patches[0, 0, :4, :4] = 1
    patches[0, 1, 4:, 4:] = 1
    patches[0, 2, :4, 4:] = 1
    masks = torch.zeros(1, 3, 8, 8)
    masks[0, 0, :4, :4] = 1           # exactly A_k
    masks[0, 1, 4:, :4] = 1           # all on the rest
    masks[0, 2, :2, 4:] = 1           # half of C
    precision, recall = mask_patch_confusion(masks, patches)
    assert precision[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert precision[0, 1].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert precision[0, 2].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert recall[0, 0, 0] == 1.0 and recall[0, 2, 2] == 0.5


def test_road_null_uses_masks_of_the_area() -> None:
    masks = null_pixel_masks(7, 7, 0.05, 224, 224)
    assert masks.shape == (8, 224, 224)
    assert torch.allclose(masks.mean(dim=(-1, -2)), torch.full((8,), 0.05), atol=1e-3)
    model = CellLinear().eval()
    x = torch.rand(2, 3, 224, 224)
    ids = model(x).topk(2, dim=-1).indices
    assert null_drops_road(model, x, ids, 0.05, 7, 7).shape == (2, 2)


def test_runner_rows_carry_the_matrix_scores_and_stage_payoffs() -> None:
    from baselines.adapter import budget_or_floor
    from evaluation.run_cell import CellSpec, _score_decomp
    from evaluation.run_methods import DecompConfig, DecompMaps, Knobs, cdea_decomp, hypotheses_for

    model = CellLinear().eval()
    x = torch.rand(2, 3, 224, 224)
    h = hypotheses_for(model, x, 2)
    spec = CellSpec(game="decomposition", dataset="cifar10", backbone="resnet50", operators=("road", "blur"))
    sample = SimpleNamespace(index=[10, 11], labels=[0, 1], patches=None, roles=None)
    maps = cdea_decomp(model, x, h, "resnet50", Knobs(cdea_steps=2), torch.device("cpu"),
                       DecompConfig(init="uniform"), area=0.05)
    rows = _score_decomp(model, x, maps, h, 0.05, spec, "cdea_decomp", sample, budget_or_floor, 0, "c", "k")
    assert len(rows) == 4
    keys = {"cd", "spec_k", "spec_l", "pur_k", "pur_l", "sv", "bal", "sep", "null_k", "null_l",
            "payoff_soft_uk", "payoff_hard_s", "payoff_scored_blur_ul", "same_as_init"}
    assert keys <= set(rows[0])
    assert all(abs(r["cd"] - (r["spec_k"] + r["spec_l"])) < 1e-5 for r in rows)
    assert all(abs(r["area_uk"] - 0.05) < 1e-3 and abs(r["area_s"] - 0.05) < 1e-3 for r in rows)

    # A pixel-map method with no shared map, as margin_gradcam_ref returns.
    ref = DecompMaps(k=torch.rand(2, 224, 224), l=torch.rand(2, 224, 224), shared=None, grid=None)
    rows = _score_decomp(model, x, ref, h, 0.05, spec, "margin_gradcam_ref", sample, budget_or_floor, 0, "c", "k")
    assert all(r["sv"] != r["sv"] and r["cd"] == r["cd"] for r in rows)   # no S: NaN; CD present
    assert "payoff_soft_uk" not in rows[0]
    assert all(r["constant_row"] == 0 for r in rows)

    # On planted patches the missing S has no recall or precision, so RR averages U_k and U_l.
    from scripts.decomp_readout import _value

    patches = torch.zeros(2, 3, 224, 224)
    patches[:, 0, :32, :32] = 1
    patches[:, 1, :32, 192:] = 1
    patches[:, 2, 192:, :32] = 1
    planted = SimpleNamespace(index=[10, 11], labels=[0, 1], patches=patches, roles=None)
    rows = _score_decomp(model, x, ref, h, 0.05, spec, "margin_gradcam_ref", planted, budget_or_floor, 0, "c", "k")
    for r in rows:
        assert r["rec_s_c"] != r["rec_s_c"] and r["prec_s_c"] != r["prec_s_c"]
        assert r["rec_uk_ak"] == r["rec_uk_ak"]
        rr = _value({key: str(r[key]) for key in ("rec_uk_ak", "rec_ul_al", "rec_s_c")}, "rr")
        assert abs(rr - (r["rec_uk_ak"] + r["rec_ul_al"]) / 2) < 1e-6

    # An all-zero map is scored as the floor and flagged.
    flat = DecompMaps(k=torch.zeros(2, 224, 224), l=torch.rand(2, 224, 224), shared=None, grid=None)
    rows = _score_decomp(model, x, flat, h, 0.05, spec, "margin_gradcam_ref", sample, budget_or_floor, 0, "c", "k")
    assert all(r["constant_row"] == 1 and r["failed_row"] == 0 for r in rows)


def test_every_decomposition_ablation_names_a_change() -> None:
    from evaluation.run_methods import DECOMP_ABLATIONS, DecompConfig

    base = DecompConfig()
    for name, change in DECOMP_ABLATIONS.items():
        assert change(base) != base, name


# 9 -------------------------------------------------------------------------

def _cifar_available() -> bool:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "data" / "cifar-10-batches-py").exists()


@pytest.mark.skipif(not _cifar_available(), reason="CIFAR-10 is not on this machine")
def test_planted_pair_test_items_carry_three_disjoint_patches() -> None:
    from evaluation.planted_cues import PAIRS, PlantedPairCIFAR

    ds = PlantedPairCIFAR(split="val", mode="test")
    pair_classes = {c for pair in PAIRS for c in pair}
    for i in range(20):
        item = ds[i]
        assert item["label"] in pair_classes
        masks = item["masks"]
        assert masks.sum(dim=(-1, -2)).tolist() == [32 * 32] * 3
        assert float((masks.sum(dim=0) > 1).sum()) == 0
        for slot in range(3):
            ys, xs = torch.nonzero(masks[slot], as_tuple=True)
            assert int(ys.min()) % 32 == 0 and int(xs.min()) % 32 == 0   # grid-aligned, one cell
        assert ds[i]["image"].equal(item["image"])                      # deterministic


def test_adjacent_cells_put_a_k_beside_a_l_and_c_apart() -> None:
    from evaluation.planted_cues import adjacent_cells

    for draw in range(200):
        ak, al, c = adjacent_cells(torch.Generator().manual_seed(draw))
        (r0, c0), (r1, c1), (r2, c2) = divmod(ak, 7), divmod(al, 7), divmod(c, 7)
        assert abs(r0 - r1) + abs(c0 - c1) == 1                       # share a side
        assert min(max(abs(r2 - r), abs(c2 - q)) for r, q in ((r0, c0), (r1, c1))) > 1


def test_the_adjacent_condition_uses_the_planted_pair_checkpoints() -> None:
    from evaluation.planted_cues import PLANTED_CONDITIONS
    from evaluation.run_models import paper_checkpoint_path

    assert PLANTED_CONDITIONS == {"planted_pair": "grid", "planted_pair_adjacent": "adjacent"}
    for seed in (0, 3):
        assert (paper_checkpoint_path("planted_pair_adjacent", "resnet50", seed)
                == paper_checkpoint_path("planted_pair", "resnet50", seed))


@pytest.mark.skipif(not _cifar_available(), reason="CIFAR-10 is not on this machine")
def test_adjacent_planted_items_touch_only_a_k_and_a_l() -> None:
    from evaluation.planted_cues import PlantedPairCIFAR

    with pytest.raises(ValueError):
        PlantedPairCIFAR(split="val", mode="train", placement="adjacent")
    ds = PlantedPairCIFAR(split="val", mode="test", placement="adjacent")
    grid = PlantedPairCIFAR(split="val", mode="test")
    for i in range(20):
        item = ds[i]
        assert item["image"].shape == grid[i]["image"].shape and item["label"] == grid[i]["label"]
        corners = []
        for slot in range(3):
            ys, xs = torch.nonzero(item["masks"][slot], as_tuple=True)
            assert int(ys.min()) % 32 == 0 and int(xs.min()) % 32 == 0 and len(ys) == 32 * 32
            corners.append((int(ys.min()) // 32, int(xs.min()) // 32))
        (r0, c0), (r1, c1), (r2, c2) = corners
        assert abs(r0 - r1) + abs(c0 - c1) == 1
        assert min(max(abs(r2 - r), abs(c2 - q)) for r, q in corners[:2]) > 1


@pytest.mark.skipif(not _cifar_available(), reason="CIFAR-10 is not on this machine")
def test_planted_pair_training_frequencies() -> None:
    from evaluation.planted_cues import PAIRS, PATCH_PROBABILITY, PlantedPairCIFAR, pair_of

    ds = PlantedPairCIFAR(split="train", mode="train")
    own = shared = counted = other = 0
    for i in range(0, len(ds), 25):
        item = ds[i]
        present = item["present"].tolist()
        if pair_of(item["label"]) is None:
            other += sum(present)
            continue
        k_class, _l_class = PAIRS[item["pair"]]
        mine, theirs = (0, 1) if item["label"] == k_class else (1, 0)
        assert not present[theirs]                 # A_k never on a class-l image
        own += present[mine]
        shared += present[2]
        counted += 1
    assert other == 0
    assert abs(own / counted - PATCH_PROBABILITY) < 0.04
    assert abs(shared / counted - PATCH_PROBABILITY) < 0.04


def test_role_check_confirms_planted_roles_on_a_linear_toy() -> None:
    """A model that reads three regions: A_k for k, A_l for l, C for both."""
    from evaluation.planted_cues import verify_roles

    class Regions(nn.Module):
        def forward(self, x):
            a_k = x[:, 0, :32, :32].mean(dim=(-1, -2))
            a_l = x[:, 0, :32, 64:96].mean(dim=(-1, -2))
            c = x[:, 0, 64:96, :32].mean(dim=(-1, -2))
            return torch.stack([4 * a_k + 4 * c, 4 * a_l + 4 * c, torch.zeros_like(c)], dim=-1)

    x = torch.full((2, 3, 224, 224), 0.1)
    x[:, 0, :32, :32] = 1.0
    x[:, 0, :32, 64:96] = 1.0
    x[:, 0, 64:96, :32] = 1.0
    patches = torch.zeros(2, 3, 224, 224)
    patches[:, 0, :32, :32] = 1
    patches[:, 1, :32, 64:96] = 1
    patches[:, 2, 64:96, :32] = 1
    roles = verify_roles(Regions().eval(), x, patches, torch.tensor([[0, 1], [0, 1]]))
    assert roles.all()
    swapped = verify_roles(Regions().eval(), x, patches[:, [2, 1, 0]], torch.tensor([[0, 1], [0, 1]]))
    assert not swapped[:, 0].any() and not swapped[:, 2].any()
