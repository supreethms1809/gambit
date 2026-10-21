"""Decomposition version 2: Omega, the size-matched null, the partition, the rule rows and the scores.

FORMULATION_DECOMPOSITION.md version 2 section 9. The toys are the cell toys of
``test_decomposition.py``: each cell carries one feature (a checkerboard
amplitude) that blur deletion removes. Omega is set by ``rho`` so that every
cell the test names is in it.
"""

from __future__ import annotations

import pytest
import torch

from test_decomposition import GRID, UNITS, CellToy, _cells, _pair, additive_toy, suppression_toy, toy_image

from baselines.decomposition import cell_partition, occlusion_partition, random_partition, rule_partition
from cdea.decompose import PartitionConfig, partition, reference_evidence
from cdea.null import interpolate_sizes, null_at, null_size_grid, size_masks, size_null
from cdea.region import (
    NEITHER,
    OUTSIDE,
    SHARED,
    U_K,
    U_L,
    evidence_region,
    normalise,
    roles_from_labels,
    rule_labels,
    rule_parts,
    top_mass_cells,
)
from core.grid import pool_sum
from core.types import HypothesisSet
from evaluation.scores import (
    _remove,
    half_verdict,
    partition_scores,
    planted_role_scores,
    removal_drops,
    resolution_shares,
    role_confirmed,
    scored_size_null,
    split_halves,
)

K_ONLY, L_ONLY, SHARED_CELL = 0, 3, 12


def _setup(model, x, rho):
    h = _pair(model, x)
    evidence = reference_evidence(model, x, h, *GRID, "gradcam")
    return h, evidence, evidence_region(evidence, rho)


def _labels(model, x, rho, steps=100):
    """``subtract``, ``occlusion_partition`` and CDEA labels of image 0."""
    h, evidence, omega = _setup(model, x, rho)
    subtract = rule_labels(rule_parts(evidence), omega)
    occlusion = occlusion_partition(model, x, h.ids, *GRID, omega)
    cdea = partition(model, x, h, *GRID, PartitionConfig(steps=steps, rho=rho), evidence=evidence)
    return subtract[0], occlusion[0], cdea


# Omega ------------------------------------------------------------------------

def test_omega_is_the_smallest_top_mass_set_with_ties_to_the_lower_index() -> None:
    flat = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0]])
    assert top_mass_cells(flat, 0.5).tolist() == [[True, True, False, False, False]]
    assert top_mass_cells(flat, 0.51).tolist() == [[True, True, True, False, False]]
    # Cells with no mass never enter, even at rho = 1.
    assert top_mass_cells(flat, 1.0).tolist() == [[True, True, True, True, False]]
    assert not top_mass_cells(torch.zeros(1, 5), 0.5).any()
    evidence = normalise(torch.tensor([[[4.0, 3, 2, 1, 0], [0, 0, 1, 1, 8]]]))
    assert evidence_region(evidence, 0.5).tolist() == [[True, True, False, False, True]]


def test_rule_takes_the_largest_part_and_never_labels_empty() -> None:
    evidence = normalise(torch.tensor([[[4.0, 3, 2, 1, 0], [0, 0, 1, 1, 8]]]))
    omega = evidence_region(evidence, 1.0)
    labels = rule_labels(rule_parts(evidence), omega)
    # Cell 2: E_k - m = m = 0.1, a tie that goes to U_k. Cell 3: only m is positive.
    assert labels.tolist() == [[U_K, U_K, U_K, SHARED, U_L]]
    assert not (labels == NEITHER).any()
    assert rule_labels(rule_parts(evidence), evidence_region(evidence, 0.5)).tolist()[0][2] == OUTSIDE


def test_cdea_step_zero_is_the_subtraction_partition() -> None:
    model, x = additive_toy()
    h, evidence, omega = _setup(model, x, 1.0)
    out = partition(model, x, h, *GRID, PartitionConfig(steps=0, rho=1.0), evidence=evidence)
    subtract = roles_from_labels(rule_labels(rule_parts(evidence), omega))
    assert torch.equal(out.initial_roles, subtract)
    assert torch.equal(out.roles, subtract)


# Toy 1: additive ----------------------------------------------------------------

def test_additive_toy_rules_recover_the_roles() -> None:
    model, x = additive_toy()
    subtract, occlusion, _ = _labels(model, x, 1.0, steps=0)
    for labels in (subtract, occlusion):
        assert int(labels[K_ONLY]) == U_K and int(labels[L_ONLY]) == U_L and int(labels[SHARED_CELL]) == SHARED
        assert int((labels != OUTSIDE).sum()) == 3


@pytest.mark.xfail(strict=True, reason=(
    "Known limit of the straight-through gradient, reported in PROGRESS.md: a pure U_k has d~_l = -nu_l < 0, "
    "so |d~_l| has slope -1 there and l-evidence looks like a gain for U_k. At lr 0.1 the shared cell "
    "drifts from S to U_k, and the final hard welfare (2.0) falls below step 0's (2.625)."))
def test_additive_toy_cdea_keeps_the_shared_cell() -> None:
    model, x = additive_toy()
    _, _, cdea = _labels(model, x, 1.0)
    labels = cdea.labels[0]
    assert int(labels[K_ONLY]) == U_K and int(labels[L_ONLY]) == U_L and int(labels[SHARED_CELL]) == SHARED


ART = 5
BACKGROUND = [6, 7, 8, 9, 10, 11, 13, 14]


class ArtefactToy(CellToy):
    """Cell ART also reads channel 1's cell mean. Channel 1 is constant, so blur deletion leaves it unchanged."""

    def features(self, x):
        f = super().features(x).clone()
        ch1 = pool_sum(x[:, 1], *GRID) / pool_sum(torch.ones_like(x[:, 1]), *GRID)
        f[:, ART] = f[:, ART] + (ch1[:, ART] - 0.5)
        return f


def test_an_attribution_artefact_is_empty_for_deletion_methods_and_unique_for_the_rule() -> None:
    """Case 2 of section 1: the evidence names the cell, deleting it changes nothing."""
    weight = _cells((0, K_ONLY, 1.0), (1, L_ONLY, 1.0), (0, SHARED_CELL, 1.0), (1, SHARED_CELL, 1.0),
                    (0, ART, 3.0), *[(0, c, 0.1) for c in BACKGROUND])
    model = ArtefactToy(weight, torch.tensor([0.5, 0.0, -5.0])).eval()
    amplitude = torch.ones(UNITS)
    amplitude[ART] = 0.0
    x = toy_image(amplitude)
    x[:, 1] = 0.9
    subtract, occlusion, cdea = _labels(model, x, 0.8)
    assert bool(cdea.omega[0, ART])
    assert int(subtract[ART]) == U_K
    assert int(occlusion[ART]) == NEITHER
    assert int(cdea.labels[0, ART]) == NEITHER


# Toy 2: suppression -------------------------------------------------------------

SUPPRESSOR, WEAK_K, ONLY_L = 0, 3, 12


def test_suppressing_the_rival_is_not_unique_evidence() -> None:
    model, x = suppression_toy()
    subtract, _, cdea = _labels(model, x, 1.0)
    assert int(subtract[SUPPRESSOR]) == U_K
    assert int(cdea.labels[0, SUPPRESSOR]) == NEITHER
    assert int(cdea.labels[0, WEAK_K]) == U_K and int(cdea.labels[0, ONLY_L]) == U_L

    from baselines.decomposition import margin_partition
    from baselines.margin import margin_gradcam

    h = _pair(model, x)
    margin = margin_partition(margin_gradcam(model, x, h), cdea.omega, *GRID)
    assert int(margin[0, SUPPRESSOR]) == U_K


# Toy 3: redundancy --------------------------------------------------------------

EITHER_A, EITHER_B, L_CELL = 0, 3, 15


def redundancy_toy():
    """z_k needs a or b: deleting one moves z_k by 2 tau log 2, below a random cell's drop. Deleting both moves it by 2."""
    tau = 0.01

    def head(f):
        either = tau * torch.logsumexp(torch.stack([f[:, EITHER_A], f[:, EITHER_B]], dim=-1) / tau, dim=-1)
        z_k = 2.0 * either + 0.1 * f[:, BACKGROUND].sum(dim=-1)
        z_l = f[:, L_CELL] + 1.0
        return torch.stack([z_k, z_l, torch.full_like(z_k, -5.0)], dim=-1)

    return CellToy(torch.zeros(3, UNITS), torch.zeros(3), head=head).eval(), toy_image(torch.ones(UNITS))


def test_joint_search_keeps_redundant_evidence_that_single_cells_miss() -> None:
    model, x = redundancy_toy()
    _, occlusion, cdea = _labels(model, x, 0.5)
    assert cdea.omega[0, [EITHER_A, EITHER_B]].all()
    assert int(occlusion[EITHER_A]) != U_K and int(occlusion[EITHER_B]) != U_K
    assert int(cdea.labels[0, EITHER_A]) == U_K and int(cdea.labels[0, EITHER_B]) == U_K


# Toy 4: mixing ------------------------------------------------------------------

K_BACKGROUND, L_BACKGROUND = [6, 7, 8, 9], [10, 11, 13, 14]


def mixing_toy():
    weight = _cells((0, K_ONLY, 1.0), (1, L_ONLY, 1.0), *[(0, c, 0.1) for c in K_BACKGROUND],
                    *[(1, c, 0.1) for c in L_BACKGROUND])
    return CellToy(weight, torch.tensor([0.5, 0.0, -5.0])).eval(), toy_image(torch.ones(UNITS))


def test_with_no_shared_cell_cdea_leaves_s_empty_or_unconfirmed() -> None:
    """Section 9 toy 4. CDEA parks a weak k cell in S (it ties with the empty role: both start at log eps
    and Adam moves both alike), so S is not empty here. The scorer does not confirm it (D8)."""
    model, x = mixing_toy()
    _, _, cdea = _labels(model, x, 1.0)
    if cdea.roles[0, SHARED].any():
        scores, _ = _score(model, x, cdea.roles)
        assert float(scores["confirmed_s"]) == 0.0


def _score(model, x, roles, operator="blur"):
    """Scores of ``roles`` (``(1, 4, R)`` bool) on a toy, as the runner computes them."""
    h, evidence, omega = _setup(model, x, 1.0)
    ids = torch.cat([h.ids, torch.tensor([[2]])], dim=1)
    sizes = roles.sum(dim=-1)
    null = scored_size_null(model, x, ids, GRID, set(sizes.flatten().tolist()) | {1}, operator)
    drops = removal_drops(model, x, roles.float(), ids, operator, GRID)
    nan = torch.full((1, 4, 3), float("nan"))
    nu, q = nan.clone(), nan.clone()
    for p in range(4):
        s = int(sizes[0, p])
        if s:
            nu[0, p], q[0, p] = null[s][0][0], null[s][1][0]
    cells = removal_drops(model, x, torch.eye(UNITS), ids, operator, GRID)
    excess = (cells - null[1][0].unsqueeze(1)).transpose(1, 2)
    from cdea.region import cell_overlap

    scores = partition_scores(drops, nu, q, sizes, excess, roles, cell_overlap(evidence), omega)
    return scores, null


def test_a_mixed_s_is_flagged_by_mix_and_by_its_halves() -> None:
    model, x = mixing_toy()
    roles = torch.zeros(1, 4, UNITS, dtype=torch.bool)
    roles[0, SHARED, [K_ONLY, L_ONLY]] = True
    scores, null = _score(model, x, roles)
    assert float(scores["sv"]) > 0          # the set-level min calls the pair shared
    assert float(scores["sv_cell"]) < 0     # each cell alone is unique
    assert float(scores["mix"]) > 0
    halves = split_halves(roles[0, SHARED])
    assert sorted(int(h.nonzero()) for h in halves) == [K_ONLY, L_ONLY]
    ids = torch.tensor([[0, 1, 2]])
    drops = removal_drops(model, x, halves.float(), ids, "blur", GRID)[0, :, :2]
    assert half_verdict(drops > null[1][1][0, :2]) == 1


# Toy 5: co-location -------------------------------------------------------------

class TwoFeatureToy(CellToy):
    """Each cell carries a channel-0 and a channel-2 checkerboard; the head reads both per cell."""

    def __init__(self, weight, bias):
        super().__init__(torch.zeros(3, UNITS), bias)
        self.register_buffer("weight2", weight.float())
        self.conv = torch.nn.Conv2d(2 * UNITS, 2 * UNITS, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(2 * UNITS).view(2 * UNITS, 2 * UNITS, 1, 1))

    def features2(self, x):
        out = []
        for channel in (0, 2):
            demod = 2.0 * (x[:, channel] - 0.5) * self.checker
            out.append(pool_sum(demod, *GRID) / pool_sum(torch.ones_like(demod), *GRID))
        return torch.cat(out, dim=1)

    def forward(self, x):
        f = self.features2(x)
        location = self.location.repeat(2, 1, 1)
        channels = self.conv(f.view(-1, 2 * UNITS, 1, 1) * location.unsqueeze(0))
        return channels.mean(dim=(-1, -2)) * UNITS @ self.weight2.t() + self.bias


COLOCATED = 5


def colocation_toy():
    both, k_cell, l_cell = COLOCATED, 0, 3
    weight = torch.zeros(3, 2 * UNITS)
    weight[0, both] = 1.0               # k reads the channel-0 feature of the cell
    weight[1, UNITS + both] = 1.0       # l reads the channel-2 feature of the same cell
    weight[0, k_cell] = 0.5
    weight[1, UNITS + l_cell] = 0.5
    model = TwoFeatureToy(weight, torch.tensor([0.5, 0.0, -5.0])).eval()
    x = toy_image(torch.ones(UNITS))
    cell = 32 // GRID[0]
    amp = torch.ones(UNITS).view(*GRID).repeat_interleave(cell, 0).repeat_interleave(cell, 1)
    x[:, 2] = 0.5 + 0.5 * amp * model.checker
    return model, x


def test_colocated_features_are_shared_for_the_cell_methods() -> None:
    """Section 10: a k-only and an l-only feature in one cell. No partition of cells can split them."""
    model, x = colocation_toy()
    subtract, occlusion, _ = _labels(model, x, 1.0, steps=0)
    assert int(subtract[COLOCATED]) == SHARED and int(occlusion[COLOCATED]) == SHARED

    from evaluation.descriptive import colocation

    # The head's contributions per channel: the cell's two channels feed different classes.
    contrib = torch.zeros(1, 2, 2, UNITS)
    contrib[0, 0, 0, COLOCATED] = 1.0
    contrib[0, 1, 1, COLOCATED] = 1.0
    out = colocation(contrib)
    assert float(out["p"]) == 1.0 and float(out["q"]) == 0.0 and float(out["share"]) == 1.0


@pytest.mark.xfail(strict=True, reason="The straight-through drift of the additive toy: the cell moves from S to U_l.")
def test_colocated_features_are_shared_for_cdea() -> None:
    model, x = colocation_toy()
    _, _, cdea = _labels(model, x, 1.0)
    assert int(cdea.labels[0, COLOCATED]) == SHARED


# The size-matched null ----------------------------------------------------------

def test_size_masks_are_whole_nested_and_of_their_size() -> None:
    masks = size_masks(49, [0, 1, 5, 49], n=8)
    assert masks.shape == (4, 8, 49)
    assert masks.sum(dim=-1).tolist() == [[0.0] * 8, [1.0] * 8, [5.0] * 8, [49.0] * 8]
    assert bool(((masks[2] - masks[1]) >= 0).all())


def test_size_null_does_not_depend_on_the_batch_and_the_grid_interpolates() -> None:
    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0, 1, UNITS))])
    ids = _pair(model, x2).ids
    both = size_null(model, x2, ids, *GRID, 6)
    one = size_null(model, x2[1:], ids[1:], *GRID, 6)
    assert both.shape == (2, 2, 7) and float(both[..., 0].abs().max()) == 0.0
    assert torch.allclose(both[1:], one, atol=1e-6)
    sparse = size_null(model, x2, ids, *GRID, 6, sizes=[0, 2, 6])
    assert torch.allclose(sparse[..., [0, 2, 6]], both[..., [0, 2, 6]], atol=1e-6)
    assert torch.allclose(sparse[..., 1], 0.5 * (both[..., 0] + both[..., 2]), atol=1e-6)
    assert null_size_grid(30) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 30]
    assert interpolate_sizes(torch.tensor([0.0, 2.0, 8.0]), torch.tensor([0.0, 2.0, 4.0]), 4).tolist() == \
        [0.0, 1.0, 2.0, 5.0, 8.0]


def test_null_at_interpolates_and_its_slope_is_the_next_cell() -> None:
    table = (torch.arange(6.0) ** 2).view(1, 1, 6).repeat(1, 2, 1)
    size = torch.tensor([[0.0, 2.5, 3.0, 7.0]], requires_grad=True)
    value = null_at(table, size)
    assert value[0, :, 0].tolist() == [0.0, 6.5, 9.0, 25.0]
    value[..., 0].sum().backward()
    # At a whole size the slope is to the next size; past the top it is 0.
    assert size.grad.tolist() == [[1.0, 5.0, 7.0, 0.0]]


# The partition ------------------------------------------------------------------

def test_partition_is_exclusive_covers_omega_and_does_not_depend_on_the_batch() -> None:
    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0.2, 1, UNITS))])
    h2 = _pair(model, x2)
    cfg = PartitionConfig(steps=8, rho=1.0)
    whole = partition(model, x2, h2, *GRID, cfg)
    again = partition(model, x2, h2, *GRID, cfg)
    part = partition(model, x2[1:], HypothesisSet(ids=h2.ids[1:], mask=h2.mask[1:]), *GRID, cfg)
    assert torch.equal(whole.plan, again.plan)
    assert torch.allclose(whole.plan[1:], part.plan, atol=1e-5)
    assert torch.equal(whole.roles.sum(dim=1), whole.omega.long())
    assert torch.allclose(whole.plan.sum(dim=1), whole.omega.float(), atol=1e-5)


def test_ablation_switches_run() -> None:
    model, x = additive_toy()
    h = _pair(model, x)
    for cfg in (PartitionConfig(steps=3, independent=True), PartitionConfig(steps=3, init="uniform"),
                PartitionConfig(steps=3, init="mean"), PartitionConfig(steps=3, binary_masks=False),
                PartitionConfig(steps=3, boundary_shift=False), PartitionConfig(steps=3, null=False),
                PartitionConfig(steps=3, null_grid=True)):
        out = partition(model, x, h, *GRID, cfg)
        assert bool(torch.isfinite(out.plan).all())
        assert bool((out.roles[:, :, ~out.omega[0]] == 0).all())
    assert partition(model, x, h, *GRID, PartitionConfig(steps=1, null=False)).null.abs().max() == 0


def test_v1_plan_is_read_as_roles() -> None:
    from cdea.decompose import DecomposeConfig, decompose, budget_roles

    model, x = additive_toy()
    h, evidence, omega = _setup(model, x, 0.5)
    out = decompose(model, x, h, *GRID, 2.0 / UNITS, DecomposeConfig(steps=20))
    roles = budget_roles(out, omega, 2.0 / UNITS)
    assert roles[:, :3].sum(dim=-1).tolist() == [[2, 2, 2]]
    assert bool((roles.sum(dim=1) <= 1).all())
    assert torch.equal(roles[:, 3], omega & ~roles[:, :3].any(dim=1))


# Rule rows and floors -----------------------------------------------------------

def test_rule_partition_pools_pixels_before_it_clamps() -> None:
    """Cell 0 of k pools to 8 - 20 < 0, so k has no evidence there and the cell is U_l.
    Clamping each pixel first would give k a third of its mass in cell 0 and call the cell S."""
    pixel_k = torch.zeros(1, 32, 32)
    pixel_k[0, 0, :8] = 1.0
    pixel_k[0, 1, 0] = -20.0
    pixel_k[0, 0:4, 8:12] = 1.0
    pixel_l = torch.zeros(1, 32, 32)
    pixel_l[0, 2, :4] = 1.0
    pixel_l[0, 2, 16:20] = 1.0
    omega = torch.zeros(1, UNITS, dtype=torch.bool)
    omega[0, :3] = True
    assert rule_partition(pixel_k, pixel_l, omega, *GRID)[0, :3].tolist() == [U_L, U_K, U_L]


def test_cell_partition_takes_the_best_positive_score_or_empty() -> None:
    g = torch.tensor([[[1.0, 0.0, 1.0, -0.5], [0.0, 1.0, 1.0, 0.5]]])
    omega = torch.tensor([[True, True, True, True]])
    assert cell_partition(g, omega).tolist() == [[U_K, U_L, SHARED, NEITHER]]


def test_random_partition_does_not_depend_on_the_batch() -> None:
    omega = torch.ones(3, UNITS, dtype=torch.bool)
    whole = random_partition(omega, [0, 1, 2], torch.Generator().manual_seed(7))
    tail = random_partition(omega[1:], [1, 2], torch.Generator().manual_seed(7))
    assert torch.equal(whole[1:], tail)
    assert set(whole.unique().tolist()) <= {0, 1, 2, 3}


# Scores -------------------------------------------------------------------------

def test_road_drops_match_one_removal_per_mask() -> None:
    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0.2, 1, UNITS))])
    ids = torch.tensor([[0, 1], [1, 0]])
    cells = torch.zeros(2, 3, UNITS)
    cells[0, 0, [0, 1]] = 1
    cells[1, 1, 5] = 1
    cells[:, 2, 12] = 1
    from core.grid import upsample_units

    batched = removal_drops(model, x2, cells, ids, "road", GRID, seed=3, offset=5, batch=4)
    for m in range(3):
        pixel = upsample_units(cells[:, m], *GRID, 32, 32)[:, 0]
        one = model(x2).gather(1, ids) - model(_remove(x2, pixel, None, 0.01, 3, 5)).gather(1, ids)
        assert torch.allclose(batched[:, m], one, atol=1e-5)


def test_scorer_null_is_measured_per_image_and_skips_empty_masks() -> None:
    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0.2, 1, UNITS))])
    ids = torch.tensor([[0, 1, 2], [1, 0, 2]])
    both = scored_size_null(model, x2, ids, GRID, {1, 3}, "blur")
    per = scored_size_null(model, x2, ids, GRID, [{1}, {1, 3}], "blur")
    assert torch.allclose(per[1][0], both[1][0])
    assert torch.isnan(per[3][0][0]).all() and torch.allclose(per[3][1][1], both[3][1][1])
    empty = removal_drops(model, x2, torch.zeros(2, 1, UNITS), ids, "road", GRID)
    assert float(empty.abs().max()) == 0.0


def test_partition_scores_follow_the_definitions() -> None:
    drops = torch.tensor([[[3.0, 1.0, 0.5], [0.5, 2.0, 0.0], [2.0, 1.5, 1.0], [0.1, 0.1, 0.0]]])
    null = torch.full((1, 4, 3), 0.5)
    q95 = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
    roles = torch.zeros(1, 4, 6, dtype=torch.bool)
    roles[0, 0, 0] = roles[0, 1, 1] = roles[0, 2, 2] = roles[0, 2, 3] = roles[0, 3, 4] = True
    sizes = roles.sum(dim=-1)
    excess = torch.zeros(1, 3, 6)
    excess[0, :, 2] = torch.tensor([1.0, -0.2, 0.0])
    excess[0, :, 3] = torch.tensor([-0.1, 0.8, 0.0])
    overlap = torch.tensor([[0.0, 0.1, 0.2, 0.2, 0.1, 0.4]])
    omega = torch.tensor([[True, True, True, True, True, False]])
    s = partition_scores(drops, null, q95, sizes, excess, roles, overlap, omega)
    assert float(s["pur_k"]) == pytest.approx(2.5 - 0.5)
    assert float(s["pur_l"]) == pytest.approx(1.5 - 0.0)
    assert float(s["sv"]) == pytest.approx(1.0) and float(s["bal"]) == pytest.approx(0.5)
    assert float(s["sv_cell"]) == pytest.approx(-0.2 - 0.1)
    assert float(s["mix"]) == pytest.approx(1.0 + 0.3)
    assert float(s["gen_s"]) == pytest.approx(0.5) and float(s["psv"]) == pytest.approx(0.5)
    # U_k counts for k only; U_l for l only; S for both; empty for neither.
    assert [float(s[f"confirmed_{r}"]) for r in ("uk", "ul", "s", "none")] == [1.0, 1.0, 1.0, 1.0]
    assert float(s["vr"]) == pytest.approx(1.0)
    assert float(s["res_shared"]) == pytest.approx(0.4 / 0.6)
    roles[0, 0] = False
    s = partition_scores(drops, null, q95, roles.sum(dim=-1), excess, roles, overlap, omega)
    assert torch.isnan(s["pur_k"]).all() and torch.isnan(s["confirmed_uk"]).all()


def test_the_confirmation_rule_is_verify_roles_rule() -> None:
    from itertools import product

    for bits in product([False, True], repeat=8):
        count = torch.tensor(bits).view(1, 4, 2)
        out = role_confirmed(count)[0].tolist()
        k, l = count[0, :, 0].tolist(), count[0, :, 1].tolist()
        assert out == [k[0] and not l[0], l[1] and not k[1], k[2] and l[2], not k[3] and not l[3]]


def test_resolution_counts_a_cell_once_when_roles_overlap() -> None:
    roles = torch.zeros(1, 4, 3, dtype=torch.bool)
    roles[0, 0, 0] = roles[0, 2, 0] = True
    roles[0, 1, 1] = True
    overlap = torch.tensor([[0.5, 0.25, 0.25]])
    omega = torch.ones(1, 3, dtype=torch.bool)
    s = resolution_shares(roles, torch.tensor([[True, False, True, False]]), overlap, omega)
    assert float(s["res_shared"]) == 0.5 and float(s["res_unique"]) == 0.0
    assert float(s["vr"]) == 0.5 and float(s["res_unconfirmed"]) == 0.5


def test_half_verdicts() -> None:
    assert half_verdict(torch.tensor([[True, False], [False, True]])) == 1
    assert half_verdict(torch.zeros(2, 2, dtype=torch.bool)) == 2
    assert half_verdict(torch.ones(2, 2, dtype=torch.bool)) == 3
    assert half_verdict(torch.tensor([[True, True], [False, False]])) == 4


def test_planted_role_accuracy_and_coverage() -> None:
    roles = torch.zeros(1, 4, 6, dtype=torch.bool)
    roles[0, 0, 0] = True                  # A_k labelled U_k
    roles[0, 2, 1] = True                  # A_l labelled S
    omega = torch.tensor([[True, True, False, True, True, True]])
    patches = torch.zeros(1, 3, 6, dtype=torch.bool)
    patches[0, 0, 0] = patches[0, 1, 1] = patches[0, 2, 2] = True    # C lies outside Omega
    s = planted_role_scores(roles, patches, omega)
    assert float(s["omega_coverage"]) == pytest.approx(2 / 3)
    assert float(s["ra_all"]) == pytest.approx(1 / 3)
    assert float(s["ra_in"]) == pytest.approx(0.5)
    assert float(s["conf_al_s"]) == 1.0 and float(s["conf_c_outside"]) == 1.0


def test_paper_run_parses_the_v2_game() -> None:
    from scripts.paper_run import parse, spec_from_args

    spec = spec_from_args(parse(["--game", "partition", "--dataset", "cifar10", "--methods", "all",
                                 "--ablations", "DA10_fixed_budgets", "--candidates", "all"]))
    assert "chefer_minsplit" not in spec.methods and "cdea" in spec.methods
    assert list(spec.ablations) == ["DA10_fixed_budgets"] and len(spec.candidates) == 4
    vit = spec_from_args(parse(["--game", "partition", "--dataset", "cifar10", "--methods", "all",
                                "--backbone", "vit_b_16"]))
    assert "chefer_minsplit" in vit.methods


def test_readout_reports_a_small_record_set(capsys) -> None:
    import numpy as np
    import pandas as pd

    from scripts.partition_readout import boot_ci, paired, report

    rng = np.random.default_rng(0)
    rows = []
    for image in range(12):
        for method, shift in (("cdea", 0.1), ("subtract", 0.0), ("occlusion_partition", 0.05)):
            row = {"dataset": "toy", "backbone": "resnet50", "operator": "road", "image_index": image,
                   "method": method, "overlap": image / 12, "overlap_omega": 0.1, "omega_size": 10,
                   "z_k_full": 3.0, "z_l_full": 2.0, "z_m_full": 1.5, "correct": 1,
                   "vr": float(np.clip(rng.uniform(0, 1) + shift, 0, 1)), "split_verdict": "mixture"}
            for name in ("res_shared", "res_unique", "res_neither", "pur_k", "pur_l", "sv", "pur_cell_k",
                         "pur_cell_l", "sv_cell", "mix", "gen_s", "confirmed_uk", "confirmed_ul", "confirmed_s",
                         "confirmed_none"):
                row[name] = float(rng.uniform())
            for name in ("size_uk", "size_ul", "size_s", "size_none"):
                row[name] = float(rng.integers(0, 4))
            if method == "cdea":
                for stage in ("soft", "hard", "init"):
                    for player in ("uk", "ul", "s"):
                        row[f"payoff_{stage}_{player}"] = float(rng.normal())
                row.update(same_as_init=0.0, trapped=0.1, empty_share=0.2)
            rows.append(row)
    selection = report(pd.DataFrame(rows), "road")
    printed = capsys.readouterr().out
    assert "CDEA - subtract" in printed and "D2 soft-hard" in printed
    assert list(selection) == ["cdea"]
    mean, lo, hi, n = boot_ci([1.0, 2.0, 3.0])
    assert mean == 2.0 and lo <= mean <= hi and n == 3
    frame = pd.DataFrame(rows)
    assert paired(frame, "cdea", "cdea", "vr")[0] == 0.0
