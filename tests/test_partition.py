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


def test_additive_toy_cdea_keeps_the_shared_cell() -> None:
    model, x = additive_toy()
    _, _, cdea = _labels(model, x, 1.0)
    labels = cdea.labels[0]
    assert int(labels[K_ONLY]) == U_K and int(labels[L_ONLY]) == U_L and int(labels[SHARED_CELL]) == SHARED


def test_the_search_alone_drifts_below_step_zero_and_the_polish_recovers() -> None:
    """Section 4.4. A pure U_k has d~_l = -nu_l < 0, where |d~_l| has slope -1, so to first order l-evidence
    looks like a gain for U_k. At lr 0.1 the straight-through search moves the shared cell to U_k and ends
    below step 0 (2.0 against 2.625). The polish starts from the better of the two and never goes below it."""
    model, x = additive_toy()
    h, evidence, _ = _setup(model, x, 1.0)
    alone = partition(model, x, h, *GRID, PartitionConfig(rho=1.0, polish=False, payoff="set"), evidence=evidence)
    assert int(alone.labels[0, SHARED_CELL]) == U_K
    from cdea.decompose import role_welfare

    full = model(x).detach()
    def welfare(roles):
        return float(role_welfare(model, x, h.ids, full, roles, alone.null, True, *GRID))
    assert welfare(alone.search_roles) < welfare(alone.initial_roles) - 0.5
    polished = partition(model, x, h, *GRID, PartitionConfig(rho=1.0, payoff="set"), evidence=evidence)
    assert float(polished.welfare) >= welfare(polished.initial_roles) - 1e-6
    assert float(polished.welfare) == pytest.approx(welfare(polished.roles), abs=1e-5)


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


def test_a_pure_suppressor_is_not_unique_evidence() -> None:
    """D3's intent, under every payoff: a cell whose deletion only raises z_l earns no gain, so it stays out of U_k.

    Grad-CAM gives such a cell no evidence, so it would lie outside Omega; the test hands it k evidence to put it
    in Omega and in U_k at step 0, so that deletion alone must move it out.
    """
    weight = _cells((0, K_ONLY, 1.0), (1, L_ONLY, 1.0), (1, ART, -1.0), (0, SHARED_CELL, 1.0), (1, SHARED_CELL, 1.0),
                    *[(0, c, 0.1) for c in K_BACKGROUND])
    model = CellToy(weight, torch.tensor([0.5, 0.5, -5.0])).eval()
    x = toy_image(torch.ones(UNITS))
    h, evidence, _ = _setup(model, x, 1.0)
    evidence = evidence.clone()
    evidence[0, 0, ART] = 0.3
    evidence = normalise(evidence)
    for payoff in ("cell_positive", "cell_abs", "set"):
        out = partition(model, x, h, *GRID, PartitionConfig(rho=1.0, payoff=payoff), evidence=evidence)
        assert bool(out.omega[0, ART]) and bool(out.initial_roles[0, U_K, ART])
        assert int(out.labels[0, ART]) == NEITHER, payoff


def test_a_unique_cell_that_also_suppresses_the_rival() -> None:
    """Toy 2's SUPPRESSOR raises z_k and lowers z_l, as a planted A_k does. Under the proposed one-sided penalty it
    is U_k, as the planted ground truth labels A_k. The two-sided penalty (DA15) keeps D3's penalty and leaves it
    empty. Margin attribution and the rule call it U_k too."""
    model, x = suppression_toy()
    subtract, _, cdea = _labels(model, x, 1.0)
    assert cdea.payoff == "cell_positive"
    assert int(subtract[SUPPRESSOR]) == U_K
    assert int(cdea.labels[0, SUPPRESSOR]) == U_K
    assert int(cdea.labels[0, WEAK_K]) == U_K and int(cdea.labels[0, ONLY_L]) == U_L
    h, evidence, _ = _setup(model, x, 1.0)
    two_sided = partition(model, x, h, *GRID, PartitionConfig(rho=1.0, payoff="cell_abs"), evidence=evidence)
    assert int(two_sided.labels[0, SUPPRESSOR]) == NEITHER

    from baselines.decomposition import margin_partition
    from baselines.margin import margin_gradcam

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


def test_with_no_shared_cell_cdea_leaves_s_empty() -> None:
    """Section 9 toy 4. The search alone parks a weak k cell in S (S and the empty role both start at log eps,
    and Adam moves both alike); the polish takes it back out."""
    model, x = mixing_toy()
    _, _, cdea = _labels(model, x, 1.0)
    assert not cdea.roles[0, SHARED].any()


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
    q05 = nan.clone()
    for p in range(4):
        s = int(sizes[0, p])
        if s:
            q05[0, p] = null[s][2][0]
    cells = removal_drops(model, x, torch.eye(UNITS), ids, operator, GRID)
    excess = (cells - null[1][0].unsqueeze(1)).transpose(1, 2)
    from cdea.region import cell_overlap

    scores = partition_scores(drops, nu, q, sizes, excess, roles, cell_overlap(evidence), omega, q05)
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
    # Measured sizes may pass the largest one; they must reach it.
    beyond = size_null(model, x2, ids, *GRID, 5, sizes=[0, 2, 6])
    assert torch.allclose(beyond[..., :6], sparse[..., :6], atol=1e-6)
    with pytest.raises(ValueError):
        size_null(model, x2, ids, *GRID, 6, sizes=[0, 2, 4])
    # The grid below an image's largest size does not depend on the batch's.
    assert null_size_grid(30, 196) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32]
    assert null_size_grid(10, 196) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]
    assert null_size_grid(6, 196) == [0, 1, 2, 3, 4, 5, 6]
    assert null_size_grid(195, 196)[-1] == 196
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
    assert torch.equal(whole.plan, again.plan) and torch.equal(whole.roles, again.roles)
    assert torch.allclose(whole.plan[1:], part.plan, atol=1e-5)
    assert torch.equal(whole.roles[1:], part.roles)
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
    q05 = torch.full((1, 4, 3), 0.2)
    roles = torch.zeros(1, 4, 6, dtype=torch.bool)
    roles[0, 0, 0] = roles[0, 1, 1] = roles[0, 2, 2] = roles[0, 2, 3] = roles[0, 3, 4] = True
    sizes = roles.sum(dim=-1)
    excess = torch.zeros(1, 3, 6)
    excess[0, :, 2] = torch.tensor([1.0, -0.2, 0.0])
    excess[0, :, 3] = torch.tensor([-0.1, 0.8, 0.0])
    overlap = torch.tensor([[0.0, 0.1, 0.2, 0.2, 0.1, 0.4]])
    omega = torch.tensor([[True, True, True, True, True, False]])
    s = partition_scores(drops, null, q95, sizes, excess, roles, overlap, omega, q05)
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
    # The empty role's drops (0.1) are below the 5th percentile (0.2): absent for both, so confirmed.
    # Above it, the lenient rule still confirms the role and the absence rule does not.
    weak = q05.clone()
    weak[0, 3] = 0.05
    s = partition_scores(drops, null, q95, sizes, excess, roles, overlap, omega, weak)
    assert float(s["confirmed_none"]) == 0.0 and float(s["confirmed_none_lenient"]) == 1.0
    assert float(s["vr"]) == pytest.approx(1.0 - 0.1 / 0.6) and float(s["vr_lenient"]) == pytest.approx(1.0)
    roles[0, 0] = False
    s = partition_scores(drops, null, q95, roles.sum(dim=-1), excess, roles, overlap, omega, q05)
    assert torch.isnan(s["pur_k"]).all() and torch.isnan(s["confirmed_uk"]).all()


def test_the_confirmation_rule_is_verify_roles_rule() -> None:
    from itertools import product

    for bits in product([False, True], repeat=8):
        count = torch.tensor(bits).view(1, 4, 2)
        out = role_confirmed(count)[0].tolist()
        k, l = count[0, :, 0].tolist(), count[0, :, 1].tolist()
        assert out == [k[0] and not l[0], l[1] and not k[1], k[2] and l[2], not k[3] and not l[3]]
        absent = ~count
        strict = role_confirmed(count, absent)[0].tolist()
        assert strict[:3] == out[:3] and strict[3] == (not k[3] and not l[3])
    # The absence test reads only the empty role's own absences.
    count = torch.zeros(1, 4, 2, dtype=torch.bool)
    absent = torch.zeros(1, 4, 2, dtype=torch.bool)
    absent[0, 3, 0] = True
    assert role_confirmed(count, absent)[0].tolist() == [False, False, False, False]
    absent[0, 3, 1] = True
    assert role_confirmed(count, absent)[0].tolist() == [False, False, False, True]


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


# DA13: the polish under ROAD ----------------------------------------------------

def test_the_edit_null_with_blur_is_the_size_null() -> None:
    from cdea.decompose import blur_edit, edit_null

    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0, 1, UNITS))])
    ids = _pair(model, x2).ids
    reference = model(x2).detach()
    with torch.no_grad():
        table = edit_null(model, ids, reference, blur_edit(x2, *GRID), 2, UNITS, 6, 8, 0)
        sparse = edit_null(model, ids, reference, blur_edit(x2, *GRID), 2, UNITS, 6, 8, 0, sizes=[0, 2, 6])
    assert torch.allclose(table, size_null(model, x2, ids, *GRID, 6), atol=1e-6)
    assert torch.allclose(sparse, size_null(model, x2, ids, *GRID, 6, sizes=[0, 2, 6]), atol=1e-6)


def test_the_road_polish_never_ends_below_its_own_step_zero_and_ignores_the_batch() -> None:
    from cdea.decompose import edit_null, role_welfare
    from evaluation.run_methods import road_polish_edit

    model, x = additive_toy()
    x2 = torch.cat([x, toy_image(torch.linspace(0.2, 1, UNITS))])
    h2 = _pair(model, x2)
    cfg = PartitionConfig(steps=4, rho=1.0, payoff="set")
    edit = road_polish_edit(x2, 0, 0, GRID)
    whole = partition(model, x2, h2, *GRID, cfg, polish_edit=edit)
    part = partition(model, x2[1:], HypothesisSet(ids=h2.ids[1:], mask=h2.mask[1:]), *GRID, cfg,
                     polish_edit=road_polish_edit(x2[1:], 1, 0, GRID))
    assert torch.equal(whole.roles[1:], part.roles)
    reference = model(x2).detach()
    with torch.no_grad():
        top = int(whole.omega.sum(dim=-1).max())
        table = edit_null(model, h2.ids, reference, edit, 2, UNITS, top, 8, 0)
        start = role_welfare(model, x2, h2.ids, reference, whole.initial_roles, table, True, *GRID, edit)
        final = role_welfare(model, x2, h2.ids, reference, whole.roles, table, True, *GRID, edit)
    assert bool((whole.welfare >= start - 1e-5).all())
    assert torch.allclose(whole.welfare, final, atol=1e-5)
    # The search is blur in both; only the polish changed.
    blur = partition(model, x2, h2, *GRID, cfg)
    assert torch.equal(blur.search_roles, whole.search_roles)


def test_the_polish_noise_never_is_the_scorers() -> None:
    from evaluation.run_methods import polish_noise_seed

    assert all(polish_noise_seed(seed) != 1000 * other for seed in range(20) for other in range(40))


def test_paper_run_takes_the_road_polish_ablation() -> None:
    from scripts.paper_run import parse, spec_from_args

    spec = spec_from_args(parse(["--game", "partition", "--dataset", "cifar10", "--methods", "none",
                                 "--ablations", "DA13_road_polish"]))
    assert list(spec.ablations) == ["DA13_road_polish"]


# The welfare's maximum (2026-10-22, formulation section 13) ---------------------

AK, AL, C_PATCH = 0, 3, 12


def _planted_toy(suppression: float):
    weight = _cells((0, AK, 1.0), (1, AK, -suppression), (1, AL, 1.0), (0, C_PATCH, 1.0), (1, C_PATCH, 1.0))
    return CellToy(weight, torch.tensor([0.5, 0.0, -5.0])).eval(), toy_image(torch.ones(UNITS))


def _roles(uk, ul, s, cells):
    roles = torch.zeros(1, 4, UNITS, dtype=torch.bool)
    for role, members in ((U_K, uk), (U_L, ul), (SHARED, s)):
        roles[0, role, members] = True
    for cell in cells:
        if not roles[0, :3, cell].any():
            roles[0, NEITHER, cell] = True
    return roles


@pytest.mark.parametrize("suppression", [0.6, 0.9, 1.5])
def test_the_cell_penalty_keeps_the_planted_roles_when_a_unique_patch_suppresses_the_rival(suppression) -> None:
    """Formulation section 13. A_k raises z_l when deleted. The set-level term pays C to join U_k; the two-sided
    cell penalty (DA15) drops A_k from s = 0.9; the one-sided penalty keeps the true partition at every s."""
    from cdea.decompose import blur_edit, role_welfare, single_cell_excess

    model, x = _planted_toy(suppression)
    h = _pair(model, x)
    full = model(x).detach()
    cells = [AK, AL, C_PATCH]
    omega = torch.zeros(1, UNITS, dtype=torch.bool)
    omega[0, cells] = True
    table = size_null(model, x, h.ids, *GRID, 3)
    with torch.no_grad():
        excess = single_cell_excess(model, h.ids, full, blur_edit(x, *GRID), omega, table, True)

    def welfare(roles, payoff):
        cell = None if payoff == "set" else excess
        return float(role_welfare(model, x, h.ids, full, roles, table, True, *GRID, payoff=payoff, cell_excess=cell))

    true = _roles([AK], [AL], [C_PATCH], cells)
    cancel = _roles([AK, C_PATCH], [AL], [], cells)
    dropped = _roles([], [AL], [C_PATCH], cells)
    assert welfare(cancel, "set") > welfare(true, "set")
    assert welfare(true, "cell_positive") > max(welfare(cancel, "cell_positive"), welfare(dropped, "cell_positive"))
    if suppression >= 0.9:
        assert welfare(dropped, "cell_abs") > welfare(true, "cell_abs")
    _, _, cdea = _labels(model, x, 1.0)
    assert [int(cdea.labels[0, c]) for c in cells] == [U_K, U_L, SHARED]


def test_the_cross_penalty_is_per_cell_and_its_gradient_is_exact() -> None:
    from cdea.payoffs import cross_penalty, partition_payoffs

    excess = torch.tensor([[[0.5, -0.2, 0.0], [-0.4, 0.3, 0.8]]])        # g_k, g_l of three cells
    masks = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]], requires_grad=True)
    pos = cross_penalty(masks, excess, "cell_positive")
    assert pos[0].tolist() == pytest.approx([0.0 + 0.3, 0.0 + 0.0, 0.0])
    assert cross_penalty(masks, excess, "cell_abs")[0].tolist() == pytest.approx([0.4 + 0.3, 0.2 + 0.0, 0.0])
    pos[0, 0].backward()
    assert masks.grad[0, 0].tolist() == pytest.approx([0.0, 0.3, 0.8])
    drops = torch.tensor([[[2.0, 0.5], [0.0, 0.0], [1.0, 1.5]]])
    null = torch.full((1, 3, 2), 0.25)
    null[0, 1] = 0.0                                                      # an empty role's null is nu(0) = 0
    u = partition_payoffs(drops, null, "cell_positive", pos.detach())
    assert u[0, 0] == pytest.approx(1.75 - 0.3)
    assert float(u[0, 1]) == 0.0                                          # an empty unique role pays exactly 0
    assert u[0, 2] == pytest.approx(0.75, abs=1e-3)


def test_the_polish_maximises_the_cell_payoff_it_reports() -> None:
    from cdea.decompose import role_welfare

    model, x = _planted_toy(0.9)
    h, evidence, _ = _setup(model, x, 1.0)
    out = partition(model, x, h, *GRID, PartitionConfig(rho=1.0), evidence=evidence)
    full = model(x).detach()

    def welfare(roles):
        return float(role_welfare(model, x, h.ids, full, roles, out.null, True, *GRID, payoff=out.payoff,
                                  cell_excess=out.cell_excess))

    assert float(out.welfare) == pytest.approx(welfare(out.roles), abs=1e-5)
    assert float(out.welfare) >= welfare(out.initial_roles) - 1e-6


def test_the_runner_diagnostics_score_each_payoff_as_it_was_optimised() -> None:
    from evaluation.run_partition import _stage_payoffs

    model, x = _planted_toy(0.9)
    h, evidence, _ = _setup(model, x, 1.0)
    for payoff in ("cell_positive", "cell_abs", "set"):
        out = partition(model, x, h, *GRID, PartitionConfig(rho=1.0, payoff=payoff), evidence=evidence)
        stage = _stage_payoffs(model, x, h.ids, out, GRID)
        hard = sum(float(stage[f"payoff_hard_{p}"]) for p in ("uk", "ul", "s"))
        assert hard == pytest.approx(float(out.welfare), abs=1e-4), payoff
        assert float(stage["polish_welfare"]) == pytest.approx(float(out.welfare), abs=1e-6)


# RA in Omega on planted pairs (2026-10-22 audit) ---------------------------------

@pytest.mark.parametrize("share", [0.3, 0.55, 0.9])
def test_rho_half_keeps_one_cue_per_class_and_ra_in_omega_ignores_the_rest(share) -> None:
    """Each class's evidence sits on its unique patch (``share``) and on C. At rho = 0.5 only the stronger cue
    of each class enters Omega, so one or two of the three patches are outside it. RA in Omega leaves them
    out and reads 1 for the rule; RA over all patch cells counts them wrong."""
    units = 49
    ak, al, c = 10, 40, 24
    ek = torch.zeros(1, units)
    ek[0, ak], ek[0, c] = share, 1 - share
    el = torch.zeros(1, units)
    el[0, al], el[0, c] = share, 1 - share
    evidence = normalise(torch.stack([ek, el], dim=1))
    omega = evidence_region(evidence, 0.5)
    patches = torch.zeros(1, 3, units, dtype=torch.bool)
    patches[0, 0, ak] = patches[0, 1, al] = patches[0, 2, c] = True
    scores = planted_role_scores(roles_from_labels(rule_labels(rule_parts(evidence), omega)), patches, omega)
    assert 1 <= int(omega[0, [ak, al, c]].sum()) <= 2
    assert float(scores["ra_in"]) == 1.0
    assert float(scores["ra_all"]) < 1.0
    assert float(scores["ra_all"]) == pytest.approx(float(scores["omega_coverage"]))


def test_the_planted_audit_reads_records(tmp_path, capsys) -> None:
    import pandas as pd

    from scripts.planted_metric_audit import main

    rows = []
    for image in range(4):
        for method in ("subtract", "cdea"):
            row = {"method": method, "operator": "road", "fast": 0, "image_index": image,
                   "role_ak": 1, "role_al": 1, "role_c": int(image % 2 == 0), "ra_in": 1.0, "ra_all": 2 / 3,
                   "omega_coverage": 2 / 3, "omega_size": 10}
            for t, outside in (("ak", 0.0), ("al", 0.0), ("c", 1.0)):
                row[f"conf_{t}_outside"] = outside
                row[f"ra_all_{t}"] = 1.0 - outside
            rows.append(row)
    out = tmp_path / "val" / "partition" / "planted_pair" / "resnet50" / "seed0"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "records.csv.gz", index=False)
    main(["--root", str(tmp_path)])
    printed = capsys.readouterr().out
    assert "subtract" in printed and "RAin|c" in printed


# The precomputed_maps row (2026-10-22) ---------------------------------------------

def test_precomputed_maps_are_read_by_their_largest_mask() -> None:
    from baselines.decomposition import precomputed_partition

    k = torch.tensor([[0.6, 0.0, 0.1, 0.0, 0.2, 0.0]])
    l = torch.tensor([[0.1, 0.5, 0.1, 0.0, 0.0, 0.3]])
    s = torch.tensor([[0.0, 0.2, 0.4, 0.0, 0.0, 0.0]])
    others = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.5, 0.0]])
    omega = torch.tensor([[True, True, True, True, True, False]])
    labels = precomputed_partition(k, l, s, others, omega)
    # Cell 3 has no mass, cell 4 belongs to another hypothesis, cell 5 lies outside Omega.
    assert labels.tolist() == [[U_K, U_L, SHARED, NEITHER, NEITHER, OUTSIDE]]


def test_precomputed_maps_are_picked_by_image_index_and_checked_against_the_pair(tmp_path) -> None:
    import numpy as np

    from baselines.precomputed import load_partition_maps

    path = tmp_path / "maps.npz"
    np.savez(path, index=np.array([7, 3]), class_k=np.array([1, 4]), class_l=np.array([2, 5]),
             k=np.eye(2, 4, dtype=np.float32), l=np.zeros((2, 4), np.float32), shared=np.zeros((2, 4), np.float32),
             others=np.zeros((2, 4), np.float32))
    maps = load_partition_maps(str(path), torch.tensor([[4, 5], [1, 2]]), [3, 7])
    assert maps["k"].tolist() == [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        load_partition_maps(str(path), torch.tensor([[4, 5], [2, 1]]), [3, 7])
    with pytest.raises(KeyError):
        load_partition_maps(str(path), torch.tensor([[1, 2]]), [9])
    assert load_partition_maps(str(tmp_path), torch.tensor([[1, 2]]), [7])["k"].shape == (1, 4)


def test_paper_run_needs_a_file_for_the_precomputed_row() -> None:
    from scripts.paper_run import parse, spec_from_args

    with pytest.raises(SystemExit):
        spec_from_args(parse(["--game", "partition", "--dataset", "cifar10", "--methods", "precomputed_maps"]))
    spec = spec_from_args(parse(["--game", "partition", "--dataset", "cifar10", "--methods", "precomputed_maps,cdea",
                                 "--precomputed-dir", "maps.npz"]))
    assert spec.precomputed_dir == "maps.npz" and "precomputed_maps" in spec.methods
    every = spec_from_args(parse(["--game", "partition", "--dataset", "cifar10", "--methods", "all"]))
    assert "precomputed_maps" not in every.methods
