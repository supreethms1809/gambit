"""Holm against hand-worked cases, and the signed-rank count.

For differences [1, 2, 3, 4, 5, −0.5] the absolute ranks are 2, 3, 4, 5, 6, 1.
The positive rank sum is 20. Of the 64 sign patterns, four are at least as
extreme, so the two-sided p is 4/64.

The family T statistic is the mean over seeds of each seed's mean difference.
The sign-flip test is checked against full enumeration on a small case.
"""

from __future__ import annotations

import itertools

import numpy as np

import analysis.stats as stats
from analysis.stats import (
    bootstrap_mean_ci,
    family_summary,
    holm_adjust,
    sign_flip_test,
    wilcoxon_signed_rank,
)


def test_holm_matches_the_hand_worked_products() -> None:
    # Sorted 0.01, 0.03, 0.04. Multipliers 3, 2, 1 give 0.03, 0.06, 0.04.
    # The cumulative maximum carries 0.06 onto the last value.
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])

    # Sorted already. 4*0.001, 3*0.008, 2*0.039, 1*0.041.
    # Cumulative max replaces 0.041 with 0.078.
    four = holm_adjust([0.001, 0.008, 0.039, 0.041])
    assert np.allclose(four, [0.004, 0.024, 0.078, 0.078])

    # 2 * 0.03 = 0.06, so neither passes 0.05, including the raw 0.04.
    stopped = holm_adjust([0.04, 0.03])
    assert np.allclose(stopped, [0.06, 0.06])
    assert all(p > 0.05 for p in stopped)


def test_wilcoxon_matches_the_enumerated_tail() -> None:
    result = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, -0.5])
    assert result["n"] == 6.0
    assert result["statistic"] == 20.0
    assert result["p"] == 4.0 / 64.0
    with_zero = wilcoxon_signed_rank([1.0, 0.0, 2.0, 3.0, 4.0, 5.0, -0.5])
    assert with_zero["n"] == 6.0
    assert with_zero["statistic"] == result["statistic"]
    assert with_zero["p"] == result["p"]


def test_bootstrap_interval_and_family_row() -> None:
    low, high = bootstrap_mean_ci([4.0, 4.0, 4.0], n_boot=20, seed=1)
    assert low == 4.0 and high == 4.0
    again = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_boot=200, seed=3)
    assert again == bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_boot=200, seed=3)
    assert again[0] <= 2.5 <= again[1]

    rows = family_summary({"margin": [1.0, 2.0, 3.0, 4.0, 5.0, -0.5]}, n_boot=50, seed=0)
    assert rows[0]["wins"] == 5.0
    assert rows[0]["n"] == 6.0
    assert rows[0]["p"] == 4.0 / 64.0
    assert rows[0]["p_holm"] == rows[0]["p"]
    assert rows[0]["ci_low"] <= rows[0]["mean"] <= rows[0]["ci_high"]


def _shifted_seeds() -> dict[int, np.ndarray]:
    generator = np.random.default_rng(1)
    return {seed: generator.normal(1.0, 0.5, size=40 + 5 * seed) for seed in range(5)}


def _enumerated_p(groups: list[list[float]], pooled: bool) -> float:
    """Exact two-sided p over all sign patterns."""
    sizes = np.cumsum([len(group) for group in groups])[:-1]
    flat = np.concatenate([np.asarray(group, dtype=np.float64) for group in groups])

    def statistic(signs: np.ndarray) -> float:
        signed = flat * signs
        if pooled:
            return float(signed.mean())
        return float(np.mean([part.mean() for part in np.split(signed, sizes)]))

    observed = abs(statistic(np.ones(flat.size)))
    patterns = [np.asarray(signs) for signs in itertools.product([1.0, -1.0], repeat=flat.size)]
    return sum(abs(statistic(signs)) >= observed - 1e-12 for signs in patterns) / len(patterns)


def test_sign_flip_all_zero_differences_give_p_one() -> None:
    result = sign_flip_test({0: np.zeros(10), 1: np.zeros(7)}, n_perm=500)
    assert result["statistic"] == 0.0
    assert result["p"] == 1.0
    assert result["n"] == 17.0 and result["seeds"] == 2.0


def test_sign_flip_consistent_shift_gives_the_smallest_p() -> None:
    result = sign_flip_test(_shifted_seeds())
    assert result["statistic"] > 0.8
    assert result["p"] == 1.0 / 10_001.0


def test_sign_flip_weights_seeds_equally() -> None:
    # Pooling the 55 images would give (50 * 1 + 5 * 3) / 55 = 1.18.
    result = sign_flip_test({0: [1.0] * 50, 1: [3.0] * 5}, n_perm=100)
    assert result["statistic"] == 2.0
    repeated = sign_flip_test({0: [1.0] * 50, 1: [3.0] * 500}, n_perm=100)
    assert repeated["statistic"] == 2.0


def test_sign_flip_is_deterministic(monkeypatch) -> None:
    seeds = {seed: values * 0.15 for seed, values in _shifted_seeds().items()}
    first = sign_flip_test(seeds, n_perm=2000)
    assert 0.0 < first["p"] < 1.0
    assert sign_flip_test(seeds, n_perm=2000) == first
    assert sign_flip_test(dict(reversed(list(seeds.items()))), n_perm=2000) == first
    assert sign_flip_test([seeds[seed] for seed in range(5)], n_perm=2000) == first
    # The flips are drawn one double each, so the block size does not matter.
    monkeypatch.setattr(stats, "_BLOCK", 7)
    assert sign_flip_test(seeds, n_perm=2000) == first


def test_sign_flip_stays_within_seeds() -> None:
    # Flipping within seeds gives p = 234/512. Pooling the images gives 454/512,
    # and flipping whole seeds gives 1.
    groups = [[3.0, 4.0], [-2.0, -1.0, 1.0, 2.0, -2.0, -1.0, -2.0]]
    within = _enumerated_p(groups, pooled=False)
    assert within == 234.0 / 512.0
    assert _enumerated_p(groups, pooled=True) == 454.0 / 512.0
    result = sign_flip_test({0: groups[0], 1: groups[1]})
    assert result["statistic"] == (3.5 - 5.0 / 7.0) / 2.0
    assert abs(result["p"] - within) < 0.02
