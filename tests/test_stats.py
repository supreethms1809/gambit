"""Holm against hand-worked cases, and the signed-rank count.

For differences [1, 2, 3, 4, 5, −0.5] the absolute ranks are 2, 3, 4, 5, 6, 1.
The positive rank sum is 20. Of the 64 sign patterns, four are at least as
extreme, so the two-sided p is 4/64.
"""

from __future__ import annotations

import numpy as np

from analysis.stats import bootstrap_mean_ci, family_summary, holm_adjust, wilcoxon_signed_rank


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
