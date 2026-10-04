"""Dataset-level confirmatory statistics.

The unit is a dataset. Wilcoxon and Holm are the family test. The bootstrap
interval is over datasets. Per-image tests stay in ``evaluation.metrics.paired``
and are descriptive.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        stop = start
        while stop + 1 < values.shape[0] and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = 0.5 * ((start + 1) + (stop + 1))
        ranks[order[start : stop + 1]] = average
        start = stop + 1
    return ranks


def _signed_rank_p(ranks: np.ndarray, w_plus: float) -> float:
    """Two-sided exact p for n <= 16, otherwise a normal approximation."""
    n = int(ranks.shape[0])
    total = float(ranks.sum())
    if n <= 16:
        scaled = np.rint(ranks * 2.0).astype(np.int64)
        target = int(round(w_plus * 2.0))
        total_scaled = int(scaled.sum())
        tail = min(target, total_scaled - target)
        if total_scaled - tail <= tail:
            return 1.0
        count = np.zeros(total_scaled + 1, dtype=np.int64)
        count[0] = 1
        for rank in scaled.tolist():
            for mass in range(total_scaled, rank - 1, -1):
                count[mass] += count[mass - rank]
        extreme = int(count[: tail + 1].sum() + count[total_scaled - tail :].sum())
        return extreme / float(2**n)
    variance = float(np.square(ranks).sum()) / 4.0
    if variance <= 0.0:
        return 1.0
    z = (w_plus - total / 2.0) / math.sqrt(variance)
    return math.erfc(abs(z) / math.sqrt(2.0))


def wilcoxon_signed_rank(differences: Sequence[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test. Exact zeros are dropped.

    The statistic is the sum of the ranks of the positive differences.
    """
    values = np.asarray(list(differences), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("differences must be a one-dimensional list")
    if not np.isfinite(values).all():
        raise ValueError("differences must be finite")
    values = values[values != 0.0]
    n = int(values.shape[0])
    if n == 0:
        return {"statistic": 0.0, "p": 1.0, "n": 0.0}
    ranks = _average_ranks(np.abs(values))
    statistic = float(ranks[values > 0.0].sum()) if np.any(values > 0.0) else 0.0
    return {"statistic": statistic, "p": _signed_rank_p(ranks, statistic), "n": float(n)}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm adjusted p-values, in the input order, capped at 1.

    Sorted p_(i) is multiplied by (m − i). The adjusted values are then the
    cumulative maximum, so a later hypothesis cannot pass after an earlier one
    fails.
    """
    raw = [float(p) for p in p_values]
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in raw):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = len(raw)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (raw[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = min(1.0, max(running, (m - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for the mean. Resamples are drawn with replacement."""
    sample = np.asarray(list(values), dtype=np.float64)
    if sample.ndim != 1 or sample.size < 1:
        raise ValueError("bootstrap needs a non-empty one-dimensional sample")
    if not np.isfinite(sample).all():
        raise ValueError("bootstrap values must be finite")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, sample.size, size=(int(n_boot), sample.size))
    means = sample[draws].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(low), float(high)


def family_summary(
    comparisons: dict[str, Sequence[float]],
    seed: int = 0,
    n_boot: int = 1000,
) -> list[dict[str, float | str]]:
    """Wins, mean difference, bootstrap interval, and Holm-adjusted Wilcoxon p.

    Each value is one number per dataset. A win is a strictly positive difference.
    """
    if not comparisons:
        raise ValueError("a family needs at least one comparison")
    rows: list[dict[str, float | str]] = []
    p_values: list[float] = []
    for name, differences in comparisons.items():
        sample = np.asarray(list(differences), dtype=np.float64)
        if sample.ndim != 1 or sample.size < 1 or not np.isfinite(sample).all():
            raise ValueError(f"{name} needs a non-empty finite list")
        test = wilcoxon_signed_rank(sample)
        low, high = bootstrap_mean_ci(sample, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "name": name,
                "n": float(sample.size),
                "wins": float(np.count_nonzero(sample > 0.0)),
                "mean": float(sample.mean()),
                "ci_low": low,
                "ci_high": high,
                "p": test["p"],
            }
        )
        p_values.append(test["p"])
    for row, adjusted in zip(rows, holm_adjust(p_values)):
        row["p_holm"] = adjusted
    return rows
