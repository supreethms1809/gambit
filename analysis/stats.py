"""Confirmatory statistics.

Family C: the unit is a dataset. Wilcoxon and Holm are the family test. The
bootstrap interval is over datasets.

Families T and D-T: the unit is the image, with seeds pooled. The statistic is
the mean over seeds of each seed's mean paired difference. The test flips signs
within each seed. The interval resamples seeds, then images within each seed.

The per-image tests in ``evaluation.metrics.paired`` stay descriptive.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

PerSeed = Mapping[int, Sequence[float]] | Sequence[Sequence[float]]

# Resamples drawn per block, to bound memory on large seeds.
_BLOCK = 1000


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


def _seed_samples(differences: PerSeed) -> list[np.ndarray]:
    """Per-seed arrays of paired differences. A mapping is read in seed order."""
    if isinstance(differences, Mapping):
        groups = [differences[key] for key in sorted(differences)]
    else:
        groups = list(differences)
    if not groups:
        raise ValueError("differences need at least one seed")
    samples = []
    for group in groups:
        sample = np.asarray(group, dtype=np.float64)
        if sample.ndim != 1 or sample.size < 1:
            raise ValueError("each seed needs a non-empty one-dimensional list")
        if not np.isfinite(sample).all():
            raise ValueError("differences must be finite")
        samples.append(sample)
    return samples


def sign_flip_test(differences: PerSeed, n_perm: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Two-sided sign-flip permutation test, flipping within each seed.

    The statistic is the mean over seeds of each seed's mean difference, so
    every seed counts equally. Each permutation flips every image's sign at
    random and recomputes the statistic with the same seeds. The p-value is
    (1 + the permutations at least as extreme) / (1 + n_perm).
    """
    samples = _seed_samples(differences)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    n_perm = int(n_perm)
    observed = sum(float(sample.mean()) for sample in samples) / len(samples)
    generator = np.random.default_rng(int(seed))
    null = np.zeros(n_perm, dtype=np.float64)
    for sample in samples:
        total = float(sample.sum())
        for start in range(0, n_perm, _BLOCK):
            stop = min(start + _BLOCK, n_perm)
            flips = generator.random((stop - start, sample.size)) < 0.5
            flipped = np.where(flips, sample, 0.0).sum(axis=1)
            null[start:stop] += (total - 2.0 * flipped) / sample.size
    null /= len(samples)
    # Ties within rounding count as extreme.
    slack = 1e-9 * max(float(np.abs(sample).max()) for sample in samples)
    extreme = int(np.count_nonzero(np.abs(null) >= abs(observed) - slack))
    return {
        "statistic": observed,
        "p": (1.0 + extreme) / (1.0 + n_perm),
        "n": float(sum(sample.size for sample in samples)),
        "seeds": float(len(samples)),
    }


def two_level_bootstrap_ci(
    differences: PerSeed,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for the mean over seeds of each seed's mean.

    Each resample draws the seeds with replacement, then draws the images of
    each drawn seed with replacement. A seed drawn twice is resampled twice.
    """
    samples = _seed_samples(differences)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    generator = np.random.default_rng(int(seed))
    picks = generator.integers(0, len(samples), size=(int(n_boot), len(samples)))
    slot_means = np.empty(picks.shape, dtype=np.float64)
    for index, sample in enumerate(samples):
        rows, cols = np.nonzero(picks == index)
        for start in range(0, rows.size, _BLOCK):
            stop = min(start + _BLOCK, rows.size)
            draws = generator.integers(0, sample.size, size=(stop - start, sample.size))
            slot_means[rows[start:stop], cols[start:stop]] = sample[draws].mean(axis=1)
    statistics = slot_means.mean(axis=1)
    low, high = np.quantile(statistics, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(low), float(high)


def seed_pooled_summary(
    comparisons: dict[str, PerSeed],
    seed: int = 0,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
) -> list[dict[str, float | str]]:
    """Mean, two-level interval, and Holm-adjusted sign-flip p (families T, D-T).

    Each value holds one comparison's per-image paired differences by seed.
    """
    if not comparisons:
        raise ValueError("a family needs at least one comparison")
    rows: list[dict[str, float | str]] = []
    for name, differences in comparisons.items():
        test = sign_flip_test(differences, n_perm=n_perm, seed=seed)
        low, high = two_level_bootstrap_ci(differences, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "name": name,
                "n": test["n"],
                "seeds": test["seeds"],
                "mean": test["statistic"],
                "ci_low": low,
                "ci_high": high,
                "p": test["p"],
            }
        )
    for row, adjusted in zip(rows, holm_adjust([float(row["p"]) for row in rows])):
        row["p_holm"] = adjusted
    return rows


def stratified_two_level_bootstrap_ci(
    differences: Mapping[int, Sequence[Sequence[float]]],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """``two_level_bootstrap_ci`` with images in strata inside each seed (EVAL_PLAN 13.11, S2).

    ``differences`` maps each seed to its strata (for S2, the two planted conditions).
    The statistic is the mean over seeds of each seed's mean over all its images,
    strata stacked. Each resample draws the seeds with replacement, then draws
    each drawn seed's images with replacement within each stratum, so every
    stratum keeps its size.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    seeds = [differences[key] for key in sorted(differences)]
    if not seeds:
        raise ValueError("differences need at least one seed")
    strata = [_seed_samples(list(groups)) for groups in seeds]
    generator = np.random.default_rng(int(seed))
    picks = generator.integers(0, len(strata), size=(int(n_boot), len(strata)))
    slot_means = np.empty(picks.shape, dtype=np.float64)
    for index, groups in enumerate(strata):
        size = sum(group.size for group in groups)
        rows, cols = np.nonzero(picks == index)
        total = np.zeros(rows.size, dtype=np.float64)
        for group in groups:
            for start in range(0, rows.size, _BLOCK):
                stop = min(start + _BLOCK, rows.size)
                draws = generator.integers(0, group.size, size=(stop - start, group.size))
                total[start:stop] += group[draws].sum(axis=1)
        slot_means[rows, cols] = total / size
    statistics = slot_means.mean(axis=1)
    low, high = np.quantile(statistics, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(low), float(high)
