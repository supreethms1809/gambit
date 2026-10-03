"""Shared scoring for CDEA and every baseline.

Every method emits a per-hypothesis map. That map is turned into a pixel mask
and a metric by this package, so method differences are not scoring differences.
"""

from evaluation.masks import mass_in, regions_to_pixels, top_fraction_mask
from evaluation.metrics import paired, spearman
from evaluation.nulls import random_translate
from evaluation.removal import noisy_linear_impute
from evaluation.sampling import seeded_indices, seeded_subset

__all__ = [
    "mass_in",
    "noisy_linear_impute",
    "paired",
    "random_translate",
    "regions_to_pixels",
    "seeded_indices",
    "seeded_subset",
    "spearman",
    "top_fraction_mask",
]
