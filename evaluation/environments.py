"""Paired shift environments.

Every unit is a pixel-aligned set of views of one image. The first view is
in-distribution. Builders live in the modules they were restored from; this
module is the list the harness imports.
"""

from evaluation.colored_mnist import ColoredMNIST, env_batch_colored_mnist
from evaluation.dogs_pairs import DogsBoxDataset, background_styles, make_env_fn
from evaluation.imagenet9_pairs import ImageNet9Pairs, env_batch_imagenet9
from evaluation.planted_cues import PlantedPatchCIFAR, env_batch_planted
from evaluation.waterbirds_pairs import WaterbirdsPairs, env_batch_waterbirds

SHIFT_DATASETS = (
    "waterbirds",
    "imagenet9",
    "stanford_dogs",
    "planted_patch",
    "colored_mnist",
)

__all__ = [
    "SHIFT_DATASETS",
    "ColoredMNIST",
    "DogsBoxDataset",
    "ImageNet9Pairs",
    "PlantedPatchCIFAR",
    "WaterbirdsPairs",
    "background_styles",
    "env_batch_colored_mnist",
    "env_batch_imagenet9",
    "env_batch_planted",
    "env_batch_waterbirds",
    "make_env_fn",
]
