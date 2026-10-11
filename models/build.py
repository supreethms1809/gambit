"""One classifier builder for the paper backbones, plus checkpoint names.

ResNet-50 and ViT-B/16 are built the same way the training script built them:
torchvision weights when ``pretrained`` is set, then a new linear head.
Checkpoint filenames follow ``paper_checkpoint_name``. The recorded input
convention lives in ``results/paper/input_convention.txt``.
"""

from __future__ import annotations

import torch.nn as nn

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "oxford_pets": 37,
    "pets": 2,
    "stanford_dogs": 120,
    "cub200": 200,
    "ham10000": 7,
    "brain_tumor": 3,
    "imagenet": 1000,
    "planted_patch": 10,
    "waterbirds": 2,
    "imagenet9": 9,
    "colored_mnist": 10,
}

PAPER_BACKBONES = ("resnet50", "vit_b_16")

# Shift datasets are full fine-tunes. Contrastive datasets are linear probes.
FINE_TUNED = {"planted_patch", "waterbirds", "imagenet9", "colored_mnist"}


def paper_checkpoint_name(
    dataset: str,
    model_name: str,
    pretrained: bool,
    freeze_backbone: bool,
    num_epochs: int,
    lr: float,
    seed: int,
    convention: str = "raw",
) -> str:
    """Filename for one paper cell. ImageNet normalisation gets its own cache key."""
    mode_tag = "lp" if freeze_backbone else "ft"
    pre_tag = "pt" if pretrained else "rand"
    lr_tag = f"lr{lr:g}"
    conv_tag = "" if convention == "raw" else f"_{convention}"
    return (
        f"{dataset}_{model_name}_{pre_tag}_{mode_tag}_ep{num_epochs}"
        f"_{lr_tag}{conv_tag}_seed{seed}.pt"
    )


def model_grid_size(model_name: str) -> tuple[int, int]:
    """``(grid_h, grid_w)`` at 224 px. ViT-B/16 is 14 by 14. ResNet-50 is 7 by 7."""
    if model_name == "vit_b_16":
        return 14, 14
    return 7, 7


def build_backbone(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """ResNet-50 or ViT-B/16 with a fresh classification head."""
    from torchvision import models

    if model_name not in PAPER_BACKBONES:
        raise ValueError(f"model_name must be one of: {', '.join(PAPER_BACKBONES)}")
    weights = "IMAGENET1K_V1" if pretrained else None
    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    model = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
