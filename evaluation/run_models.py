"""Load the classifier for one paper cell.

Sources, in order:

* ``checkpoint``: an explicit path. Loaded through ``load_checkpoint_into``,
  so an ImageNet-convention checkpoint gets its normalisation layer back.
* ``paper``: the checkpoint ``get_or_train`` writes for (dataset, backbone, seed)
  under ``results/paper/checkpoints``.
* ``imagenet``: the torchvision ``IMAGENET1K_V1`` weights, wrapped in
  ``NormalizedModel``. Used for the ImageNet unit, which has no training.
* ``smoke``: the ImageNet backbone with a randomly initialised head. It only
  exercises the code path. Its numbers mean nothing, and every record from it
  carries ``model_source="smoke"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
PAPER_CKPT_DIR = REPO / "results" / "paper" / "checkpoints"

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "oxford_pets": 37,
    "stanford_dogs": 120,
    "cub200": 200,
    "ham10000": 7,
    "brain_tumor": 3,
    "imagenet": 1000,
    "waterbirds": 2,
    "imagenet9": 9,
    "planted_patch": 10,
    "colored_mnist": 10,
}

# Shift datasets are full fine-tunes; contrastive datasets are linear probes.
FINE_TUNED = {"waterbirds", "imagenet9", "planted_patch", "colored_mnist"}


@dataclass
class LoadedModel:
    model: nn.Module
    source: str
    path: Optional[str]
    num_classes: int


def _bare(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    from scripts.ablation_contrastive import _build_model

    return _build_model(backbone, num_classes, pretrained=pretrained)


def paper_checkpoint_path(dataset: str, backbone: str, seed: int, *,
                          ckpt_dir: Optional[Path] = None, num_epochs: int = 15) -> Path:
    """Where ``get_or_train`` writes this cell under the recorded input convention.

    ``ckpt_dir`` and ``num_epochs`` locate smoke checkpoints; paper cells use the defaults.
    """
    from models.wrapper import read_input_convention
    from scripts.train_backbone import paper_checkpoint_name

    fine_tune = dataset in FINE_TUNED
    name = paper_checkpoint_name(
        dataset,
        backbone,
        pretrained=True,
        freeze_backbone=not fine_tune,
        num_epochs=num_epochs,
        lr=1e-4 if fine_tune else 1e-3,
        seed=seed,
        convention=read_input_convention(),
    )
    return Path(ckpt_dir or PAPER_CKPT_DIR) / name


def load_cell_model(
    dataset: str,
    backbone: str,
    seed: int,
    *,
    source: str = "auto",
    checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> LoadedModel:
    """``source`` is ``auto``, ``checkpoint``, ``paper``, ``imagenet`` or ``smoke``.

    ``auto`` uses ``checkpoint`` if given, ImageNet weights for the ImageNet
    unit, and otherwise the paper checkpoint. It never falls back to ``smoke``:
    a missing paper checkpoint raises, so a real run cannot silently score a
    random head.
    """
    from models.wrapper import NormalizedModel, load_checkpoint_into

    if dataset not in NUM_CLASSES:
        raise ValueError(f"unknown dataset {dataset!r}")
    num_classes = NUM_CLASSES[dataset]
    if source == "auto":
        if checkpoint is not None:
            source = "checkpoint"
        elif dataset == "imagenet":
            source = "imagenet"
        else:
            source = "paper"

    path: Optional[Path] = None
    if source in {"checkpoint", "paper"}:
        path = Path(checkpoint) if source == "checkpoint" else paper_checkpoint_path(dataset, backbone, seed)
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint for {dataset}/{backbone}/seed{seed}: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "num_classes" in blob:
            num_classes = int(blob["num_classes"])
        model = load_checkpoint_into(_bare(backbone, num_classes, pretrained=False), blob)
    elif source == "imagenet":
        if dataset != "imagenet":
            raise ValueError("ImageNet weights only explain the ImageNet unit")
        model = NormalizedModel(_imagenet_classifier(backbone))
    elif source == "smoke":
        torch.manual_seed(seed)
        model = NormalizedModel(_bare(backbone, num_classes, pretrained=True))
    else:
        raise ValueError(f"unknown model source {source!r}")

    # Parameters keep requires_grad: Grad-CAM needs gradients at activations, and
    # the allocator and TorchRay freeze and restore parameters themselves.
    model.eval()
    if device is not None:
        model.to(device)
    return LoadedModel(model=model, source=source, path=str(path) if path else None, num_classes=num_classes)


def _imagenet_classifier(backbone: str) -> nn.Module:
    from torchvision import models

    if backbone == "resnet50":
        return models.resnet50(weights="IMAGENET1K_V1")
    if backbone == "vit_b_16":
        return models.vit_b_16(weights="IMAGENET1K_V1")
    raise ValueError(f"no ImageNet weights wired for {backbone!r}")
