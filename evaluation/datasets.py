"""Eval loaders for the paper cells.

Preprocessing is the eval path: CIFAR uses ``ToTensor`` then a tensor
``Resize`` (antialiased). It is not the training transform. A missing
dataset raises. There is no random-image fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

MEDICAL_EVAL_ROOTS = {
    "ham10000": Path("ham10000") / "val",
    "brain_tumor": Path("brain_tumor") / "Testing",
}
DATASET_CHOICES = (
    "mnist", "cifar10", "cifar100", "pets", "oxford_pets", "stanford_dogs",
    "cub200", "ham10000", "brain_tumor", "imagenet",
)


def _finalize_eval_dataset(
    dataset_name: str,
    ds,
    *,
    split: str,
    final: bool,
    config_hash: Optional[str],
    num_images: Optional[int],
    seed: int,
):
    """Apply the recorded split, then a seeded subset. Never a class-ordered prefix."""
    from torch.utils.data import Subset

    from evaluation.sampling import seeded_subset
    from evaluation.splits import load_indices

    indices = load_indices(
        dataset_name,
        split,
        final=final,
        config_hash=config_hash,
        n_items=len(ds),
    )
    ds = Subset(ds, indices)
    if num_images is not None:
        ds = seeded_subset(ds, num_images, seed)
    return ds


def get_eval_loader(
    dataset: str,
    batch_size: int,
    data_root: Path,
    image_size: Optional[int] = None,
    *,
    seed: int = 0,
    num_images: Optional[int] = None,
    split: str = "val",
    final: bool = False,
    config_hash: Optional[str] = None,
):
    try:
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import MNIST, CIFAR10, ImageFolder
    except ImportError as e:
        raise ImportError("torchvision is required for dataset loading") from e

    from evaluation.splits import load_spec

    resize = transforms.Resize((image_size, image_size)) if image_size is not None else None
    spec = load_spec(dataset)
    root_rel = spec["roots"][split]

    if dataset == "mnist":
        ops = [transforms.ToTensor(), transforms.Lambda(lambda t: t.repeat(3, 1, 1))]
        if resize is not None:
            ops.append(resize)
        t = transforms.Compose(ops)
        ds = MNIST(root=str(data_root), train=root_rel == "train", download=False, transform=t)
        num_classes = 10
    elif dataset == "cifar10":
        ops = [transforms.ToTensor()]
        if resize is not None:
            ops.append(resize)
        t = transforms.Compose(ops)
        ds = CIFAR10(root=str(data_root), train=root_rel == "train", download=False, transform=t)
        num_classes = 10
    elif dataset == "pets":
        target_size = image_size if image_size is not None else 64
        t = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor()])
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
        num_classes = len(ds.classes)
    elif dataset == "stanford_dogs":
        target_size = image_size if image_size is not None else 224
        t = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor()])
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
        num_classes = len(ds.classes)
    elif dataset in {"cifar100", "oxford_pets", "cub200"}:
        from evaluation.paper_datasets import open_unsplit

        target_size = image_size if image_size is not None else 224
        t = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        ds = open_unsplit(dataset, split, data_root, transform=t)
        num_classes = len(set(ds.targets))
    elif dataset == "imagenet":
        t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size if image_size is not None else 224),
            transforms.ToTensor(),
        ])
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
        num_classes = len(ds.classes)
    elif dataset in MEDICAL_EVAL_ROOTS:
        target_size = image_size if image_size is not None else 224
        t = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor()])
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
        num_classes = len(ds.classes)
    else:
        raise ValueError("dataset must be one of: " + ", ".join(DATASET_CHOICES))

    if len(ds) == 0:
        raise FileNotFoundError(f"no images for {dataset} split {split} under {data_root}")
    ds = _finalize_eval_dataset(
        dataset, ds, split=split, final=final, config_hash=config_hash,
        num_images=num_images, seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), num_classes
