"""Openers for the paper datasets added in S07.

``pets`` stays the 2-class cats-vs-dogs folder. The 37-breed paper dataset is
``oxford_pets``. ImageNet-S is not opened here: its images are ImageNet-1k,
which this machine does not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset

# Official test is held out. Val is 10% of each class inside the official train pool.
S07_DATASETS = ("cifar100", "oxford_pets", "cub200")


class CubBirds(Dataset):
    """CUB-200-2011 in ``images.txt`` order, so split indices stay stable.

    ``root`` is either ``data/`` or ``data/CUB_200_2011``.
    """

    def __init__(self, root: Path, transform=None):
        root = Path(root)
        if (root / "CUB_200_2011").is_dir():
            root = root / "CUB_200_2011"
        images = _indexed_file(root / "images.txt")
        labels = _indexed_file(root / "image_class_labels.txt")
        class_names = _indexed_file(root / "classes.txt")
        ids = sorted(images)
        self.root = root
        self.transform = transform
        self.samples = [(root / "images" / images[i], int(labels[i]) - 1) for i in ids]
        self.targets = [label for _, label in self.samples]
        self.classes = [class_names[i].split(".", 1)[-1] for i in range(1, len(class_names) + 1)]
        if len(self.classes) != 200:
            raise RuntimeError(f"CUB should have 200 classes, found {len(self.classes)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _indexed_file(path: Path) -> dict[int, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        out[int(key)] = value.strip()
    return out


def cub_train_flags(root: Path) -> dict[int, int]:
    """Map 1-based image id to 1 (official train) or 0 (official test)."""
    root = Path(root)
    if (root / "CUB_200_2011").is_dir():
        root = root / "CUB_200_2011"
    flags = {}
    for image_id, value in _indexed_file(root / "train_test_split.txt").items():
        flags[image_id] = int(value)
    return flags


def open_unsplit(name: str, split: str, data_root: Path, transform=None):
    """Dataset whose length matches ``index_length[split]``. Does not download.

    Train and val share one dataset object. Test is the official test set, which
    for CUB is the same image list with different indices.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    data_root = Path(data_root)
    if name == "cifar100":
        from torchvision.datasets import CIFAR100

        ds = CIFAR100(
            root=str(data_root),
            train=split != "test",
            download=False,
            transform=transform,
        )
        return ds
    if name == "oxford_pets":
        from torchvision.datasets import OxfordIIITPet

        ds = OxfordIIITPet(
            root=str(data_root),
            split="test" if split == "test" else "trainval",
            target_types="category",
            download=False,
            transform=transform,
        )
        ds.targets = [int(t) for t in ds._labels]
        return ds
    if name == "cub200":
        return CubBirds(data_root, transform=transform)
    raise ValueError(f"unknown paper dataset {name}")


def label_list(dataset) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError(f"{type(dataset).__name__} has no targets")
    return [int(t) for t in targets]


def missing_classes(labels: Sequence[int], indices: Sequence[int], n_classes: int) -> list[int]:
    present = {int(labels[i]) for i in indices}
    return [c for c in range(n_classes) if c not in present]


def class_counts(labels: Sequence[int], indices: Sequence[int], n_classes: int) -> dict[str, int]:
    counts = {str(c): 0 for c in range(n_classes)}
    for i in indices:
        counts[str(int(labels[i]))] += 1
    return counts


def require_every_class(
    labels: Sequence[int],
    indices: Sequence[int],
    n_classes: int,
    split: str,
) -> dict[str, int]:
    """Class counts for one split. Every class must appear at least once."""
    missing = missing_classes(labels, indices, n_classes)
    if missing:
        shown = ", ".join(str(c) for c in missing[:8])
        raise RuntimeError(f"{split} is missing {len(missing)} classes, including {shown}")
    return class_counts(labels, indices, n_classes)
