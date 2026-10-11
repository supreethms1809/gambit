"""Paired ImageNet-9 background environments.

``original``, ``mixed_same``, and ``mixed_rand`` are three views of one
foreground. Files pair by class folder plus foreground id. A mixed file named
``fg_<id>_bg_<other>.JPEG`` pairs with ``<id>.JPEG`` in the same class folder.
Identical relative paths pair as well.

The Madry challenge release (``bg_challenge``) is the held-out test set. This
loader refuses that directory until the eval plan is frozen. The training
archives' own ``val`` folders are the split used for checks.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from core.types import EnvBatch
from evaluation.splits import assert_split_allowed

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("original", "mixed_same", "mixed_rand")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


def foreground_key(path: Path) -> str:
    """``class/foreground-id``, shared by an original file and its mixes."""
    name = path.name
    if name.startswith("fg_") and "_bg_" in name:
        stem = name[3:].split("_bg_", 1)[0]
    else:
        stem = path.stem
    return f"{path.parent.name}/{stem}"


def pair_variant_files(directories: dict[str, Path]) -> list[dict[str, Path]]:
    """One dict per foreground that exists in every variant."""
    grouped: dict[str, dict[str, Path]] = {}
    for variant, directory in directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"missing ImageNet-9 variant directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.suffix not in _IMAGE_SUFFIXES or not path.is_file():
                continue
            key = foreground_key(path.relative_to(directory))
            slot = grouped.setdefault(key, {})
            if variant in slot:
                raise RuntimeError(f"two {variant} files share foreground key {key}")
            slot[variant] = path
    pairs = [grouped[key] for key in sorted(grouped) if all(v in grouped[key] for v in VARIANTS)]
    if not pairs:
        raise FileNotFoundError(
            "no foreground is present in original, mixed_same, and mixed_rand"
        )
    return pairs


def _variant_dir(root: Path, variant: str, split: str) -> Path:
    direct = root / variant
    nested = direct / split
    if nested.is_dir():
        return nested
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"missing {direct} (looked for {nested} as well)")


class ImageNet9Pairs(Dataset):
    """One item is original, mixed_same, mixed_rand, and the class index."""

    def __init__(
        self,
        root: Path | None = None,
        split: str = "val",
        image_size: int = 224,
        final: bool = False,
        config_hash: str | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.root = Path(root) if root else REPO / "data" / "imagenet9"
        self.split = split
        self.image_size = int(image_size)
        if "bg_challenge" in self.root.parts or self.root.name == "bg_challenge":
            assert_split_allowed("test", final=final, config_hash=config_hash)
        directories = {
            variant: _variant_dir(self.root, variant, split) for variant in VARIANTS
        }
        self.pairs = pair_variant_files(directories)
        self._to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        paths = self.pairs[index]
        views = {name: self._load(paths[name]) for name in VARIANTS}
        class_name = paths["original"].parent.name
        # Class folders are ``00_dog`` ... ``08_insect``, or a plain name.
        prefix = class_name.split("_", 1)[0]
        label = int(prefix) if prefix.isdigit() else 0
        return {**views, "label": label}

    def _load(self, path: Path) -> torch.Tensor:
        image = self._to_tensor(Image.open(path).convert("RGB"))
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image.clamp(0, 1)


class ImageNet9Classifier(Dataset):
    """The original ImageNet-9 image and its class. Mixes are evaluation environments."""

    def __init__(self, split: str = "val", image_size: int = 224, **kwargs):
        self.inner = ImageNet9Pairs(split=split, image_size=image_size, **kwargs)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.inner[index]
        return item["original"], int(item["label"])


def env_batch_imagenet9(
    original: torch.Tensor,
    mixed_same: torch.Tensor,
    mixed_rand: torch.Tensor,
) -> EnvBatch:
    return EnvBatch(
        xs=[original, mixed_same, mixed_rand],
        env_ids=["original", "mixed_same", "mixed_rand"],
    )
