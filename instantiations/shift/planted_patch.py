"""CIFAR-10 with two planted patches, and the environments that move or remove them.

Patch A is a solid class-colored square. Patch B is a checker. Both positions
are drawn from a seed so the ground-truth masks are known. ``present`` has
both patches, ``moved`` has the same patches in disjoint new positions, and
``removed`` is the untouched image.

The confirmatory two-patch recovery metric is ``evaluation.scores.two_patch_recovery``.
This loader is the shift pair.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.types import EnvBatch
from evaluation.splits import load_indices
from instantiations.shift.biased_data import _class_hue

REPO = Path(__file__).resolve().parents[2]


def _box(origin: tuple[int, int], patch: int) -> tuple[int, int, int, int]:
    y, x = origin
    return y, x, y + patch, x + patch


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def patch_layout(
    index: int,
    seed: int,
    canvas: int = 224,
    patch: int = 32,
    margin: int = 8,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Present and moved origins for patch A and patch B. All four boxes are disjoint."""
    span = canvas - patch - 2 * margin
    if span < 2:
        raise ValueError("canvas is too small for two patches")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(index) * 10007)
    present: list[tuple[int, int]] = []
    for _ in range(400):
        coords = torch.randint(0, span, (2,), generator=generator)
        origin = (margin + int(coords[0]), margin + int(coords[1]))
        if all(not _overlap(_box(origin, patch), _box(other, patch)) for other in present):
            present.append(origin)
        if len(present) == 2:
            break
    if len(present) != 2:
        raise RuntimeError(f"could not place two patches at index {index}")

    def shift(origin: tuple[int, int], dy: int, dx: int) -> tuple[int, int]:
        y, x = origin
        return (
            margin + (y - margin + dy) % span,
            margin + (x - margin + dx) % span,
        )

    for attempt in range(span):
        dy = span // 2 + attempt
        dx = span // 3 + attempt
        moved = [shift(origin, dy, dx) for origin in present]
        boxes = [_box(origin, patch) for origin in present + moved]
        if all(
            not _overlap(boxes[i], boxes[j])
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
        ):
            return present[0], present[1], moved[0], moved[1]
    raise RuntimeError(f"could not move patches at index {index}")


def _solid(color: tuple[float, float, float], patch: int) -> torch.Tensor:
    image = torch.zeros(3, patch, patch)
    image[0], image[1], image[2] = color
    return image


def _checker(patch: int) -> torch.Tensor:
    image = torch.zeros(3, patch, patch)
    cells = torch.arange(patch)
    board = ((cells[:, None] // 4 + cells[None, :] // 4) % 2).float()
    image[0] = board
    image[1] = 1.0 - board
    return image


def _stamp(
    image: torch.Tensor,
    origin: tuple[int, int],
    patch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = origin
    height, width = patch.shape[-2:]
    out = image.clone()
    out[:, y:y + height, x:x + width] = patch
    mask = torch.zeros(image.shape[-2], image.shape[-1])
    mask[y:y + height, x:x + width] = 1.0
    return out, mask


class PlantedPatchCIFAR(Dataset):
    """Val and train index the CIFAR-10 train pool. Test stays locked."""

    def __init__(
        self,
        split: str = "val",
        root: Path | None = None,
        image_size: int = 224,
        patch: int = 32,
        seed: int = 0,
        final: bool = False,
        config_hash: str | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        from torchvision import transforms
        from torchvision.datasets import CIFAR10

        self.split = split
        self.image_size = int(image_size)
        self.patch = int(patch)
        self.seed = int(seed)
        data_root = Path(root) if root else REPO / "data"
        train_pool = split != "test"
        self.cifar = CIFAR10(
            root=str(data_root),
            train=train_pool,
            download=False,
            transform=transforms.ToTensor(),
        )
        self.indices = load_indices(
            "cifar10",
            split,
            final=final,
            config_hash=config_hash,
            n_items=len(self.cifar),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        image, label = self.cifar[self.indices[index]]
        if image.shape[-1] != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        a, b, a_moved, b_moved = patch_layout(
            index, self.seed, self.image_size, self.patch
        )
        patch_a = _solid(_class_hue(int(label), 10), self.patch)
        patch_b = _checker(self.patch)
        present, mask_a = _stamp(image, a, patch_a)
        present, mask_b = _stamp(present, b, patch_b)
        moved, mask_a_moved = _stamp(image, a_moved, patch_a)
        moved, mask_b_moved = _stamp(moved, b_moved, patch_b)
        return {
            "present": present.clamp(0, 1),
            "moved": moved.clamp(0, 1),
            "removed": image.clamp(0, 1),
            "mask_a": mask_a,
            "mask_b": mask_b,
            "mask_a_moved": mask_a_moved,
            "mask_b_moved": mask_b_moved,
            "label": int(label),
        }


class PlantedPatchClassifier(Dataset):
    """The image with both patches present. Moved and removed are evaluation environments.

    Patch positions use ``patch_seed`` and stay fixed across model seeds.
    """

    def __init__(self, split: str = "val", patch_seed: int = 0, **kwargs):
        self.inner = PlantedPatchCIFAR(split=split, seed=patch_seed, **kwargs)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.inner[index]
        return item["present"], int(item["label"])


def env_batch_planted(
    present: torch.Tensor,
    moved: torch.Tensor,
    removed: torch.Tensor,
) -> EnvBatch:
    return EnvBatch(
        xs=[present, moved, removed],
        env_ids=["present", "moved", "removed"],
    )
