"""CIFAR-10 with two planted patches.

``patch_layout``, ``_class_hue`` and ``PlantedPatchCIFAR`` are the geometry
the planted-patch checkpoints were trained on. The dataset key stays
``planted_patch`` because it appears in those checkpoint names.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from evaluation.splits import load_indices

REPO = Path(__file__).resolve().parents[1]


def _hue_rgb(hue: float) -> tuple[float, float, float]:
    r = 0.5 * (1.0 + math.cos(2.0 * math.pi * hue))
    g = 0.5 * (1.0 + math.cos(2.0 * math.pi * hue + 2.0943951))
    b = 0.5 * (1.0 + math.cos(2.0 * math.pi * hue + 4.1887902))
    return r, g, b


def _class_hue(class_idx: int, num_classes: int) -> tuple[float, float, float]:
    return _hue_rgb(class_idx / max(num_classes, 1))


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

    def separated(dy: int, dx: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
        moved = [shift(origin, dy, dx) for origin in present]
        boxes = [_box(origin, patch) for origin in (*present, *moved)]
        if any(
            _overlap(boxes[i], boxes[j])
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
        ):
            return None
        return moved[0], moved[1]

    for attempt in range(span):
        found = separated(span // 2 + attempt, span // 3 + attempt)
        if found is not None:
            return present[0], present[1], found[0], found[1]
    for dy in range(span):
        for dx in range(span):
            if dy == 0 and dx == 0:
                continue
            found = separated(dy, dx)
            if found is not None:
                return present[0], present[1], found[0], found[1]
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


def env_batch_planted(
    present: torch.Tensor,
    moved: torch.Tensor,
    removed: torch.Tensor,
) -> EnvBatch:
    """Present, moved, and removed views of the same image. Present is in-distribution."""
    from core.types import EnvBatch

    return EnvBatch(
        xs=[present, moved, removed],
        env_ids=["present", "moved", "removed"],
    )


class PlantedPatchClassifier(Dataset):
    """The image with both patches present. Patch positions use ``patch_seed``."""

    def __init__(self, split: str = "val", patch_seed: int = 0, **kwargs):
        self.inner = PlantedPatchCIFAR(split=split, seed=patch_seed, **kwargs)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.inner[index]
        return item["present"], int(item["label"])


# ---------------------------------------------------------------------------
# Planted pairs: ground truth for the decomposition game
# (FORMULATION_DECOMPOSITION.md section 6). For each class pair (k, l), patch
# A_k appears only in class-k training images, A_l only in class-l images, and
# a texture C in both, each with probability 0.9. C is evidence for {k, l}
# against the rest and carries no information between k and l.
# ---------------------------------------------------------------------------

# CIFAR-10: cat/dog, automobile/truck, deer/horse.
PAIRS: tuple[tuple[int, int], ...] = ((3, 5), (1, 9), (4, 7))
PATCH_PROBABILITY = 0.9
PAIR_PATCH = 32          # one 7x7 cell at 224


def pair_of(label: int) -> int | None:
    for index, pair in enumerate(PAIRS):
        if int(label) in pair:
            return index
    return None


def _stripes(pair_index: int, patch: int) -> torch.Tensor:
    """C of one pair: black-and-white stripes, period 8 px. Diagonal, anti-diagonal, horizontal."""
    y = torch.arange(patch).view(-1, 1)
    x = torch.arange(patch).view(1, -1)
    phase = (x + y, x - y + patch, y + 0 * x)[pair_index]
    board = ((phase // 4) % 2).float()
    return board.unsqueeze(0).expand(3, -1, -1).clone()


def pair_cells(generator: torch.Generator, count: int, grid: int = 7) -> list[int]:
    """``count`` distinct cells, no two touching (not even at a corner)."""
    chosen: list[int] = []
    for cell in torch.randperm(grid * grid, generator=generator).tolist():
        r, c = divmod(cell, grid)
        if all(max(abs(r - q // grid), abs(c - q % grid)) > 1 for q in chosen):
            chosen.append(cell)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("could not place the patches")


def _origins(generator: torch.Generator, count: int, canvas: int, patch: int, placement: str) -> list[tuple[int, int]]:
    if placement == "grid":
        side = canvas // patch
        return [(divmod(cell, side)[0] * patch, divmod(cell, side)[1] * patch)
                for cell in pair_cells(generator, count, side)]
    if placement != "random":
        raise ValueError(f"unknown placement {placement!r}")
    origins: list[tuple[int, int]] = []
    for _ in range(1000):
        y, x = (int(v) for v in torch.randint(0, canvas - patch + 1, (2,), generator=generator))
        box = _box((y, x), patch)
        grown = (box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8)
        if all(not _overlap(grown, _box(o, patch)) for o in origins):
            origins.append((y, x))
        if len(origins) == count:
            return origins
    raise RuntimeError("could not place the patches")


class PlantedPairCIFAR(Dataset):
    """CIFAR-10 with per-pair planted patches. Val and train index the CIFAR-10 train pool.

    ``mode="train"``: each patch of the image's pair appears with probability
    0.9; classes outside the pairs get none. ``mode="test"``: the image is of a
    pair class and carries A_k, A_l and C of its pair at disjoint positions.
    Items are deterministic in ``(seed, index)``.
    """

    def __init__(self, split: str = "val", mode: str = "train", root: Path | None = None,
                 image_size: int = 224, patch: int = PAIR_PATCH, seed: int = 0, placement: str = "grid",
                 final: bool = False, config_hash: str | None = None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if mode not in {"train", "test"}:
            raise ValueError("mode must be train or test")
        from torchvision import transforms
        from torchvision.datasets import CIFAR10

        self.split, self.mode, self.placement = split, mode, placement
        self.image_size, self.patch, self.seed = int(image_size), int(patch), int(seed)
        data_root = Path(root) if root else REPO / "data"
        self.cifar = CIFAR10(root=str(data_root), train=split != "test", download=False,
                             transform=transforms.ToTensor())
        indices = load_indices("cifar10", split, final=final, config_hash=config_hash, n_items=len(self.cifar))
        if mode == "test":
            targets = self.cifar.targets
            indices = [i for i in indices if pair_of(int(targets[i])) is not None]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def _image(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.cifar[self.indices[index]]
        if image.shape[-1] != self.image_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False).squeeze(0)
        return image, int(label)

    def __getitem__(self, index: int) -> dict:
        image, label = self._image(index)
        generator = torch.Generator(device="cpu").manual_seed(self.seed * 1_000_003 + int(index))
        pair_index = pair_of(label)
        masks = torch.zeros(3, self.image_size, self.image_size)
        present = [False, False, False]
        if pair_index is not None:
            k_class, l_class = PAIRS[pair_index]
            patches = (_solid(_class_hue(k_class, 10), self.patch), _solid(_class_hue(l_class, 10), self.patch),
                       _stripes(pair_index, self.patch))
            if self.mode == "test":
                present = [True, True, True]
            else:
                draws = torch.rand(2, generator=generator)
                own = 0 if label == k_class else 1
                present[own] = bool(draws[0] < PATCH_PROBABILITY)
                present[2] = bool(draws[1] < PATCH_PROBABILITY)
            origins = _origins(generator, 3, self.image_size, self.patch, self.placement)
            for slot in range(3):
                if present[slot]:
                    image, masks[slot] = _stamp(image, origins[slot], patches[slot])
        return {"image": image.clamp(0, 1), "label": label, "pair": -1 if pair_index is None else pair_index,
                "masks": masks, "present": torch.tensor(present)}


class PlantedPairClassifier(Dataset):
    """Training-mode planted pairs as ``(image, label)``, for ``scripts/train_backbone.py``."""

    def __init__(self, split: str = "train", **kwargs):
        self.inner = PlantedPairCIFAR(split=split, mode="train", **kwargs)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.inner[index]
        return item["image"], int(item["label"])


def verify_roles(model, x: torch.Tensor, patches: torch.Tensor, ids: torch.Tensor, n_random: int = 32,
                 seed: int = 0, patch: int = PAIR_PATCH) -> torch.Tensor:
    """``(B, 3)`` bool: is each planted patch doing its role in this model, by intervention?

    Each patch (A_k, A_l, C) is ROAD-deleted alone. Its drop in a class counts
    when it exceeds the 95th percentile of that class's drops over ``n_random``
    seeded patch-sized squares. A_k is confirmed when its k drop counts and its
    l drop does not; A_l the mirror; C when both count.
    """
    from evaluation.scores import _remove

    ids = ids[:, :2].long()
    batch, _, height, width = x.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    origins = torch.randint(0, height - patch + 1, (n_random, 2), generator=generator)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full = model(x).gather(1, ids)

            def drop(mask: torch.Tensor) -> torch.Tensor:
                return full - model(_remove(x, mask, 24, 0.01, seed)).gather(1, ids)

            random_drops = []
            for y0, x0 in origins.tolist():
                square = torch.zeros(batch, height, width, device=x.device)
                square[:, y0:y0 + patch, x0:x0 + patch] = 1.0
                random_drops.append(drop(square))
            threshold = torch.quantile(torch.stack(random_drops, dim=0), 0.95, dim=0)
            counts = [drop(patches[:, slot].to(x.device)) > threshold for slot in range(3)]
    finally:
        model.train(was_training)
    a_k = counts[0][:, 0] & ~counts[0][:, 1]
    a_l = counts[1][:, 1] & ~counts[1][:, 0]
    shared = counts[2][:, 0] & counts[2][:, 1]
    return torch.stack([a_k, a_l, shared], dim=1).cpu()
