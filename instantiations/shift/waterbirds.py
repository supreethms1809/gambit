"""Paired Waterbirds environments.

Each CUB bird is drawn twice: once on a land Places background and once on a
water Places background. The bird pixels stay. The background is the shortcut.
This is the counterfactual pair. The unpaired natural-groups objective is
``GroupStatisticsObjective``.

Backgrounds are the Places365 validation images of the four categories used by
Sagawa et al.: bamboo forest and broadleaf forest (land), lake/natural and
ocean (water). There are 100 images in each category. Each pool is split
160/20/20 into train/val/test with a fixed seed, so a place photo is not in
two splits. Birds use the CUB paper indices in ``data/splits/cub200.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from core.types import EnvBatch
from evaluation.paper_datasets import CubBirds
from evaluation.splits import VAL_CARVE_SEED, load_indices

REPO = Path(__file__).resolve().parents[2]

# Category ids in categories_places365.txt.
LAND_CATEGORY_IDS = (36, 150)    # bamboo_forest, forest/broadleaf
WATER_CATEGORY_IDS = (205, 243)  # lake/natural, ocean
LAND_SPLIT_SEED = VAL_CARVE_SEED
WATER_SPLIT_SEED = VAL_CARVE_SEED + 1

# Substring match from the group_DRO construction. "tern" also matches "bittern".
WATER_BIRD_TOKENS = (
    "Albatross",
    "Auklet",
    "Cormorant",
    "Frigatebird",
    "Fulmar",
    "Gull",
    "Jaeger",
    "Kittiwake",
    "Pelican",
    "Puffin",
    "Tern",
    "Gadwall",
    "Grebe",
    "Mallard",
    "Merganser",
    "Guillemot",
    "Pacific_Loon",
)


def is_waterbird(image_path: Path) -> bool:
    """1 if the CUB folder name contains a water-bird token."""
    species = Path(image_path).parent.name.split(".", 1)[-1].lower()
    return any(token.lower() in species for token in WATER_BIRD_TOKENS)


def split_backgrounds(names: Sequence[str], seed: int) -> dict[str, list[str]]:
    """Disjoint train/val/test. Val and test are each ``len // 10``."""
    names = sorted(names)
    if len(names) < 10:
        raise ValueError(f"need at least 10 backgrounds, got {len(names)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(len(names), generator=generator).tolist()
    ordered = [names[i] for i in perm]
    n_test = len(ordered) // 10
    n_val = len(ordered) // 10
    return {
        "test": ordered[:n_test],
        "val": ordered[n_test:n_test + n_val],
        "train": ordered[n_test + n_val:],
    }


def place_pools(label_file: Path) -> dict[str, list[str]]:
    """Filenames of the land and water validation images."""
    land: list[str] = []
    water: list[str] = []
    for line in Path(label_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, idx_text = line.split()
        idx = int(idx_text)
        if idx in LAND_CATEGORY_IDS:
            land.append(name)
        elif idx in WATER_CATEGORY_IDS:
            water.append(name)
    if len(land) != 200 or len(water) != 200:
        raise RuntimeError(
            f"expected 200 land and 200 water Places images, got {len(land)} and {len(water)}"
        )
    return {"land": land, "water": water}


def check_category_names(categories_file: Path) -> None:
    """The hardcoded ids must still name the four official categories."""
    expected = {
        36: "bamboo_forest",
        150: "broadleaf",
        205: "lake/natural",
        243: "ocean",
    }
    found: dict[int, str] = {}
    for line in Path(categories_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, idx_text = line.rsplit(maxsplit=1)
        found[int(idx_text)] = name
    for idx, needle in expected.items():
        if needle not in found.get(idx, ""):
            raise RuntimeError(f"Places category {idx} is {found.get(idx)!r}, expected {needle}")


def crop_and_resize(source: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Center-crop ``source`` to the target aspect, then resize to (width, height)."""
    target_width, target_height = target_size
    source = source.convert("RGB")
    source_width, source_height = source.size
    if source_width < target_width or source_height < target_height:
        width_resize = (
            target_width,
            max(1, int(round(target_width / source_width * source_height))),
        )
        if width_resize[0] >= target_width and width_resize[1] >= target_height:
            source = source.resize(width_resize, Image.Resampling.LANCZOS)
        else:
            height_resize = (
                max(1, int(round(target_height / source_height * source_width))),
                target_height,
            )
            source = source.resize(height_resize, Image.Resampling.LANCZOS)
        return crop_and_resize(source, target_size)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        box = (offset, 0, source_width - offset, source_height)
    else:
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        box = (0, offset, source_width, source_height - offset)
    return source.crop(box).resize((target_width, target_height), Image.Resampling.LANCZOS)


def composite_foreground(
    bird: Image.Image,
    mask: np.ndarray,
    background: Image.Image,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep bird pixels and replace the rest with ``background``.

    ``mask`` is H×W, 1 on the bird. Values above 1 are treated as 0–255.
    Returns image (3, H, W) and mask (H, W), both float in [0, 1], at the
    bird's resolution.
    """
    bird = bird.convert("RGB")
    mask_np = np.asarray(mask).astype(np.float32)
    if mask_np.ndim != 2 or mask_np.shape != (bird.size[1], bird.size[0]):
        raise ValueError(
            f"mask shape {mask_np.shape} does not match bird {bird.size[0]}×{bird.size[1]}"
        )
    if float(mask_np.max()) > 1.0:
        mask_np = mask_np / 255.0
    background = crop_and_resize(background, bird.size)
    bird_np = np.asarray(bird).astype(np.float32) / 255.0
    bg_np = np.asarray(background).astype(np.float32) / 255.0
    out = bird_np * mask_np[..., None] + bg_np * (1.0 - mask_np[..., None])
    image = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).contiguous()
    mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).contiguous()
    return image.clamp(0, 1), mask_t.clamp(0, 1)


def resize_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = F.interpolate(
        image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)
    mask = F.interpolate(
        mask.view(1, 1, *mask.shape), size=(size, size), mode="nearest"
    ).view(size, size)
    return image.clamp(0, 1), mask.clamp(0, 1)


class WaterbirdsPairs(Dataset):
    """One item is the land view, the water view, the bird mask, and the label.

    ``label`` is 1 for a water bird and 0 for a land bird. Both views share it.
    """

    def __init__(
        self,
        split: str = "val",
        cub_root: Path | None = None,
        places_root: Path | None = None,
        image_size: int = 224,
        final: bool = False,
        config_hash: str | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.split = split
        self.image_size = int(image_size)
        self.cub_root = Path(cub_root) if cub_root else REPO / "data" / "CUB_200_2011"
        self.places_root = Path(places_root) if places_root else REPO / "data" / "places365"
        self.birds = CubBirds(self.cub_root)
        self.indices = load_indices(
            "cub200",
            split,
            final=final,
            config_hash=config_hash,
            n_items=len(self.birds),
        )
        categories = self.places_root / "categories_places365.txt"
        if categories.is_file():
            check_category_names(categories)
        pools = place_pools(self.places_root / "places365_val.txt")
        land = split_backgrounds(pools["land"], LAND_SPLIT_SEED)[split]
        water = split_backgrounds(pools["water"], WATER_SPLIT_SEED)[split]
        image_dir = self.places_root / "backgrounds" / "val_256"
        self.land_paths = [image_dir / name for name in land]
        self.water_paths = [image_dir / name for name in water]
        missing = [p for p in self.land_paths + self.water_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} Places backgrounds missing, first: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        cub_index = self.indices[index]
        path, _fine = self.birds.samples[cub_index]
        rel = path.relative_to(self.birds.root / "images")
        mask_img = Image.open(self.birds.root / "segmentations" / rel.with_suffix(".png"))
        mask = np.asarray(mask_img.convert("L")).astype(np.float32) / 255.0
        bird = Image.open(path).convert("RGB")
        land_bg = Image.open(self.land_paths[index % len(self.land_paths)])
        water_bg = Image.open(self.water_paths[index % len(self.water_paths)])
        land, mask_t = resize_pair(
            *composite_foreground(bird, mask, land_bg), self.image_size
        )
        water, _same_mask = resize_pair(
            *composite_foreground(bird, mask, water_bg), self.image_size
        )
        return {
            "land": land,
            "water": water,
            "mask": mask_t,
            "label": int(is_waterbird(path)),
        }


# Train uses the group_DRO correlation. Val is balanced across backgrounds.
# The draw is fixed, so model seeds change initialization and not the images.
CONFOUNDER_STRENGTH = 0.95
CONFOUNDER_SEED = VAL_CARVE_SEED


def confounder_uses_water(labels: Sequence[int], split: str, seed: int = CONFOUNDER_SEED) -> list[bool]:
    """Whether each image is composited on water.

    Training puts a water bird on water with probability ``CONFOUNDER_STRENGTH``.
    Val uses 0.5. ``seed`` does not follow the model seed.
    """
    if split == "train":
        strength = CONFOUNDER_STRENGTH
    elif split == "val":
        strength = 0.5
    else:
        raise ValueError("confounder assignment is only defined for train and val")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    draw = torch.rand(len(labels), generator=generator)
    return [
        bool(draw[i] < (strength if int(label) == 1 else 1.0 - strength))
        for i, label in enumerate(labels)
    ]


class WaterbirdsClassifier(Dataset):
    """One correlated view per bird, for training a model that can use the background."""

    def __init__(self, split: str = "val", image_size: int = 224, **kwargs):
        self.pairs = WaterbirdsPairs(split=split, image_size=image_size, **kwargs)
        self.labels = [
            int(is_waterbird(self.pairs.birds.samples[cub_index][0]))
            for cub_index in self.pairs.indices
        ]
        self.use_water = confounder_uses_water(self.labels, split)
        self.groups = [label * 2 + int(water) for label, water in zip(self.labels, self.use_water)]
        self.targets = self.labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.pairs[index]
        image = item["water"] if self.use_water[index] else item["land"]
        return image, self.labels[index]


def env_batch_waterbirds(land: torch.Tensor, water: torch.Tensor) -> EnvBatch:
    """``land`` and ``water`` are (B, 3, H, W)."""
    return EnvBatch(xs=[land, water], env_ids=["land", "water"])
