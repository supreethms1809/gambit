"""Images for one paper cell.

The sample for seed ``s`` is a seeded draw with generator seed ``1000 + s``
(EVAL_PLAN.md section 2.4), the same for every method. The test split goes
through the same lock as every other loader, so ``split="test"`` raises until
the plan is frozen and ``final`` is set with the frozen config hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from core.types import EnvBatch

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"
IMAGE_SIZE = 224
SAMPLE_SEED_OFFSET = 1000


def sample_seed(seed: int) -> int:
    return SAMPLE_SEED_OFFSET + int(seed)


@dataclass
class ContrastiveSample:
    x: torch.Tensor          # (B, 3, 224, 224), raw [0, 1]
    labels: torch.Tensor     # (B,)
    index: list[int]         # index into the split's dataset


@dataclass
class ShiftSample:
    x_id: torch.Tensor                   # (B, 3, 224, 224)
    env: EnvBatch                        # xs[0] is x_id
    labels: torch.Tensor                 # (B,)
    index: list[int]
    foreground: Optional[torch.Tensor] = None   # (B, H, W) ground truth, if the unit has one
    notes: dict = field(default_factory=dict)


def contrastive_sample(
    dataset: str,
    split: str,
    n: int,
    seed: int,
    *,
    final: bool = False,
    config_hash: Optional[str] = None,
) -> ContrastiveSample:
    from scripts.ablation_contrastive import _get_eval_loader

    loader, _ = _get_eval_loader(
        dataset, max(1, n), DATA_ROOT, image_size=IMAGE_SIZE,
        seed=sample_seed(seed), num_images=n, split=split,
        final=final, config_hash=config_hash,
    )
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(torch.as_tensor(y))
    index = root_indices(loader.dataset)
    return ContrastiveSample(x=torch.cat(xs)[:n], labels=torch.cat(ys)[:n].long(), index=index[:n])


def root_indices(ds) -> list[int]:
    """Indices into the underlying dataset, through any nesting of Subsets.

    The eval loader is Subset(Subset(dataset, split), sample); a record's
    image_index must name the image in the dataset, not its rank in the sample.
    """
    from torch.utils.data import Subset

    if not isinstance(ds, Subset):
        return list(range(len(ds)))
    inner = root_indices(ds.dataset)
    return [inner[i] for i in ds.indices]


def _resize(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == IMAGE_SIZE and x.shape[-2] == IMAGE_SIZE:
        return x
    return F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).clamp(0, 1)


def _draw(length: int, n: int, seed: int) -> list[int]:
    from evaluation.sampling import seeded_indices

    return seeded_indices(length, n, sample_seed(seed))


def shift_sample(
    dataset: str,
    split: str,
    n: int,
    seed: int,
    *,
    final: bool = False,
    config_hash: Optional[str] = None,
) -> ShiftSample:
    lock = dict(final=final, config_hash=config_hash)
    if dataset == "waterbirds":
        return _waterbirds(split, n, seed, lock)
    if dataset == "imagenet9":
        return _imagenet9(split, n, seed, lock)
    if dataset == "planted_patch":
        return _planted(split, n, seed, lock)
    if dataset == "colored_mnist":
        return _colored_mnist(split, n, seed, lock)
    if dataset == "stanford_dogs":
        return _dogs(split, n, seed, lock)
    raise ValueError(f"no shift loader for {dataset!r}")


def _waterbirds(split: str, n: int, seed: int, lock: dict) -> ShiftSample:
    from instantiations.shift.waterbirds import WaterbirdsPairs

    ds = WaterbirdsPairs(split=split, image_size=IMAGE_SIZE, **lock)
    index = _draw(len(ds), n, seed)
    items = [ds[i] for i in index]
    labels = torch.tensor([int(it["label"]) for it in items])
    land = torch.stack([it["land"] for it in items])
    water = torch.stack([it["water"] for it in items])
    # In-distribution is the background the training correlation pairs with the
    # label: water birds on water, land birds on land.
    is_water = labels.bool().view(-1, 1, 1, 1)
    x_id = torch.where(is_water, water, land)
    x_ood = torch.where(is_water, land, water)
    mask = torch.stack([it["mask"].reshape(IMAGE_SIZE, IMAGE_SIZE) for it in items])
    return ShiftSample(
        x_id=x_id, env=EnvBatch(xs=[x_id, x_ood], env_ids=["id", "ood"]),
        labels=labels, index=index, foreground=mask,
    )


def _imagenet9(split: str, n: int, seed: int, lock: dict) -> ShiftSample:
    from instantiations.shift.imagenet9 import ImageNet9Pairs, env_batch_imagenet9

    ds = ImageNet9Pairs(split=split, image_size=IMAGE_SIZE, **lock)
    index = _draw(len(ds), n, seed)
    items = [ds[i] for i in index]
    views = {k: torch.stack([it[k] for it in items]) for k in ("original", "mixed_same", "mixed_rand")}
    env = env_batch_imagenet9(views["original"], views["mixed_same"], views["mixed_rand"])
    labels = torch.tensor([int(it["label"]) for it in items])
    return ShiftSample(x_id=views["original"], env=env, labels=labels, index=index,
                       notes={"foreground": "not provided by ImageNet9Pairs"})


def _planted(split: str, n: int, seed: int, lock: dict) -> ShiftSample:
    from instantiations.shift.planted_patch import PlantedPatchCIFAR, env_batch_planted

    ds = PlantedPatchCIFAR(split=split, image_size=IMAGE_SIZE, seed=0, **lock)
    index = _draw(len(ds), n, seed)
    items = [ds[i] for i in index]
    views = {k: torch.stack([it[k] for it in items]) for k in ("present", "moved", "removed")}
    patches = torch.stack([
        (it["mask_a"].reshape(IMAGE_SIZE, IMAGE_SIZE) + it["mask_b"].reshape(IMAGE_SIZE, IMAGE_SIZE)).clamp(0, 1)
        for it in items
    ])
    labels = torch.tensor([int(it["label"]) for it in items])
    return ShiftSample(
        x_id=views["present"], env=env_batch_planted(views["present"], views["moved"], views["removed"]),
        labels=labels, index=index, foreground=1.0 - patches,
        notes={"foreground": "complement of the planted patches"},
    )


def _colored_mnist(split: str, n: int, seed: int, lock: dict) -> ShiftSample:
    from evaluation.splits import load_indices, load_spec
    from instantiations.shift.biased_data import ColoredMNIST, env_batch_colored_mnist
    from scripts.train_backbone import COLORED_MNIST_CORRELATION, SHIFT_DATA_SEED

    train_pool = load_spec("mnist")["roots"][split] == "train"
    base = ColoredMNIST(root=str(DATA_ROOT), train=train_pool, download=False,
                        correlation=COLORED_MNIST_CORRELATION, seed=SHIFT_DATA_SEED)
    pool = load_indices("mnist", split, n_items=len(base), **lock)
    pick = _draw(len(pool), n, seed)
    index = [pool[i] for i in pick]
    xs, ys = zip(*(base[i] for i in index))
    x = torch.stack(xs)
    y = torch.tensor([int(v) for v in ys])
    env = env_batch_colored_mnist(x, y)
    env = EnvBatch(xs=[_resize(v) for v in env.xs], env_ids=env.env_ids)
    return ShiftSample(x_id=env.xs[0], env=env, labels=y, index=index,
                       notes={"foreground": "colour shortcut is not spatial"})


def _dogs(split: str, n: int, seed: int, lock: dict) -> ShiftSample:
    """Stanford Dogs restyled outside the box, restricted to the paper split.

    ``eval_robust_shortcut_dogs.DogsBoxDataset`` reads every annotated image,
    which would mix training images into the unit. Here the split's ImageFolder
    indices are mapped to it by path. Styles come from train images only.
    """
    from torchvision.datasets import ImageFolder

    from evaluation.splits import load_indices, load_spec
    from scripts.eval_robust_shortcut_dogs import DogsBoxDataset, make_env_fn

    spec = load_spec("stanford_dogs")
    folder = ImageFolder(root=str(DATA_ROOT / spec["roots"][split]))
    pool = load_indices("stanford_dogs", split, n_items=len(folder), **lock)
    boxes = DogsBoxDataset(size=IMAGE_SIZE)
    by_path = {str(Path(p).resolve()): i for i, (p, _c, _b) in enumerate(boxes.samples)}
    annotated = [i for i in pool if str(Path(folder.samples[i][0]).resolve()) in by_path]
    pick = _draw(len(annotated), n, seed)
    index = [annotated[i] for i in pick]
    items = [boxes[by_path[str(Path(folder.samples[i][0]).resolve())]] for i in index]
    x = torch.stack([it[0] for it in items])
    labels = torch.tensor([int(it[1]) for it in items])
    box = torch.stack([it[2] for it in items])

    styles = _dog_styles(boxes, by_path, folder, lock)
    restyled = make_env_fn(styles)(x, box)
    env = EnvBatch(xs=[x, *restyled.xs], env_ids=["id", *restyled.env_ids])
    return ShiftSample(x_id=x, env=env, labels=labels, index=index, foreground=box)


def _dog_styles(boxes, by_path, folder, lock) -> torch.Tensor:
    """Three background styles from k-means over train-split images. Cached."""
    from torch.utils.data import Subset

    from evaluation.splits import load_indices
    from scripts.eval_robust_shortcut_dogs import background_styles

    cache = REPO / "results" / "paper" / "shift_pairs" / "dogs_styles_train.pt"
    if cache.is_file():
        return torch.load(cache, weights_only=True)
    train = load_indices("stanford_dogs", "train", n_items=len(folder))
    members = [by_path[p] for p in (str(Path(folder.samples[i][0]).resolve()) for i in train) if p in by_path]
    styles = background_styles(Subset(boxes, members), n_sample=400, k=3, seed=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(styles, cache)
    return styles
