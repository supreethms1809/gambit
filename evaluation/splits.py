"""Train / val / test index files, and the lock that keeps test data out of decisions.

Training uses train. Checkpoint selection and every hyperparameter use val.
Reported numbers use test, and only after ``EVAL_PLAN.md`` is frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO / "data" / "splits"
EVAL_PLAN_PATH = REPO / "docs" / "paper" / "EVAL_PLAN.md"
FROZEN_MARKER = "eval-plan-frozen"

# Historical pets/dogs training split in scripts/train_backbone.py.
HOLDOUT_SEED = 42
HOLDOUT_TRAIN_FRACTION = 0.8
# Carved from the train pool only, so the historical holdout stays the test set.
VAL_FRACTION_OF_TRAIN = 0.1
VAL_CARVE_SEED = HOLDOUT_SEED + 1


class SplitLockedError(RuntimeError):
    """Raised when a loader is asked for the test split before the plan is frozen."""


class MissingSplitError(FileNotFoundError):
    """Raised when a dataset has no recorded train/val/test file."""


def plan_is_frozen(text: str) -> bool:
    """True only when the plan carries the freeze marker and does not say otherwise.

    A draft that names the marker while saying it is not frozen stays locked.
    """
    if FROZEN_MARKER not in text:
        return False
    if "not frozen" in text:
        return False
    return True


def assert_split_allowed(
    split: str,
    *,
    final: bool,
    config_hash: Optional[str],
    eval_plan_path: Path = EVAL_PLAN_PATH,
) -> None:
    """Refuse ``split='test'`` unless this is a final run of a frozen config."""
    if split != "test":
        return
    if not final:
        raise SplitLockedError("split='test' requires --final")
    if not config_hash:
        raise SplitLockedError("split='test' requires the run's config hash")
    if not eval_plan_path.is_file():
        raise SplitLockedError(f"frozen eval plan not found: {eval_plan_path}")
    text = eval_plan_path.read_text(encoding="utf-8")
    if not plan_is_frozen(text):
        raise SplitLockedError("EVAL_PLAN.md is not frozen")
    if config_hash not in text:
        raise SplitLockedError("config hash is not listed in the frozen EVAL_PLAN.md")


def holdout_complement_indices(
    n: int,
    *,
    seed: int = HOLDOUT_SEED,
    train_fraction: float = HOLDOUT_TRAIN_FRACTION,
) -> list[int]:
    """Indices of the complement of ``random_split(..., seed)``.

    Matches ``torch.utils.data.random_split``: the generator permutation's first
    ``int(train_fraction * n)`` entries are train, and the rest are the holdout.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    n_train = int(train_fraction * n)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    return perm[n_train:]


def carve_val_from_train_pool(
    train_pool: Sequence[int],
    *,
    val_fraction: float = VAL_FRACTION_OF_TRAIN,
    seed: int = VAL_CARVE_SEED,
) -> tuple[list[int], list[int]]:
    """Split a train pool into ``(train, val)``. The pool itself is not reshuffled into test."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    pool = list(train_pool)
    if len(pool) <= 1 or val_fraction == 0.0:
        return pool, []
    n_val = int(round(val_fraction * len(pool)))
    n_val = min(max(n_val, 1), len(pool) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(len(pool), generator=generator).tolist()
    val = [pool[i] for i in order[:n_val]]
    train = [pool[i] for i in order[n_val:]]
    return train, val


def imagefolder_three_way(
    n: int,
    *,
    seed: int = HOLDOUT_SEED,
    train_fraction: float = HOLDOUT_TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION_OF_TRAIN,
    val_seed: int = VAL_CARVE_SEED,
) -> dict[str, list[int]]:
    """Pets/dogs: test is the historical holdout; val is carved from the train pool only."""
    n_train = int(train_fraction * n)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    train_pool, test = perm[:n_train], perm[n_train:]
    train, val = carve_val_from_train_pool(train_pool, val_fraction=val_fraction, seed=val_seed)
    return {"train": train, "val": val, "test": test}


def per_class_val_carve(
    indices: Sequence[int],
    label_of: Mapping[int, str],
    *,
    val_fraction: float = VAL_FRACTION_OF_TRAIN,
    seed: int = VAL_CARVE_SEED,
) -> tuple[list[int], list[int]]:
    """Hold out ``val_fraction`` of the images inside each class.

    Each image is its own group, so the cut is by image. Every class with at
    least two images stays on both sides.
    """
    group_of = {int(i): str(i) for i in indices}
    by_class = {int(i): str(label_of[int(i)]) for i in indices}
    return carve_grouped(
        list(indices),
        group_of,
        val_fraction=val_fraction,
        seed=seed,
        by_class=by_class,
    )


def torchvision_train_val(
    n_train: int,
    *,
    val_fraction: float = VAL_FRACTION_OF_TRAIN,
    seed: int = HOLDOUT_SEED,
) -> dict[str, list[int]]:
    """Official train set, with ``val_fraction`` held out. Test is the official test set."""
    if n_train < 2:
        raise ValueError("n_train must be >= 2")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_train, generator=generator).tolist()
    n_val = int(round(val_fraction * n_train))
    n_val = min(max(n_val, 1), n_train - 1)
    return {"train": perm[n_val:], "val": perm[:n_val]}


def carve_grouped(
    indices: Sequence[int],
    group_of: Mapping[int, str],
    *,
    val_fraction: float = VAL_FRACTION_OF_TRAIN,
    seed: int = VAL_CARVE_SEED,
    by_class: Optional[Mapping[int, str]] = None,
) -> tuple[list[int], list[int]]:
    """Carve val so that no group id lands in both sides.

    When ``by_class`` is given, the fraction is applied inside each class, which
    keeps every class in both sides when the class has at least two groups.
    """
    if by_class is None:
        classes = {"all": list(indices)}
    else:
        classes = {}
        for idx in indices:
            classes.setdefault(str(by_class[idx]), []).append(idx)

    # Deterministic and independent of dict order: shuffle group ids per class.
    import random

    rng = random.Random(int(seed))
    train: list[int] = []
    val: list[int] = []
    for cls in sorted(classes):
        members = classes[cls]
        groups: dict[str, list[int]] = {}
        for idx in members:
            groups.setdefault(str(group_of[idx]), []).append(idx)
        gids = sorted(groups)
        rng.shuffle(gids)
        if len(gids) <= 1 or val_fraction == 0.0:
            n_val = 0
        else:
            n_val = int(round(val_fraction * len(gids)))
            n_val = min(max(n_val, 1), len(gids) - 1)
        val_g = set(gids[:n_val])
        for gid, idxs in groups.items():
            (val if gid in val_g else train).extend(idxs)
    return sorted(train), sorted(val)


def split_path(dataset: str, splits_dir: Path = SPLITS_DIR) -> Path:
    return splits_dir / f"{dataset}.json"


def load_spec(dataset: str, splits_dir: Path = SPLITS_DIR) -> dict:
    path = split_path(dataset, splits_dir)
    if not path.is_file():
        raise MissingSplitError(
            f"No split file for {dataset} at {path}. "
            "Run PYTHONPATH=. python scripts/write_paper_splits.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_indices(
    dataset: str,
    split: str,
    *,
    final: bool = False,
    config_hash: Optional[str] = None,
    splits_dir: Path = SPLITS_DIR,
    eval_plan_path: Path = EVAL_PLAN_PATH,
    n_items: Optional[int] = None,
) -> list[int]:
    """Indices for ``split``. ``test`` is refused until the eval plan is frozen."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    assert_split_allowed(
        split, final=final, config_hash=config_hash, eval_plan_path=eval_plan_path
    )
    spec = load_spec(dataset, splits_dir)
    indices = spec["indices"][split]
    expected = spec["index_length"][split]
    if n_items is not None and n_items != expected:
        raise RuntimeError(
            f"{dataset} {split} was indexed into a dataset of length {expected}, "
            f"but the loader has length {n_items}. The folder order may have changed."
        )
    return list(indices)


def write_spec(dataset: str, spec: dict, splits_dir: Path = SPLITS_DIR) -> Path:
    splits_dir.mkdir(parents=True, exist_ok=True)
    path = split_path(dataset, splits_dir)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def spec_hash(spec: Mapping) -> str:
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
