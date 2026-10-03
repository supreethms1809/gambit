"""Write data/splits/<dataset>.json for the paper protocol.

Train is what optimisation sees. Val is checkpoint selection and hyperparameters.
Test is the official test set, or the historical holdout, and the loader refuses
it until EVAL_PLAN.md is frozen.

Pets and dogs: test is the complement of train_backbone's old
``random_split(seed=42)`` 80/20 cut. Val is 10% of that train pool, carved with
seed 43, so the historical holdout is unchanged.

HAM10000: the current lesion-grouped ``val`` folder is test. Val is a
lesion-grouped 10% carve of ``train``.

Brain tumor: the current patient-grouped ``Testing`` folder is test. Val is a
patient-grouped 10% carve of ``Training``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import argparse

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.splits import (  # noqa: E402
    HOLDOUT_SEED,
    VAL_CARVE_SEED,
    VAL_FRACTION_OF_TRAIN,
    carve_grouped,
    imagefolder_three_way,
    torchvision_train_val,
    write_spec,
)

DATA = REPO / "data"


def _spec(dataset: str, protocol: str, roots: dict, indices: dict, lengths: dict) -> dict:
    return {
        "dataset": dataset,
        "protocol": protocol,
        "holdout_seed": HOLDOUT_SEED,
        "val_carve_seed": VAL_CARVE_SEED,
        "val_fraction_of_train": VAL_FRACTION_OF_TRAIN,
        "roots": roots,
        "indices": indices,
        "index_length": lengths,
    }


def _torchvision(name: str) -> dict:
    from torchvision.datasets import CIFAR10, MNIST

    if name == "cifar10":
        train = CIFAR10(root=str(DATA), train=True, download=False)
        test = CIFAR10(root=str(DATA), train=False, download=False)
    elif name == "mnist":
        train = MNIST(root=str(DATA), train=True, download=False)
        test = MNIST(root=str(DATA), train=False, download=False)
    else:
        raise ValueError(name)
    carved = torchvision_train_val(len(train), seed=HOLDOUT_SEED)
    return _spec(
        name,
        "official test set; 10% of the official train set is val",
        {"train": "train", "val": "train", "test": "test"},
        {"train": carved["train"], "val": carved["val"], "test": list(range(len(test)))},
        {"train": len(train), "val": len(train), "test": len(test)},
    )


def _imagefolder(name: str, relative: str) -> dict:
    from torchvision.datasets import ImageFolder

    root = DATA / relative
    ds = ImageFolder(str(root))
    splits = imagefolder_three_way(len(ds), seed=HOLDOUT_SEED)
    return _spec(
        name,
        "test is the complement of random_split(seed=42, 80%); val is 10% of that train pool",
        {"train": relative, "val": relative, "test": relative},
        splits,
        {split: len(ds) for split in ("train", "val", "test")},
    )


def _ham() -> dict:
    from torchvision.datasets import ImageFolder

    train_ds = ImageFolder(str(DATA / "ham10000" / "train"))
    test_ds = ImageFolder(str(DATA / "ham10000" / "val"))
    lesion_of = _ham_lesion_ids()
    group_of = {}
    class_of = {}
    missing = 0
    for index, (path, label) in enumerate(train_ds.samples):
        stem = Path(path).stem
        group = lesion_of.get(stem)
        if group is None:
            missing += 1
            group = f"image:{stem}"
        group_of[index] = group
        class_of[index] = train_ds.classes[label]
    if missing:
        print(f"WARNING: {missing} HAM train images had no lesion_id; each is its own group")
    train_idx, val_idx = carve_grouped(
        list(range(len(train_ds))),
        group_of,
        val_fraction=VAL_FRACTION_OF_TRAIN,
        seed=VAL_CARVE_SEED,
        by_class=class_of,
    )
    return _spec(
        "ham10000",
        "current val/ folder is test; val is a lesion-grouped carve of train/",
        {
            "train": "ham10000/train",
            "val": "ham10000/train",
            "test": "ham10000/val",
        },
        {"train": train_idx, "val": val_idx, "test": list(range(len(test_ds)))},
        {"train": len(train_ds), "val": len(train_ds), "test": len(test_ds)},
    )


def _ham_lesion_ids() -> dict[str, str]:
    csv_path = DATA / "ham10000_raw" / "HAM10000_metadata.csv"
    out = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["image_id"]] = row["lesion_id"]
    return out


def _brain() -> dict:
    from torchvision.datasets import ImageFolder

    train_ds = ImageFolder(str(DATA / "brain_tumor" / "Training"))
    test_ds = ImageFolder(str(DATA / "brain_tumor" / "Testing"))
    pid_of = _brain_patient_ids()
    group_of = {}
    class_of = {}
    missing = 0
    for index, (path, label) in enumerate(train_ds.samples):
        stem = Path(path).stem
        group = pid_of.get(stem)
        if group is None:
            missing += 1
            group = f"slice:{stem}"
        group_of[index] = group
        class_of[index] = train_ds.classes[label]
    if missing:
        print(f"WARNING: {missing} brain train slices had no patient id; each is its own group")
    train_idx, val_idx = carve_grouped(
        list(range(len(train_ds))),
        group_of,
        val_fraction=VAL_FRACTION_OF_TRAIN,
        seed=VAL_CARVE_SEED,
        by_class=class_of,
    )
    return _spec(
        "brain_tumor",
        "current Testing/ folder is test; val is a patient-grouped carve of Training/",
        {
            "train": "brain_tumor/Training",
            "val": "brain_tumor/Training",
            "test": "brain_tumor/Testing",
        },
        {"train": train_idx, "val": val_idx, "test": list(range(len(test_ds)))},
        {"train": len(train_ds), "val": len(train_ds), "test": len(test_ds)},
    )


def _brain_patient_ids() -> dict[str, str]:
    import h5py

    mats = DATA / "brain_tumor_raw" / "mats"
    out = {}
    paths = sorted(mats.glob("*.mat"))
    for i, path in enumerate(paths, start=1):
        with h5py.File(path, "r") as handle:
            pid = "".join(chr(c[0]) for c in handle["cjdata"]["PID"][:])
        out[path.stem] = pid
        if i % 500 == 0:
            print(f"  read {i}/{len(paths)} patient ids")
    return out


def _per_class_torchvision(name: str, train_ds, test_ds, protocol: str, roots: dict) -> dict:
    from evaluation.paper_datasets import label_list, require_every_class
    from evaluation.splits import per_class_val_carve

    train_labels = label_list(train_ds)
    test_labels = label_list(test_ds)
    n_classes = len(set(train_labels))
    label_of = {i: train_labels[i] for i in range(len(train_labels))}
    train_idx, val_idx = per_class_val_carve(range(len(train_ds)), label_of)
    test_idx = list(range(len(test_ds)))
    require_every_class(train_labels, train_idx, n_classes, "train")
    require_every_class(train_labels, val_idx, n_classes, "val")
    require_every_class(test_labels, test_idx, n_classes, "test")
    return _spec(
        name,
        protocol,
        roots,
        {"train": train_idx, "val": val_idx, "test": test_idx},
        {"train": len(train_ds), "val": len(train_ds), "test": len(test_ds)},
    )


def _cifar100() -> dict:
    from torchvision.datasets import CIFAR100

    train = CIFAR100(root=str(DATA), train=True, download=False)
    test = CIFAR100(root=str(DATA), train=False, download=False)
    return _per_class_torchvision(
        "cifar100",
        train,
        test,
        "official test set; val is 10% of each class in the official train set",
        {"train": "train", "val": "train", "test": "test"},
    )


def _oxford_pets() -> dict:
    from evaluation.paper_datasets import open_unsplit

    train = open_unsplit("oxford_pets", "train", DATA)
    test = open_unsplit("oxford_pets", "test", DATA)
    return _per_class_torchvision(
        "oxford_pets",
        train,
        test,
        "official test.txt; val is 10% of each breed in trainval.txt. This is the 37-breed set, not cats-vs-dogs.",
        {"train": "oxford-iiit-pet", "val": "oxford-iiit-pet", "test": "oxford-iiit-pet"},
    )


def _cub200() -> dict:
    from evaluation.paper_datasets import cub_train_flags, label_list, open_unsplit, require_every_class
    from evaluation.splits import per_class_val_carve

    ds = open_unsplit("cub200", "train", DATA)
    labels = label_list(ds)
    flags = cub_train_flags(DATA)
    if len(flags) != len(ds):
        raise RuntimeError(f"CUB split file has {len(flags)} rows, images list has {len(ds)}")
    train_pool = [i for i in range(len(ds)) if flags[i + 1] == 1]
    test_idx = [i for i in range(len(ds)) if flags[i + 1] == 0]
    label_of = {i: labels[i] for i in train_pool}
    train_idx, val_idx = per_class_val_carve(train_pool, label_of)
    n_classes = 200
    require_every_class(labels, train_idx, n_classes, "train")
    require_every_class(labels, val_idx, n_classes, "val")
    require_every_class(labels, test_idx, n_classes, "test")
    return _spec(
        "cub200",
        "official train_test_split.txt; val is 10% of each class inside the official train images",
        {"train": "CUB_200_2011", "val": "CUB_200_2011", "test": "CUB_200_2011"},
        {"train": train_idx, "val": val_idx, "test": test_idx},
        {"train": len(ds), "val": len(ds), "test": len(ds)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write data/splits/<dataset>.json")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Dataset names to write. Default: the original six plus the S07 sets.",
    )
    args = parser.parse_args()
    jobs = [
        ("mnist", lambda: _torchvision("mnist")),
        ("cifar10", lambda: _torchvision("cifar10")),
        ("pets", lambda: _imagefolder("pets", "PetImages")),
        ("stanford_dogs", lambda: _imagefolder("stanford_dogs", "stanford_dogs/images/Images")),
        ("ham10000", _ham),
        ("brain_tumor", _brain),
        ("cifar100", _cifar100),
        ("oxford_pets", _oxford_pets),
        ("cub200", _cub200),
    ]
    if args.only:
        wanted = set(args.only)
        jobs = [(name, build) for name, build in jobs if name in wanted]
        missing = wanted - {name for name, _ in jobs}
        if missing:
            raise SystemExit(f"unknown dataset: {sorted(missing)}")
    for name, build in jobs:
        print(f"writing {name}")
        spec = build()
        path = write_spec(name, spec)
        counts = {split: len(spec["indices"][split]) for split in ("train", "val", "test")}
        print(f"  {path}  {counts}")


if __name__ == "__main__":
    main()
