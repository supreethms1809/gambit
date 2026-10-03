"""
scripts/prepare_ham10000.py

Convert the raw HAM10000 download into the class-per-folder layout the CDEA
contrastive pipeline expects.

The raw release is a metadata CSV plus two flat image directories, which
``ImageFolder`` cannot read. This script writes::

    data/ham10000/train/<dx>/ISIC_xxxxxxx.jpg
    data/ham10000/val/<dx>/ISIC_xxxxxxx.jpg

Two details matter for the split:

- **Grouped by lesion.** HAM10000 contains multiple images of the same lesion
  (same ``lesion_id``). Splitting per-image leaks near-duplicates into val and
  inflates accuracy, so whole lesion groups are assigned to one split.
- **Stratified by class.** The split is done per diagnosis, so all 7 classes
  appear in both train and val despite the heavy imbalance.

``--max_per_class`` caps each class before splitting, which is the simplest way
to stop the dominant ``nv`` class (~67% of the data) from making every top-K
hypothesis set identical.

Expected raw layout (as downloaded from Harvard Dataverse / Kaggle)::

    data/ham10000_raw/HAM10000_metadata.csv
    data/ham10000_raw/HAM10000_images_part_1/*.jpg
    data/ham10000_raw/HAM10000_images_part_2/*.jpg

Usage (from repo root)::

    PYTHONPATH=. python scripts/prepare_ham10000.py
    PYTHONPATH=. python scripts/prepare_ham10000.py --max_per_class 1000 --link
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent

DEFAULT_RAW = REPO / "data" / "ham10000_raw"
DEFAULT_OUT = REPO / "data" / "ham10000"

METADATA_NAME = "HAM10000_metadata.csv"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _index_images(raw_root: Path) -> Dict[str, Path]:
    """Map image_id -> file path, searching raw_root recursively."""
    index: Dict[str, Path] = {}
    for path in raw_root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def _read_metadata(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")
    return rows


def _emit(src: Path, dst: Path, link: bool, resize: int | None) -> None:
    """Place one image at dst, optionally pre-resized."""
    if resize is not None:
        from PIL import Image
        with Image.open(src) as im:
            im.convert("RGB").resize((resize, resize), Image.BILINEAR).save(dst, quality=95)
    elif link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def prepare(
    raw_root: Path,
    out_root: Path,
    val_frac: float = 0.2,
    max_per_class: int | None = None,
    seed: int = 0,
    link: bool = False,
    resize: int | None = None,
    force: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Build data/ham10000/{train,val}/<dx>/. Returns per-split class counts."""
    csv_path = raw_root / METADATA_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download HAM10000 and place the metadata CSV and "
            f"image folders under {raw_root} (see docs/MEDICAL_DATASETS.md)."
        )

    if out_root.exists():
        if not force:
            raise FileExistsError(
                f"{out_root} already exists. Pass --force to rebuild it from scratch."
            )
        shutil.rmtree(out_root)

    rows = _read_metadata(csv_path)
    images = _index_images(raw_root)
    print(f"Read {len(rows)} metadata rows; indexed {len(images)} image files under {raw_root}")

    # Group image ids by (diagnosis, lesion) so a lesion never spans both splits.
    groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    n_missing = 0
    for row in rows:
        image_id = row["image_id"]
        if image_id not in images:
            n_missing += 1
            continue
        groups[row["dx"]][row["lesion_id"]].append(image_id)
    if n_missing:
        print(f"WARNING: {n_missing} metadata rows had no matching image file (skipped)")
    if not groups:
        raise RuntimeError("No metadata row matched an image file — check the raw layout.")

    rng = random.Random(seed)
    counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}}

    for dx in sorted(groups):
        lesion_ids = sorted(groups[dx])
        rng.shuffle(lesion_ids)

        # Cap per class by taking whole lesion groups until the image budget is hit.
        if max_per_class is not None:
            kept, n_kept = [], 0
            for lesion_id in lesion_ids:
                if n_kept >= max_per_class:
                    break
                kept.append(lesion_id)
                n_kept += len(groups[dx][lesion_id])
            lesion_ids = kept

        # Stratified split: at least one lesion group in each split per class.
        n_val = max(1, round(val_frac * len(lesion_ids)))
        n_val = min(n_val, len(lesion_ids) - 1) if len(lesion_ids) > 1 else 0
        split_of = {lid: ("val" if i < n_val else "train") for i, lid in enumerate(lesion_ids)}

        for split in ("train", "val"):
            (out_root / split / dx).mkdir(parents=True, exist_ok=True)
            counts[split][dx] = 0

        for lesion_id, split in split_of.items():
            for image_id in groups[dx][lesion_id]:
                src = images[image_id]
                _emit(src, out_root / split / dx / src.name, link, resize)
                counts[split][dx] += 1
        n_done = counts["train"][dx] + counts["val"][dx]
        print(f"  {dx:<6} {n_done:>5} images")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 for the CDEA contrastive pipeline")
    parser.add_argument("--raw_root", type=str, default=str(DEFAULT_RAW),
                        help="Directory holding HAM10000_metadata.csv and the image folders")
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT),
                        help="Output root; train/ and val/ are created inside it")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of each class's lesion groups held out for validation")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap images per class before splitting (mitigates the nv imbalance)")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for the split")
    parser.add_argument("--link", action="store_true",
                        help="Symlink instead of copying (saves ~2.5 GB; breaks if raw_root moves)")
    parser.add_argument("--resize", type=int, default=None,
                        help="Pre-resize images to NxN on disk. Strongly recommended: decoding "
                             "600x450 JPEGs every epoch is the training bottleneck. Use the same "
                             "value as the model input size (224 for torchvision CNNs).")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild an existing out_root")
    args = parser.parse_args()

    if not (0.0 < args.val_frac < 1.0):
        parser.error("--val_frac must be in (0, 1)")
    if args.max_per_class is not None and args.max_per_class <= 0:
        parser.error("--max_per_class must be > 0")
    if args.resize is not None and args.resize <= 0:
        parser.error("--resize must be > 0")
    if args.resize is not None and args.link:
        parser.error("--resize writes new files, so it cannot be combined with --link")

    counts = prepare(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        val_frac=args.val_frac,
        max_per_class=args.max_per_class,
        seed=args.seed,
        link=args.link,
        resize=args.resize,
        force=args.force,
    )

    print(f"\nWrote {args.out_root}")
    print(f"{'class':<8}{'train':>8}{'val':>8}")
    for dx in sorted(counts["train"]):
        print(f"{dx:<8}{counts['train'][dx]:>8}{counts['val'][dx]:>8}")
    print(f"{'total':<8}{sum(counts['train'].values()):>8}{sum(counts['val'].values()):>8}")


if __name__ == "__main__":
    sys.exit(main())
