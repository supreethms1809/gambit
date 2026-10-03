"""
scripts/prepare_brain_tumor.py

Convert the Cheng et al. brain tumor dataset (figshare, CC BY 4.0) into the
class-per-folder layout the CDEA contrastive pipeline expects.

The raw release is 3,064 MATLAB v7.3 ``.mat`` files, one per slice, each holding a
``cjdata`` struct with:

    PID          patient identifier (char codes)
    image        512x512 int16 MRI slice, per-image intensity range
    label        1 = meningioma, 2 = glioma, 3 = pituitary
    tumorMask    512x512 binary tumor mask
    tumorBorder  boundary coordinates

Two things this buys over the Kaggle repackaging of the same data:

- **Patient-grouped splits.** The 3,064 slices come from only 233 patients, with
  many slices per patient. Splitting per-slice puts adjacent slices of the same
  tumor in both train and test, which inflates accuracy badly. Whole patients are
  assigned to one split here.
- **Tumor masks.** Written alongside the images so ``scripts/eval_localization.py``
  can score mask localization, exactly as it does for HAM10000.

Note this release has no "no tumor" class — it is a 3-way tumor-type problem.

Expected raw layout::

    data/brain_tumor_raw/mats/*.mat        (1.mat ... 3064.mat)

Usage (from repo root)::

    PYTHONPATH=. python scripts/prepare_brain_tumor.py --resize 224
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent

DEFAULT_RAW = REPO / "data" / "brain_tumor_raw" / "mats"
DEFAULT_OUT = REPO / "data" / "brain_tumor"
DEFAULT_MASKS = REPO / "data" / "brain_tumor_raw" / "masks"

# Label codes are defined by the dataset's own README.
LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def _read_mat(path: Path):
    """Return (pid, label, image_uint8, mask_uint8) from one cjdata .mat file."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        d = f["cjdata"]
        pid = "".join(chr(c[0]) for c in d["PID"][:])
        label = int(d["label"][0][0])
        img = np.array(d["image"], dtype=np.float64)
        mask = (np.array(d["tumorMask"]) > 0).astype("uint8") * 255

    # Slices have per-image intensity ranges (int16, not windowed), so scale each
    # to 0-255 independently. Without this the darker scans wash out to near-black.
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo) if hi > lo else img * 0.0
    return pid, label, (img * 255).astype("uint8"), mask


def prepare(
    raw_root: Path,
    out_root: Path,
    mask_root: Path,
    val_frac: float = 0.2,
    resize: int | None = 224,
    seed: int = 0,
    force: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Build data/brain_tumor/{Training,Testing}/<class>/. Returns per-split counts."""
    from PIL import Image

    mats = sorted(raw_root.glob("*.mat"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not mats:
        raise FileNotFoundError(
            f"No .mat files under {raw_root}. Download the figshare release and unzip it "
            f"there (see docs/MEDICAL_DATASETS.md)."
        )
    if out_root.exists():
        if not force:
            raise FileExistsError(f"{out_root} already exists. Pass --force to rebuild it.")
        shutil.rmtree(out_root)
    if mask_root.exists() and force:
        shutil.rmtree(mask_root)
    mask_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading {len(mats)} .mat files from {raw_root}")
    records = []
    for i, p in enumerate(mats):
        pid, label, img, mask = _read_mat(p)
        records.append((p.stem, pid, label, img, mask))
        if (i + 1) % 500 == 0:
            print(f"  read {i + 1}/{len(mats)}")

    # Group slices by (class, patient) so no patient spans both splits.
    groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (_stem, pid, label, _img, _mask) in enumerate(records):
        groups[LABELS[label]][pid].append(idx)

    n_patients = len({pid for _s, pid, _l, _i, _m in records})
    print(f"{len(records)} slices from {n_patients} patients across {len(groups)} classes")

    rng = random.Random(seed)
    counts: Dict[str, Dict[str, int]] = {"Training": {}, "Testing": {}}

    for cls in sorted(groups):
        pids = sorted(groups[cls])
        rng.shuffle(pids)
        n_val = max(1, round(val_frac * len(pids)))
        n_val = min(n_val, len(pids) - 1) if len(pids) > 1 else 0
        split_of = {pid: ("Testing" if i < n_val else "Training") for i, pid in enumerate(pids)}

        for split in ("Training", "Testing"):
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)
            counts[split][cls] = 0

        for pid, split in split_of.items():
            for idx in groups[cls][pid]:
                stem, _pid, _label, img, mask = records[idx]
                im = Image.fromarray(img).convert("L")
                mk = Image.fromarray(mask).convert("L")
                if resize is not None:
                    im = im.resize((resize, resize), Image.BILINEAR)
                    mk = mk.resize((resize, resize), Image.NEAREST)
                im.save(out_root / split / cls / f"{stem}.png")
                mk.save(mask_root / f"{stem}_segmentation.png")
                counts[split][cls] += 1
        print(f"  {cls:<12} {len(pids):>4} patients -> "
              f"{counts['Training'][cls]:>5} train / {counts['Testing'][cls]:>4} test slices")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Cheng et al. brain tumor dataset")
    parser.add_argument("--raw_root", type=str, default=str(DEFAULT_RAW))
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--mask_root", type=str, default=str(DEFAULT_MASKS),
                        help="Where tumor masks are written, for eval_localization.py")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of each class's PATIENTS held out for testing")
    parser.add_argument("--resize", type=int, default=224,
                        help="Resize slices to NxN on disk (set 0 to keep native 512)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.val_frac < 1.0):
        parser.error("--val_frac must be in (0, 1)")

    counts = prepare(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        mask_root=Path(args.mask_root),
        val_frac=args.val_frac,
        resize=args.resize if args.resize and args.resize > 0 else None,
        seed=args.seed,
        force=args.force,
    )

    print(f"\nWrote {args.out_root}")
    print(f"{'class':<12}{'train':>8}{'test':>8}")
    for cls in sorted(counts["Training"]):
        print(f"{cls:<12}{counts['Training'][cls]:>8}{counts['Testing'][cls]:>8}")
    print(f"{'total':<12}{sum(counts['Training'].values()):>8}{sum(counts['Testing'].values()):>8}")


if __name__ == "__main__":
    sys.exit(main())
