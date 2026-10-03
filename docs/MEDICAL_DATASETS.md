# Medical Image Datasets

Two medical datasets are wired into the contrastive pipeline: **HAM10000** (dermoscopic
skin lesions) and **Brain Tumor MRI**. Neither ships with the repo — download them into
`data/` as described below.

Both use a pre-split `<root>/<split>/<class>/` layout so that train and validation never
share a patient or lesion. The split roots are declared in `MEDICAL_SPLIT_ROOTS`
(`examples/contrastive_explanation.py`); the mirrored constants in
`scripts/train_backbone.py` and `scripts/ablation_contrastive.py` must stay in sync.

---

## HAM10000 — 7 skin lesion classes

10,015 dermoscopic images. The headline contrastive question is *"why melanoma rather
than a benign nevus?"* — a genuine clinical distinction rather than a toy one.

| Folder  | Diagnosis             | Approx. count |
| ------- | --------------------- | ------------- |
| `nv`    | melanocytic nevus     | 6705          |
| `mel`   | melanoma              | 1113          |
| `bkl`   | benign keratosis      | 1099          |
| `bcc`   | basal cell carcinoma  | 514           |
| `akiec` | actinic keratosis     | 327           |
| `vasc`  | vascular lesion       | 142           |
| `df`    | dermatofibroma        | 115           |

### 1. Download

From Harvard Dataverse (`doi:10.7910/DVN/DBW86T`) or the Kaggle mirror. Arrange the raw
files as:

```
data/ham10000_raw/HAM10000_metadata.csv
data/ham10000_raw/HAM10000_images_part_1/*.jpg
data/ham10000_raw/HAM10000_images_part_2/*.jpg
```

### 2. Convert to the pipeline layout

The raw release is a metadata CSV plus flat image directories, which `ImageFolder` cannot
read. Convert it:

```bash
PYTHONPATH=. python scripts/prepare_ham10000.py
```

This writes `data/ham10000/{train,val}/<dx>/`. Two things it handles that a naive split
does not:

- **Lesion grouping.** HAM10000 contains multiple images of the same lesion. Splitting
  per-image leaks near-duplicates into validation and inflates accuracy, so whole
  `lesion_id` groups are assigned to a single split.
- **Class stratification.** The split runs per diagnosis, so all 7 classes appear in both
  splits despite the imbalance.

Useful flags:

```bash
PYTHONPATH=. python scripts/prepare_ham10000.py --max_per_class 1000 --link
```

- `--max_per_class N` caps each class before splitting. Worth using: `nv` is ~67% of the
  data, and left unchecked it dominates every top-K hypothesis set, which makes the
  contrastive explanations repetitive.
- `--link` symlinks instead of copying (saves ~2.5 GB, but breaks if `ham10000_raw` moves).
- `--force` rebuilds an existing output directory.

### 3. Run

```bash
PYTHONPATH=. python scripts/train_backbone.py --dataset ham10000 --epochs 10 --no-freeze-backbone
```

```bash
PYTHONPATH=. python examples/contrastive_explanation.py --dataset ham10000 --train --epochs 10
```

---

## Brain Tumor MRI — 3 classes (Cheng et al.)

3,064 T1-weighted contrast-enhanced slices from **233 patients**, in 3 classes:
`meningioma` (708), `glioma` (1426), `pituitary` (930). This is the figshare original
(CC BY 4.0, credential-free), not the Kaggle repackaging — it keeps the patient IDs and
tumor masks that the Kaggle version strips.

Note there is **no "no tumor" class**: this is a 3-way tumor-type problem.

### 1. Download

From figshare `doi:10.6084/m9.figshare.1512427`, four zips totalling ~880 MB. Unzip all
of them into a single directory:

```
data/brain_tumor_raw/mats/1.mat ... 3064.mat
```

### 2. Convert

The release is MATLAB v7.3 files, so it needs `h5py` (`pip install h5py`) and a
conversion pass:

```bash
PYTHONPATH=. python scripts/prepare_brain_tumor.py --resize 224
```

This writes `data/brain_tumor/{Training,Testing}/<class>/` plus tumor masks to
`data/brain_tumor_raw/masks/`. Two things it handles:

- **Patient-grouped splits.** 3,064 slices come from only 233 patients, and adjacent
  slices of the same tumor are near-identical. A per-slice split puts them on both sides
  and inflates accuracy sharply. Whole patients go to one split.
- **Per-image intensity scaling.** Slices are int16 with per-scan ranges, not windowed,
  so each is normalized to 0–255 independently. Skipping this washes darker scans out.

### 3. Run

```bash
PYTHONPATH=. python examples/contrastive_explanation.py --dataset brain_tumor --train --epochs 20 --lr 1e-4 --pretrained
```

### Caveat: the 7×7 grid is too coarse for these tumors

Measured over all 3,064 masks, the tumor covers a **mean 1.7%** of the frame (median
1.3%), while one cell of the default 7×7 grid is 2.0%. **70% of tumors are smaller than a
single grid cell.** Classification works well, but `eval_localization.py` on this dataset
mostly measures grid quantization rather than allocation quality. For a fair localization
test here, use a backbone with finer feature maps — `vit_b_16` gives a 14×14 grid
(0.51% per cell) via `model_grid_size()` in `scripts/train_backbone.py`.

HAM10000 does not have this problem: its lesions average ~37% of the frame.

---

## Both datasets also work with

```bash
PYTHONPATH=. python examples/contrastive_explanation_ig.py --dataset ham10000
```

```bash
PYTHONPATH=. python scripts/ablation_contrastive.py --dataset brain_tumor --num_images 200
```

```bash
PYTHONPATH=. python scripts/run_experiments.py --datasets ham10000 brain_tumor --seeds 0 1 2
```

---

## Caveats worth stating in any writeup

- **Not clinical evidence.** These explanations are research artifacts. Nothing here is
  validated for diagnostic use.
- **Licensing.** HAM10000 is CC BY-NC 4.0 — non-commercial, and derived figures inherit
  that constraint. Check the Kaggle brain tumor dataset's own license before
  redistributing anything derived from it.
- **HAM10000 has known shortcut artifacts** — ruler markings, surgical ink, and dark
  corner vignetting correlate with malignancy because images came from different
  acquisition sites. That is a liability for a straight accuracy claim, but an asset for
  the shift game in `instantiations/shift/`: the same dataset can show a shortcut mask
  locking onto rulers while the robust mask stays on the lesion.
- **Brain tumor provenance.** The Kaggle set is an aggregation of several sources with
  duplicate/leakage concerns, so accuracy numbers on it are not publication-grade. Use it
  for plumbing and qualitative figures.
