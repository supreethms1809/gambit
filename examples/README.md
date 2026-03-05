# Examples

Run from the repository root with `PYTHONPATH=.` so that `core`, `modality`, `base_evidence`, and `instantiations` are importable.

## Contrastive explanation

```bash
PYTHONPATH=. python examples/contrastive_explanation.py [--dataset DATASET] [--model MODEL] [--pretrained] [--batch_size N] [--num_alloc_steps N] [--interaction none|attention|transformer] [--game_mode cooperative|competitive|mixed|manual] [--interaction_attn_mix 0.35] [--interaction_weight_blend 0.5]
```

**Models** (torchvision): `--model resnet18` (default), `resnet34`, `mobilenet_v2`, `efficientnet_b0`. Use `--pretrained` for ImageNet weights (final layer is still replaced for your num_classes). Inputs are resized to 224×224 for these models.

**Training (recommended for interpretable contrastive masks):** Use `--train` to train the model on the chosen dataset before running the explanation. Training uses cross-entropy, Adam, and saves the best checkpoint by validation accuracy. Example:
```bash
PYTHONPATH=. python examples/contrastive_explanation.py --dataset cifar10 --train --epochs 10
```
Checkpoint is saved to `examples/out/checkpoints/<dataset>_<model>.pt` by default. To run explanation with a previously trained model: `--checkpoint path/to/checkpoint.pt`.
`interaction` is off by default (`--interaction none`), so add `--interaction attention` or `--interaction transformer` to enable hypothesis interaction during the explanation phase of that same run.

Stanford Dogs train+explain with interaction:
```bash
PYTHONPATH=. python /Users/ssuresh/gambit/examples/contrastive_explanation.py \
  --dataset stanford_dogs \
  --model resnet18 \
  --train \
  --epochs 50 \
  --batch_size 64 \
  --num_alloc_steps 25 \
  --max_viz_classes 3 \
  --interaction attention \
  --interaction_dim 64 \
  --interaction_heads 4 \
  --interaction_ffn 256 \
  --interaction_dropout 0.0 \
  --checkpoint /Users/ssuresh/gambit/examples/out/checkpoints/stanford_dogs_resnet18.pt
```

**Datasets** (from `data/`): choose with `--dataset`
- **`mnist`** — MNIST test set (28×28, 10 classes). Uses `data/MNIST`.
- **`cifar10`** — CIFAR-10 test set (32×32, 10 classes). Uses `data/cifar-10-batches-py` (or `data` as root).
- **`pets`** — PetImages Cat vs Dog (resized to 64×64, 2 classes). Uses `data/PetImages` (Cat/ and Dog/ subdirs).
- **`stanford_dogs`** — Stanford Dogs breeds (120 classes). Uses `data/stanford_dogs/images/Images` (class subdirs from ImageFolder).

Default: `--dataset cifar10`, `--batch_size 4`, `--num_alloc_steps 25`, `--max_viz_classes 3`, `--interaction none`, `--game_mode mixed`. If the dataset is not found, a random batch is used and a message is printed.

**Interaction (optional):** choose `--interaction attention` or `--interaction transformer` to enable top-K hypothesis interaction. The script automatically enables deterministic unit embeddings when interaction is on.

**Game mode:** choose from:
- `--game_mode cooperative`: shared evidence favored, no overlap/disjoint competition terms.
- `--game_mode competitive`: no shared mask, stronger unique-mask separation terms.
- `--game_mode mixed` (default): shared + unique with moderate competition terms.
- `--game_mode manual`: manually set `--use_shared/--no-use_shared`, `--lambda_margin`, `--lambda_overlap`, `--lambda_disjoint`, `--lambda_partition`.

Example:
```bash
PYTHONPATH=. python examples/contrastive_explanation.py --dataset cifar10 --interaction attention --interaction_dim 64 --interaction_heads 4 --game_mode mixed
```

Builds a CDEA explainer with:
- **Selector:** top-K hypotheses per sample
- **Evidence:** Grad-CAM pooled to a spatial grid (7×7)
- **Allocator:** optimization over unique masks (disjoint, sparse)
- **Objective:** contrastive (sufficiency, margin, overlap, sparsity)

Runs `explain(x)` on one batch and **saves a figure** to `examples/out/contrastive_explanation_{dataset}.png` (compact 3-row layout):
- **Row 0:** Input image + summary text with **true class**, **predicted class**, and top shown classes
- **Row 1:** Evidence maps for shown classes
- **Row 2:** Contrastive overlays for shown classes (green = unique, red = shared)
- **Legend:** Included at the bottom of the figure for evidence/unique/shared color coding.

Use `--max_viz_classes` to control how many top classes are shown in the figure (default 3 to avoid clutter).

Also prints a **probability split report** for sample 0:
- `shared-only`: class probabilities using `keep(m_shared)`
- `shared+unique`: class probabilities using `keep(m_shared + m_unique[k])`
- `delta`: contribution of the unique component for each top-m hypothesis

Also writes machine-readable reports:
- `examples/out/contrastive_<dataset>_metrics.json`
- `examples/out/contrastive_<dataset>_metrics.csv`
- `examples/out/contrastive_<dataset>_split.csv`
- `examples/out/contrastive_<dataset>_pairwise.csv` (`why k rather than l` margins)

Requires `torchvision` (dataset loading) and `matplotlib` (visualization).

## Contrastive explanation (Integrated Gradients evidence)

```bash
PYTHONPATH=. python examples/contrastive_explanation_ig.py [--dataset DATASET] [--model MODEL] [--pretrained] [--batch_size N] [--num_alloc_steps N] [--ig_steps N] [--ig_baseline zero|mean] [--game_mode cooperative|competitive|mixed|manual] [--interaction_attn_mix 0.35] [--interaction_weight_blend 0.5]
```

This runs the same contrastive CDEA pipeline but swaps base evidence from Grad-CAM to Integrated Gradients pooled to grid regions.

Defaults:
- `--ig_steps 24`
- `--ig_baseline zero`
- `--max_viz_classes 3`

Output figure is saved to `examples/out/contrastive_explanation_ig_{dataset}.png`.
Reports are also saved to:
- `examples/out/contrastive_ig_<dataset>_metrics.json`
- `examples/out/contrastive_ig_<dataset>_metrics.csv`
- `examples/out/contrastive_ig_<dataset>_split.csv`
- `examples/out/contrastive_ig_<dataset>_pairwise.csv` (`why k rather than l` margins)

## Shift-aware robust vs shortcut (Instantiation II)

Run the second CDEA game (robust evidence vs shortcut evidence) with:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
PYTHONPATH=. python /Users/ssuresh/gambit/scripts/eval_robust_shortcut.py --game_mode mixed
```

Default run settings in the script:
- `num_images=200`
- `batch_size=16`
- `num_steps=40`
- `game_mode=mixed`

Shift game mode options:
- `cooperative`: no shortcut-gap rivalry/disjointness terms and positive shared shortcut utility term.
- `competitive`: stronger shortcut-gap and robust-vs-shortcut disjointness terms.
- `mixed` (default): balanced settings.
- `manual`: pass all `--lambda_*` values explicitly (`--lambda_mean`, `--lambda_var`, `--lambda_gap`, `--lambda_shortcut`, `--lambda_disjoint`, `--lambda_sparse`).

Outputs:
- Figure: `scripts/out/robust_shortcut_masks.png`
- Summary JSON: `scripts/out/robust_shortcut_metrics.json`
- Scalar CSV: `scripts/out/robust_shortcut_metrics.csv`
- Per-batch CSV: `scripts/out/robust_shortcut_per_batch.csv`

To run with custom values:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
PYTHONPATH=. python /Users/ssuresh/gambit/scripts/eval_robust_shortcut.py \
  --num_images 400 \
  --batch_size 16 \
  --num_steps 60 \
  --game_mode competitive \
  --export_prefix robust_shortcut_custom
```

## Jupyter notebook usage

For an end-to-end notebook workflow (Grad-CAM and Integrated Gradients) use:

- Guide: `examples/JUPYTER_GUIDE.md`
- Starter notebook: `examples/notebooks/cdea_contrastive_quickstart.ipynb`

Quick start:
```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
jupyter lab
```
