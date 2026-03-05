# CDEA Prelims: Experiments and Preliminary Results Template

Use this as a fill-in template after longer training runs complete.

## 1) Experimental Setup

### 1.1 Environment

- Repository: `/Users/ssuresh/gambit`
- Python environment: `conda activate marl`
- Device used: `[TODO: mps/cuda/cpu]`
- Seed policy: `[TODO: single seed / multi-seed list]`
- Game mode policy: `[TODO: mixed (default) / ablation list]`

### 1.2 Datasets

- `mnist`
- `cifar10`
- `pets`
- `stanford_dogs` (confusable fine-grained classes)

### 1.3 Backbone

- Model family: `resnet18` (as currently used in examples)
- Training epochs: `[TODO]`
- Batch size: `[TODO]`
- Checkpoint path: `/Users/ssuresh/gambit/examples/out/checkpoints/[TODO].pt`

### 1.4 Evidence Providers

- Grad-CAM regions (`base_evidence/gradcam_regions.py`)
- Integrated Gradients regions (`base_evidence/integrated_gradients_regions.py`)

## 2) Commands Used

### 2.1 Train + Contrastive explain (Grad-CAM)

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
PYTHONPATH=. python /Users/ssuresh/gambit/examples/contrastive_explanation.py \
  --dataset stanford_dogs \
  --model resnet18 \
  --train \
  --epochs [TODO] \
  --batch_size [TODO] \
  --num_alloc_steps 25 \
  --max_viz_classes 3 \
  --interaction attention \
  --interaction_dim 64 \
  --interaction_heads 4 \
  --interaction_ffn 256 \
  --interaction_dropout 0.0 \
  --game_mode mixed \
  --checkpoint /Users/ssuresh/gambit/examples/out/checkpoints/stanford_dogs_resnet18.pt
```

### 2.2 Contrastive explain (IG)

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
PYTHONPATH=. python /Users/ssuresh/gambit/examples/contrastive_explanation_ig.py \
  --dataset stanford_dogs \
  --model resnet18 \
  --checkpoint /Users/ssuresh/gambit/examples/out/checkpoints/stanford_dogs_resnet18.pt \
  --batch_size [TODO] \
  --num_alloc_steps 25 \
  --ig_steps 8 \
  --interaction attention \
  --interaction_dim 64 \
  --interaction_heads 4 \
  --interaction_ffn 256 \
  --interaction_dropout 0.0 \
  --game_mode mixed \
  --max_viz_classes 3
```

### 2.3 Instantiation II (robust-shortcut)

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
PYTHONPATH=. python /Users/ssuresh/gambit/scripts/eval_robust_shortcut.py \
  --game_mode mixed
```

## 3) Metrics Definitions (copy-ready)

### 3.1 Contrastive metrics

1. `suff`: target-class logit under keep-mask intervention (higher is better).
2. `margin`: contrastive logit margin \(z_k - \max(z_{\text{foil}})\) under keep-mask intervention (higher is better).
3. `overlap`: pairwise overlap among unique masks (lower is better).
4. `sparse`: L1 mask mass of unique masks (lower is better).
5. `mass_dev`: absolute deviation between optimized mask mass and base-evidence mass (lower is better).

### 3.2 Shift-aware metrics

1. `rob_mean`: mean robust sufficiency across environments (higher is better).
2. `rob_var`: variance of robust sufficiency across environments (lower is better).
3. `sho_gap`: ID shortcut sufficiency minus mean OOD shortcut sufficiency (higher is better).
4. `sho_mean`: mean shortcut sufficiency across environments (higher in cooperative mode, often near-zero in mixed/competitive).
5. `disjoint`: overlap between robust and shortcut masks (lower is better).
6. `sparse`: average mask mass penalty (lower is better).
7. `id_ood_gap`: quantitative ID-OOD gap for shortcut mask (higher is better if shortcut is ID-specific).

## 4) Preliminary Results Table Template

### 4.1 Contrastive: per dataset and evidence

| Dataset | Evidence | Suff | Margin | Overlap | Sparse | Mass Dev | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| MNIST | Grad-CAM | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| MNIST | IG | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| CIFAR10 | Grad-CAM | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| CIFAR10 | IG | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| Pets | Grad-CAM | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| Pets | IG | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| Stanford Dogs | Grad-CAM | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| Stanford Dogs | IG | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

### 4.2 Shift-aware: robust-shortcut decomposition

| Run | Rob Mean | Rob Var | Sho Gap | Sho Mean | Disjoint | Sparse | ID-OOD Gap | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Robust-shortcut eval | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## 5) Figure Checklist

Figures to include in prelim proposal:

1. Contrastive visualization with legend for one example per dataset.
2. Stanford Dogs confusable-class case study with true label, predicted label, and shared/unique overlays.
3. Grad-CAM vs IG side-by-side for the same sample.
4. Robust vs shortcut mask figure (`robust_shortcut_masks.png`) with concise interpretation.

## 6) Interpretation Template

### 6.1 What worked

- [TODO: summarize where overlap dropped while suff remained stable]
- [TODO: summarize any positive margin shifts]
- [TODO: summarize visual evidence quality on Stanford Dogs]

### 6.2 What remains weak

- [TODO: cases with diffuse masks or weak/negative margins]
- [TODO: instability/failure cases]
- [TODO: limitations of current robust-shortcut setup]

### 6.3 Proposal-positioned claim

The preliminary results demonstrate implementation feasibility and initial directional improvements in explanation structure (especially overlap reduction), motivating the planned next phase focused on strengthening faithfulness, stability, and cross-dataset generalization.

## 7) Artifact Locations

### 7.1 Contrastive outputs

- `/Users/ssuresh/gambit/examples/out/contrastive_*_metrics.json`
- `/Users/ssuresh/gambit/examples/out/contrastive_*_metrics.csv`
- `/Users/ssuresh/gambit/examples/out/contrastive*_pairwise.csv`
- `/Users/ssuresh/gambit/examples/out/contrastive*_split.csv`
- `/Users/ssuresh/gambit/examples/out/contrastive_explanation*.png`

### 7.2 Shift outputs

- `/Users/ssuresh/gambit/scripts/out/robust_shortcut*_metrics.json`
- `/Users/ssuresh/gambit/scripts/out/robust_shortcut*_metrics.csv`
- `/Users/ssuresh/gambit/scripts/out/robust_shortcut*_per_batch.csv`
- `/Users/ssuresh/gambit/scripts/out/robust_shortcut_masks.png`
