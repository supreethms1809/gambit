# GAMBIT Journal Experiment Report

**Setup:** model=resnet18, datasets=['cifar10', 'mnist', 'pets', 'stanford_dogs'], evidence=['gradcam', 'ig'], mean ± std, n=3 seeds [0, 1, 2]

> Metrics marked ↑ are better when higher; ↓ are better when lower.
> Overlap is an unnormalized pairwise dot-product sum (scale depends on evidence magnitude).
> Suff is baseline-subtracted vs all-zeros input (positive = mask carries real signal).
> Margin = logit_k − max(logit_foil) under the kept mask.

## Key Findings

- **Overlap reduction** (optimized vs base): **93.2%** mean across 8 (dataset × evidence) combos
- **Sufficiency cost**: 0.83293 mean absolute change
- **Margin change**: +1.24834 mean delta vs base

> Suff is baseline-subtracted: f(keep(x, m))[k] − f(zeros)[k]. Positive = kept regions carry
> more signal than seeing nothing. Margin is logit_k − max(logit_foil) under the kept mask.

## Instantiation I: Contrastive Explanation

### Main Results: CDEA (optimized) vs Baselines

One row per (dataset, evidence). Columns show absolute values for the **optimized** method,
plus % improvement over the base evidence baseline.

| Dataset | Evidence | Suff ↑ | Margin ↑ | Overlap ↓ | Overlap ↓% vs Base | Sparse ↓ | Sparse ↓% vs Base |
|---|---|---|---|---|---|---|---|
| cifar10 | gradcam | 0.7865 ± 0.0714 | -1.2532 ± 0.0462 | 0.0085 ± 0.0001 | +97.9% | 1.0376 ± 0.0013 | -3.8% |
| cifar10 | ig | 0.8201 ± 0.0690 | -1.2112 ± 0.0505 | 0.0141 ± 0.0002 | +97.3% | 1.0344 ± 0.0021 | -3.4% |
| mnist | gradcam | 2.7935 ± 0.0190 | -1.9392 ± 0.0275 | 0.0096 ± 0.0003 | +97.6% | 1.0443 ± 0.0024 | -4.4% |
| mnist | ig | 2.7451 ± 0.0201 | -2.0259 ± 0.0273 | 0.0259 ± 0.0003 | +98.5% | 1.0169 ± 0.0038 | -1.7% |
| pets | gradcam | 1.1164 ± 0.1822 | 2.2899 ± 0.0718 | 0.0004 ± 0.0001 | +68.0% | 1.3272 ± 0.0194 | -32.8% |
| pets | ig | 1.0599 ± 0.1778 | 2.1630 ± 0.0735 | 0.0032 ± 0.0002 | +94.7% | 1.2242 ± 0.0220 | -22.4% |
| stanford_dogs | gradcam | -2.5525 ± 0.0885 | -1.4244 ± 0.1224 | 0.0217 ± 0.0002 | +95.8% | 1.0725 ± 0.0039 | -7.3% |
| stanford_dogs | ig | -2.5798 ± 0.0905 | -1.4535 ± 0.1274 | 0.0257 ± 0.0008 | +96.2% | 1.0578 ± 0.0042 | -5.8% |

### Full Method Breakdown (base / naive / optimized)

Suff ↑ and Margin ↑ are better higher. Overlap ↓ and Sparse ↓ are better lower.

#### cifar10

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | 0.2544 ± 0.0803 | -2.0184 ± 0.0486 | 0.3981 ± 0.0025 | 0.9997 ± 0.0003 |
| gradcam | naive | 0.2588 ± 0.0804 | -2.0127 ± 0.0481 | 0.2329 ± 0.0010 | 0.9997 ± 0.0003 |
| gradcam | **CDEA** | 0.7865 ± 0.0714 | -1.2532 ± 0.0462 | 0.0085 ± 0.0001 | 1.0376 ± 0.0013 |
| ig | base | 0.2470 ± 0.0796 | -2.0280 ± 0.0506 | 0.5259 ± 0.0011 | 1.0000 |
| ig | naive | 0.2488 ± 0.0777 | -2.0239 ± 0.0504 | 0.2962 ± 0.0013 | 1.0000 |
| ig | **CDEA** | 0.8201 ± 0.0690 | -1.2112 ± 0.0505 | 0.0141 ± 0.0002 | 1.0344 ± 0.0021 |

#### mnist

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | 2.1376 ± 0.0141 | -2.8212 ± 0.0203 | 0.3955 ± 0.0006 | 1.0000 |
| gradcam | naive | 2.1408 ± 0.0137 | -2.8156 ± 0.0196 | 0.2254 ± 0.0018 | 1.0000 |
| gradcam | **CDEA** | 2.7935 ± 0.0190 | -1.9392 ± 0.0275 | 0.0096 ± 0.0003 | 1.0443 ± 0.0024 |
| ig | base | 2.1272 ± 0.0144 | -2.8240 ± 0.0225 | 1.7476 ± 0.0015 | 1.0000 |
| ig | naive | 2.1539 ± 0.0114 | -2.8376 ± 0.0187 | 0.9669 ± 0.0025 | 1.0000 |
| ig | **CDEA** | 2.7451 ± 0.0201 | -2.0259 ± 0.0273 | 0.0259 ± 0.0003 | 1.0169 ± 0.0038 |

#### pets

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | -0.0291 ± 0.2417 | 0.0530 ± 0.0037 | 0.0012 ± 0.0004 | 0.9996 ± 0.0007 |
| gradcam | naive | -0.0283 ± 0.2413 | 0.0540 ± 0.0032 | 0.0000 | 0.9996 ± 0.0007 |
| gradcam | **CDEA** | 1.1164 ± 0.1822 | 2.2899 ± 0.0718 | 0.0004 ± 0.0001 | 1.3272 ± 0.0194 |
| ig | base | -0.0624 ± 0.2481 | -0.0001 ± 0.0012 | 0.0606 ± 0.0001 | 1.0000 |
| ig | naive | -0.0520 ± 0.2481 | 0.0055 ± 0.0410 | 0.0000 | 1.0000 |
| ig | **CDEA** | 1.0599 ± 0.1778 | 2.1630 ± 0.0735 | 0.0032 ± 0.0002 | 1.2242 ± 0.0220 |

#### stanford_dogs

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | -3.5886 ± 0.0967 | -2.5980 ± 0.1325 | 0.5173 ± 0.0040 | 1.0000 |
| gradcam | naive | -3.5090 ± 0.0973 | -2.5327 ± 0.1385 | 0.3202 ± 0.0008 | 1.0000 |
| gradcam | **CDEA** | -2.5525 ± 0.0885 | -1.4244 ± 0.1224 | 0.0217 ± 0.0002 | 1.0725 ± 0.0039 |
| ig | base | -3.5603 ± 0.0964 | -2.6044 ± 0.1311 | 0.6834 ± 0.0022 | 1.0000 |
| ig | naive | -3.5156 ± 0.0951 | -2.5692 ± 0.1369 | 0.3990 ± 0.0032 | 1.0000 |
| ig | **CDEA** | -2.5798 ± 0.0905 | -1.4535 ± 0.1274 | 0.0257 ± 0.0008 | 1.0578 ± 0.0042 |

## Instantiation II: Robust vs Shortcut

Columns: Rob Mean ↑ (robust sufficiency across envs), Rob Var ↓ (stability),
Sho Gap ↑ (shortcut is ID-specific), Disjoint ↓ (mask separation), Sparse ↓, ID-OOD Gap ↑.

| Dataset | Game Mode | Rob Mean ↑ | Rob Var ↓ | Sho Gap ↑ | Disjoint ↓ | Sparse ↓ | ID-OOD Gap ↑ |
|---|---|---|---|---|---|---|---|

## Artifact Locations

- Per-seed ablations: `scripts/out/ablation_<dataset>_<evidence>_seed<N>_metrics.csv`
- Contrastive summary (all methods, all seeds): `scripts/out/journal/summary_contrastive.csv`
- Shift summary: `scripts/out/journal/summary_shift.csv`
- Robust/shortcut mask visualization: `scripts/out/robust_shortcut_masks.png`