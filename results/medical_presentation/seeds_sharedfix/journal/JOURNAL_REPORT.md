# GAMBIT Journal Experiment Report

**Setup:** model=resnet18, datasets=['ham10000', 'brain_tumor'], evidence=['gradcam', 'ig'], mean ± std, n=3 seeds [0, 1, 2]

> Metrics marked ↑ are better when higher; ↓ are better when lower.
> Overlap is an unnormalized pairwise dot-product sum (scale depends on evidence magnitude).
> Suff is baseline-subtracted vs all-zeros input (positive = mask carries real signal).
> Margin = logit_k − max(logit_foil) under the kept mask.

## Key Findings

- **Overlap reduction** (optimized vs base): **82.7%** mean across 4 (dataset × evidence) combos
- **Sufficiency cost**: 1.85032 mean absolute change
- **Margin change**: +3.07230 mean delta vs base

> Suff is baseline-subtracted: f(keep(x, m))[k] − f(zeros)[k]. Positive = kept regions carry
> more signal than seeing nothing. Margin is logit_k − max(logit_foil) under the kept mask.

## Instantiation I: Contrastive Explanation

### Main Results: CDEA (optimized) vs Baselines

One row per (dataset, evidence). Columns show absolute values for the **optimized** method,
plus % improvement over the base evidence baseline.

| Dataset | Evidence | Suff ↑ | Margin ↑ | Overlap ↓ | Overlap ↓% vs Base | Sparse ↓ | Sparse ↓% vs Base |
|---|---|---|---|---|---|---|---|
| ham10000 | gradcam | 2.0068 ± 0.5335 | 0.4236 ± 0.8001 | 0.0574 ± 0.0034 | +84.5% | 1.0970 ± 0.1080 | -15.3% |
| ham10000 | ig | 2.1343 ± 0.6058 | 0.5673 ± 0.8604 | 0.1184 ± 0.0067 | +75.8% | 1.1429 ± 0.1082 | -14.3% |
| brain_tumor | gradcam | 1.8301 ± 0.2523 | 1.2077 ± 0.3737 | 0.0081 ± 0.0022 | +86.8% | 1.3986 ± 0.0746 | -40.9% |
| brain_tumor | ig | 2.0501 ± 0.2866 | 1.3304 ± 0.4505 | 0.0482 ± 0.0046 | +83.8% | 1.4143 ± 0.0916 | -41.4% |

### Full Method Breakdown (base / naive / optimized)

Suff ↑ and Margin ↑ are better higher. Overlap ↓ and Sparse ↓ are better lower.

#### ham10000

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | 0.1670 ± 0.1719 | -3.0384 ± 1.1806 | 0.3709 ± 0.0216 | 0.9517 ± 0.0038 |
| gradcam | naive | 0.1831 ± 0.1771 | -3.0178 ± 1.1782 | 0.2033 ± 0.0145 | 0.9517 ± 0.0038 |
| gradcam | **CDEA** | 2.0068 ± 0.5335 | 0.4236 ± 0.8001 | 0.0574 ± 0.0034 | 1.0970 ± 0.1080 |
| ig | base | 0.1235 ± 0.1553 | -3.0782 ± 1.1829 | 0.4891 ± 0.0034 | 1.0000 |
| ig | naive | 0.1916 ± 0.1899 | -3.0063 ± 1.1636 | 0.2797 ± 0.0037 | 1.0000 |
| ig | **CDEA** | 2.1343 ± 0.6058 | 0.5673 ± 0.8604 | 0.1184 ± 0.0067 | 1.1429 ± 0.1082 |

#### brain_tumor

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | 0.2433 ± 0.1233 | -1.0717 ± 0.4364 | 0.0610 ± 0.0081 | 0.9925 ± 0.0123 |
| gradcam | naive | 0.2790 ± 0.1305 | -1.0325 ± 0.4407 | 0.0062 ± 0.0010 | 0.9925 ± 0.0123 |
| gradcam | **CDEA** | 1.8301 ± 0.2523 | 1.2077 ± 0.3737 | 0.0081 ± 0.0022 | 1.3986 ± 0.0746 |
| ig | base | 0.0862 ± 0.0723 | -1.5719 ± 0.3188 | 0.2976 ± 0.0122 | 1.0000 |
| ig | naive | 0.1593 ± 0.1316 | -1.5603 ± 0.4071 | 0.1397 ± 0.0171 | 1.0000 |
| ig | **CDEA** | 2.0501 ± 0.2866 | 1.3304 ± 0.4505 | 0.0482 ± 0.0046 | 1.4143 ± 0.0916 |

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