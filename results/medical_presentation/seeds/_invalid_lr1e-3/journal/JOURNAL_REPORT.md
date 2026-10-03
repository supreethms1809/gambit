# GAMBIT Journal Experiment Report

**Setup:** model=resnet18, datasets=['ham10000', 'brain_tumor'], evidence=['gradcam', 'ig'], mean ± std, n=3 seeds [0, 1, 2]

> Metrics marked ↑ are better when higher; ↓ are better when lower.
> Overlap is an unnormalized pairwise dot-product sum (scale depends on evidence magnitude).
> Suff is baseline-subtracted vs all-zeros input (positive = mask carries real signal).
> Margin = logit_k − max(logit_foil) under the kept mask.

## Key Findings

- **Overlap reduction** (optimized vs base): **67.4%** mean across 4 (dataset × evidence) combos
- **Sufficiency cost**: 0.47659 mean absolute change
- **Margin change**: +1.72281 mean delta vs base

> Suff is baseline-subtracted: f(keep(x, m))[k] − f(zeros)[k]. Positive = kept regions carry
> more signal than seeing nothing. Margin is logit_k − max(logit_foil) under the kept mask.

## Instantiation I: Contrastive Explanation

### Main Results: CDEA (optimized) vs Baselines

One row per (dataset, evidence). Columns show absolute values for the **optimized** method,
plus % improvement over the base evidence baseline.

| Dataset | Evidence | Suff ↑ | Margin ↑ | Overlap ↓ | Overlap ↓% vs Base | Sparse ↓ | Sparse ↓% vs Base |
|---|---|---|---|---|---|---|---|
| ham10000 | gradcam | 1.0035 ± 0.1890 | -2.7650 ± 0.6904 | 0.0615 ± 0.0072 | +83.0% | 0.9691 ± 0.0372 | -1.2% |
| ham10000 | ig | 1.0166 ± 0.1829 | -2.7286 ± 0.6914 | 0.1618 ± 0.0022 | +65.0% | 1.0080 ± 0.0060 | -0.8% |
| brain_tumor | gradcam | 0.0448 ± 0.2735 | -1.0837 ± 0.3533 | 0.0123 ± 0.0015 | +72.3% | 0.8252 ± 0.1656 | -0.1% |
| brain_tumor | ig | 0.0514 ± 0.2717 | -1.0803 ± 0.3645 | 0.1145 ± 0.0039 | +49.4% | 0.9976 ± 0.0020 | +0.2% |

### Full Method Breakdown (base / naive / optimized)

Suff ↑ and Margin ↑ are better higher. Overlap ↓ and Sparse ↓ are better lower.

#### ham10000

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | 0.1647 ± 0.2435 | -5.7678 ± 1.0896 | 0.3624 ± 0.0173 | 0.9580 ± 0.0304 |
| gradcam | naive | 0.1695 ± 0.2462 | -5.7686 ± 1.0859 | 0.2045 ± 0.0132 | 0.9580 ± 0.0304 |
| gradcam | **CDEA** | 1.0035 ± 0.1890 | -2.7650 ± 0.6904 | 0.0615 ± 0.0072 | 0.9691 ± 0.0372 |
| ig | base | 0.1569 ± 0.2425 | -5.7755 ± 1.0919 | 0.4620 ± 0.0072 | 1.0000 |
| ig | naive | 0.1698 ± 0.2436 | -5.7817 ± 1.0856 | 0.2722 ± 0.0118 | 1.0000 |
| ig | **CDEA** | 1.0166 ± 0.1829 | -2.7286 ± 0.6914 | 0.1618 ± 0.0022 | 1.0080 ± 0.0060 |

#### brain_tumor

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|
| gradcam | base | -0.0540 ± 0.1503 | -1.5030 ± 0.1902 | 0.0444 ± 0.0081 | 0.8244 ± 0.1674 |
| gradcam | naive | -0.0546 ± 0.1503 | -1.5038 ± 0.1892 | 0.0085 ± 0.0025 | 0.8244 ± 0.1674 |
| gradcam | **CDEA** | 0.0448 ± 0.2735 | -1.0837 ± 0.3533 | 0.0123 ± 0.0015 | 0.8252 ± 0.1656 |
| ig | base | -0.0577 ± 0.1528 | -1.5026 ± 0.1985 | 0.2265 ± 0.0086 | 1.0000 |
| ig | naive | -0.0557 ± 0.1572 | -1.4979 ± 0.2049 | 0.0942 ± 0.0115 | 1.0000 |
| ig | **CDEA** | 0.0514 ± 0.2717 | -1.0803 ± 0.3645 | 0.1145 ± 0.0039 | 0.9976 ± 0.0020 |

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