# GAMBIT Journal Experiment Report

**Setup:** model=resnet18, datasets=['mnist', 'cifar10', 'pets', 'stanford_dogs'], evidence=['gradcam', 'ig'], mean ± std, n=3 seeds [0, 1, 2]

> Metrics marked ↑ are better when higher; ↓ are better when lower.
> Overlap is an unnormalized pairwise dot-product sum (scale depends on evidence magnitude).
> Suff is baseline-subtracted vs all-zeros input (positive = mask carries real signal).
> Margin = logit_k − max(logit_foil) under the kept mask.

## Instantiation I: Contrastive Explanation

### Main Results: CDEA (optimized) vs Baselines

One row per (dataset, evidence). Columns show absolute values for the **optimized** method,
plus % improvement over the base evidence baseline.

| Dataset | Evidence | Suff ↑ | Margin ↑ | Overlap ↓ | Overlap ↓% vs Base | Sparse ↓ | Sparse ↓% vs Base |
|---|---|---|---|---|---|---|---|
| mnist | gradcam | — | — | — | — | — | — |
| mnist | ig | — | — | — | — | — | — |
| cifar10 | gradcam | — | — | — | — | — | — |
| cifar10 | ig | — | — | — | — | — | — |
| pets | gradcam | — | — | — | — | — | — |
| pets | ig | — | — | — | — | — | — |
| stanford_dogs | gradcam | — | — | — | — | — | — |
| stanford_dogs | ig | — | — | — | — | — | — |

### Full Method Breakdown (base / naive / optimized)

Suff ↑ and Margin ↑ are better higher. Overlap ↓ and Sparse ↓ are better lower.

#### mnist

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|

#### cifar10

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|

#### pets

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|

#### stanford_dogs

| Evidence | Method | Suff ↑ | Margin ↑ | Overlap ↓ | Sparse ↓ |
|---|---|---|---|---|---|

## Instantiation II: Robust vs Shortcut

Columns: Rob Mean ↑ (robust sufficiency across envs), Rob Var ↓ (stability),
Sho Gap ↑ (shortcut is ID-specific), Disjoint ↓ (mask separation), Sparse ↓, ID-OOD Gap ↑.

| Dataset | Game Mode | Rob Mean ↑ | Rob Var ↓ | Sho Gap ↑ | Disjoint ↓ | Sparse ↓ | ID-OOD Gap ↑ |
|---|---|---|---|---|---|---|---|
| colored_mnist | cooperative | 1.4048 ± 0.0999 | 0.0213 ± 0.0043 | 0.1341 ± 0.0631 | 12.0210 ± 1.0005 | 14.4576 ± 0.9691 | 5.3412 ± 0.0839 |
| colored_mnist | mixed | 1.3285 ± 0.0931 | 0.0153 ± 0.0039 | 0.1086 ± 0.0243 | 0.7223 ± 0.0460 | 8.0851 ± 0.4645 | 5.3521 ± 0.1480 |
| colored_mnist | competitive | 1.2385 ± 0.0884 | 0.0119 ± 0.0031 | 0.1414 ± 0.0309 | 0.6200 ± 0.0440 | 7.5843 ± 0.4391 | 5.3310 ± 0.0902 |
| colored_cifar10 | cooperative | 6.8704 ± 0.2164 | 0.0736 ± 0.0283 | 0.4375 ± 0.1044 | 18.5522 ± 0.3187 | 22.1274 ± 0.3083 | 2.8497 ± 0.0650 |
| colored_cifar10 | mixed | 6.7615 ± 0.2248 | 0.0359 ± 0.0205 | 1.0004 ± 0.1003 | 1.2098 ± 0.0296 | 12.8279 ± 0.1197 | 3.4432 ± 0.1175 |
| colored_cifar10 | competitive | 6.6621 ± 0.2430 | 0.0230 ± 0.0170 | 1.0013 ± 0.0922 | 1.1020 ± 0.0174 | 12.4148 ± 0.0812 | 3.3828 ± 0.1582 |
| texture_mnist | cooperative | 5.0299 ± 0.3695 | 0.2592 ± 0.0391 | 0.6868 ± 0.0835 | 13.2935 ± 0.6697 | 16.5923 ± 0.6933 | 3.4170 ± 0.4506 |
| texture_mnist | mixed | 4.8761 ± 0.3604 | 0.2285 ± 0.0335 | 2.6465 ± 0.2536 | 1.7997 ± 0.0743 | 14.0409 ± 0.5094 | 5.3915 ± 0.5434 |
| texture_mnist | competitive | 4.7417 ± 0.3300 | 0.1974 ± 0.0226 | 2.6645 ± 0.2596 | 1.5243 ± 0.0790 | 13.8191 ± 0.4818 | 5.4010 ± 0.5069 |

## Artifact Locations

- Per-seed ablations: `scripts/out/ablation_<dataset>_<evidence>_seed<N>_metrics.csv`
- Contrastive summary (all methods, all seeds): `scripts/out/journal/summary_contrastive.csv`
- Shift summary: `scripts/out/journal/summary_shift.csv`
- Robust/shortcut mask visualization: `scripts/out/robust_shortcut_masks.png`