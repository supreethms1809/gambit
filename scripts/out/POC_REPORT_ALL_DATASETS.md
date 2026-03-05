# CDEA POC Report (All Datasets, Grad-CAM + IG)

This report summarizes 6 concrete runs (3 datasets x 2 evidence providers).

## Run Setup

- Datasets: `mnist`, `cifar10`, `pets`
- Evidence: `gradcam`, `ig`
- Backbone(s): `resnet18`
- ImageNet pretrained weights: `disabled`
- `lambda_disjoint`: `1`
- `lambda_mass`: `2`
- Compared methods per run: `base_evidence`, `naive_contrastive`, `optimized (CDEA)`

## Advantage Summary

| dataset | evidence | overlap_red_vs_base_pct | overlap_red_vs_naive_pct | sparse_red_vs_base_pct | suff_change_vs_base_abs |
|---|---:|---:|---:|---:|---:|
| mnist | gradcam | 97.72% | 96.13% | -0.32% | -0.000111 |
| mnist | ig | 95.93% | 93.45% | 22.49% | -0.000767 |
| cifar10 | gradcam | 97.68% | 95.48% | -0.16% | -0.000018 |
| cifar10 | ig | 88.23% | 79.83% | 9.94% | -0.000221 |
| pets | gradcam | 82.54% | NA | -1.35% | -0.000005 |
| pets | ig | 22.92% | NA | 1.03% | +0.000047 |

## Aggregate Takeaway

- Mean overlap reduction vs base: **80.84%**
- Mean sparsity reduction vs base: **5.27%**
- Mean absolute sufficiency change vs base: **0.000195**

Interpretation: with this run setup, optimized CDEA reduces overlap and reduces mask mass, while sufficiency stays close to baseline.

## Metric Definitions

- `suff`: Mean target-class logit under keep-mask intervention.
- `margin`: Mean contrastive margin (`z_k - max(z_foil)`) under keep-mask intervention.
- `overlap`: Pairwise overlap among unique masks (lower is better).
- `sparse`: Mean L1 mask mass (lower is better).
- `overlap_red_vs_base_pct`: Percent overlap reduction of optimized vs base evidence.
- `overlap_red_vs_naive_pct`: Percent overlap reduction of optimized vs naive contrastive.
- `sparse_red_vs_base_pct`: Percent sparsity reduction of optimized vs base evidence.
- `suff_change_vs_base_abs`: Absolute sufficiency delta (`optimized - base`).

## Artifacts

- Consolidated matrix: `/Users/ssuresh/gambit/scripts/out/poc_all_datasets_matrix.csv`
- Consolidated JSON: `/Users/ssuresh/gambit/scripts/out/poc_all_datasets_matrix.json`
- Per-run ablations: `/Users/ssuresh/gambit/scripts/out/ablation_contrastive_<dataset>_<evidence>_metrics.csv`