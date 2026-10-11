# Shift evaluation plan

Status: not frozen. No shift number has been read. Two Waterbirds cells from the earlier objective exist and stay unscored. This file has no version name.

## Datasets

Waterbirds, ImageNet-9, Stanford Dogs restyled outside the box, planted-patch CIFAR-10, ColoredMNIST (non-spatial), and a sixth dataset (COCO-on-Places, still to be chosen). That is `n = 6` once the sixth is chosen. The five that exist are the units this branch can run.

Image counts per backbone are the contrastive minimums until this plan is frozen. The gate uses `n = 64`.

## Metrics

| Angle | Metric | Optimised by |
|---|---|---|
| Shortcut necessity, primary | `ΔD@a`: drop in `D = mean_{e ∈ E'} \|m(x_id) - m(x_e)\|` after ROAD-deleting `M_S` in every environment, minus the drop for a random mask of equal area | Nothing directly. The allocation transplants. It does not delete. The probability gap is kept as a companion. |
| Robust necessity | `RN@a`: min over environments of the drop in `m` under ROAD deletion of `M_R`, minus a random mask | The robust player, disclosed |
| Transplant closure | `TC@a`: the shortcut payoff of the binary `M_S` | The shortcut player, disclosed. Descriptive. |
| Ground truth, family T-shift | `SPC@a` on the planted patch: mass of `M_S` on the class-tied patch A minus mass on the neutral checker B. Both patches change across environments. | Nothing |
| Construction masses | `M_S` on the background and `M_R` on the foreground, against a random mask of equal area | Descriptive. `M_S` on the background is near-tautological for a transplant, and the text says so. |
| Use case | Worst-group accuracy after masking `M_S` | Exploratory |

## Methods

Core: CDEA-shift, gap attribution, attribution difference, SpRAy, per-environment Extremal Perturbations differenced, and a random floor.

Extended: R2R, fetched by `scripts/fetch_r2r.sh`, unlicensed, research use, never vendored.

## Selection

Per dataset, at most four configurations per method, maximising val `ΔD@a`. The CDEA-shift grid is initialisation `{Grad-CAM, IG}` times learning rate `{0.05, 0.2}`, with `T = 100`. The area pilot for methods other than CDEA-shift is `{5, 10, 25}` percent. The family S comparator is fixed on val before CDEA-shift's val score is read.

## Statistics

Family S is one comparison, ResNet-50, six datasets, exact Wilcoxon. It needs six wins out of six for `p = 0.031`.

Family T-shift is `SPC` against the val-chosen comparator. The unit is the image. Seeds are pooled. The test is a sign-flip permutation within a seed and a two-level bootstrap.

Robustness: the blur operator, the other two areas, ViT, and leaving one dataset out.

## Gate

Default configuration on Waterbirds and planted-patch val, ResNet-50, seeds 0 and 1, `n = 64`.

Read-out: `ΔD@a` against gap attribution and against attribution difference.

Stop rule: continue as a method paper only if CDEA-shift is ahead of both on both datasets. Otherwise take the evaluation-paper route, with no further formulation revision.

`scripts/gate_shift.py` prints this read-out and this stop rule. It does not launch the pilot unless `--run` is passed. Fast knobs cannot satisfy the gate.

## After the gate

Per-dataset selection, the sixth dataset, shift seeds 1–4 on the GH200, and the family S and family T statistics. Those wait until the gate has been read.
