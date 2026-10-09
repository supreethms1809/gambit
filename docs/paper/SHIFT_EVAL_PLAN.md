# Shift evaluation plan (separate paper)

**Status.** The shift game moved to its own paper (user decision, recorded 2026-10-09). This file holds the shift sections of the Oct 7 `EVAL_PLAN.md`, **not reassessed**. Their wording is kept, a few cross-references are condensed, and one dated note is added. Nothing on the CVPR contrastive path depends on it.

The shared rules still apply wherever these sections refer to the contrastive plan: splits, models, the input convention, cost accounting, execution, records and freeze mechanics. Section numbers in parentheses refer to the Oct 7 plan.

Before this plan is used, it needs the same review the contrastive plan had. The open questions are:
- CDEA-shift still uses the v1-style objective (`docs/paper/FORMULATION.md` covers the contrastive game only);
- R2R is an extended baseline;
- ΔD is aligned with which methods.

---

## Claim (§1)

| # | Claim | Metric | Compared against | Test |
|---|---|---|---|---|
| S1 | CDEA's shortcut mask captures what makes the model change across environments, more than the strongest baseline | Logit ΔD at the piloted area | The model's own logits | Family S (section 7) |

## Datasets (§2.2)

| Dataset | Shortcut | Environments (paired) | Ground truth for secondary scoring | Model |
|---|---|---|---|---|
| Waterbirds *(paired)* | Background (land/water) | Same bird composited on land and water backgrounds | CUB bird segmentation | Fine-tuned, 5 seeds |
| ImageNet-9 Backgrounds | Background | `original` / `mixed_same` / `mixed_rand` | Foreground box | Fine-tuned, 5 seeds |
| Stanford Dogs backgrounds | Background style | Background restyled outside the box | Bounding box | Contrastive Dogs checkpoint |
| Planted-patch CIFAR-10 | Two class-tied patches | Patch present / moved / removed | Exact patch pixels | Fine-tuned, 5 seeds |
| ColoredMNIST | Digit hue | Re-coloured digit | Digit vs hue (non-spatial; see note) | Fine-tuned, 5 seeds |
| **Sixth dataset [set before freeze]** | Spatial background or context shortcut | Paired by compositing with segmentation masks | Object segmentation | Fine-tuned, 5 seeds |

- **Sixth dataset.** It must meet four criteria:
  - a spatially separable shortcut;
  - paired environments that can be built from a segmentation mask;
  - images that do not appear in any other shift unit;
  - a published construction.

  The recommendation is COCO-on-Places (Ahmed et al., ICLR 2021), built with the existing Waterbirds compositing code. The choice, its licence check, and its split go into `PROGRESS.md` before the freeze.
- **Waterbirds natural groups** (the unpaired objective) share images and a checkpoint with Waterbirds paired. They are an ablation (section 8), not a unit. Counting them as a unit would double-count Waterbirds.
- **ColoredMNIST** has a colour shortcut, not a spatial one. A spatial mask can only remove it by covering the digit. It stays a unit because the primary score is model-centric, and the paper notes this.
- The ImageNet-9 challenge archive is the test set and stays unextracted until the freeze.

Test-sample minimums (§2.4): shift ResNet-50 128, shift ViT-B/16 32.

Models (§3): shift datasets use a full fine-tune at lr 1e-4. Waterbirds uses class-weighted loss. The model table reports each model's shortcut reliance (ID–OOD logit gap on the full image, plus worst-group accuracy where groups exist). The ID–OOD gap is a model property and does not appear in the method table.

## Methods (§4.3)

Each method gives a robust map and a shortcut map. ΔD uses the shortcut map.
- **CDEA-shift:** paired objective, hard mass target. Its config is selected on val.
- **Attribution difference:** the shortcut map is the maximum over environments of |E(x_id) − E(x_e)|, and the robust map is the minimum over environments.
- **SpRAy:** CoRelAy spectral clustering of Zennit EpsilonPlus relevance, fitted on the same test sample. Each image takes its cluster's mean relevance. The shortcut map is that cluster map.
- **Per-environment Extremal Perturbations, differenced.**
- **Random floor.**

## Metrics (§5)

- **Logit ΔD** (family S): the drop in mean |z_y(x_id) − z_y(x_e)| after removing the shortcut mask in every environment, minus the same drop for an independent random mask of equal area (`evaluation.scores.logit_disagreement_reduction`). Here y is the model's predicted class on x_id, so the score is about the model, not the labels. It is logit rather than probability because confident models saturate probabilities. Probability ΔD is reported beside it.
- **Shift area** **[set at freeze]**, by the pilot rule below.
- Secondary: M_sho mass on background, M_rob mass on foreground, against area and translated nulls (correctness against dataset construction); probability ΔD.
- Exploratory: worst-group accuracy after test-time masking of M_sho, with DFR and GroupDRO as reference points. They are not explanation baselines.

## Selection (§6)

| Method | Candidates | Count |
|---|---|---|
| CDEA-shift | λ_gap {0.5, 1, 1.5} × λ_mass {0.1, 0.5} × λ_disjoint {0.2, 0.4} | 12 |
| Shift baselines | Backend {Grad-CAM, IG} where applicable | ≤ 2 each |

- Rule: maximise mean logit ΔD over the six shift val sets, at the area fixed by the pilot.
- **Degenerate routes:** D6 (robust-mask fraction above 0.5, complement deviation, shortcut mass fraction) and D7 (ID–OOD gap recorded as a model property, out of the method table), as implemented in `scripts/check_degenerate.py`.

### Shift area pilot (§6.3)
- Run every **non-CDEA** shift method on val at a ∈ {5%, 10%, 25%}.
- The primary area is the smallest a at which the best non-CDEA baseline's mean logit ΔD is above 0 with one-sided paired p < 0.05 on at least 4 of the 6 datasets. If no area qualifies, a = 25%.
- CDEA is not run in this pilot. The other two areas are robustness checks.

Note added 2026-10-09: v1 CDEA-shift masks have a fixed mass of one cell. At 25% most of a mask would be filled by the random tie-break (measured for contrastive v1 masks, 32% of kept pixels on average at 25%). Recheck before this pilot is used.

### Family S comparator (§6.4)
The strongest non-CDEA shift baseline is the one with the highest mean val logit ΔD at the chosen area. It is selected before CDEA-shift's val score is looked at.

## Statistics (§7)

- **Family S** (ResNet-50, 6 datasets). A single comparison: CDEA-shift vs the comparator above, on logit ΔD at the piloted area, two-sided exact Wilcoxon.
- Power: with n = 6, p = 0.031 needs 6/6 wins.
- Robustness: the two non-primary pilot areas.

## Ablation (§8)

| ID | Variant | What it isolates |
|---|---|---|
| AS1 | Shift: paired vs unpaired objective (Waterbirds), mass target on/off | Shift objective design |

## Execution and freeze (§10, §12)

- The Stanford Dogs shift unit is restricted to the paper split by path. The older `scripts/eval_robust_shortcut_dogs.py` read every annotated image. Background styles come from train images only.
- Freeze items specific to shift: the sixth shift dataset chosen and prepared; the shift area and comparator recorded.
