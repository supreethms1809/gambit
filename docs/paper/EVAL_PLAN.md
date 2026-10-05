# Evaluation plan

**Status.** This plan is not frozen. The tag `eval-plan-frozen` was not created. The test split stays locked. No run reads the test split or passes `--final` until every item in "Freeze procedure" (section 12) is done and the tag exists. The selection rule in section 6 has not been applied. Sections marked **[set at freeze]** are filled from val runs only, never from test.

This version replaces the Oct 3 draft (S17). Changes from that draft:
- The input convention is ImageNet normalisation inside the model.
- Contrastive masks have a hard mass budget.
- CD@a uses the unique masks only.
- Every method's foil mask is defined.
- ImageNet is the 8th contrastive dataset, run on Spark.
- Waterbirds-groups is an ablation, not a sixth shift dataset. A new independent sixth shift dataset is added.
- The shift primary is the logit ΔD on the predicted class, at an area chosen by a val pilot.
- Selection eligibility was updated for the hard budget.
- Power limits, ablations, cost accounting, machine assignment, the record schema, and the freeze procedure are all specified.

---

## 1. Claims and how each is tested

Every claim names its metric, what the metric compares against, and its test. Claims not listed here are not made (see `FRAMING.md`).

| # | Claim | Metric | Compared against | Test |
|---|---|---|---|---|
| C1 | CDEA's unique evidence is more contrastive than margin attribution, contrastive Extremal Perturbations, and CVE at matched area | CD@5% (CD1@5% for CVE), ROAD removal | The model's own logits | Family C (section 7) |
| C2 | Allocation makes explanations more model-dependent than the evidence they start from | Spearman ρ under cascading model randomisation | The same method on a randomised model | Secondary, descriptive |
| S1 | CDEA's shortcut mask captures what makes the model change across environments, more than the strongest baseline | Logit ΔD at the piloted area | The model's own logits | Family S (section 7) |
| N1 | Lesion overlap cannot validate contrastive explanations | Lesion-mass fraction against the null family | Human annotation | Exploratory, reported as a negative finding |

## 2. Datasets and splits

### 2.1 Contrastive game (family C units)

| Dataset | Classes | Test split | Val split | Models | Machine |
|---|---|---|---|---|---|
| CIFAR-10 *(dev)* | 10 | official test | 10% of train, seed 43 | trained, 5 seeds | **[set at freeze]** |
| CIFAR-100 | 100 | official test | 10% of train, seed 43 | trained, 5 seeds | **[set at freeze]** |
| Oxford-IIIT Pet | 37 | official test | 10% of train, seed 43 | trained, 5 seeds | **[set at freeze]** |
| Stanford Dogs | 120 | complement of `random_split(seed=42)` | 10% of train pool, seed 43 | trained, 5 seeds | **[set at freeze]** |
| CUB-200 | 200 | official test | 10% of train, seed 43 | trained, 5 seeds | **[set at freeze]** |
| HAM10000 *(dev)* | 7 | lesion-grouped former `val/` | lesion-grouped 10% of train | trained, 5 seeds | **[set at freeze]** |
| Brain tumor MRI | 3 | patient-grouped `Testing/` | patient-grouped 10% of `Training/` | trained, 5 seeds | **[set at freeze]** |
| ImageNet-1k | 1000 | ImageNet val minus our val carve (40,000) | 10 per class from ImageNet val (10,000), seed 43 | torchvision pretrained, fixed | Spark |

- CIFAR-10 and HAM10000 are the **dev datasets**: hyperparameter selection uses their val splits. The main result is reported on all 8 datasets and again on the 6 non-dev datasets.
- **ImageNet** uses the published torchvision `IMAGENET1K_V1` weights for both backbones, the same weights the other datasets' linear probes start from. Neither was trained on our test images, but both were developed against ImageNet val. That is disclosed, and it affects model accuracy, not the explanation comparison. Because the weights are fixed, a "seed" on ImageNet is a test-sample draw only. Its seed variance therefore excludes model variance, and the model table says so.
- ImageNet-S segmentations are not needed. The primary metric uses no annotation.

### 2.2 Shift game (family S units)

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

### 2.3 Split rules
- Training uses train. Checkpoint selection and every method's hyperparameters use val. Every reported number uses test.
- The split files in `data/splits/` are fixed. The loader refuses `split="test"` without `--final` and a frozen plan (`evaluation/splits.py`).

### 2.4 Test samples **[set at freeze]**
- For seed s ∈ {0,…,4}, the test sample is the first n indices of `randperm(test_size)`, using generator seed 1000 + s. The same images are used for every method in that seed.
- n per seed is fixed at the freeze from a timed val cell, so that each machine's grid fits its compute window. Minimums:
  - contrastive ResNet-50: 200;
  - contrastive ViT-B/16: 64;
  - shift ResNet-50: 128;
  - shift ViT-B/16: 32.
- Images are not filtered on correctness. Correct vs incorrect is an exploratory split.

## 3. Models

- **ResNet-50 is primary.** Families C and S use it. ViT-B/16 is the second-backbone robustness check.
- **Training.**
  - Contrastive datasets: an ImageNet linear probe (frozen backbone with BN in eval mode; Adam, cosine schedule, lr 1e-3, 15 epochs).
  - Shift datasets: a full fine-tune at lr 1e-4. Waterbirds uses class-weighted loss.
  - Selection: val balanced accuracy.
  - Seeds: 0–4.
- **Input convention:** ImageNet normalisation is a layer inside the model (`NormalizedModel`). Every method receives raw [0, 1] images, and every intervention is defined in [0, 1]. Every checkpoint load goes through `models.wrapper.load_checkpoint_into`, which restores the wrapper from the checkpoint's `input_convention` tag. The probe that chose this is `results/paper/logs/probe_input/cifar10_resnet50_seed0.json` (val balanced accuracy 0.829 raw vs 0.915 normalised).
- **Model table** (main text): top-1 and balanced accuracy per dataset × backbone × seed on test. For shift datasets it also reports each model's shortcut reliance (ID–OOD logit gap on the full image, plus worst-group accuracy where groups exist). The ID–OOD gap is a model property and does not appear in the method table.

## 4. Methods and their masks

### 4.1 Shared rules
- **Hypotheses.** The model's top K = min(5, C) classes on the full image. The contrastive pair is k = rank 0 and l = rank 1. Every method receives the same k and l.
- **Budget.** Every map goes through `baselines.adapter.adapt_scores`: bilinear upsampling, then the top a of pixels, with a seeded tie-break. The primary area is a = 5%.
- **Failures.** A row with a non-finite or empty map is replaced by the seeded random floor and stays in the sample. Failure counts are reported per method × dataset. No image is dropped.

### 4.2 Contrastive methods

| Method | M_k | M_l | Tier |
|---|---|---|---|
| **CDEA** | Unique mask of rank 0 | Unique mask of rank 1 | — |
| Base evidence | E_k from CDEA's selected backend | E_l | Core |
| Margin attribution | Attribution of z_k − z_l | Attribution of z_l − z_k | Core |
| Contrastive Extremal Perturbations | TorchRay mask at area a, maximising z_k − z_l | Same, maximising z_l − z_k | Core |
| CVE (Goyal et al. 2019) | Query cells of x replaced in edit order while flipping k → l | Not defined | Core, ResNet-50 only |
| Random floor | Seeded area-a mask | Independent seeded area-a mask | Core |
| Extremal Perturbations, per class | Mask for z_k | Mask for z_l | Extended (K×K) |
| RISE, margin weights | Masks weighted by z_k − z_l | Weighted by z_l − z_k | Extended |
| Contrastive Grad-CAM | Cross-entropy toward l | Cross-entropy toward k | Extended |
| `naive_contrastive` | E_k − mean(E_foils) | E_l − mean(others) | Sanity row |

Notes on these definitions:
- **CDEA:** joint allocation over K hypotheses with a hard mass budget. Unique plus shared on both sides is a sensitivity check.
- **Margin attribution:** the Grad-CAM variant needs a separate call for M_l, because the ReLU means M_l is not simply the negated map. For IG, M_l is the negated signed map, because IG is linear in the target.
- **Extremal Perturbations:** area a is the method's own constraint.
- **CVE:** its map is one-sided, so it is scored with CD1 (section 5).
  - **Distractor:** a seeded train-split image whose label and prediction are both l. If no such image exists, the row fails.
  - **Feature map and decision network:** the `layer4` output (7×7), with avgpool and fc as the decision network.
  - **Not run on ViT:** the CLS head does not read patch tokens after the last block, so there is no spatial decision network.
- **Random floor:** each side gets its own independent seeded mask.
- **ViT backends:** CDEA and base evidence use IG on ViT, because Grad-CAM's field degenerates on LayerNorm'd tokens. The margin-attribution variant is selected on val per backbone.

### 4.3 Shift methods
Each method gives a robust map and a shortcut map. ΔD uses the shortcut map.
- **CDEA-shift:** paired objective, hard mass target. Its config is selected on val.
- **Attribution difference:** the shortcut map is the maximum over environments of |E(x_id) − E(x_e)|, and the robust map is the minimum over environments.
- **SpRAy:** CoRelAy spectral clustering of Zennit EpsilonPlus relevance, fitted on the same test sample. Each image takes its cluster's mean relevance. The shortcut map is that cluster map.
- **Per-environment Extremal Perturbations, differenced.**
- **Random floor.**

## 5. Metrics

All removals are ROAD noisy-linear imputation (`evaluation.removal.noisy_linear_impute`, 24 iterations, noise 0.01) unless marked "blur". Optimisation uses blur keep, so the primary scores use a different operator from the one CDEA was fitted to. The blur versions are robustness checks.

### 5.1 Primary
- **CD@5%** (family C): m(x without M_l) − m(x without M_k), with m = z_k − z_l (`evaluation.scores.contrastive_deletion`). Positive means removing the foil's evidence helps k relative to l, and removing k's evidence does not.
- **CD1@5%** (the CVE comparison only): m(x) − m(x without M_k). It is computed identically for CDEA and CVE, and it is reported for every method as a secondary score.
- **Logit ΔD** (family S): the drop in mean |z_y(x_id) − z_y(x_e)| after removing the shortcut mask in every environment, minus the same drop for an independent random mask of equal area (`evaluation.scores.logit_disagreement_reduction`). Here y is the model's predicted class on x_id, so the score is about the model, not the labels. It is logit rather than probability because confident models saturate probabilities. Probability ΔD is reported beside it.
- **Shift area** **[set at freeze]**, by the pilot rule in section 6.3.

### 5.2 Secondary (reported in the main text, not confirmatory)

| Metric | Measures (Co-12 property, Nauta et al. 2023) |
|---|---|
| K×K deletion matrix: diagonal − off-diagonal, and \|diag\|/off, for methods with per-class maps | Contrastivity |
| CD1@5% for every method | Correctness |
| Components of CD: z_k and z_l drops reported separately | Guards against margins won by foil suppression alone |
| Model randomisation ρ (cascade to full), every method, ResNet-50, seed 0, n = 64, every contrastive dataset. A method with \|ρ\| ≥ 0.3 at full randomisation is flagged in the main table | Correctness |
| Two-patch recovery on planted-patch CIFAR-10. Images whose top-2 are the two planted classes. Share of M_k on patch A and of M_l on patch B. Excluded images are counted | Correctness against exact ground truth |
| Shift: M_sho mass on background, M_rob mass on foreground, against area and translated nulls | Correctness against dataset construction |
| Shift: probability ΔD | Correctness |
| Mask area check, connected components | Compactness |

### 5.3 Exploratory (labelled as such)
- Deletion and insertion AUC, ROAD MoRF/LeRF.
- Seed and input-noise stability.
- Lesion overlap on HAM10000 and brain tumor, with the null family.
- Correct vs incorrect stratification.
- Worst-group accuracy after test-time masking of M_sho, with DFR and GroupDRO as reference points. They are not explanation baselines.
- Per-image paired tests.

## 6. Selection on val (before the freeze)

All of this uses the val splits, seed-0 checkpoints, and n = 64 images per dataset. Nothing here reads test.

### 6.1 Degenerate routes after the hard budget
Rerun `scripts/check_degenerate.py` on the dev val sets. The stored report predates the hard budget.
- **D1 and D5 close by construction.** The unique and shared masks are both projected to a fixed mass. The rerun asserts this rather than assuming it.
- **D2** (soft vs hard gap ≤ 0.5) and **D4** (evidence capture ≥ 1.2× chance and above the translated null) are eligibility constraints.
- **D3** (the share of the margin coming from foil suppression) is reported, not used as a gate. For "k rather than l", evidence that lowers l is legitimate contrastive evidence. The reader sees z_k and z_l separately (section 5.2). This was decided before any val selection ran.
- **D6 and D7** stay closed, and the rerun confirms them.

### 6.2 Candidate grids (each method's own knobs, scored by the same rule)

| Method | Candidates | Count |
|---|---|---|
| CDEA | backend {Grad-CAM, IG} × λ_margin {0.5, 1, 2} × λ_overlap {0.1, 0.2, 0.4}. 50 steps, lr 0.2, λ_shared_sparse 0.25. ViT: IG only | 18 (9 on ViT) |
| Margin attribution | {Grad-CAM, IG-16, IG-32} | 3 |
| Contrastive Extremal Perturbations | max_iter {300, 800} × smoothing {0, TorchRay default} | 4 |
| RISE | n_masks {2000, 4000} × cell size {7, 8}, p1 0.5 | 4 |
| CVE | Distractor rule fixed. No knobs | 1 |
| CDEA-shift | λ_gap {0.5, 1, 1.5} × λ_mass {0.1, 0.5} × λ_disjoint {0.2, 0.4} | 12 |
| Shift baselines | Backend {Grad-CAM, IG} where applicable | ≤ 2 each |

- **Rule** (`analysis.selection.select_config`):
  - Contrastive: maximise mean CD@5% over the two dev val sets, among eligible candidates. Ties keep the earliest. If no CDEA candidate is eligible, the freeze stops and the formulation is revisited. There is no fallback.
  - Shift: maximise mean logit ΔD over the six shift val sets, at the area fixed in 6.3.
- Every method's grid, scores, and selection go into `results/paper/selection/` and are summarised in the appendix. Both the authors' default and the val-tuned row are reported for each baseline.

### 6.3 Shift area pilot
- Run every **non-CDEA** shift method on val at a ∈ {5%, 10%, 25%}.
- The primary area is the smallest a at which the best non-CDEA baseline's mean logit ΔD is above 0 with one-sided paired p < 0.05 on at least 4 of the 6 datasets. If no area qualifies, a = 25%.
- CDEA is not run in this pilot. The other two areas are robustness checks.

### 6.4 Family S comparator
The strongest non-CDEA shift baseline is the one with the highest mean val logit ΔD at the chosen area. It is selected before CDEA-shift's val score is looked at.

## 7. Statistical protocol

- **Unit:** the dataset. Average over seeds inside each dataset, then compare methods across datasets. Every statistic is computed from raw per-image records by `analysis/stats.py`.
- **Family C** (ResNet-50, 8 datasets). Two-sided exact Wilcoxon signed-rank, Holm over 3 comparisons:
  1. CDEA vs margin attribution, on CD@5%;
  2. CDEA vs contrastive Extremal Perturbations, on CD@5%;
  3. CDEA vs CVE, on CD1@5%.
- **Family S** (ResNet-50, 6 datasets). A single comparison: CDEA-shift vs the section 6.4 comparator, on logit ΔD at the piloted area, two-sided exact Wilcoxon.
- **Report:** wins out of N, the mean difference, a 95% percentile bootstrap interval over datasets (10,000 resamples, seed 0), and the Holm-adjusted p.
- **Power, stated in advance.**
  - Family C with n = 8: the exact two-sided p is 0.0078 for 8/8 wins, 0.0156 when the one loss has the smallest |difference|, 0.0234 at T = 2, and 0.039 at T = 3. Against Holm's thresholds (0.0167, 0.025, 0.05), the first rejection needs T ≤ 1, the second T ≤ 2, and the third T ≤ 3.
  - Family S with n = 6: p = 0.031 needs 6/6 wins.

  A non-significant result means the experiment cannot separate the methods. It does not mean the methods are equivalent.
- **CDEA vs base evidence** appears in the main table as a secondary comparison with the same statistics. It is not in family C.
- **Robustness checks.** Each confirmatory claim is checked under:
  - leave each dataset out;
  - drop the two dev datasets;
  - ViT-B/16;
  - the blur operator;
  - a ∈ {2.5%, 10%} (shift: the two non-primary pilot areas);
  - native vs grid-pooled resolution for pixel-level baselines;
  - Extremal Perturbations at CDEA's forward-pass count;
  - unique plus shared on both sides (CDEA);
  - effect size against measured run-to-run variation (MPS and CUDA, section 10).

  A claim is called robust only if it survives every check. Otherwise the text names the check it fails.

## 8. Ablations

- **Setup:** ResNet-50 on CIFAR-100, CUB-200, brain tumor, and ImageNet (all non-dev), seeds 0–2, the same n as the main grid, CD@5% plus its components.
- **Format:** one table, one row per removed component. Exploratory: it reports means and per-dataset wins, not tests.

| ID | Variant | What it isolates |
|---|---|---|
| A1 | Independent per-class optimisation (no overlap term, K separate runs) | Joint allocation, the central mechanism |
| A2 | No margin term | Margin |
| A3 | No overlap term | Overlap |
| A4 | No shared mask | Shared mask |
| A5 | Zero vs evidence initialisation, across 6 evidence backends (`library_adapters`) | Dependence on the base evidence (the "any attribution" claim) |
| A6 | Interaction: none / attention / transformer | Interaction module |
| A7 | Steps {10, 25, 50, 100}, against Extremal Perturbations at the same forward-pass count | Cost–quality and compute confound |
| A8 | Presets: mixed / cooperative / competitive | The "game mode" claim |
| AS1 | Shift: paired vs unpaired objective (Waterbirds), mass target on/off | Shift objective design |

Confounds are stated per row: forward passes, number of masks, K.

## 9. Cost accounting
- **Hardware-independent unit:** forward and backward passes per explanation, counted by a wrapper around the model. CDEA's count includes its base-evidence pass.
- **Wall-clock** per image per method, reported per platform (MPS, CUDA).
- **Excluded:** classifier training and data loading. This is stated in the table caption.

## 10. Execution
- **One dataset, one machine.** Every method and seed of a dataset runs on the same machine. Platform differences therefore never enter a within-dataset method comparison. Assignments go into `PROGRESS.md` before launch. ImageNet runs on Spark.
- **Cross-platform check.** One cell (CIFAR-10, ResNet-50, seed 0, every core method, n = 64) runs on both machines. Report the maximum absolute per-image difference in CD@5%. Results from the two machines are compared only through dataset-level means.
- **Run-to-run variation:** repeat that same cell 3× on each machine.
- **Records.** One row per image × method × seed × backbone × area × operator. Each row holds:
  - dataset, method, config hash, seed, image index, k, l;
  - CD, CD1, z_k and z_l before and after each removal, and failure flag;
  - mask area, forward/backward counts, wall-clock;
  - code commit and platform.

  Rows are stored as `.csv.gz` under `results/paper/runs/<game>/<dataset>/<backbone>/<method>/seed<s>.csv.gz`, with a JSON summary. A cell is complete only when its done marker exists (`scripts/completion_check.py`). Record assertions: masks in [0, 1], area = a ± tolerance, finite values, and n as planned.
- **Executor (still to be built).**
  - `evaluation/executor.py` currently scores a given pair of masks.
  - The full per-cell runner must load the checkpoint through `load_checkpoint_into`, draw the frozen test sample, run each method's map function, apply sections 4–5, and write records.
  - It is built and tested on val before the freeze. Its smoke test is the G1 pilot.

## 11. Gates this plan depends on
- **G0:** every Core baseline has its reproduction (B3) or a recorded, justified omission. VOC is unpacked. ImageNet val is on Spark, so the RISE reproduction runs there.
- **G1:** the family-C pilot on dev val. If CDEA does not beat margin attribution on val, stop and decide with the user between the critique-paper route and continuing.

## 12. Freeze procedure
The plan is frozen only after all of the following are recorded in this file and in `PROGRESS.md`:
1. G0 passed (tag `g0-baselines`) and G1 passed (tag `g1-pilot`).
2. Degenerate rerun under the hard budget (6.1). D2 and D4 eligible for the selected config.
3. Selected configs and their hashes for every method (6.2), and the shift area (6.3) and comparator (6.4).
4. The sixth shift dataset chosen and prepared (2.2).
5. n per seed and backbone, test-sample seeds (2.4), and machine assignments (10).
6. The executor built, with tests passing on val.
7. The cross-platform cell and the run-to-run repeats done on val.

Then add a line reading `frozen: true` at the top of this file, commit, and tag `eval-plan-frozen`. After the freeze, any change to this file is a new dated section with its reason. Results computed before that section are reported under the old plan.
