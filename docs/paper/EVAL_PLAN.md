# Evaluation plan

**Status.** This plan is not frozen. The tag `eval-plan-frozen` was not created. The test split stays locked. No run reads the test split or passes `--final` until every item in "Freeze procedure" (section 12) is done and the tag exists. The selection rule in section 6 has not been applied. Sections marked **[set at freeze]** are filled from val runs only, never from test.

**Scope.** This plan covers the contrastive paper. The shift game moved to a separate paper. Its sections are in `SHIFT_EVAL_PLAN.md` and were not reassessed.

**Revision 2026-10-09.** This rewrite replaces the Oct 7 version.

The earlier-formulation val read motivated it: seed 0, all 8 datasets, both backbones. The earlier formulation (tag `framing-v1`) trailed the better margin-attribution variant in all 16 cells. The diagnosis is in `FORMULATION.md` section 1.

What changed:
- **The method.** CDEA is this formulation (`FORMULATION.md`). The earlier formulation (tag `framing-v1`) is ablation A8.
- **The metrics.** The suite is rebuilt around the separate angles of "why k rather than l": necessity, sufficiency, ground truth, decomposition, and sanity. Every metric states which methods optimise it (section 5.0).
  - The primary metric is unchanged: CD@5%.
  - Two ground-truth metrics that no method optimises are new: discriminative-part contrast on CUB-200, and two-cue recovery.
- **Family C's Extremal Perturbations comparator is the deletion variant,** which asks the same question as CD. The preservation variant becomes the comparator for sufficiency.
- **Selection is per dataset,** on that dataset's val split, with at most 4 configurations per method. Before, it was on the two dev datasets only.
- **A new secondary family T** (ground truth) is pre-registered.
- **Gate G1 is replaced by G1,** which has a stop rule (section 11).
- **Shift moved** to `SHIFT_EVAL_PLAN.md`.

Disclosures:
- The earlier-formulation val numbers on all 8 datasets were read before this formulation and this plan were written. The test split was not read.
- 15 of the 16 earlier-formulation CDEA rows carried the chunk-scale bug (PR #28). The earlier formulation is tag `framing-v1`.

---

## 1. Claims and how each is tested

Every claim names its metric, what the metric compares against, and its test. Claims not listed here are not made (`FRAMING.md`).

| # | Claim | Metric | Compared against | Test |
|---|---|---|---|---|
| C1 | CDEA's unique evidence is more necessary for the model's preference of k over l than margin attribution, deletion Extremal Perturbations, and CVE at matched area | CD@5% (CD1@5% for CVE), ROAD removal | The model's own logits | Family C, primary (section 7) |
| C2 | On CUB-200, CDEA's evidence for k over l falls on the parts whose annotated attributes separate the two species, rather than on parts that do not, more than the strongest baseline | DPC@5% | Expert part keypoints and class attributes | Family T, secondary |
| C3 | On two-cue CIFAR-10, each side of CDEA's explanation recovers its own class's planted cue, more than the strongest baseline | TCS@5% | Construction | Family T, secondary |
| C4 | CDEA's per-hypothesis evidence is specific to its hypothesis under intervention, and its shared evidence lowers the belief in H without reordering H | K×K specificity, shared validity | The model's own logits | Descriptive with intervals (section 7) |
| C5 | CDEA's maps depend on the model | Spearman ρ under cascading randomisation | The same method on a randomised model | Sanity filter, descriptive |
| N1 | Lesion overlap cannot validate contrastive explanations | Lesion-mass fraction against the null family | Human annotation | Exploratory, reported as a negative finding |

C2 is a plausibility claim against expert annotation. It is not a localisation claim, and "good localisation" is still not claimed.

## 2. Datasets and splits

### 2.1 Contrastive units

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

- **Dev datasets.** CIFAR-10 and HAM10000 are where the formulation was developed and where gate G1 is read (section 11). Hyperparameters are selected per dataset (section 6). Because this formulation was designed after reading earlier-formulation val on all 8 datasets, the result without the dev datasets is a robustness check, not a hold-out claim.
- **ImageNet** uses the published torchvision `IMAGENET1K_V1` weights for both backbones, the same weights the other datasets' linear probes start from.
  - Neither was trained on our test images, but both were developed against ImageNet val. That is disclosed. It affects model accuracy, not the explanation comparison.
  - Because the weights are fixed, a "seed" on ImageNet is a test-sample draw only. Its seed variance therefore excludes model variance, and the model table says so.

### 2.2 Annotations and constructed data used by family T
- **CUB-200 parts and attributes**, used only for scoring:
  - `parts/part_locs.txt`: 15 keypoints with visibility flags.
  - `attributes/class_attribute_labels_continuous.txt`: 200 classes × 312 class-level attribute percentages.
  - The eval transform is a plain resize to 224 × 224, so a keypoint (u, v) maps to (224u/W, 224v/H).
- **Two-cue CIFAR-10** (section 5.4). CIFAR-10 images with two class-tied cues, scored with the planted-patch classifiers (section 3). It is built from CIFAR-10 val for selection and the gate, and from the official test split for final runs.

### 2.3 Split rules
- Training uses train. Checkpoint selection and every method's hyperparameters use val. Every reported number uses test.
- The split files in `data/splits/` are fixed. The loader refuses `split="test"` without `--final` and a frozen plan (`evaluation/splits.py`).

### 2.4 Test samples **[set at freeze]**
- **Main grid.** For seed s ∈ {0,…,4}, the test sample is the first n indices of `randperm(test_size)`, using generator seed 1000 + s. Every method in that seed uses the same images.
- **n per seed** is fixed at the freeze from a timed val cell, so that each machine's grid fits its compute window. Minimums:
  - ResNet-50: 200;
  - ViT-B/16: 64.
- **CUB part scoring** uses the CUB images of the main grid. Images with fewer than 6 visible parts are excluded from DPC and counted.
- **Two-cue CIFAR-10.** Images are taken in the same seeded order until 200 eligible images per seed are reached, or 2,000 have been tried (section 5.4). The eligibility rate is reported.
- Images are not filtered on correctness. Correct vs incorrect is an exploratory split.

## 3. Models

- **ResNet-50 is primary.** Families C and T use it. ViT-B/16 is the second-backbone robustness check.
- **Contrastive datasets.** An ImageNet linear probe: frozen backbone with BN in eval mode, Adam, cosine schedule, lr 1e-3, 15 epochs. Selection is by val balanced accuracy. Seeds 0–4.
- **Two-cue classifiers.** The planted-patch CIFAR-10 classifiers: a full fine-tune at lr 1e-4, trained with one class-tied hue patch and one class-neutral checker per image (`instantiations/shift/planted_patch.py`).
  - Seeds 0–4 are needed.
  - Seed 0 exists from the shift training.
  - Seeds 1–4 are a family T prerequisite (section 12).
- **Input convention.** ImageNet normalisation is a layer inside the model (`NormalizedModel`).
  - Every method receives raw [0, 1] images, and every intervention is defined in [0, 1].
  - Every checkpoint load goes through `models.wrapper.load_checkpoint_into`, which restores the wrapper from the checkpoint's `input_convention` tag.
  - The probe that chose this is `results/paper/logs/probe_input/cifar10_resnet50_seed0.json` (val balanced accuracy 0.829 raw vs 0.915 normalised).
- **Model table** (main text). Top-1 and balanced accuracy per dataset × backbone × seed on test, plus the two-cue classifiers' clean accuracy and two-cue eligibility rate.

## 4. Methods and their masks

### 4.1 Shared rules
- **Hypotheses.** The model's top K = min(5, C) classes on the full image (H). The contrastive pair is k = rank 0 and l = rank 1. Every method receives the same H, k and l.
- **Budget.** Every map goes through `baselines.adapter.adapt_scores`: bilinear upsampling, then the top a of pixels, with a seeded tie-break.
  - Exceptions, each named in the dossier:
    - Extremal Perturbations' native mask at area a;
    - CVE's and SC-CVE's edit order.
  - The primary area is a = 5%. Robustness areas are 2.5% and 10%.
- **Per-area optimisation.** CDEA and both Extremal Perturbations variants are optimised separately at each area. Every other method produces one map, which is thresholded at each area.
- **Failures.** A row with a non-finite or empty map is replaced by the seeded random floor and stays in the sample. Failure counts are reported per method × dataset. No image is dropped.

### 4.2 Contrastive methods

| Method | M_k | M_l | Maps for all of H (C4) | Tier | Backbones |
|---|---|---|---|---|---|
| **CDEA** | Unique allocation A_k | Unique allocation A_l | A_j for every j ∈ H; shared A_S | — | Both |
| Base evidence | E_k from CDEA's selected backend | E_l | E_j | Core | Both |
| Margin attribution | Attribution of z_k − z_l | Attribution of z_l − z_k | Attribution of z_j − z_{r(j)}, where r(j) is the strongest other member of H | Core | Both |
| Contrastive Extremal Perturbations, **deletion** | TorchRay `DELETE_VARIANT` mask at area a, reward z_k − z_l | Same, reward z_l − z_k | — | Core | Both |
| CVE (Goyal et al. 2019) | Query cells replaced in edit order while flipping k → l | Not defined | — | Core | ResNet-50 |
| Random floor | Seeded area-a mask | Independent seeded mask | Independent seeded masks | Core | Both |
| Contrastive Extremal Perturbations, preservation | TorchRay `PRESERVE_VARIANT`, reward z_k − z_l | Same, reward z_l − z_k | — | Extended (sufficiency comparator) | Both |
| Chefer et al. 2021, transformer attribution | Class map for k | Class map for l | Class map for j | Extended | ViT-B/16 |
| SC-CVE (Vandenhende et al. 2022) | Query cells replaced in joint-search edit order | Not defined | — | Extended | ResNet-50 |
| RISE | Masks weighted by z_k − z_l | Weighted by z_l − z_k | Official class maps for j | Extended | Both |
| Contrastive Grad-CAM | Cross-entropy toward l | Cross-entropy toward k | — | Extended | Both |
| Analytic decomposition | E_k − min_{j∈H} E_j | E_l − min_{j∈H} E_j | E_j − min_{i∈H} E_i; shared map min_{j∈H} E_j | Sanity row | Both |

Notes on these definitions:
- **CDEA.** Defined in `FORMULATION.md`. Unique plus shared on both sides is a sensitivity check.
- **Analytic decomposition.** This is CDEA's split into shared and unique with no optimisation, computed from the same base evidence. It answers "does the allocation add anything over its own input's arithmetic?" It replaces `naive_contrastive`.
- **Margin attribution.** The variant (Grad-CAM, IG-16, IG-32) is selected per dataset on val (section 6), because the better variant changes with image resolution: earlier-formulation val showed IG ahead on the high-resolution datasets and degenerate on CIFAR-10 and HAM10000.
  - The Grad-CAM variant needs a separate call for M_l, because the ReLU means M_l is not simply the negated map.
  - For IG, M_l is the negated signed map, because IG is linear in the target.
- **Extremal Perturbations, deletion variant.** It answers the same necessity question as CD with the same family of blur deletions. It is the closest single-player counterpart of CDEA. The preservation variant answers sufficiency and is scored there. TorchRay's `contrastive_reward` is z_c − max_{c'≠c} z_{c'}, which is a different target and is not used.
- **CVE.** Its map is one-sided, so it is scored with CD1 (section 5).
  - **Distractor:** a seeded train-split image whose label and prediction are both l. If no such image exists, the row fails.
  - **Feature map and decision network:** the `layer4` output (7×7), with avgpool and fc as the decision network.
  - **Not run on ViT:** the CLS head does not read patch tokens after the last block, so there is no spatial decision network.
- **SC-CVE.** One-sided like CVE, so it is scored with CD1.
  - **Distractor:** up to 20 seeded train-split images (the authors' default) whose label and prediction are both l, searched jointly.
  - **Semantic prior:** SwAV ResNet-50 trunk features, pinned (`facebookresearch/swav` @ `06b1b7c`, weights hash-recorded at download). Weak on HAM10000 and brain MRI by construction.
  - **Chunking:** it seeds distractors by image index, so it runs on the whole sample (`_whole_sample`).
- **Chefer.** The authors' `transformer_attribution` on the torchvision ViT, through the `from_torchvision` bridge (logit-equivalence tested).
- **Random floor.** Each side, and each member of H, gets its own independent seeded mask.
- **ViT backends.** CDEA's initialisation and base evidence use IG on ViT, because Grad-CAM's field degenerates on LayerNorm'd tokens.

## 5. Metrics

Every removal is ROAD noisy-linear imputation (`evaluation.removal.noisy_linear_impute`, 24 iterations, noise 0.01) unless marked "blur". CDEA and both Extremal Perturbations variants optimise with blur, so every primary and secondary score uses a different operator from the one they were fitted to. Blur scores are robustness checks.

### 5.0 The angles of the question, and who optimises each metric

| Angle | Metric | Section | Optimised by (with some removal operator) | Lineage |
|---|---|---|---|---|
| Necessity | CD@a, CD1@a | 5.1 | CDEA and deletion Extremal Perturbations (blur). Margin attribution is CDEA's first-order solution (`FORMULATION.md` section 9). CVE and SC-CVE minimise edits to a flip, by replacement | Deletion (Petsiuk et al. 2018), ROAD (Rong et al. 2022) |
| Sufficiency | SC@a | 5.2 | Preservation Extremal Perturbations (blur), the earlier formulation (tag `framing-v1`) | Insertion (Petsiuk et al. 2018) |
| Discriminative parts | DPC@a on CUB-200 | 5.3 | **None** | Counterfactual part triplets (SCOUT, Wang & Vasconcelos 2020); keypoint scoring (Goyal et al. 2019; Vandenhende et al. 2022) |
| Exact ground truth | TCS@a on two-cue CIFAR-10 | 5.4 | **None** | Synthetic ground truth (SHAPES in Goyal et al. 2019) |
| Decomposition | K×K specificity, shared validity | 5.5 | Partly CDEA: its payoffs are the log-odds analogues | Class-specificity tests (Chefer et al. 2021) |
| Sanity | Model randomisation | 5.6 | **None** | Adebayo et al. 2018 |

The claims that rest on metrics no method optimises (C2, C3) are the strongest evidence. The metrics CDEA partly optimises (CD, the K×K matrix) are reported with that alignment stated.

### 5.1 Primary
- **CD@5%** (family C): m(x without M_l) − m(x without M_k), with m = z_k − z_l (`evaluation.scores.contrastive_deletion`). Positive means removing the foil's evidence helps k relative to l, and removing k's evidence does not.
- **CD1@5%** (the CVE comparison only): m(x) − m(x without M_k). It is computed identically for CDEA and CVE, and it is reported for every method as a secondary score.
- **Components:** z_k and z_l before and after each removal are reported beside CD. This guards against margins won by suppressing the foil alone.

### 5.2 Sufficiency contrast
- **SC@a** = m(x ⊕ M_k) − m(x ⊕ M_l), where x ⊕ M keeps M and ROAD-imputes every other pixel. Positive means k's evidence alone favours k, and l's evidence alone favours l.
- It is reported for every method that defines both sides. Descriptive, with intervals.

### 5.3 Discriminative-part contrast (CUB-200)
For an image x with pair (k, l):
1. **Part dissimilarity.** For each attribute group g, P_{c,g} is class c's attribute percentages in g, plus 1e-3, renormalised. Then d_p(k, l) is the mean, over the groups mapped to part p, of the total variation TV(P_{k,g}, P_{l,g}).
2. **Sets.** V(x) is the set of visible parts. D is the 3 parts of V with the largest d_p, and N is the 3 with the smallest. Ties are broken by part index. Images with |V| < 6 are excluded and counted.
3. **Hits.** A part is hit when its keypoint pixel lies inside the binary M_k at area a.
4. **Score.** DPC@a = |hits ∩ D|/3 − |hits ∩ N|/3, in [−1, 1].

A mask that covers the bird uniformly, a centre prior, and a random mask all score about 0. Only a mask that prefers the parts separating the two species scores above 0.

Part-to-attribute map (fixed now, before any scoring):

| Part(s) | Attribute groups |
|---|---|
| beak | bill_shape, bill_length, bill_color |
| crown | crown_color, head_pattern |
| forehead | forehead_color, head_pattern |
| nape | nape_color, head_pattern |
| left eye, right eye | eye_color |
| throat | throat_color |
| breast | breast_color, breast_pattern, underparts_color |
| belly | belly_color, belly_pattern, underparts_color |
| back | back_color, back_pattern, upperparts_color |
| left wing, right wing | wing_color, wing_shape, wing_pattern |
| tail | tail_shape, tail_pattern, upper_tail_color, under_tail_color |
| left leg, right leg | leg_color |
| not mapped | size, shape, primary_color (whole-bird attributes with no part) |

Departures from SCOUT, decided before scoring:
- Total variation instead of an exponentiated symmetric KL: it is bounded and has no temperature.
- Per-image D and N sets instead of a global top-80% triplet cut, so every scored image contributes the same number of parts.

DPC scores M_k only, because the species-separating parts of the image are evidence for the class it shows. It applies to every method, including CVE and SC-CVE. SC-CVE's own Near-KP and Same-KP are reported as exploratory, as its reproduction metrics.

### 5.4 Two-cue recovery (CIFAR-10)
- **Construction.** For a CIFAR-10 image of label y:
  - draw a second class y' ≠ y with generator seed 2000 + s + index;
  - stamp 32-px solid hue cues for y and y' (`_class_hue`) at the two seeded, non-overlapping positions from `patch_layout(index, seed)`;
  - swap which cue takes the first position on odd indices, so position is not tied to role;
  - add no checker.
- **Eligibility.** The classifier's top-2 classes are {y, y'}. Then k = rank 0 and l = rank 1, and cue(k), cue(l) are known exactly.
- **Score.** TCS@a = ½[(mass(M_k on cue(k)) − mass(M_k on cue(l))) + (mass(M_l on cue(l)) − mass(M_l on cue(k)))], where mass is `evaluation.masks.mass_in`, in [−1, 1]. A mask that covers both cues equally scores 0, and a random mask scores about 0.
- Methods that define only one side (CVE, SC-CVE) are not scored.

### 5.5 Decomposition (descriptive)
- **K×K specificity.**
  - For each j ∈ H, M^(j) is the method's map for j at area a.
  - Δ_{j→i} = z_i(x) − z_i(x ⊖ M^(j)), with ROAD removal.
  - Spec@a = mean over j of [Δ_{j→j} − mean_{i≠j} Δ_{j→i}].
  - It is invariant to shifts common to all logits. Scored for the methods with a map for every member of H (section 4.2).
- **Shared validity**, for datasets with C > K (all except brain tumor):
  - SV@a = [s_H(x) − s_H(x ⊖ M_S)] − mean_{j∈H} |c_j(x) − c_j(x ⊖ M_S)|, with s_H and c_j as in `FORMULATION.md` section 3.
  - Removing shared evidence should lower the belief in H without reordering H.
  - Scored for CDEA, the analytic decomposition, and the random floor.

### 5.6 Sanity and reporting
- **Model randomisation.**
  - Spearman ρ under cascading randomisation, through to full randomisation.
  - Every method, ResNet-50, seed 0, n = 64, every contrastive dataset.
  - A method with |ρ| ≥ 0.3 at full randomisation is flagged in the main table.
- **Mask checks.** Area (a ± tolerance), connected components, and CDEA's allocation entropy.
- **Support.** n images, the fraction with valid K, and failure counts, for every number.

### 5.7 Exploratory (labelled as such)
- Deletion and insertion AUC, ROAD MoRF/LeRF.
- Seed and input-noise stability.
- Lesion overlap on HAM10000 and brain tumor, with the null family (claim N1).
- Correct vs incorrect stratification.
- SC-CVE's Near-KP and Same-KP on CUB.
- DPC with a 16-px keypoint tolerance instead of the exact pixel.
- Per-image paired tests outside family T.

## 6. Selection on val (before the freeze)

All of this uses the val splits, seed-0 checkpoints, and n = 64 images per dataset. Nothing here reads test.

### 6.1 Degenerate routes
The routes are redefined for this formulation in `FORMULATION.md` section 11. `scripts/check_degenerate.py` is updated to match before it runs.
- **Gates:** D1, D2, D4 and D5 (`analysis.selection.GATING_ROUTES`).
- **Reported only:** D3, the share of u_k that comes from rivals rising. For "k rather than l", evidence that lowers l is legitimate contrastive evidence.
- The reader sees z_k and z_l separately (section 5.1).

### 6.2 Candidate grids (at most 4 configurations per method and backbone)

| Method | Candidates | Count |
|---|---|---|
| CDEA | ResNet-50: initialisation {Grad-CAM, IG} × η {0.05, 0.2}, T = 100. ViT: IG initialisation × η {0.05, 0.2} × T {50, 100} | 4 |
| Margin attribution | {Grad-CAM, IG-16, IG-32} (ViT: IG-16, IG-32) | 3 (2) |
| Extremal Perturbations, deletion | max_iter {300, 800} × `smooth` {0 (TorchRay default), 0.1} | 4 |
| Extremal Perturbations, preservation | Same | 4 |
| RISE | n_masks {2000, 4000} × cell size {7, 8}, p1 0.5 | 4 |
| Chefer | `start_layer` {0, 1} | 2 |
| Base evidence, analytic decomposition | Backend {Grad-CAM, IG} (ViT: IG) | 2 (1) |
| CVE, SC-CVE, contrastive Grad-CAM | Authors' defaults | 1 |

- **Rule** (`analysis.selection.select_config`).
  - For each dataset and backbone, maximise mean CD@5% on that dataset's val, among eligible candidates.
  - Ties keep the earliest.
  - The same rule applies to every method.
  - If no CDEA candidate is eligible on a dataset, the freeze stops and the formulation is revisited. There is no fallback.
- **Cost fallback, fixed now.** The grid cost is estimated from the seed-0 summaries before launch. If it does not fit the compute window, both Extremal Perturbations grids drop to the authors' default on the six non-dev datasets. That is recorded in `PROGRESS.md`. No other method's grid changes.
- **Reporting.** Every method's grid, scores, and selection go into `results/paper/selection/`. Both the authors'-default row and the val-tuned row are reported for each baseline.

### 6.3 Family T comparators
- For DPC (on CUB val) and for TCS (on two-cue val), the comparator is the non-CDEA method with the highest val mean on that metric.
- It is chosen before CDEA's val score on that metric is computed. The baselines' family T val scores are run and the choice is logged in `PROGRESS.md` first.

## 7. Statistical protocol

- **Family C** (primary; ResNet-50, 8 datasets).
  - **Unit:** the dataset. Average over seeds inside each dataset, then compare methods across datasets. Every statistic is computed from raw per-image records by `analysis/stats.py`.
  - **Test:** two-sided exact Wilcoxon signed-rank, Holm over 3 comparisons:
    1. CDEA vs margin attribution (the variant selected per dataset), on CD@5%;
    2. CDEA vs contrastive Extremal Perturbations, deletion variant, on CD@5%;
    3. CDEA vs CVE, on CD1@5%. This comparison stays in the family only if the CVE reproduction on CUB is recorded before the freeze (`docs/paper/G0.md`). Otherwise the family is comparisons 1 and 2, Holm over 2, and CVE is an exploratory row.
- **Family T** (secondary, pre-registered; ResNet-50). Holm over 2 comparisons:
  1. CDEA vs the section 6.3 comparator, on DPC@5% (CUB-200);
  2. CDEA vs the section 6.3 comparator, on TCS@5% (two-cue CIFAR-10).
  - **Unit:** the image, with seeds pooled. Each claim concerns one dataset, so the image is the sampling unit. Model variance enters through the seeds.
  - **Statistic:** the mean over seeds of the per-seed mean paired difference.
  - **Test:** a two-sided sign-flip permutation test on the per-image differences, flipping within each seed (10,000 permutations, generator seed 0).
  - **Interval:** a two-level percentile bootstrap that resamples seeds, then images within each seed (10,000 resamples, seed 0).
  - C2 and C3 are claimed only if their Holm-adjusted p is below 0.05.
- **Descriptive, with intervals, not tested:**
  - SC@5%;
  - K×K specificity and shared validity (C4). They are aligned with CDEA's own payoffs, so a test would add little evidence beyond C1;
  - randomisation (C5);
  - ViT-B/16 rows;
  - CDEA vs base evidence, and vs the analytic decomposition, in the main table with the family C statistics.
- **Report:** wins out of N (family C), the mean difference, the 95% interval, and the Holm-adjusted p.
- **Power, stated in advance.**
  - Family C with n = 8: the exact two-sided p is 0.0078 for 8/8 wins, 0.0156 when the one loss has the smallest |difference|, 0.0234 at T = 2, and 0.039 at T = 3. Against Holm's thresholds (0.0167, 0.025, 0.05), the first rejection needs T ≤ 1, the second T ≤ 2, and the third T ≤ 3.
  - Family C reduced to 2 comparisons: Holm's thresholds are 0.025 and 0.05, so the first rejection needs T ≤ 2 and the second T ≤ 3.
  - Family T: with 1,000 or more scored images per comparison (5 seeds × at least 200), the binding risk is the comparator choice, which is fixed on val (6.3), not the sample size.

  A non-significant result means the experiment cannot separate the methods. It does not mean the methods are equivalent.
- **Robustness checks.** Each confirmatory claim is checked under:
  - leave each dataset out (family C);
  - drop the two dev datasets;
  - ViT-B/16 (with Chefer and margin IG);
  - the blur operator;
  - a ∈ {2.5%, 10%};
  - margin attribution with each variant held fixed across datasets;
  - native vs grid-pooled resolution for pixel-level baselines;
  - Extremal Perturbations at CDEA's forward-pass count;
  - CDEA's 5% allocation scored at other areas, against re-optimisation per area;
  - unique plus shared on both sides (CDEA);
  - per-seed sign consistency (family T);
  - effect size against measured run-to-run variation (section 10).

  A claim is called robust only if it survives every check. Otherwise the text names the check it fails.

## 8. Ablations

- **Setup:** ResNet-50 on CIFAR-100, CUB-200, brain tumor, and ImageNet (all non-dev), seeds 0–2, the same n as the main grid. Scores: CD@5% and its components, K×K specificity, and DPC on CUB.
- **Format:** one table, one row per change. Exploratory: it reports means and per-dataset wins, not tests.

| ID | Variant | What it isolates |
|---|---|---|
| A1 | Independent players: no exclusivity, no shared player, K separate problems | Joint allocation, the central mechanism |
| A2 | No shared player | The chain-rule split into shared and unique |
| A3 | Preservation payoff (keep instead of delete) | The choice of necessity |
| A4 | Pair-only game (K = 2) | The context of the other hypotheses |
| A5 | Uniform vs evidence initialisation, across 6 evidence backends (`library_adapters`) | Dependence on the base evidence |
| A6 | the earlier formulation (tag `framing-v1`)'s hard top-mass projection instead of Sinkhorn | The trapped-support fix |
| A7 | Steps {25, 50, 100, 200}, against deletion Extremal Perturbations at the same forward-pass count | Cost–quality and the compute confound |
| A8 | the earlier formulation (tag `framing-v1`) (the Oct 7 objective, runner default) | What the revision changed |
| A9 | First-order solution: top units of the margin gradient × (x − b(x)), `FORMULATION.md` section 9 | What optimisation adds beyond its own linearisation |
| A10 | Boundary-robust payoff, `FORMULATION.md` section 4.1 (under test). If it is adopted, the roles swap and A10 is the fixed-boundary payoff | Dependence of the payoff on the exact cell border |

Confounds are stated per row: forward passes, number of masks, and K.

## 9. Cost accounting
- **Hardware-independent unit:** forward and backward passes per explanation, counted by a wrapper around the model. CDEA's count includes its base-evidence pass. CDEA uses T × P per image per area (`FORMULATION.md` section 7).
- **Wall-clock** per image per method, reported per platform (MPS, CUDA).
- **Excluded:** classifier training and data loading. This is stated in the table caption.

## 10. Execution
- **One dataset, one machine.** Every method and seed of a dataset runs on the same machine. Platform differences therefore never enter a within-dataset method comparison. Assignments go into `PROGRESS.md` before launch. ImageNet runs on Spark.
- **Cross-platform check.** One cell (CIFAR-10, ResNet-50, seed 0, every core method, n = 64) runs on both machines. Report the maximum absolute per-image difference in CD@5%. Results from the two machines are compared only through dataset-level means.
- **Run-to-run variation:** repeat that same cell 3× on each machine.
- **Chunking.** Cells run in image chunks (`--image-batch`). A chunked cell must equal the whole-sample cell. `tests/test_image_batch.py` checks this end to end through `run_contrastive` (PR #28). CDEA's per-image loss makes it hold by construction.
- **Records.** One file set per cell, at `results/paper/cells/<split>/contrastive/<dataset>/<backbone>/seed<s>/`, beside a `summary.json`:
  - `records.csv.gz`: one row per image × method × area × operator. It holds:
    - dataset, method, config hash, seed, image index, k, l;
    - CD, CD1, SC, z_k and z_l before and after each removal, the failure flag;
    - mask area, forward/backward counts, wall-clock;
    - code commit and platform.
  - `kxk.csv.gz`: one row per image × method × area × (j, i) with Δ_{j→i}, plus s_H and c_j before and after removing the shared map where one exists.
  - `parts.csv.gz` (CUB only): one row per image × method × area. It holds the visible parts, the D and N sets, the hits, and DPC.
  - `twocue.csv.gz` (two-cue only): one row per tried image. It holds y, y′, the eligibility flag, and per method and area the four cue masses and TCS.

  A cell is complete only when its done marker exists (`scripts/completion_check.py`). Record assertions: masks in [0, 1], area = a ± tolerance, finite values, n as planned, and CDEA marginals within 1e-3.
- **Executor.**
  - `scripts/paper_run.py` runs one cell. `scripts/launch_paper_eval.py` runs the grid, resumably, with `--datasets` per machine. `scripts/smoke_e2e.py` runs every unit, method, ablation, and selection candidate on a few val images.
  - Code:
    - `evaluation/run_models.py`: loading, always through `load_checkpoint_into`;
    - `evaluation/run_data.py`: samples with seed 1000 + s through the locked loaders;
    - `evaluation/run_methods.py`: section 4 maps, ablations, candidates;
    - `evaluation/run_cell.py`: scoring, records.
  - New code before the freeze:
    - CDEA;
    - deletion Extremal Perturbations;
    - the analytic decomposition;
    - the scorers for SC, DPC, TCS, K×K and SV;
    - the two-cue builder;
    - the stratified permutation test and the two-level bootstrap in `analysis/stats.py`.
  - Run instructions are in `docs/paper/SPARK_RUNBOOK.md`.

## 11. Gates this plan depends on
- **G0.** The entry rule in `docs/paper/G0.md`. Every Core baseline has:
  - pinned or named code;
  - passing toy tests and cross-checks;
  - a merged PR;
  - a dossier entry.

  Deletion Extremal Perturbations and the analytic decomposition are new Core entries and need all four. Reproductions (B3) are reported in the appendix and do not gate, except CVE (section 7). The CVE reproduction runs through the Goyal port in the vendored SC-CVE code on CUB.
- **G1** (replaces the earlier read-out in `docs/paper/G1.md`). The earlier formulation is tag `framing-v1`. The rule below is fixed before any number from this formulation is read.
  - **Pilot.**
    - CDEA at its default configuration (`FORMULATION.md` section 8), not a selected one.
    - Dev val: CIFAR-10 and HAM10000. ResNet-50, seeds 0 and 1, n = 64 per cell.
  - **Read-out.** For each dev dataset, the mean paired CD@5% difference against:
    - the better margin-attribution variant on that dataset;
    - deletion Extremal Perturbations at 5%.

    Each comes with a 95% bootstrap interval over images, and the z_k and z_l drops are reported beside them.
  - **Stop rule.**
    - If CDEA is ahead of both comparators on both dev datasets (mean difference above 0), the paper continues as a method paper: selection (section 6), seeds 2–4, families C and T.
    - Otherwise, the paper takes the evaluation-paper route (`G1.md`), with the earlier formulation (tag `framing-v1`) and CDEA reported as ablations, and **the formulation is not revised again**.
  - **Prerequisites:** this formulation merged with its tests, deletion Extremal Perturbations merged, and the chunk-scale fix (PR #28) on `main`.

## 12. Freeze procedure
The plan is frozen only after all of the following are recorded in this file and in `PROGRESS.md`:
1. G0 passed (tag `g0-baselines`). The G1 decision logged (tag `g1-pilot`). The CVE reproduction recorded, or family C reduced to two comparisons as in section 7.
2. The this formulation degenerate routes (6.1) checked on val. D1, D2, D4 and D5 closed for every selected CDEA configuration.
3. Selected configurations and their hashes for every method, dataset and backbone (6.2). The family T comparators (6.3).
4. The CUB part scorer and the two-cue builder committed with tests. The two-cue classifiers trained for seeds 0–4.
5. n per seed and backbone, test-sample seeds (2.4), and machine assignments (10).
6. The executor's smoke run (`results/paper/smoke/REPORT.md`) shows no method errors on the machine that runs each unit.
7. The cross-platform cell and the run-to-run repeats done on val.

Then add a line reading `frozen: true` at the top of this file, commit, and tag `eval-plan-frozen`. After the freeze, any change to this file is a new dated section with its reason. Results computed before that section are reported under the old plan.
