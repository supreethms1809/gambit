# Baselines

Dossier for the methods in the baseline protocol (`docs/paper/PLAN.md`, B1–B7).
A comparison number is written by `evaluation/` when that run exists. This file
records the pin, the conversion, and the check each method has to pass first.

## Shared rules

- Hypotheses come from `baselines.hypotheses.shared_hypotheses`. The foil is rank 1 of that list. A method does not choose its own classes.
- Every map goes through `baselines.adapter.adapt_scores`, which calls `evaluation.masks.to_budget_mask`: bilinear upsample, then the top-a fraction of pixels, ties broken by a seeded jitter.
- The budget conversion does not clamp. An all-negative map keeps its order, and the least-negative entries receive the area.
- A row with a non-finite value, or a row with no entries, is a failure. `budget_or_floor` replaces that row with `random_floor`, a seeded area-a mask, and leaves the row in the batch.
- Library pins, checked in `tests/test_baselines.py`: `grad-cam` 1.5.7 and `captum` 0.9.0 (`baselines/versions.py`). This repo has no `environment.yml`.
- GradientShap draws its path coefficients with NumPy and its noise with `torch.normal`. A repeat seeds both `torch.manual_seed` and `numpy.random.seed` before the call. RISE and SpRAy get the same rule when those methods land.
- Paper checkpoints take raw `[0, 1]` input. `NormalizedModel` is the wrapper for a library call that expects ImageNet normalisation inside the model. The CIFAR-10 checkpoint used for the cross-check was trained on raw input, so both implementations call that model directly.

## Margin attribution

- **Reviewer question.** Why not just attribute the margin \(z_k - z_l\)?
- **Citation.** Grad-CAM (Selvaraju et al. 2017) and Integrated Gradients (Sundararajan et al. 2017), applied to the logit margin.
- **Source.** `pytorch-grad-cam` 1.5.7 with `MarginOutputTarget`. Captum 0.9.0 `IntegratedGradients` on a forward that returns \(z_k - z_l\), `method="riemann_right"`.
- **Patches.** None.
- **Defaults.** IG uses 8 steps in the harness tests and in `scripts/crosscheck_evidence.py`. Authors' IG step counts and the val-tuned step count are recorded when the method is tuned.
- **Reproduction.** The toy-model test in `tests/test_baselines.py`. Class 0 is a red square, class 1 is a blue square, and the margin map has to land on the kept square. There is no single published number for this target.
- **Conversion.** The shared budget conversion. The map is already at pixel resolution.
- **Failure modes.** A model with fewer than two valid hypotheses has no foil. That call raises, and the caller records a failure rather than inventing a class.
- **Cost.** One Grad-CAM backward, or one integrated-gradient path, per image.

## Base evidence (Grad-CAM, integrated gradients)

- **Reviewer question.** Does allocation add anything over its own input?
- **Source.** `GradCAMRegionsProvider` and `IntegratedGradientsRegionsProvider`. The references are the same library versions as above.
- **Patches.** None. The check does not go through pytorch-grad-cam's display resize. That function min-max scales the map and resizes it to the image. The comparison pools the target-layer ReLU CAM with the same adaptive average pool our provider uses. IG is compared under the right Riemann sum. Both sides sum channels and clamp at zero, which is what `IntegratedGradientsRegionsProvider` returns. Captum's default quadrature is Gauss-Legendre.
- **Defaults.** IG steps match on both sides of a check. The 100-image run uses 8 steps.
- **Reproduction.** `scripts/crosscheck_evidence.py` on 100 CIFAR-10 val images. The process exits 0 only when the mean Spearman of each pair is at least 0.95. The JSON is `results/paper/crosscheck/val100.json`.
- **Conversion.** The shared budget conversion after pooling to the region grid.
- **Failure modes.** A Grad-CAM map can be all zeros when the target-layer activations are not the non-negative maps Grad-CAM assumes. The provider warns. An all-zero map that both sides produce still agrees. A non-finite map is a failure under the shared rule.
- **Cost.** One backward per class for Grad-CAM. One path of forward and backward passes per class for IG.

## Counterfactual visual explanations

- **Reviewer question.** Goyal et al. already answer why class k rather than class l on fine-grained data.
- **Citation.** Yash Goyal, Ziyan Wu, Jan Ernst, Dhruv Batra, Devi Parikh, and Stefan Lee, Counterfactual Visual Explanations, ICML 2019.
- **Source.** No official release is assumed. `baselines/cve.py` is Algorithm 1: greedy sequential search. Each step replaces one query cell with one distractor cell, the pair that most raises the distractor's log-probability, and stops when the argmax is the distractor class. Query cells already replaced are excluded. A distractor cell may be copied again. Ties keep the earliest query index, then the earliest source index.
- **Patches.** None. The pool-then-linear scorer is the same replacement, written as a change to the pooled vector. `tests/test_cve.py` checks it against editing the cells and calling the head.
- **Defaults.** The paper's reported CUB and MNIST numbers use this exhaustive search, on the last convolutional feature map. The continuous relaxation in their Section 2.3 is a different procedure and is not this adapter. The search runs until the class flips, or until every query cell has been replaced.
- **Reproduction target.** CUB-200, their VGG-16 (79.4% test accuracy), last conv map 7×7×512. Mean edits until the decision changes: 7.4 for a random distractor class, 5.3 for an attribute nearest-neighbor class. After the flip, edited regions fall inside the bird segmentation 97% of the time, near a keypoint 75% (query) and 80% (distractor), and on the same keypoint in both birds 20% of the time. Their MNIST CNN (98.4% test, features 4×4×20) takes 2.67 edits on average. That CUB run has not been started: it needs their classifier and the keypoint and segmentation annotations. The CI check is the toy square.
- **Conversion.** `earliest_edits` keeps a prefix of the query cells in edit order. Cells the search never replaced stay off. This is not the shared top-a threshold.
- **Failure modes.** A non-finite feature map raises. If the query map is already predicted as the distractor class, the edit list is empty. If the class never flips, `flipped` is false and the caller counts a failure rather than dropping the image.
- **Cost.** One decision-network evaluation per candidate pair per step. On the paper's 7×7 map that is 2,401 evaluations per edit.

## Not in this harness yet

| Method | Reviewer question | Stage |
|---|---|---|
| Extremal Perturbations | Fong et al. already optimise a mask at a fixed area. | S11 |
| CVE on the CUB edit counts | The published 7.4 / 5.3 edit counts above. | Still needs their VGG-16 and the CUB keypoint annotations. |
| Random area-a mask | Is the metric satisfied by chance? | The floor is `baselines.adapter.random_floor`. |
| Attribution difference | Why not subtract two heatmaps? | S13 |
| SpRAy | Explanation-based shortcut discovery already exists. | S13 |
| Per-environment Extremal Perturbations | Same mask method, differenced across environments. | S13 |
| Contrastive Grad-CAM | Gradient-based contrastive saliency exists. | S14 |
| SCOUT | Discriminant why-A-not-B explanations exist. | S14 |
| RISE | Sampling-based perturbation would do the same. | S14 |

Method-specific conversions that are not the shared top-a rule (CVE edit order, SpRAy cluster relevance, Extremal Perturbations' native area mask) are added with the method that needs them.
