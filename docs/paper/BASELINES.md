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

## Extremal Perturbations

- **Reviewer question.** Fong et al. already optimise a mask at a fixed area.
- **Citation.** Ruth C. Fong, Mandela Patrick, and Andrea Vedaldi, Understanding Deep Networks via Extremal Perturbations and Smooth Masks, ICCV 2019.
- **Source.** TorchRay `extremal_perturbation`, commit `6a198ee61d229360a3def590410378d2ed6f1f06`, vendored at `third_party/torchray`. Licence CC BY-NC 4.0.
- **Patches.** `Perturbation.to` keeps the moved pyramid. Upstream called `tensor.to` and discarded the result. `MaskGenerator` passes `indexing="ij"` to `torch.meshgrid`, which is the order current PyTorch uses and which a later release will require by name. Neither change alters the mask update.
- **Defaults.** Area list `[0.1]`, blur perturbation, preserve variant, `simple_reward`, 800 iterations, step 7, sigma 21, jitter on (a deterministic horizontal flip on even steps), SGD learning rate 0.01, momentum 0.9. The pointing-game setting in the TorchRay benchmark uses areas `{0.025, 0.05, 0.1, 0.2}`, sums those masks, and smooths with a Gaussian whose standard deviation is 9% of the short side. The contrastive adapter scores \(z_k - z_l\) through `reward_func`. TorchRay's own `contrastive_reward` is \(z_c - \max_{c' \neq c} z_{c'}\), which is a different target and is not the one used here.
- **Reproduction target.** TorchRay's published pointing-game table, VOC 2007 test, all / difficult: VGG16 88.0 / 76.1, ResNet50 88.9 / 78.7, for `extremal_perturbation`, averaged over 3 runs. That run uses the excitation-backprop paper's fine-tuned classifiers and the full 4952-image test set. Those classifiers and PASCAL VOC are not on disk, and TorchRay describes this benchmark as cluster-scale. It has not been run. The CI check is the toy square and an identity check against `extremal_perturbation` itself.
- **Conversion.** The native mask at the requested area. It does not go through the shared top-a threshold. The area is the function's `areas` argument.
- **Failure modes.** The call is one image at a time. A batch is a loop. TorchRay disables `requires_grad` on the classifier; the wrapper turns those flags back on. The kernel defaults (step 7, sigma 21) are for images around 224 px. A 32 px toy uses step 2 and sigma 4 so the mask can sit on an 8 px square. That is an argument to the same function.
- **Cost.** One forward and backward of the classifier per iteration. The authors' default is 800 iterations per class per image.

## Not in this harness yet

| Method | Reviewer question | Stage |
|---|---|---|
| Extremal Perturbations on the full pointing game | The published VOC number above. | Still needs VOC 2007 and the fine-tuned classifiers. |
| Counterfactual Visual Explanations | Goyal et al. already answer why k rather than l on fine-grained data. | S12 |
| Random area-a mask | Is the metric satisfied by chance? | The floor is `baselines.adapter.random_floor`. |
| Attribution difference | Why not subtract two heatmaps? | S13 |
| SpRAy | Explanation-based shortcut discovery already exists. | S13 |
| Per-environment Extremal Perturbations | Same mask method, differenced across environments. | S13 |
| Contrastive Grad-CAM | Gradient-based contrastive saliency exists. | S14 |
| SCOUT | Discriminant why-A-not-B explanations exist. | S14 |
| RISE | Sampling-based perturbation would do the same. | S14 |

Method-specific conversions that are not the shared top-a rule (CVE edit order, SpRAy cluster relevance, Extremal Perturbations' native area mask) are added with the method that needs them.
