# Baselines

Dossier for the methods in the baseline protocol (`docs/paper/PLAN.md`, B1–B7).
A comparison number is written by `evaluation/` when that run exists. This file
records the pin, the conversion, and the check each method has to pass first.

## Shared rules

- Hypotheses come from `baselines.hypotheses.shared_hypotheses`. The foil is rank 1 of that list. A method does not choose its own classes.
- Every map goes through `baselines.adapter.adapt_scores`, which calls `evaluation.masks.to_budget_mask`: bilinear upsample, then the top-a fraction of pixels, ties broken by a seeded jitter.
- The budget conversion does not clamp. An all-negative map keeps its order, and the least-negative entries receive the area.
- A row with a non-finite value, or a row with no entries, is a failure. `budget_or_floor` replaces that row with `random_floor`, a seeded area-a mask, and leaves the row in the batch.
- Library pins, checked in `tests/test_baselines.py`: `grad-cam` 1.5.7 and `captum` 0.9.0 (`baselines/versions.py`). The project environment is `requirements.txt`, conda env name `gambit`.
- GradientShap draws its path coefficients with NumPy and its noise with `torch.normal`. A repeat seeds both `torch.manual_seed` and `numpy.random.seed` before the call. RISE draws its masks from NumPy's global state, so a repeat seeds `numpy.random.seed` before the call. SpRAy seeds k-means with `random_state`.
- Paper checkpoints take raw `[0, 1]` input. `NormalizedModel` is the wrapper for a library call that expects ImageNet normalisation inside the model. The CIFAR-10 checkpoint used for the cross-check was trained on raw input, so both implementations call that model directly.
- `NormalizedModel` is currently unused outside its own test: every adapter calls the raw checkpoint directly, which is correct for raw-trained checkpoints and would be wrong for checkpoints trained on normalised input. Wiring it in for a normalised-weights setting (e.g. ImageNet-S public weights) is open work, not a silent default: wrapping a raw-trained checkpoint would score a broken model.

## Margin attribution

- **Reviewer question.** Why not just attribute the margin \(z_k - z_l\)?
- **Citation.** Grad-CAM (Selvaraju et al. 2017) and Integrated Gradients (Sundararajan et al. 2017), applied to the logit margin.
- **Source.** `pytorch-grad-cam` 1.5.7 with `MarginOutputTarget`. Captum 0.9.0 `IntegratedGradients` on a forward that returns \(z_k - z_l\), `method="riemann_right"`.
- **Patches.** None.
- **Defaults.** IG uses 8 steps in the harness tests and in `scripts/crosscheck_evidence.py`. Authors' IG step counts and the val-tuned step count are recorded when the method is tuned.
- **Reproduction.** The toy-model test in `tests/test_baselines.py`. Class 0 is a red square, class 1 is a blue square, and the margin map has to land on the kept square. There is no single published number for this target.
- **Clamp asymmetry.** Base-evidence IG clamps the channel sum at zero (negatives discarded) while margin IG keeps the signed map so opposing channels lose the top-a ranking. CDEA therefore sees only positive evidence and the margin baseline sees signed evidence: different information, not just different processing. The direction of any resulting bias is open.
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

## SC-CVE (semantically consistent counterfactuals)

- **Reviewer question.** Fine-grained "why k rather than l", with a semantic-consistency prior (the recent ResNet-50 row).
- **Citation.** Sam Vandenhende, Stamatios Georgoulis, and Luc Van Gool, Making Heads or Tails: Towards Semantically Consistent Visual Counterfactuals, ECCV 2022.
- **Source.** `facebookresearch/visual-counterfactuals` at `cd879d6d41e2a24a6d74f398811b8226b57c39b4` (archived), vendored at `third_party/sc_cve`. Licence CC BY-NC 4.0, the same terms as the TorchRay we already vendor.
- **Patches.** A `device` argument on `compute_counterfactual` and `_find_single_best_edit` (upstream called `.cuda()`; the driver still asserts CUDA for the published repro). `_CallableHead` in `baselines/sc_cve.py` wraps our closure split-heads in an `nn.Module` because the vendored search calls `.eval()`. Neither changes the edit scores.
- **Defaults.** `counterfactuals_ours_cub_res50.yaml`: λ=0.4, temperature 0.1, `topk=0.2` kNN prefilter, up to 20 distractors. The Goyal configs in the same repo (`counterfactuals_goyal_cub_*.yaml`) set λ=0, no prefilter, 1 distractor. SwAV ResNet-50 at `facebookresearch/swav` @ `06b1b7c`, weights `swav_800ep_pretrain.pth.tar` hash-recorded at download and loaded offline.
- **Reproduction target.** Their README, ResNet-50 CUB, all edits: 2.9 edits, Near KP 64.6, Same KP 31.1. Their Goyal port in the same repo: 3.5 edits, Near KP 54.0, Same KP 7.4. Not run: needs a CUDA node plus the CUB ResNet checkpoint (191.7 MB) and SwAV weights (113.7 MB) downloads, each with approval at run time.
- **Audit (2026-10-08).** The objective in `_find_single_best_edit` is argmax of `log p(distractor | edit) + λ · log semantic_prob(edit)` (Eq. 4). The exclusion drops every pair sharing the used query cell *or* the used distractor cell, while our CVE reimplementation lets a distractor cell be copied again. The distractor class comes from the confusion matrix (most-confused class); ours is the foil. Their ResNet head keeps `layer4[1:]` on `layer4[0]` features; ours is avgpool+fc on the full `layer4` output. The `while` loop has no exit when edits run out (`argmax` on empty); the wrapper maps that to `flipped=False`.
- **Conversion.** Edit-order rank encoding shared with `cve_pair`: earliest edit scores highest, untouched cells 0. One-sided, `l=None`, scored with CD1.
- **Failure modes.** No correctly-classified foil image in the tries writes a NaN row (floored downstream), as in CVE. A non-finite feature map raises. No flip after exhaustion records `flipped=False`, never hangs. Missing SwAV weights raise an informative error. SwAV is ImageNet self-supervised, so its semantic prior is weak on HAM10000 and brain MRI.
- **Cost.** One head evaluation per candidate pair per step over the joint `49 × (N × 49)` set after the prefilter, per edit. Per-image wall time and memory at 200 images on CIFAR-10 and HAM10000 feed `scripts/launch_paper_eval.py` scheduling before the grid runs.

## Extremal Perturbations, deletion variant

- **Reviewer question.** The preservation mask answers sufficiency. Necessity needs a deletion mask.
- **Source.** TorchRay `extremal_perturbation` with `variant=DELETE_VARIANT` and the margin reward `z_k - z_l`. The preservation call stays as the sufficiency comparator.
- **Status.** Not re-audited. The deletion flag is new, and `scripts/crosscheck_evidence.py` has not been re-run for this variant.
- **Conversion.** The shared budget conversion, same area as the preservation mask.

## Analytic decomposition

- **Reviewer question.** Why not subtract the weakest hypothesis map from each class map?
- **Source.** `baselines` analytic pair: unique map `E_k - min_j E_j`, shared map `min_j E_j`, with `E` the same base evidence used to initialise CDEA.
- **Status.** Not re-audited. This entry replaces the earlier pairwise subtraction.
- **Conversion.** The shared budget conversion.

## Contrastive Grad-CAM

- **Reviewer question.** Gradient-based contrastive saliency exists.
- **Citation.** Mohit Prabhushankar, Gukyeong Kwon, Dogancan Temel, and Ghassan AlRegib, Contrastive Explanations in Neural Networks, ICIP 2020.
- **Source.** `pytorch-grad-cam` 1.5.7 with `CrossEntropyContrastTarget`. The scalar is the cross-entropy of the logits toward the contrast class Q, which is hypothesis rank 1. The official repository is https://github.com/olivesgatech/Contrastive-Explanations. Their qualitative examples (spoonbill versus flamingo, a bull mastiff, Stanford Cars) have not been run.
- **Patches.** None. The target is the paper's recognition loss. It is not the logit margin in `baselines/margin.py`.
- **Defaults.** The same last-convolution Grad-CAM pooling as the margin adapter.
- **Reproduction.** The toy-model test in `tests/test_extended_baselines.py`. Class 0 is a red square, class 1 is a blue square, and the map has to land on the kept square. The authors' qualitative figures are the published check and have not been run.
- **Conversion.** The shared budget conversion.
- **Failure modes.** A model with fewer than two valid hypotheses has no contrast class. That call raises.
- **Cost.** One Grad-CAM backward per image.

## SCOUT

- **Reviewer question.** Discriminant why-A-not-B explanations exist.
- **Citation.** Pei Wang and Nuno Vasconcelos, SCOUT: Self-aware Discriminant Counterfactual Explanations, CVPR 2020.
- **Source.** https://github.com/peiwang062/SCOUT is public. It is a training pipeline for CUB and ADE: an AlexNet, VGG, or ResNet plus a separate hardness predictor, with weights on Google Drive. It does not explain an arbitrary frozen classifier.
- **Attempt.** The repository was inspected and not vendored. A drop-in call would require their hardness predictor and a CUB or ADE retrain. That retrain is not this adapter.
- **Reproduction.** Not run. The dossier records the omission. No Google Drive weights were downloaded.

## RISE

- **Reviewer question.** Sampling-based perturbation would do the same.
- **Citation.** Vitali Petsiuk, Abir Das, and Kate Saenko, RISE: Randomized Input Sampling for Explanation of Black-box Models, BMVC 2018.
- **Source.** Official PyTorch class `RISE` in `explanations.py`, commit `d91ea006d4bb9b7990347fe97086bdc0f5c1fe10`, vendored at `third_party/rise`. Licence MIT. TorchRay's RISE is a different reimplementation and is not this pin.
- **Patches.** `RISE` takes a `device` and moves the masks there. When CUDA is available the default is still CUDA. `generate_masks` skips the `masks.npy` write when `savepath` is `None`. The mask draw and the weighted sum are unchanged. The official forward uses the model's raw output, not a softmax.
- **Defaults.** The ImageNet ResNet-50 setting: 8000 masks, cell grid 7, keep-probability 0.5, input 224. VGG-16 in the paper uses 4000 masks. The toy check passes a smaller mask count and grid as arguments of the same function.
- **Reproduction target.** Table 1, ImageNet validation, deletion (lower is better) and insertion (higher is better). ResNet-50: deletion \(0.1076 \pm 0.0005\), insertion \(0.7267 \pm 0.0006\). VGG-16: deletion \(0.0980 \pm 0.0025\), insertion \(0.6663 \pm 0.0014\). That run has not been started: ImageNet-1k is not on disk. The CI check is the toy square, plus an identity check against the vendored `RISE.forward`.
- **Conversion.** The shared budget conversion. The per-class map is the official weighted sum. The margin map uses the same masks and weights each one by \(z_k - z_l\).
- **Signed-weight variant.** Official RISE weights by nonnegative class scores. The margin map weights by a signed logit margin, so masks can cancel each other — a novel variant, not published RISE. Characterise negative-weight behaviour before it anchors a confirmatory comparison.
- **Failure modes.** Fewer than one mask raises. A keep-probability outside \((0, 1]\) raises. A non-finite map raises. The margin call raises when the foil is missing.
- **Cost.** One forward of the classifier per mask. The authors' ResNet-50 setting is 8000 forwards per image.

## Random area-a mask

- **Reviewer question.** Is the metric satisfied by chance?
- **Source.** `baselines.adapter.random_floor`. A constant field makes every entry a tie, and the shared top-a conversion draws the area from the seed.
- **Patches.** None.
- **Defaults.** The area is the same fraction `a` as the method it is compared with. The seed is the caller's seed.
- **Reproduction.** There is no published number. `tests/test_baselines.py` checks that a constant map keeps the area, that the seed fixes the ties, and that a non-finite row is replaced by this floor and kept in the batch.
- **Conversion.** The shared budget conversion. The floor is that conversion applied to a constant map.
- **Failure modes.** The floor is the score for a failed row. It is not itself a failure.
- **Cost.** One seeded ranking of the pixels. No model call.

## Gap attribution

- **Reviewer question.** Does the shortcut payoff's own first-order path integral already find the units that change the prediction?
- **Source.** `baselines/gap_attribution.py`. Integrated gradients of the log-odds from `x_e` to `x_id`, pooled with `pool_sum` and signed by `g_e`. The robust companion is the min over environments of the same integral from the deletion baseline.
- **Patches.** None. Captum is not used. The sum is the path integral in `cdea/shift.py`.
- **Defaults.** 16 steps. Fast smoke uses 4.
- **Reproduction.** On a model linear in the pixels the pooled map sums to `g_e`. `tests/test_shift.py` checks that identity.
- **Conversion.** The shared top-a conversion.
- **Failure modes.** Fewer than two environments raises. A one-class model raises inside the log-odds.
- **Cost.** One backward per step per environment other than id, plus the same for the robust companion.

## Attribution difference

- **Reviewer question.** Does an elementwise difference of per-environment maps already separate the shortcut?
- **Source.** `baselines/shift_maps.py`. The robust map is the elementwise minimum. The shortcut map is the largest absolute gap from the id map. The input maps are Grad-CAM on ResNet-50 and integrated gradients on ViT-B/16.
- **Patches.** None. The reduction is the one in the file.
- **Defaults.** The evidence step count is the shared knob.
- **Reproduction.** There is no published number for this reduction.
- **Conversion.** The shared top-a conversion.
- **Failure modes.** Fewer than two maps raises. A non-finite map raises.
- **Cost.** One evidence pass per environment.

## SpRAy

- **Reviewer question.** Does clustering relevance maps separate a shortcut from the object?
- **Source.** `baselines/spray.py`, with Zennit and CoRelAy under `third_party/`. Each environment is attributed with Zennit's `EpsilonPlus`, clustered, and replaced by its cluster mean. The robust and shortcut maps are then the attribution-difference reduction.
- **Patches.** None in the clustering steps. The eigenvalue count, the neighbor count, the cluster count, and the k-means seed are chosen here.
- **Defaults.** Two clusters. Eigenvalues and neighbors shrink when the batch is smaller than the published setting.
- **Reproduction.** The published Fisher-vector analysis used a different attribution implementation. That number is not claimed here.
- **Conversion.** The shared top-a conversion, applied to the cluster-mean maps.
- **Failure modes.** Fewer than three maps skips the clustering and keeps the raw relevance. A non-finite map raises.
- **Cost.** One LRP backward per image per environment, then a clustering of that stack. The whole sample is one call.

## Per-environment Extremal Perturbations

- **Reviewer question.** Does differencing a mask that was optimised in each environment find the shortcut?
- **Source.** `baselines/shift_maps.py` `per_environment_extremal`. The class mask in each environment uses the deletion Extremal Perturbations entry above. The reduction is the same elementwise min and max-gap.
- **Patches.** None beyond that entry.
- **Defaults.** The shared extremal step count and smoothing.
- **Reproduction.** The published pointing-game number is the contrastive entry's, not this reduction's.
- **Conversion.** The native area mask, then the shared budget conversion of the reduced map.
- **Failure modes.** The deletion entry's failures, plus fewer than two environments.
- **Cost.** One extremal solve per environment per image.

## Not in this harness yet

| Method | Reviewer question | Stage |
|---|---|---|
| Extremal Perturbations on the full pointing game | The published VOC number above. | Still needs VOC 2007 and the fine-tuned classifiers. |
| CVE on the CUB edit counts | The published 7.4 / 5.3 edit counts above. | Still needs their VGG-16 and the CUB keypoint annotations. |
| RISE deletion and insertion | The published ImageNet table above. | Still needs ImageNet-1k. |
| Grad-CAM pointing game | TorchRay's published Grad-CAM number on VOC 2007. | Still needs VOC 2007 and the fine-tuned classifiers. |

Method-specific conversions that are not the shared top-a rule (CVE edit order, SpRAy cluster relevance, Extremal Perturbations' native area mask) are added with the method that needs them.
