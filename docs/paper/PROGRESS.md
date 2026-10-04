# Progress

Source of truth for the stage sequence: `docs/paper/PLAN.md`.
Branch: `paper/p2-shift-baselines`.

## Stage status

| ID | Status | Commit | Notes |
|---|---|---|---|
| S01 | done | `1f322c3` through `a449b54` | Existing work split into commits. Plan, kickoff, `AGENTS.md`, and the workflow rule are in the repo. |
| S02 | done | `93aa355`, `49ffdc7` | Seeded subsets, three-way splits, test lock, interaction moved to the input device. |
| S03 | done | `46b1bfe` | Provenance in `save_json`, `NormalizedModel`, `--final` refuses a dirty tree. |
| S04 | done | `da589bb` | `evaluation/` scorer. Eval scripts import it. Throughput measured; see below. |
| S05 | done | `fd447a1`, `3b979ae` | One overlap weight, shared-sparse default 0.25, shift mass target, equation audit. |
| S06 | done | `7724885` | D1–D7 on val. Open routes recorded in `results/paper/degenerate/REPORT.md`. |
| S07 | done | `e956015` | CIFAR-100, Oxford-IIIT Pet (37), CUB-200. ImageNet-S is waiting on ImageNet-1k. |
| S08 | done | `0090a78` | Waterbirds pairs, ImageNet-9 backgrounds, planted-patch CIFAR-10, ColoredMNIST recolor check. |
| S09 | launched | `1981031` | Training queue is running. See the background job below. |
| S10 | done | `c15aeed`, merge `ec9239e` | Harness merged as PR #7. |
| S11 | done | merge `061ddaa` | Adapter merged as PR #8. The VOC pointing game is not run. |
| S12 | done | merge `a965754` | Algorithm 1 merged as PR #9. The CUB edit-count reproduction is not run. |
| S13 | in review | `8510013` | Attribution difference and per-environment Extremal Perturbations. SpRAy is not started. |
| S14–S28 | todo | | Not started. |

## Decisions

- 2026-10-03. Pets and dogs: test is the complement of `random_split(seed=42)` at 80% train. Val is 10% of that train pool, carved with seed 43. The historical holdout is unchanged. Checked against `torch.utils.data.random_split`.
- 2026-10-03. HAM10000: the current `val/` folder is test. Val is a lesion-grouped 10% carve of `train/`. No lesion sits in both. All 7 classes are in both.
- 2026-10-03. Brain tumor: the current `Testing/` folder is test. Val is a patient-grouped 10% carve of `Training/`. No patient sits in both (168 train patients, 19 val patients).
- 2026-10-03. Checkpoint selection uses that val split for every paper dataset. Medical selection no longer reads the old holdout folders. Adam, cosine schedule, and cross-entropy are unchanged. Class weights for the medical sets are computed from the train subset.
- 2026-10-03. The throughput benchmark used randomly initialised ResNet-50 and ViT-B/16. Weights do not change the flop count, and ResNet-50 ImageNet weights were not downloaded.
- 2026-10-03. Provisional image budget, not frozen and not written into an eval plan: contrastive ResNet-50 `n=200`, contrastive ViT `n=64`, shift ResNet-50 `n=128`, shift ViT `n=32`. Revise after one timed CDEA cell. At 24.8 model steps/s, 200 images in batches of 4 is 50 batches; 50 allocator steps is about a minute per method before heavier baselines.
- 2026-10-03. CDEA means Contrastive Decomposition via Evidence Allocation. The method is Adam on the mask logits for a fixed number of steps, with the classifier frozen. Pairwise overlap has one weight, `lambda_overlap`. `lambda_disjoint` on the contrastive allocator is rejected. The shift game keeps its own product weight.
- 2026-10-03. Contrastive `kept_logit` is the raw logit \(z_k\). The returned key `suff` is an alias of that number. The shift quantity is the baseline-subtracted kept logit, \(z - z_0\).
- 2026-10-03. `lambda_shared_sparse` defaults to 0.25, including the script flags that used to pass 0. `RobustShortcutObjective.lambda_mass` defaults to 0.1 with target mass `max(1, R / 49)` per mask. S06 is the check of whether those defaults close D1 and D5.
- 2026-10-03. These changes invalidate the ablation `suff` and `overlap` columns, any contrastive run that applied both overlap weights, and any shared mask trained at `lambda_shared_sparse=0`. The map is `docs/paper/EQUATION_AUDIT.md`.
- 2026-10-03. The ID-OOD gap is a model property on the full image. It is not a column of the shift method table.
- 2026-10-03. Degenerate checks used the val split, seed 0, 48 images, 50 steps, ResNet-18 checkpoints already on disk. CIFAR-10 `results/paper_rerun/checkpoints/cifar10_resnet18_pt_lp_ep15_lr0.001_seed0.pt` (raw input). HAM10000 `examples/out/checkpoints/ham10000_resnet18.pt` (raw input). `lambda_mass` stays 0.1. A 16-image sweep did not find a weight pair that closes D1, D2, D3, and D5 together without collapsing the CIFAR kept logit. Phase 4 has to select those weights under the thresholds in `evaluation/degenerate.py`.
- 2026-10-03. The paper pet dataset is `oxford_pets`, the 37-breed Oxford-IIIT Pet set. `pets` remains the 2-class cats-vs-dogs folder and is not a paper dataset.
- 2026-10-03. CIFAR-100, Oxford-IIIT Pet, and CUB-200 use the official test split. Val is 10% of each class inside the official train pool, seed 43. Script counts: CIFAR-100 45000/5000/10000 (100 classes, 450/50/100). Oxford-IIIT Pet 3311/369/3669 (37 breeds). CUB-200 5394/600/5794 (200 classes). Every class is on all three sides.
- 2026-10-03. ImageNet-S was not downloaded. The public release is a split of ImageNet-1k plus segmentation masks, and ImageNet-1k is not on this machine.
- 2026-10-03. Waterbirds pairs use CUB segmentations and the Places365 validation images of bamboo forest, broadleaf forest, lake/natural, and ocean (200 land, 200 water). Backgrounds are split 160/20/20 with seeds 43 and 44. Birds use the CUB paper indices. The water-bird list is the group_DRO substring list, so "tern" also marks bittern. This is not the full Places training pool.
- 2026-10-03. ImageNet-9 training images came from the S3 mirror. The Dropbox links return HTML. Val folders pair completely: 4185 images in original, mixed_same, and mixed_rand, and 45405 original train images. The GitHub challenge release is the held-out test set. It is downloaded and not extracted. The loader refuses a `bg_challenge` path.
- 2026-10-03. Planted-patch CIFAR-10 has two patches. The environments are present, moved, and removed, on the CIFAR-10 paper indices. The two-patch recovery metric stays in S17.
- 2026-10-03. ColoredMNIST OOD views recolor the digit recovered by inverting the label hue. The previous view averaged the colored channels, which scales the digit by the mean of the hue. That invalidates stored ColoredMNIST environment views built the old way. Those runs are pre-audit. Stanford Dogs background restyling is unchanged (`make_env_fn`).
- 2026-10-03. The val contact sheet was inspected. The same bird sits on a forest photo and a beach photo, with the mask on that bird. The two CIFAR patches move and then disappear. A ColoredMNIST digit keeps its shape across three hues. An ImageNet-9 dog keeps its pose across original, mixed-same, and mixed-rand backgrounds.
- 2026-10-03. Paper training is an ImageNet linear probe for the contrastive datasets (frozen backbone, Adam, cosine schedule, cross-entropy, lr 1e-3, 15 epochs). Shift datasets are a full fine-tune at lr 1e-4. Checkpoint selection stays val balanced accuracy. ViT-B/16 uses batch 16. ResNet-50 linear probes use batch 32. Shift fine-tunes use batch 16. Waterbirds also uses class-weighted loss. Seeds are 0–4. ImageNet-S is not in the grid.
- 2026-10-03. ColoredMNIST colors are fixed with seed 43, and training uses the MNIST paper train indices. Older ColoredMNIST checkpoints were fit on the full 60k training pool, including the val images. Those checkpoints are not the paper models.
- 2026-10-03. `IntegratedGradientsRegionsProvider` sums channels and clamps at zero. The previous absolute channel sum counted evidence against the class as support. Stored IG maps from before `c15aeed` are invalid.
- 2026-10-03. The Grad-CAM cross-check compares the target-layer ReLU map, pooled with the same adaptive average pool as our provider. pytorch-grad-cam's returned image is min-max scaled and resized for display, and that display step is not the comparison. IG uses the right Riemann sum on both sides. Captum's default quadrature is Gauss-Legendre.
- 2026-10-03. Library pins for that check are grad-cam 1.5.7 and captum 0.9.0, in `baselines/versions.py`. The CIFAR-10 checkpoint in the check was trained on raw `[0, 1]` input, so both implementations call it directly.
- 2026-10-03. GradientShap draws path coefficients with NumPy. A repeat seeds `torch.manual_seed` and `numpy.random.seed`. Failed rows stay in the batch and take a seeded area-a floor.
- 2026-10-03. TorchRay is vendored at `6a198ee` under CC BY-NC 4.0. Two patches: `Perturbation.to` keeps the moved pyramid, and the mask kernel passes `indexing="ij"` to `meshgrid`. The margin adapter scores \(z_k - z_l\). TorchRay's `contrastive_reward` is a different target and is not used. The published pointing-game numbers are the reproduction target in `docs/paper/BASELINES.md`. VOC 2007 and the fine-tuned classifiers are not on disk, so that run has not been started.
- 2026-10-03. CVE is Goyal et al. Algorithm 1: greedy replacement of one spatial cell at a time until the argmax is the distractor class. The pool-then-linear scorer is that same replacement. Their Section 2.3 relaxation is not used. The CUB edit counts (7.4 random, 5.3 attribute nearest neighbor) are the reproduction target and have not been run.
- 2026-10-03. The shift attribution difference takes the elementwise minimum across environments as the robust map and the largest absolute gap from the in-distribution map as the shortcut. Per-environment Extremal Perturbations applies that reduction to native class masks. SpRAy still needs official CoRelAy and Zennit.

## Open issues

- Stored numbers under `results/` are pre-audit. The ablation tables are invalid: they used a class-ordered prefix, and pets/dogs eval included training images. Do not quote them.
- S05 invalidates stored ablation `suff` and `overlap` columns. Those files were already pre-audit.
- `docs/paper/EVAL_PLAN.md` does not exist. The test split stays locked.
- ImageNet-S still needs a local ImageNet-1k copy. Do not download ImageNet-1k without being asked.
- Waterbirds backgrounds are the 400 Places365 validation photos of the four official categories, not the Places training set. A later training run may need a larger background pool.
- The ImageNet-9 challenge test archive is `data/imagenet9/backgrounds_challenge_data.tar.gz`. Do not extract it before the eval plan is frozen.
- Training queue, started 2026-10-03. PID 74115 (`python scripts/launch_paper_training.py`), caffeinate PID 74116, parent shell 74099. Still alive. Log: `results/paper/logs/train/driver.log`. Done markers: `results/paper/logs/train/*.done`. Checkpoints: `results/paper/checkpoints/`. 110 cells. Code `1981031`. First cell: CIFAR-10 ResNet-50 linear probe, seed 0, on MPS. The log had reached epoch 10 of 15.

## Throughput (MPS, batch 4, 224, random init, 15 steps)

Log (local, gitignored): `results/paper/logs/throughput_mps.json`.

| Model | Forward+backward / s |
|---|---|
| ResNet-50 | 24.8 |
| ViT-B/16 | 10.1 |

## Exit check (S13)

```
101 passed, 11 warnings in 43.64s
```

SpRAy was not run. The test split was not read.

## Exit check (S12)

```
94 passed, 10 warnings in 33.92s
```

After merging the Extremal Perturbations adapter, the combined suite was:

```
98 passed, 11 warnings in 30.21s
```

The CUB edit-count reproduction was not run. The test split was not read.

## Exit check (S11)

```
93 passed, 11 warnings in 36.43s
```

The VOC pointing game was not run. The test split was not read.

## Exit check (S10)

```
89 passed, 10 warnings in 39.79s
```

`PYTHONPATH=. python scripts/crosscheck_evidence.py` exited 0. Code `c15aeed`. Output: `results/paper/crosscheck/val100.json`. 100 CIFAR-10 val images. The test split was not read.

## Next session

Review the shift-baseline pull request. SpRAy is the remainder of S13 and needs official CoRelAy and Zennit. The VOC pointing game and the CUB edit counts stay open. Leave the training queue running and check `results/paper/logs/train/driver.log`. Do not pass `--final`. ImageNet-S is still blocked on ImageNet-1k. The ImageNet-9 challenge test archive stays unextracted.
