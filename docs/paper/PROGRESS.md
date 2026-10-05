# Progress

Source of truth for the stage sequence: `docs/paper/PLAN.md`.
Branch: `main`.

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
| S09 | launched | `dcb5f71` | Live Mac queue is still that process. New launches, including Spark, use `5e78f56`. |
| S10 | done | `c15aeed`, merge `ec9239e` | Harness merged as PR #7. |
| S11 | done | merge `061ddaa` | Adapter merged as PR #8. The VOC pointing game is not run. |
| S12 | done | merge `a965754` | Algorithm 1 merged as PR #9. The CUB edit-count reproduction is not run. |
| S13 | done | merge `4fb5895` | Attribution difference merged as `db760b0`. SpRAy merged as PR #11. The VOC horse analysis is not run. |
| S14 | done | merge `6d038e2` | Contrastive Grad-CAM and official RISE merged as PR #12. SCOUT is recorded as not a drop-in explainer. ImageNet deletion/insertion and the qualitative figures are not run. |
| S15 | done | merge `d870a1d` | Gate G0 does not pass. The tag `g0-baselines` was not created. Open Core reproductions are listed in `docs/paper/G0.md`. |
| S16 | done | merge `8715ded` | Unpaired group-statistics objective merged as PR #14. The paired objective is unchanged. |
| S17 | done | merge `a0ff742` | Scores and the draft plan merged as PR #15. `EVAL_PLAN.md` is not frozen. |
| S18 | done | merge `25c8842` | Gate G1 does not run. Merged as PR #16. The tag `g1-pilot` was not created. |
| S19 | done | merge `9f4f090` | Val selection rule merged as PR #17. It has not been applied. `EVAL_PLAN.md` is not frozen. |
| S20 | done | merge `17c8fc5` | Resumable grid runner merged as PR #18. The contrastive grid was not launched. `--final` stays refused. |
| S21 | done | merge `59f4109` | Shift-grid and ablation manifest merged as PR #19. Neither job was launched. |
| S22 | done | merge `aea98e1` | Completion check merged as PR #20. The grids are unfinished. The tag `final-runs-v1` was not created. |
| S23 | done | merge `535e94e` | Results builder merged as PR #21. `results/paper/RESULTS.md` was not written. |
| S24 | done | merge `608562b` | Read-only audit merged as PR #22. The paper file is absent. Masks were not checked. The audit does not pass. |
| S25–S28 | todo | | Not started. |

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
- 2026-10-03. The shift attribution difference takes the elementwise minimum across environments as the robust map and the largest absolute gap from the in-distribution map as the shortcut. Per-environment Extremal Perturbations applies that reduction to native class masks.
- 2026-10-03. SpRAy clustering is CoRelAy `SpectralClustering` at `bc80524`. Relevance is Zennit `EpsilonPlus` at `3e98348`. Each image receives its cluster's mean relevance. The published VOC horse analysis has not been run. `metrohash-python` 1.1.3.3 is the CoRelAy import dependency.
- 2026-10-03. Contrastive Grad-CAM backpropagates cross-entropy toward the foil class. That target is not the logit margin. Official RISE is vendored at `d91ea00`. The device follows the machine, and a missing save path skips `masks.npy`. The margin map reweights those masks by \(z_k - z_l\). ResNet-50 deletion \(0.1076 \pm 0.0005\) and insertion \(0.7267 \pm 0.0006\) are the reproduction target and have not been run. SCOUT's public repository trains a hardness predictor on CUB and ADE, so it was not vendored.
- 2026-10-03. Gate G0 does not pass. Core reproductions still open: the Grad-CAM and Extremal Perturbations pointing games, the CVE edit counts, and the SpRAy horse analysis. The tag `g0-baselines` was not created. No pilot comparison runs. The random area-a floor has a dossier entry.
- 2026-10-03. The unpaired shift objective scores each image in one group. The reward is the mean of the per-group baseline-subtracted kept logits, minus the variance of the robust group means, plus the variance of the shortcut group means. `lambda_gap` weights that shortcut variance. The paired in-distribution gap is a different quantity and is unchanged. Allocation stays Adam on the mask logits for a fixed number of steps, with the classifier frozen and no learning-rate schedule.
- 2026-10-03. Primary scores live in `evaluation/scores.py`. CD@a uses ROAD. ΔD subtracts a random mask of equal area. Holm adjusted p-values are the cumulative maximum of (m − i) p_(i). `docs/paper/EVAL_PLAN.md` is a draft. The tag `eval-plan-frozen` was not created. The test split stays locked.
- 2026-10-03. Gate G1 does not run. G0 is still closed, and the paper dev checkpoints are 2 of 110 cells, both CIFAR-10. HAM10000 has no paper checkpoint yet. The tag `g1-pilot` was not created. No stop-or-continue decision is logged.
- 2026-10-03. Val selection maximises the mean CD@5% among candidates that close D1–D5. Ties keep the earliest candidate. An open route is not a fallback. The rule has not been applied to val. The tag `eval-plan-frozen` was not created.
- 2026-10-03. The final-grid runner skips a cell whose done marker exists. `--final` is refused while `EVAL_PLAN.md` says it is not frozen, even if the file names the freeze marker. The contrastive grid was not launched.
- 2026-10-03. The shift manifest lists the six shift datasets, five seeds, ResNet-50 and ViT-B/16, and the five core shift methods. Ablation cells are one variant per removed component, including paired versus unpaired and the mass target on or off. Neither list was started. S20's contrastive log directory is absent, so that grid is not launched.
- 2026-10-03. A final cell is complete only when its done marker exists. A failed marker is a rerun. The contrastive manifest names eight datasets, including ImageNet-S. Run-record checks cover mask range, area, probabilities, finiteness, and `n`. No paper record was checked. The tag `final-runs-v1` was not created.
- 2026-10-03. `analysis/build_results.py` formats every table number from the records through `family_summary` or a seed-averaged mean. Family S stays without a baseline until val selection supplies one. `results/paper/RESULTS.md` was not written.
- 2026-10-03. The results audit compares a markdown file with a fresh render and does not rewrite the file. A hand-edited number fails that comparison. RESULTS.md is absent. Masks were not checked. The fresh-session audit has not been run. The audit does not pass.
- 2026-10-04. Review fixes on `paper/p6-review-fixes` (`e31c897`–`7a3f12f` plus this entry): partition penalty is a batch/region mean (preset numbers unchanged, effective strength weaker — re-tune under S19); allocators and Grad-CAM paths hold eval mode; linear-probe training holds backbone BN/dropout in eval; single-foil margin is 0 and penalty means cover valid hypotheses only; shift shortcut init gets seeded jitter; interaction blends default to 0; logit-gap companions added beside prob ΔD; Spearman is tie-aware with 0.0 for constant rows; translated null never returns a zero roll; `frozen: true` line wins the plan lock; manual game mode no longer requires retired lambda_disjoint. Suite: 166 passed (149 before + 17 new).
- 2026-10-04. Cells 1–3 (CIFAR-10 ResNet-50 LP seeds 0–2, code `1981031`) predate the backbone-eval fix and must be retrained before S19 selection. Do not restart the queue until the branch merges; the rerun starts from a clean log dir or explicitly overwrites those three done markers.
- 2026-10-04. `paper/p6-review-fixes` is on `main` (`6b23854`, audit line refs `dcb5f71`). The pre-fix checkpoints and done markers for seeds 0–2 are in `results/paper/checkpoints/stale_1981031/` and `results/paper/logs/train/stale_1981031/`. The queue was relaunched from `dcb5f71`. Parent shell 1571, python 1587, caffeinate 1588. Log: `results/paper/logs/train/driver.log`. 110 cells. Cell 1 restarted as CIFAR-10 ResNet-50 linear probe, seed 0.
- 2026-10-04. Raw-[0,1] training stays: train and eval are consistently unnormalised, which is correct for raw-trained checkpoints but leaves ImageNet-pretrained features underused (measured 0.389 raw vs 0.816 normalised on a CIFAR-10 probe). `NormalizedModel` stays unused outside its test until a normalised-weights setting needs it. Revisit only with a val-measured comparison, not by wrapping silently.
- 2026-10-04. Shift split discipline verified, no fix needed: `WaterbirdsPairs` and `PlantedPatchCIFAR` read the paper indices via `load_indices` (cub200/cifar10), backgrounds are disjointly split, and `ColoredMNIST` preserves MNIST row order so the mnist indices align.
- 2026-10-04. Environment pin is `c71fd37` (`requirements.txt`, conda env name `gambit`). Input convention is `e7a1a24`: absent `results/paper/input_convention.txt` means raw, and an ImageNet winner uses a separate checkpoint name. Hard budget and the family-C foil executor are `5e78f56`. Suite before these commits: 174 passed. The cached CIFAR-10 probe has not finished, so the convention file is not written. Contrastive allocation changed; stored degenerate reports are stale and must be rerun. Classifier training on the raw path is otherwise the same Adam, cosine, and cross-entropy loop.
- 2026-10-05. Branch `paper/fix-checkpoint-convention` (Claude Code review). Fixed: (1) no checkpoint loader read `input_convention`, so ImageNet-convention checkpoints would be scored on raw input. Every load now goes through `models.wrapper.load_checkpoint_into` (`ablation_contrastive._build_model`, `model_table`, `examples/contrastive_explanation.load_checkpoint`, `eval_robust_shortcut`). (2) A wrapped ViT picked its patch embedding as the Grad-CAM layer. `_find_target_layer` unwraps. (3) Margin and contrastive Grad-CAM raised on ViT and the library CAM adapters used the wrong layer. pytorch-grad-cam now gets the ViT token reshape. (4) GradientShap noise left [0, 1] and would fail every row on a wrapped model. Its path points are clamped. Suite: 184 passed (`marl` env).
- 2026-10-05. Decisions with the user: ImageNet val (on Spark) is the 8th contrastive unit, using torchvision `IMAGENET1K_V1` weights, with no ImageNet-S masks. Waterbirds groups are ablation AS1, not a shift unit. A sixth independent shift dataset will be added (recommended COCO-on-Places), with placeholder `sixth_shift_tbd` in `scripts/shift_grid.py`. CD@a uses unique masks only. Shared on both sides is the sensitivity check (`evaluation/foil_masks.py`). The shift area is piloted on val before the freeze.
- 2026-10-05. `docs/paper/EVAL_PLAN.md` rewritten as the full plan (still a draft, not frozen). Selection gates on D1, D2, D4, D5. D3 is recorded, not gated (`analysis/selection.py`). The shift primary is logit ΔD on the predicted class. CVE is ResNet-50 only and is compared on CD1@5%.

## Open issues

- Stored numbers under `results/` are pre-audit. The ablation tables are invalid: they used a class-ordered prefix, and pets/dogs eval included training images. Do not quote them.
- S05 invalidates stored ablation `suff` and `overlap` columns. Those files were already pre-audit.
- `docs/paper/EVAL_PLAN.md` is the full draft (2026-10-05). It is not frozen. The test split stays locked. Its section 12 lists what the freeze needs.
- ImageNet val is on Spark, not on this Mac. ImageNet cells (contrastive unit 8 and the RISE reproduction) run on Spark. ImageNet-S masks are not needed.
- Waterbirds backgrounds are the 400 Places365 validation photos of the four official categories, not the Places training set. A later training run may need a larger background pool.
- The ImageNet-9 challenge test archive is `data/imagenet9/backgrounds_challenge_data.tar.gz`. Do not extract it before the eval plan is frozen.
- VOC 2007 is unpacked at `data/voc2007/VOCdevkit/VOC2007` (9963 images, 4952 test ids). The public CUB VGG-16 file is `data/weights/cub_vgg16_model.ckpt` (a later reimplementation, not yet scored against the 7.4 / 5.3 edit counts). ImageNet val is not on this Mac. It is on Spark.
- This Mac has no `gambit` conda env. The rules name `gambit`, which exists on Spark only. The Mac suite runs in `marl`. Create `gambit` here from `requirements.txt`, or note per machine in `AGENTS.md`.
- The Mac training queue is not running. The log ends at the restart of cell 1 after the probe, with no process alive. Confirm Spark owns training before restarting anything here.
- Checkpoints trained before `load_checkpoint_into` are fine. Their metadata carries `input_convention`, so they now reload correctly.

## Throughput (MPS, batch 4, 224, random init, 15 steps)

Log (local, gitignored): `results/paper/logs/throughput_mps.json`.

| Model | Forward+backward / s |
|---|---|
| ResNet-50 | 24.8 |
| ViT-B/16 | 10.1 |

## Exit check (review fixes)

```
166 passed, 13 warnings in 19.23s
```

Branch `paper/p6-review-fixes`, 12 commits over `main`. The training queue stays stopped (cells 1–3 predate the backbone-eval fix and must be retrained). The test split was not read.

## Exit check (S24)

```
149 passed, 11 warnings in 53.00s
```

The audit does not pass. RESULTS.md is absent. Masks were not checked. The test split was not read.

## Exit check (S23)

```
147 passed, 11 warnings in 35.86s
```

`results/paper/RESULTS.md` was not written. The test split was not read.

## Exit check (S22)

```
144 passed, 11 warnings in 31.46s
```

Final runs are not complete. The tag `final-runs-v1` was not created. The test split was not read.

## Exit check (S21)

```
139 passed, 11 warnings in 54.14s
```

The shift grid and the ablations were not launched. The contrastive grid was not launched. The test split was not read.

## Exit check (S20)

```
134 passed, 11 warnings in 67.43s
```

The contrastive grid was not launched. The test split was not read.

## Exit check (S19)

```
130 passed, 11 warnings in 53.73s
```

The selection rule was not applied to val. The tag `eval-plan-frozen` was not created. The test split was not read.

## Exit check (S18)

```
125 passed, 11 warnings in 54.80s
```

Gate G1 does not run. The tag `g1-pilot` was not created. The test split was not read.

## Exit check (S17)

```
124 passed, 11 warnings in 47.62s
```

`EVAL_PLAN.md` is a draft. The tag `eval-plan-frozen` was not created. The test split was not read.

## Exit check (S16)

```
115 passed, 11 warnings in 49.02s
```

The smoke run used four Waterbirds val images, two on land and two on water. The test split was not read.

## Exit check (S15)

```
111 passed, 11 warnings in 61.07s
```

Gate G0 does not pass. The tag `g0-baselines` was not created. The test split was not read.

## Exit check (S14)

```
109 passed, 11 warnings in 43.77s
```

The ImageNet deletion/insertion table and the contrastive qualitative figures were not run. The test split was not read.

## Exit check (S13 SpRAy)

```
104 passed, 11 warnings in 32.65s
```

The VOC horse analysis was not run. The test split was not read.

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

Read `docs/paper/EVAL_PLAN.md` first. It is the full plan, and its section 12 is the freeze checklist. Work in this order:

1. Merge `paper/fix-checkpoint-convention` (the user reviews the PR). Spark must pull it before any evaluation, because its checkpoints are ImageNet-convention and need `load_checkpoint_into`.
2. Build the per-cell executor (EVAL_PLAN section 10). It loads through `load_checkpoint_into`, draws the frozen test sample, produces every method's M_k and M_l as defined in section 4.2 (including CD1 for CVE and the shift maps in 4.3), and writes the record schema. Test it on val only.
3. Rerun `scripts/check_degenerate.py` under the hard budget (section 6.1), on the paper seed-0 checkpoints once they exist. Use `--device cpu` if a trainer holds the GPU.
4. G0 reproductions:
   - Grad-CAM and Extremal Perturbations pointing game on VOC (on disk);
   - CVE edit counts on CUB;
   - RISE on ImageNet val (Spark);
   - a feasible SpRAy target, or a recorded omission.
5. Choose and prepare the sixth shift dataset (section 2.2). Replace `sixth_shift_tbd` in `scripts/shift_grid.py`.
6. Then the shift-area pilot (6.3), selection (6.2), and G1 (section 11).

Do not freeze `EVAL_PLAN.md` or tag `eval-plan-frozen`, `g0-baselines`, `g1-pilot`, or `final-runs-v1` before their conditions hold. Do not pass `--final`. Do not launch the final grids. Do not write `results/paper/RESULTS.md`. The ImageNet-9 challenge archive stays unextracted. This Mac must not start a second trainer while Spark owns training.
