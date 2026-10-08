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
- 2026-10-05. End-to-end runner on branch `paper/e2e-runner`.
  - Entry points: `scripts/paper_run.py` runs one cell, `scripts/launch_paper_eval.py` runs the resumable grid, `scripts/smoke_e2e.py` runs everything on a few val images. Modules: `evaluation/run_{models,data,methods,cell}.py`.
  - The ImageNet split and loader are in (`write_paper_splits.py --only imagenet`).
  - **Smoke run:** 26 cells, every unit except ImageNet (no data on this Mac), both backbones, every method, all 22 ablations, all selection candidates. Method errors: none. Report: `results/paper/smoke/REPORT.md`. Cells ran at clean commits `aeabe4c` (23 cells) and `f66d44c` (3 cells, resume flag only). Commit `0965528`'s message cites a wrong range; the commits above are the right ones.
  - **Bugs the smoke run found and fixed:**
    - SpRAy cast to float64 on MPS;
    - `CamLibraryProvider` never released pytorch-grad-cam hooks (activations piled up on every later forward pass);
    - RISE ran with autograd on (33 GB at 200 masks);
    - runner outputs were not git-ignored, so `--final` would have refused every cell after the first;
    - the Stanford Dogs shift unit read every annotated image instead of the split (the runner restricts it).
  - **Cost** (pass counts, hardware-independent): Extremal Perturbations is 3200 passes per image per area, about 6× CDEA. It dominates the budget; see `docs/paper/SPARK_RUNBOOK.md`.
- 2026-10-06. Spark training for the paper checkpoints was restarted into `paper_final/` after the earlier `smake_run` checkpoints and game outputs were removed. Two seed queues share the one GB10 (`GAMBIT_LOADER_WORKERS=4`). Seeds 0 and 1 are the live pair (PIDs 3522551 and 3522553, started 2026-10-06T00:00:54Z). The supervisor that was supposed to start seeds 2 and 3, then seed 4, exited immediately (`wait` could not see those PIDs). Seeds 2–4 will not start on their own.
- 2026-10-07. Training progress, from `paper_final/logs/timing.tsv` and `driver_seed0.log` / `driver_seed1.log`. Each of seeds 0 and 1 has saved 17 of 22 checkpoints (34 files). Both are on cell 18, planted-patch ViT-B/16, epoch 7 of 15. Finished two-at-a-time wall times, seed 0: CIFAR-10 ResNet 3097s, CIFAR-10 ViT 11602s, CIFAR-100 ResNet 3095s, CIFAR-100 ViT 11601s, Pets ResNet 236s, Pets ViT 865s, Dogs ResNet 1028s, Dogs ViT 3829s, CUB ResNet 382s, CUB ViT 1402s, HAM10000 ResNet 500s, HAM10000 ViT 1864s, brain tumor ResNet 157s, brain tumor ViT 582s, ColoredMNIST ResNet 10785s, ColoredMNIST ViT 25861s, planted-patch ResNet 8937s. Seed 1 matches within a few seconds. Queue time at the planted-patch ResNet checkpoint: 85821s. Compared with the earlier one-at-a-time seed-0 run, two jobs take about 2.2× as long per cell (CIFAR-10 ResNet 3097s vs 1416s; ColoredMNIST ViT 25861s vs 11664s).
- 2026-10-07. Training time left, scaled from those measured cells, not from the smoke run. Planted-patch ViT alone took 29448s; at 2.2× and with 8 of 15 epochs still to go, about 10h remain on the current cell. ImageNet-9 and Waterbirds, both models, were 15356s alone, about 9h at 2.2×. Seeds 0 and 1 therefore have about 19h left. A full two-seed wave is about 51h. Seeds 2 and 3 are another wave. Seed 4 alone is about 23h. From this point the five-seed training queue is about 94h, about 4 days, if the next pairs are started when the current pair exits.
- 2026-10-06. Spark end-to-end smoke (`scripts/smoke_e2e.py --device cuda`, val only, report was `results/paper/smoke_spark/REPORT.md` and was removed with the other pre-`paper_final` outputs). 28 cells, no method errors. Paper-knob seconds per image, one CIFAR-10 image, core methods, one area: ResNet-50 base evidence 0.0, CDEA 1.9, CVE 0.0, Extremal Perturbations 27.2, margin Grad-CAM 0.0, margin IG 0.1, random floor 0.0 (29.2s). ViT-B/16 base evidence 1.3, CDEA 5.2, Extremal Perturbations 27.7, margin Grad-CAM 0.0, margin IG 0.2, random floor 0.0 (34.4s). CVE is not run on ViT. The timer stops before ROAD scoring, so the blur operator does not multiply these seconds. Extremal Perturbations is inside the area loop, so the launcher default of three areas (`0.025,0.05,0.10`) multiplies only that method.
- 2026-10-07. Explanation-grid estimate from that smoke table and the plan's minimum n. Not a frozen n. Contrastive only, 8 datasets × 5 seeds, core methods, no extended methods, no ablations, no shift. One area: ResNet-50 200 images × 29.2s × 40 cells = 65h; ViT-B/16 64 × 34.4s × 40 = 24h; together 89h. Three areas: ResNet-50 84s/image and 186h, ViT-B/16 90s/image and 64h, together 250h. Shift was timed only at the fast knobs, so it is not in this estimate. Training the classifiers is also not in it.
- 2026-10-07. Seeds 0 and 1 training finished: both drivers printed `queue finished` (22/22 cells each, 44 checkpoints in `paper_final/checkpoints/`). Final cells: ImageNet-9 ResNet-50 seed 0 best val_balanced_acc 0.9438, ViT 0.9357; Waterbirds ResNet-50 seed 0 0.8271, ViT seed 0 0.7345 / seed 1 0.7231. GPU idle after ~2026-10-07T15:24Z.
- 2026-10-07. Val games launched on seeds 0–1 per `SPARK_RUNBOOK.md` step 2 (dev units for G1, then the shift-area pilot), three areas and plan-minimum n, core methods only, no extended/ablations/candidates. Command pair: `launch_paper_eval.py --split val --datasets cifar10,ham10000 --seeds 0,1` (8 cells) then `--split val --game shift --seeds 0,1` (20 cells). Launcher PID 2269007 under bash 2269006, `python -u`, log `results/paper/logs/eval_val_seeds01.log`, markers under `results/paper/runs/_markers/val/`. Prep: 44 checkpoints symlinked into `results/paper/checkpoints/` (all 48 non-ImageNet cell paths resolve; ImageNet uses torchvision weights), `data/imagenet/val` present (1000 classes), suite 197 passed + 1 skipped in `gambitEnv`. Seeds 2–4 training waits until the games analysis (G1) says continue. `git pull` failed (no network from sandbox); tree still dirty (pre-`paper_final` cleanup deletions, `imagenet.json` untracked, corelay fix uncommitted).
- 2026-10-07. Second OOM (~13:42 local): the relaunched dev grid died the same way — device mapping failed with only 567MB free of the 130GB unified pool, right after `margin_gradcam` in cell 1 (trace: cdea 359MB → margin_gradcam 435MB, then EP mapping failed). Note the node itself never rebooted (uptime continuous at 17d+; the "reboot" was the session/GUI going down). Forensics: after killing the orphan shift pilot, system use fell 120→44GB, so our jobs were holding ~55GB+ of unified memory — the whole-sample-on-device design (n=200) plus EP 800 iters × 3 areas is the hog, not CPU threads. Conservative relaunch (launcher PID 2447479, log `results/paper/logs/eval_val_seeds01.log`, old log rotated to `eval_val_seeds01_oom1.log`): n=64 everywhere (`--n-contrastive resnet50:64,vit_b_16:64`, `--n-shift resnet50:64,vit_b_16:32`), which is exactly EVAL_PLAN section 6's val size ("n = 64 images per dataset"); three areas kept (areas cost time, not peak — EP runs once per area sequentially — and 6.3 needs all three). Thread caps, expandable_segments, trace, nice all kept; mem watch restarted on the new PID. If this still OOMs, the remaining lever is code-level per-image batching in `run_cell.py`.
- 2026-10-07. Gates revised with the user. G0 is an entry check: pinned or named code, toy tests and cross-checks, merged PR, dossier entry. Every Core baseline meets it, so G0 passes; the tag `g0-baselines` goes on the commit that lands this on `main`. The B3 reproductions run in parallel and go into the appendix. CVE is the exception: its CUB reproduction must be recorded before the freeze for CVE to stay in family C, otherwise family C is two comparisons under Holm over 2. G1 is the framing decision, not a writing gate, and does not wait for G0 reproductions. Its read-out (mean paired CD@5% difference per dev dataset, seeds 0–1, against the better margin-attribution variant) is fixed in `docs/paper/G1.md` before the pilot is scored. Writing the formulation, method, protocol, and related work starts now (S25 no longer needs S23). Files: `G0.md`, `G1.md`, `PLAN.md` (S15, S18, S25, B3, B8, G1 paragraph), `EVAL_PLAN.md` (sections 7, 11, 12), `FRAMING.md`, `FINAL_RUNS.md`, `tests/test_g0.py`, `tests/test_g1.py`.

- 2026-10-07. GH200 node (`gpugh200-02-05`, one GH200 96GB, 72 cores). This is a separate machine from Spark. It has no Spark checkpoints, so it trains its own. Env `gambitEnv` matches every pin in `requirements.txt` (torch 2.9.1+cu128, CUDA available). Suite: 197 passed, 1 skipped. The ResNet-50 `IMAGENET1K_V1` weights were downloaded with user approval. Branch `paper/parallel-runs`: `--jobs` on `launch_paper_training.py`, `launch_paper_eval.py` and `smoke_e2e.py` runs one process per cell (`scripts/parallel_cells.py`), with the longest training cells first and per-job thread caps. The training loop, loss, optimizer and schedule are unchanged. Smoke checkpoints come from `--epochs 1 --ckpt-dir results/paper/smoke/checkpoints` and are read through `smoke_e2e.py --checkpoint-dir`, so they never sit beside paper checkpoints. Suite: 205 passed, 1 skipped.
- 2026-10-07. GH200 smoke run. All 22 seed-0 cells trained for 1 epoch with 11 jobs, with no failures and none at chance (logs `results/paper/smoke/logs/train/`). Then `smoke_e2e.py --jobs 13 --checkpoint-dir ...`: 28 cells, every model a smoke checkpoint except ImageNet, no method errors (`results/paper/smoke/REPORT.md`). The paper-knob timing cells ran alone after the parallel pass. On this node a single explanation job is bound by kernel launches, not GPU compute. Its per-image seconds are higher than Spark's (see the report), and GPU utilisation during training already reads 100% at 11 jobs, with CPU load about 15 of 72. Parallel cells are therefore what speeds up the eval grid here.
- 2026-10-07. User decision: run strictly by seed on the GH200. Train all 22 cells of a seed in parallel, then that seed's val games, then the next seed. `scripts/run_seed_queue.sh` (`c52420e`) does that for seeds 0–4: training with 11 jobs, then val dev units (`--datasets cifar10,ham10000`, n 64/64) and the shift pilot (`--game shift`, n 64/32) at the same time, 7 jobs each, at three areas with core methods. Started 22:24Z, PID 1019376, log `results/paper/logs/seed_queue.log`. Per-cell logs: `results/paper/logs/train/<cell>.log` and `results/paper/runs/_markers/val/<cell>.log`. This overrides the earlier "seeds 2–4 wait for G1" decision only on this node. Stop the queue after seed 1 if G1 says stop.

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

**GH200 node:** the seed-major queue owns the GPU (PID 1019376, `results/paper/logs/seed_queue.log`). Do not start another GPU job. Check `results/paper/logs/train/*.done` and `results/paper/runs/_markers/val/*.{done,failed}`. The queue is resumable: `bash scripts/run_seed_queue.sh` skips finished work. When seed 0's val games finish, check GPU memory headroom in the per-cell logs and raise or lower `EVAL_JOBS`. Then score G1 on the seed-0 dev val records. Branch `paper/parallel-runs` needs review before it merges.
- 2026-10-07. User decision: stop after seed 0 for the G0/G1 review. The queue driver (script PID 1019393, launched with `SEEDS="0 1 2 3 4"`) must be killed at the seed-0 boundary, before seed-1 training starts; then with the user: (1) score G1 from the seed-0 dev val records (CDEA vs margin attribution, mean CD@5% under ROAD on CIFAR-10 + HAM10000, via `analysis/build_results.py` and the rule in `analysis/selection.py`); (2) review G0's open reproductions (G0.md) and decide run / justify-and-omit / stop; (3) only then relaunch with `SEEDS="1 2 3 4"` or stop. Seeds 1–4 do not start on a timer.
- 2026-10-07. Merged the gate revision (`b6fb773`, `79849b1`) into `paper/parallel-runs`, clean merge, suite 207 passed + 1 skipped. Consequences for this node: (1) G0 passes under the entry rule — the queue owes it nothing; the B3 reproductions are a parallel workstream. (2) G1's read-out is fixed (mean paired CD@5% difference per dev dataset on ResNet-50, seeds 0–1, vs the better margin variant, EP at 5%, runner-default CDEA) — so the seed-0 review can no longer finalize G1; it decides the seed-1 launch, and G1 is scored after seed 1's dev games. The queue already runs everything the read-out needs (both margin variants, 5% among the three areas, n=64, no candidates), so no queue or monitor changes. (3) Open for the review: Spark's relaunched grid is also running the seeds 0–1 dev games, which duplicates this node's seed 0–1 dev cells — keep both for the cross-platform check or stand one down to save compute.
- 2026-10-07. Seed-0 completion monitor: `scripts/monitor_seed0.sh` (branch `paper/parallel-runs`), running as a background shell job with a notification armed on its `SEED0_COMPLETE` / `SEED0_DRIVER_GONE` lines. On the driver's `seed 0: done` line it kills the queue driver plus any slipped seed-1 launcher, so seeds 1–4 cannot start before the review. Process matching is anchored to real interpreter command lines and every kill is re-verified through `/proc`, so the shell-tool wrappers are never matched. Both branches were tested against fake inputs; a live-fire of the kill path was deliberately not run. Tested: completion prints the marker and exits 0 without touching the real driver; a dead driver prints the gone marker and exits 1; the matcher finds the real driver PID and rejects the wrapper. Suite: 205 passed, 1 skipped.

**Spark (earlier):** Seeds 0–1 training is done; seeds 2–4 training waits for the G1 verdict on the val games (user decision 2026-10-07). Do not start a second job on the GPU: the val-games grid owns it (conservative n=64 relaunch after two OOMs; launcher PID 2447479, log `results/paper/logs/eval_val_seeds01.log`, 8 dev cells then 20 shift-pilot cells, resumable via `results/paper/runs/_markers/val/`).

When the grid finishes: score G1 (CDEA vs margin attribution on dev val), run the shift-area pilot analysis (EVAL_PLAN 6.3–6.4), then decide with the user between seeds 2–4 training and the critique-paper route.

The ImageNet split is written at `data/splits/imagenet.json` and is not committed. The Python 3.14 SpRAy annotation fix in `third_party/corelay` is not committed. `paper_final/` is gitignored.

Still before any test run: commit those two, decide whether Extremal Perturbations runs at one area or three, set n per seed, choose the sixth shift dataset, record the CVE reproduction (or reduce family C), and finish the hard-budget degenerate rerun, val selection, and the G1 decision. The other G0 reproductions run in parallel and do not block. Do not freeze `EVAL_PLAN.md` or pass `--final` before EVAL_PLAN section 12 is complete.
