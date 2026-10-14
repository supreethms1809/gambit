# Progress

History before this branch is on tag `framing-v1`. Do not score records written under `results/paper/runs/`. New records go to `results/paper/cells/`.

The seed-1 training chain on the GH200 stays pinned at `cf07ac2` (the archive commit). Its checkpoints are used. Its CDEA evaluation rows are not.

## Running

- **Planted-pair ResNet-50 training: done 2026-10-14 01:08 UTC.** All 5 seeds trained. Best val balanced accuracy 0.986–0.987.
- **Planted-pair ViT-B/16 training, started 01:13 UTC.** `launch_paper_training.py --seeds 0,1,2,3,4 --datasets planted_pair --models vit_b_16 --jobs 3`. Log `results/paper/logs/train_planted_pair_vit_b_16.log`.
- **Planted-pair val cells, ResNet-50, started 01:13 UTC.** Seeds 0–4 run one after another, n = 200, every method and ablation, areas 2.5/5/10%, at hash `a5deb9fe`. Records go to `results/paper/decomp_dev/`. Loop log `results/paper/logs/decomp_dev/planted_pair_resnet50_loop.log`.

- **Decomposition dev cells, started 2026-10-14 ~00:20 UTC (user decision).** These run *before* the win conditions in `EVAL_PLAN.md` 13.4 were signed off, which departs from the handoff's "pre-declare before any run". They are development runs: they select CDEA's steps and lr and the best min-split row on dev val, and no win condition is read from them.
  - Setup: CIFAR-10 and HAM10000, ResNet-50, seed 0, val, n = 64, areas 2.5/5/10%, ROAD and blur.
  - Methods: every method, ablations DA1–DA9, and the 4-config grid. They use the existing seed-0 probes.
  - Code: branch `feat/cdea-decomposition` at `0e529b1`, method hash `a5deb9fe`.
  - Output: records in `results/paper/decomp_dev/`, logs in `results/paper/logs/decomp_dev/`.
  - Read with `scripts/decomp_readout.py --records results/paper/decomp_dev`, from the worktree.

- **Planted-pair training, started 2026-10-14 (user approved, ResNet-50 first).** `scripts/launch_paper_training.py --seeds 0,1,2,3,4 --datasets planted_pair --models resnet50 --jobs 5`, run from the worktree `../gambit-decomp` (branch `feat/cdea-decomposition`, `8642eb4`). It uses the planted-patch recipe: fine-tune, lr 1e-4, 15 epochs. Checkpoints go to `results/paper/checkpoints/`. The launcher log is `results/paper/logs/train_planted_pair_resnet50.log`, with per-seed logs `results/paper/logs/train/planted_pair_resnet50_ft_lr0.0001_seed*.log`. ViT-B/16 follows once ResNet-50 is done.
- **Seed-0 chain stopped, 2026-10-14 (user decision).** The launcher (PID 873953) and its four running cells were killed: CUB-200 R50, Oxford Pets R50, ImageNet-9 R50 and ColoredMNIST ViT. 10 of 24 units had finished, and their done markers are in `results/paper/cells/_markers/val/`: imagenet ViT/R50, cifar10, cifar100, ham10000, planted_patch ViT/R50, waterbirds ViT/R50, imagenet9 ViT. The rest are not run. Rerunning the chain skips the finished units while the hash is `9d84e7f2`.

- **Seed-0 chain, restarted 2026-10-12.** `scripts/launch_paper_chain.py --seeds 0 --jobs 4` on the GH200, from `27f4065` (branch `fix/vit-gradcam-layer`), method hash `9d84e7f2`. Launcher PID 873953. All 24 units rerun from scratch, eval only, with the existing checkpoints.
- **Why it restarted.** Grad-CAM on ViT hooked the last encoder block's output, where the patch tokens get zero gradient because the head reads only CLS. Margin Grad-CAM on ViT was zero everywhere (CD 0.00 on ImageNet ViT-B/16). The fix hooks that block's `ln_1`. Only the ViT margin Grad-CAM rows were wrong, but the fix changes the method hash, so the whole seed was rerun (user decision). The first run (hash `9ca0d89a`, launcher PID 247230, 8 units finished) is in `results/paper/superseded/seed0_9ca0d89a/`, as is the readout built from it. Keep it out of any table.
- **Seed-0 chain, first run, started 2026-10-11, superseded.** `scripts/launch_paper_chain.py --seeds 0 --jobs 4` on the GH200. PID 247230. Log `results/paper/logs/chain_seed0.log`. Per-cell logs are `results/paper/logs/train/chain_<dataset>_<backbone>_seed0.log`. Val split, existing checkpoints, two big and two small at a time. Seed 1 is not in this command.
- **Seed-0 cells.** CIFAR-10 and HAM10000 val cells (ResNet-50, n = 64) are in `results/paper/cells/`. Their CDEA rows were merged in after the Sinkhorn catch-up (`d271283`). Seed 1 is not trained.
- **CDEA was not reproducible before `d010f02`.**
  - Two runs at the same method hash (`540acbe0`: `results/paper/cdea_rerun/` and `results/paper/cdea_stages/`) shared no identical row.
  - At 5% on CIFAR-10 val, the median per-image difference in CD was 0.71 nats, and the mean moved from 2.39 to 1.99.
  - The cause was the backward of the cell upsampling, which accumulates in a varying order on GPUs; Adam amplified it.
  - Since `d010f02`, a repeated allocation on MPS is identical. CDEA rows from before it carry run-to-run noise of that size.
- **D2 fails** in `results/paper/cdea_stages/` (`scripts/check_degenerate.py --records`). At 5% the soft-minus-hard gap is above 0.5 nats for ranks 0 and 1 on both dev datasets.
- **Declared departure from the G1 stop rule.** One formulation change is under test: the boundary-robust payoff, `FORMULATION.md` section 4.1. Its read-out and decision rule are in `G1.md` and were fixed before it ran.
  - The fixed-boundary and boundary-robust CDEA cells run together at `f011c8e3` into `results/paper/cdea_boundary/`.
  - The baseline rows are not rerun: no baseline calls the changed code.
  - Read with `scripts/g1_readout.py --records results/paper/cells results/paper/cdea_boundary --cdea cdea cdea:A10_boundary_shift --seeds 0`.

## Decomposition game, 2026-10-14 (merged to `main` 2026-10-14)

- **Decision.** The user adopted the handoff `docs/paper/handoff10-13-26.md`, going with its recommendations. CDEA returns to splitting one pair's evidence into unique-to-k, unique-to-l and shared. The contrast game stays as ablation DA2.
- **Specification.** `FORMULATION_DECOMPOSITION.md` (method, comparators, scores, planted pairs, decisions, and departures from the handoff). The evaluation is `EVAL_PLAN.md` section 13.
- **Built.**
  - `cdea/_plan.py` (the refactor; `allocate()` unchanged, checked by `tests/test_plan_refactor.py`), `cdea/null.py` and `cdea/decompose.py`;
  - `baselines/decomposition.py`;
  - the scores, the runner (`--game decomposition`), `scripts/decomp_readout.py` and D2/D8/D9 in `scripts/check_degenerate.py`;
  - the planted-pair dataset and its role check.
  - The toys pass, including the interaction toy: the joint search finds redundant evidence that single-cell occlusion misses.
- **Worktree.** Built in `../gambit-decomp` and merged to `main` on 2026-10-14. The merge changes the method hash, so the seed-0 chain records (`9d84e7f2`) no longer match `main`.
- **Signed off, 2026-10-14.** The user signed off the win conditions (`EVAL_PLAN.md` 13.4) as drafted. The merge that records it is tagged `decomp-declared`. Planted-pair training was approved and the dev runs were started before sign-off (see Running).
- **Fractional-mask test on seed 1** (`G1.md`, reopened 2026-10-13): held, by the same decision. It was never launched.

## G1 declared read-out, 2026-10-11 (seed 0 only)

`scripts/g1_readout.py --records results/paper/cells results/paper/cdea_boundary --cdea cdea cdea:A10_boundary_shift --seeds 0`

- **Boundary-robust payoff** (`cdea:A10_boundary_shift`): "ahead of both comparators on both datasets". The declared rule is met.
- **Fixed-boundary payoff** (`cdea`): not ahead of margin Grad-CAM on either dataset.

**Open before section 4.1 becomes the method.**
- The 95% intervals against margin Grad-CAM include 0 on both datasets.
- D2 still fails at 5% on CIFAR-10, though the gap is smaller than with the fixed boundary.
- The G1 read is to be repeated on seed 1, which is not trained.

The default is unchanged until the user confirms adoption.

## Offset-seed check, declared 2026-10-11 before it ran

- **What runs.** The boundary-robust payoff with offset seeds 0, 1 and 2: `A10_boundary_shift`, `_o1`, `_o2`. Seed-0 val, both dev datasets, n = 64, CD@5% under ROAD, same baseline rows. Seed 0 is rerun so all three share one code hash, and so its rows can be compared with `results/paper/cdea_boundary/` as an end-to-end reproducibility check.
- **How it is read.** Every seed is reported; none is selected.
- **Verdict.** The G1 result counts as robust to the offset draw only if all three seeds are ahead (mean paired difference above 0) of margin Grad-CAM and of deletion Extremal Perturbations on both datasets. Otherwise the margin cannot be told apart from the method's sensitivity to its offsets.

**Result, 2026-10-11.**

`scripts/g1_readout.py --records results/paper/cells results/paper/cdea_offset_seeds --cdea cdea:A10_boundary_shift cdea:A10_boundary_shift_o1 cdea:A10_boundary_shift_o2 --seeds 0`

- **Verdict.** All three offset seeds print "ahead of both comparators on both datasets". By the declared rule, the G1 result is robust to the offset draw.
- **Margins.** Against margin Grad-CAM, every interval still includes 0, and the HAM10000 margins are the smallest.
- **Reproducibility.** The seed-0 rows reproduce `results/paper/cdea_boundary/` exactly.
- **D2.** Still fails at 5% on CIFAR-10 for all three seeds. It passes on HAM10000.

## Adopted, 2026-10-11

- **Default.** By user decision, CDEA's default is the boundary-robust payoff (`5b3d99b`). The fixed-boundary payoff is ablation `A10_fixed_boundary`, and offset seeds 1 and 2 are `A11_offset_seed_1` and `A11_offset_seed_2`.
- **Labels in older records.** Records written before `5b3d99b` use the earlier labels:
  - `cdea` was the fixed-boundary payoff;
  - `cdea:A10_boundary_shift` is the adopted default;
  - `cdea:A10_boundary_shift_o1` and `_o2` are A11.

  Their method hashes no longer match the tree.
- **Open.**
  - D2 fails at 5% on CIFAR-10.
  - The G1 read on seed 1 needs seed-1 checkpoints. This section says not to train seeds 1–4, so the user decides when.
  - Every paper cell, including the baseline rows, is rerun at the current hash before anything is reported.

## D2 fix, declared 2026-10-11 before the full run

- **Diagnosis.** Diagnostic on 24 seed-0 CIFAR-10 val images at 5% (scratch, not a record):
  - With more Adam steps, D2 grew instead of shrinking. The soft objective rewards spreading partial deletions over many cells, which no binary mask reproduces.
  - Two changes closed D2 on that sample: payoffs evaluated on binary masks drawn from the plan, and annealing the Sinkhorn temperature.
- **Choice.** By a rule fixed now, binary masks are adopted, because they add no setting. Annealing is used only if binary masks fail D2 on a dev cell.
  - **Draw.** Each unit goes to one row of the plan, with probability given by its column. The draw is shared across images, so allocations stay chunk-invariant, and players stay exclusive in every draw.
  - **Gradient.** Straight-through.
- **Checks.** D2 at all three areas on both dev datasets (`scripts/check_degenerate.py`), then the G1 read-out. CD did not enter the choice and is reported beside it.

**Result, 2026-10-11.**
- **D2.** Binary draws (`532c686`, hash `4c623f66`) pass D2 at 2.5%, 5% and 10% on both dev datasets (`results/paper/cdea_binary/`). The annealing fallback was not needed.
- **G1 read-out.** With D2 closed, the read-out is no longer met: CDEA is behind margin Grad-CAM on CIFAR-10 and HAM10000. See `G1.md`.
- **Decision, 2026-10-11.** Evaluation-paper route (`G1.md`). There is no further formulation change.
- **Labels.** Records under `results/paper/cdea_offset_seeds/` use the earlier `A10_boundary_shift*` labels.

## Decomposition game, 2026-10-14

The handoff `handoff10-13-26.md` was adopted with its recommendations. It was built on branch `feat/cdea-decomposition` and merged to `main` on 2026-10-14. Its entry is above; the specification is `FORMULATION_DECOMPOSITION.md`. The fractional-mask test below is held.

## Reopened, 2026-10-13: fractional-mask test, declared before it runs (held 2026-10-14)

- By user decision, the method question is reopened for one test: fractional masks (A12) against binary CDEA. It runs on seed 1, val, ResNet-50, all 8 contrastive datasets, n = 200. The rule, the decision and the commands are in `G1.md`, "Reopened, 2026-10-13".
- The read-out is `scripts/fractional_readout.py`.
- **Not launched.** Before launch, the user signs off on how D2 is treated, and the section and script are committed.

## Checkpoints, 2026-10-11

The formulation in `cdea/` does not change the classifier. `train_model` is the same routine as at `cf07ac2`: Adam, cosine schedule, cross-entropy, best val balanced accuracy, 15 epochs. Contrastive datasets stay frozen linear probes at lr 1e-3. Shift datasets stay full fine-tunes at lr 1e-4. ColoredMNIST's color draw and the planted-patch geometry are the ones those checkpoints were trained on. Input convention is still `imagenet`.

Seed 0 on this machine is complete: 22 checkpoints, both backbones, for the seven contrastive probes and the four shift fine-tunes. ImageNet uses the torchvision weights. Those files are reused. Seed 1 has the 14 contrastive probes only; its shift checkpoints are not trained.

Retrain a cell only when that cell's training is the thing that failed. `--force-train` does that for the cells named on the command. It does not walk the rest of the seed.

## Next session: full run on the GH200

**Before merging to `main`, 2026-10-11.**
- **Smoke run.** `scripts/smoke_e2e.py` at fast knobs ran every contrastive unit on both backbones, with every method and ablation. ImageNet was skipped because its data is not on the Mac.
  - The only error was `cdea:A8_precomputed`, which needs maps exported from tag `framing-v1`.
  - There were no non-finite scores.
  - CVE and SC-CVE fell back to the random floor only on random-head smoke models.
- **Shift game.** Only Waterbirds ResNet-50 ran before the run was stopped. It was clean.

**Known gaps.**
- No launcher passes an A8 maps directory, and no export script exists. Every `--ablations` cell records A8 as an error until one does. The cell finishes everything else.
- The shift units other than Waterbirds ResNet-50 were not smoke-tested.
- On a smoke model, the analytic decomposition gave the same masks as base evidence. Check this at full n.

**On the GH200.**
1. Pull `main`, then run `PYTHONPATH=. python -m pytest tests/`. `test_a_resnet_allocation_repeats_exactly_on_an_accelerator` checks CUDA determinism there.
2. Launch seed 0 with `scripts/launch_paper_chain.py --seeds 0 --jobs 4`.
   - Each cell trains its checkpoint only when the file is missing, then runs that checkpoint's games before the next dataset. Seed 0's checkpoints are already on disk, so this seed is the games.
   - Stanford Dogs trains once, then runs contrastive and shift. ImageNet skips training.
   - Four cells at a time, and the cap holds for the whole seed: two big (shift, either backbone, and contrastive ViT) and two small (contrastive ResNet). A failed cell starts nothing further. `--image-batch` trades memory for `--jobs` and does not change scores.
   - Done markers carry the code and knobs hashes, so cells written by older code run again.
   - Seed 1 starts only after seed 0 exits 0, and only if the command names it (`--seeds 0,1`). Shift seed 1 would train on that pass. Do not add seeds 1–4 until seed 0's games have finished and the user says to continue.

The test split stays locked until `eval-plan-frozen`.

Shift lives beside contrastive. `cdea/shift.py` is the transplant payoff and its own Sinkhorn loop. `scripts/gate_shift.py` prints the read-out and the stop rule. Do not train seeds 1–4 without a user decision. The sixth shift dataset is still unchosen.
