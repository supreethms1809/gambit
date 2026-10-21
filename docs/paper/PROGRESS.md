# Progress

This file tracks the decomposition paper from 2026-10-14 on. The method is `FORMULATION_DECOMPOSITION.md`. The evaluation is `EVAL_PLAN.md`, headed by section 13. The shift game is a separate paper (`SHIFT_EVAL_PLAN.md`). Its runs are tracked under "Shift paper" below.

History before 2026-10-14 is not carried forward. That covers the contrast-game route, gate G1, the D2 fix and the fractional-mask test. It is in git: `git show 62aecf6:docs/paper/PROGRESS.md`. The earlier method is on tag `framing-v1`.

**How to use this file.** At the end of every session:
- update "Next session";
- list every running job with its PID, command and log;
- tick the plan;
- add a dated line to the log, newest first.

Results are not typed here. Name the command and the records it read.

## Next session

On the GH200:
- **First, read open decision 9 (version 2 re-scope).**
  - If the user has signed off version 2, hold the decomposition phases (`dev`, `planted-val`, `natural-val`). Version 1 records will not match the version 2 hash.
  - The shift phases are unaffected.
0. The 2026-10-20 review fixes and the B7 review's fixes (log) moved the method hash to `d69b1ac6` and the knobs hash to `38f5d7c0`. No earlier record counts. ROAD now solves to convergence, so time one decomposition cell first (C2) and check the launcher's cell budget.
1. Pull `main` and run `PYTHONPATH=. python -m pytest tests/`. Every test should pass there. Two of them only mean something there:
   - `test_a_resnet_allocation_repeats_exactly_on_an_accelerator` checks CUDA determinism;
   - `test_plan_refactor` compares against a CPU reference that a Mac does not reproduce, even at the commit that wrote it.
2. Check the jobs under "Running". The planted-pair ViT-B/16 training must have finished: the launcher trains any checkpoint it cannot find, and it cannot see a trainer started outside it.
3. Dry-run, then launch the val phases in one queue, two large and two small cells at once:
   ```bash
   PYTHONPATH=. python scripts/launch_runs.py --phases dev,planted-val,natural-val,shift-gate --dry-run
   nohup env PYTHONPATH=. python scripts/launch_runs.py --phases dev,planted-val,natural-val,shift-gate > results/paper/logs/launch_runs.log 2>&1 &
   ```
   - Record the launcher's PID under "Running".
   - Cell logs are in `results/paper/logs/runs/`.
   - The same command resumes after a stop and skips done cells.
4. When dev is done, make the dev selection (A2) and run the degenerate checks (A3). Record the chosen labels in the log.
5. When the shift gate is done, read it (SH2). Only then launch `shift-val` (SH3).

## Current state, 2026-10-14

- **Code.** Method hash `d69b1ac6`, knobs hash `38f5d7c0` (2026-10-20, the review fixes and the B7 review's fixes). Before that it was `11bdeb54`. On 2026-10-14 it moved from `a5deb9fe` to `8486647c` (the shift-game commits), then to `11bdeb54` (the built-in input convention). See the log.
  - The 2026-10-20 fixes change scores: ROAD's fill, the shift random floor and `extremal_shift`. Rows written at `11bdeb54` are not comparable.
  - The decomposition game did not change. A CPU smoke cell at all three hashes gives identical rows, every column except the hash (7 core methods, 28 rows).
  - The input convention is built in (`models.wrapper.INPUT_CONVENTION = "imagenet"`). No file is read, and every paper checkpoint name is unchanged.
  - On `main`: the decomposition method, its runner (`paper_run.py --game decomposition`) and the planted-pair dataset with `verify_roles`.
  - Also on `main`: `scripts/decomp_readout.py` (RR, val-only selection, the candidate ranking and family D-T statistics) and D2/D8/D9 in `scripts/check_degenerate.py`.
- **Tags.** `framing-v1` and `decomp-declared` (on `80caf2a`) exist. `eval-plan-frozen`, `g0-baselines` and `final-runs` do not.
- **Formulation.** `FORMULATION_DECOMPOSITION.md` is version 2, a DRAFT written on 2026-10-20 and not signed off. It is implemented on branch `paper/decomp-partition` (not merged; method hash there `251951b4`), with two changes made after the first dev read: the polish (section 4.4) and the ∅ rule (section 6.4). `main` implements version 1 (tag `decomp-declared`). See plan R and open decision 9.
- **Plan.** `EVAL_PLAN.md` is not frozen, and the test split is locked. Section 13.4 reads "signed off" and is tagged, but the user asked to keep it a draft (open decision 1).
- **Checkpoints** (on the GH200).

  | Set | State |
  |---|---|
  | Natural datasets, seed 0 | 7 contrastive probes on both backbones; ImageNet uses torchvision weights |
  | Natural datasets, seed 1 | 7 contrastive probes on both backbones, from the seed-1 chain pinned at `cf07ac2` |
  | Natural datasets, seeds 2–4 | not trained; the launcher trains each one before the first cell that needs it |
  | Planted pairs, ResNet-50, seeds 0–4 | trained 2026-10-14; best val balanced accuracy 0.986–0.987 |
  | Planted pairs, ViT-B/16, seeds 0–4 | training started 2026-10-14 01:13 UTC |

- **Records.** A record counts only when its hashes match the tree.

  | Where | What | Counts on `main` |
  |---|---|---|
  | `results/paper/cells/val/decomposition/`, `.../shift/` | Written by `scripts/launch_runs.py` from now on | yes, when the hash matches |
  | `results/paper/gate_shift/` | The shift gate cells, written by the `shift-gate` phase | yes, when the hash matches |
  | `results/paper/decomp_dev/` | The first dev and planted ResNet-50 val cells | no. Written at `a5deb9fe` and rerun at the current hash by user decision (2026-10-14); kept, not scored |
  | `results/paper/cells/val/contrastive/` | The contrast-game seed-0 chain, stopped at 10 of 24 units | no, `9d84e7f2` |
  | `results/paper/runs/`, `results/paper/superseded/`, `results/paper/cdea_*` | Earlier objectives and diagnostics | no, never scored |

## Running

Last recorded on 2026-10-14. No PIDs were recorded, so check these on the GH200 and add the PIDs.
- **Planted-pair ViT-B/16 training.**
  - Command: `launch_paper_training.py --seeds 0,1,2,3,4 --datasets planted_pair --models vit_b_16 --jobs 3`.
  - Log: `results/paper/logs/train_planted_pair_vit_b_16.log`.
- **Planted-pair ResNet-50 val cells.**
  - Seeds 0–4 run one after another, writing into `results/paper/decomp_dev/`.
  - They run from the worktree `../gambit-decomp`, which has the same hash as `main`.
  - Log: `results/paper/logs/decomp_dev/planted_pair_resnet50_loop.log`.
- **Dev cells for CIFAR-10 and HAM10000.** Started 2026-10-14 around 00:20 UTC. Logs are in `results/paper/logs/decomp_dev/`.

## Plan

**A. Dev selection on val (GH200)**
- [ ] A1. Run the `dev` and `planted-val` phases of `scripts/launch_runs.py` at the current hash (Next session, step 3). A cell that finishes with a method error is marked failed.
- [ ] A2. Read them:
  ```bash
  PYTHONPATH=. python scripts/decomp_readout.py --records results/paper/cells --split val
  ```
  Record the chosen `cdea_decomp@lr…_t…` setting and the best min-split row per dataset and backbone (13.3).
- [ ] A3. Run the degenerate checks on each dev cell, for the chosen setting:
  ```bash
  PYTHONPATH=. python scripts/check_degenerate.py --records results/paper/cells/val/decomposition/<dataset>/resnet50/seed0 --method <chosen label>
  ```
  - D2 is gated. D8, D9 and the outside-prior share are reported.
  - If D9 (`same_as_init`) is high, CDEA stayed on its min-split start. Stop and decide before going on.
- [ ] A4. Planted ViT-B/16 val cells are in `planted-val`. Choose their min-split row by RR (13.3) with the same read-out.

**B. Wiring**
- [x] B1. `scripts/launch_runs.py` (2026-10-14) schedules both games by phase, two large and two small cells at once, and trains missing checkpoints. `--dry-run` lists every cell as done or pending.
- [ ] B2. Give `scripts/completion_check.py` the decomposition manifest. `FINAL_RUNS.md` and `tests/test_completion_check.py` pin today's contrastive manifest, so they change in the same commit. Until then, `launch_runs.py --dry-run` is the completion check.
- [ ] B3. Give `analysis/build_results.py` the D-T and D-C tables. `AUDIT.md` and `tests/test_audit_results.py` pin today's dataset list, so they change with it.
- [ ] B4. Add the decomposition units to `scripts/smoke_e2e.py`, and run the smoke on the GH200 (section 12, item 6).
- [ ] B5. Save the scored masks for figures and for G2's eye-check. No runner writes them today, so G2 cannot look at a mask without rerunning a cell.

**C. Natural-image val (family D-C selection)**
- [ ] C1. The `natural-val` phase runs the other 6 natural datasets on ResNet-50 and all 8 on ViT-B/16, seed 0, n = 64. Choose the best min-split row per dataset and backbone (13.3).
- [ ] C2. Time a cell, then set n per seed and the test-sample seeds (2.4).

**D. Checkpoints**
- [ ] D1. Natural seeds 2–4: the launcher trains each checkpoint before the first cell that needs it (user decision, 2026-10-14).

**E. Freeze**
- [ ] E1. Rewrite section 12 for the decomposition paper (open decision 4).
- [ ] E2. Record the selections, hashes and n in `EVAL_PLAN.md`. Add `frozen: true` and tag `eval-plan-frozen`.
- [ ] E3. Make DA1–DA9 start from the chosen setting. Today `_decomp_maps` applies them to `DecompConfig()` (lr 0.1 and the knobs' steps), which is not a candidate. That is fine for dev, but the test ablations must vary the chosen setting.

**F. Confirmatory runs (test, `--final`, after E2 only)**
- F1–F3 are one launch, with the n set at the freeze:
  ```bash
  PYTHONPATH=. python scripts/launch_runs.py --phases test --cdea <chosen> --final --config-hash <frozen hash> --n-test resnet50:<n>,vit_b_16:<n>
  ```
- [ ] F1. Family D-T: planted pairs, ResNet-50, seeds 0–4, n = 200 kept images per seed. ViT-B/16 is the robustness row.
- [ ] F2. Family D-C: the 8 natural datasets, ResNet-50, seeds 0–4.
- [ ] F3. Robustness and ablations: blur, the other areas, ViT-B/16, every image. DA1–DA9 run on section 8's units and seeds (CIFAR-100, CUB-200, brain tumor and ImageNet; seeds 0–2; ResNet-50).
- [ ] F4. Read out and apply 13.4.

**G. Results and paper**
- [ ] G1. Generate `results/paper/RESULTS.md` with `analysis/build_results.py`, and check it with `analysis/audit_results.py`.
- [ ] G2. A fresh session, one that did not write the results, audits them and eye-checks a sample of masks on every dataset.
- [ ] G3. Write the paper from `FRAMING.md`. Use `PAPER_LESSONS_AND_WRITING_GUIDE.md` as the checklist.

**R. Re-scope to version 2 (2026-10-20, `FORMULATION_DECOMPOSITION.md` version 2, `EVAL_PLAN.md` 13.5–13.8)**
- [ ] R1. The user signs off version 2 and 13.5–13.8, or amends them. Settle the formulation's open decisions (section 12) first.
- [x] R2. Descriptive measurements on dev val, before the version 2 implementation (formulation open decision 10). Both need only forward passes on existing checkpoints:
  - the co-location share at ResNet-50's last layer;
  - the pairwise interaction on the top Ω cells.

  Done on the Mac on 2026-10-20 with the local linear probes, not the paper checkpoints (log). Repeat on the GH200 with the paper checkpoints.
- [x] R3. Implement version 2 (branch `paper/decomp-partition`; the files differ from the list below: `cdea/region.py`, `evaluation/run_partition.py`, `scripts/partition_readout.py`):
  - Ω and the size-matched null (`cdea/null.py`);
  - the four-role per-cell plan (`cdea/decompose.py`);
  - the rule partitions, `occlusion_partition`, `gradient_partition` and `random_partition` (`baselines/decomposition.py`);
  - whole-cell scoring with the k, l and m columns, the cell-level scores, MIX and the confirmations (`evaluation/scores.py`, `evaluation/run_cell.py`);
  - VR, the resolution shares, RA and the tertiles (`scripts/decomp_readout.py`);
  - D8, D9 and the trapped roles (`scripts/check_degenerate.py`).
- [x] R4. Tests: formulation section 9, including the mixing and co-location toys (`tests/test_partition.py`).
- [ ] R5. Rerun the dev and planted val cells at the version 2 hash. Make the version 2 selection (13.7).
  - A local dev read (CIFAR-10 and HAM10000, local probes) is in the log. It is not R5: the hash, the checkpoints and the planted pairs are the GH200's.
  - Commands: `paper_run.py --game partition` with `--methods core --candidates all --ablations DA3_gradient_partition,DA8_no_null,DA10_fixed_budgets,DA11_no_polish,DA12_polish_only`. Planted pairs: `--dataset planted_pair`.
- [ ] R6. Bring `PAPER_NOTES.md`, the guide (`docs/guide/`) and the project's CVPR outline in line with version 2. Done after R1, so that they follow the signed-off text.

## Shift paper

A separate paper: `SHIFT_FORMULATION.md` and `SHIFT_EVAL_PLAN.md`. Implemented on 2026-10-14 and ready for the val gate. The test split stays locked.
- **Checkpoints** (on the GH200):
  - seed 0: the four shift fine-tunes (`colored_mnist`, `planted_patch`, `imagenet9`, `waterbirds`) on both backbones;
  - seed 1: not trained; the `shift-gate` phase trains Waterbirds and `planted_patch` ResNet-50 first;
  - Stanford Dogs restyled trains in the chain.
- **Known errors.** `cdea_shift:S6_precomputed` errors without a maps directory, like A8. Every other core method and ablation ran on a fast smoke cell (`planted_patch`, n = 2).

**Plan**
- [ ] SH1. The gate: Waterbirds and `planted_patch` val, ResNet-50, seeds 0 and 1, n = 64, area 5%. The `shift-gate` phase runs all four cells (Next session, step 3) and trains the missing seed-1 checkpoints first (user decision, 2026-10-14). Records go to `results/paper/gate_shift/`, apart from `shift-val`.
- [ ] SH2. Read the gate and check the degenerate routes:
  ```bash
  PYTHONPATH=. python scripts/gate_shift.py --records results/paper/gate_shift --split val
  PYTHONPATH=. python scripts/check_degenerate.py --records results/paper/gate_shift/val/shift/<dataset>/resnet50/seed<s>
  ```
  - D2 and D4 are gates. D3 and D9 are reported.
  - The stop rule is in `SHIFT_EVAL_PLAN.md`.
- [ ] SH3. After the gate, per-dataset selection: `launch_runs.py --phases shift-val`. That covers the 5 shift datasets on both backbones, seed 0, areas 5/10/25%, the candidates, and S1–S8 without S6 on Waterbirds and `planted_patch` ResNet-50. The sixth dataset and seeds 1–4 come after that.

## Open decisions

1. **Section 13.4: draft or signed off.** PR #34 recorded a sign-off and tagged `decomp-declared`. On 2026-10-14 the user asked to keep it a draft, because the plan and metrics may change after the dev results. If it stays a draft, remove the tag and restore the DRAFT heading. If it is signed off, a later change is a dated entry.
2. ~~Seeds 2–4~~ Resolved 2026-10-14: the launcher trains a missing checkpoint before the first cell that needs it.
3. **Contrast-game rows.** C1–C5 are descriptive (section 1). Either DA2 inside the decomposition cells is enough, or the contrast-game chain is rerun at the current hash.
   - The K×K specificity, contrast SV, DPC and TCS scorers (`EVAL_PLAN.md` sections 5.3–5.5) do not exist. They are descoped with this decision (user, 2026-10-20). If the contrast-game chain is rerun, they are implemented or dropped from the plan before the freeze.
4. **Section 12's freeze list** was written for the contrast-game paper: G0 tag, CVE reproduction, CUB part scorer, two-cue classifiers. Decide which items apply, and add the decomposition ones: the CDEA selection, the D2 and D9 checks, and the planted ViT.

5. ~~Records written at `a5deb9fe`~~ Resolved 2026-10-14: the dev and planted ResNet-50 val cells are rerun at the current hash (`dev`, `planted-val`). `results/paper/decomp_dev/` is kept and not scored.
6. **Shift gate and plan.**
   - The 2026-10-14 edit of `SHIFT_EVAL_PLAN.md` dropped:
     - the robustness list (blur, the other areas, ViT, leave-one-dataset-out);
     - the "After the gate" order;
     - the worst-group-accuracy use case;
     - the family S power line;
     - the "No shift number has been read" disclosure.
   - The `SHIFT_FORMULATION.md` edit replaced S7 (steps matched to per-environment Extremal Perturbations) with `S7_fixed_boundary`.
   - Confirm these drops were intended. Restore the disclosure if it is still true.
7. **RISE mask count.** The harness runs 4000 masks with the val grid {2000, 4000}. That is the RISE paper's VGG-16 count; its ResNet-50 setting is 8000 (`BASELINES.md`). Either add 8000 to the grid or keep 4000 and say so in the paper.
8. **The adapter floor is scattered pixels.** `baselines.adapter.random_floor`, used for failed rows and constant maps, ranks a map of ones, so every pixel ties and the tie-break picks scattered pixels, not cells. ROAD fills scattered pixels far better than a cell (mean fill error 0.032 against 0.104 at 5%), so a floored row likely drops less than the cell-shaped null and scores below chance. Either draw it from seeded cell scores, as the floor methods and the null do, or keep it and report the floored share (both read-outs print it).

9. **Version 2 re-scope (2026-10-20).** The decomposition's question was re-scoped (`FORMULATION_DECOMPOSITION.md` version 2):
   - divide the evidence region Ω with free role sizes;
   - measure the overlap before and the verified separation after;
   - keep K = 2, and score the third class.

   Both documents are drafts. Decide:
   - the formulation's open decisions (section 12);
   - whether to sign off 13.5–13.8;
   - whether the CVPR timeline (paper deadline Nov 16) allows R3–R5 before the freeze, or the venue changes.

   Until then, version 1 (13.1–13.4) stands.

   On 2026-10-20 and 21 the user asked for version 2 to be built and read locally before the sign-off (log). The sign-off now covers the two changes that read led to: the polish and the ∅ rule. Two questions are open:
   - Should CDEA polish under ROAD as well as blur? Its VR gains under its own blur are two to three times those under ROAD.
   - Should the ∅ absence test stay at the 5th percentile? Under ROAD almost no ∅ passes it, so VR now counts the unique and shared roles alone.

## Parked

Not part of this plan unless reopened:
- the fractional-mask test (held 2026-10-14);
- ablation A8's map export from `framing-v1`, and S6's from the same tag.
- `results/paper/shift_pairs/preview.png` (2026-10-03) shows the `planted_patch` patch moved, from before the in-place counterfactual. No script regenerates it. Do not use it in a figure.

## Log

- **2026-10-21, version 2 fixed and read again (local).** After the CIFAR-10 read (below), the user asked for two fixes, made on `paper/decomp-partition` (hash `251951b4`):
  - **The ∅ rule.** ∅ is now confirmed by an absence test. The old rule is kept as `vr_lenient`.
  - **The polish.** CDEA's search is now followed by a discrete polish. DA11 is the search alone; DA12 is the polish alone.
  - The original-spec HAM10000 cell was stopped before it wrote records. So no HAM10000 method row was read before the fixes.
  - Records: `results/paper/smoke/v2_local/cells_fixed/val/partition/{cifar10,ham10000}/resnet50/seed0`, in the worktree `.claude/worktrees/decomp-v2`, not under `main`. Same command as below, plus DA11 and DA12.
  - Read with `PYTHONPATH=. python scripts/partition_readout.py --root results/paper/smoke/v2_local/cells_fixed`, with `--operator road` and `--operator blur`.
  - The read is summarised in the session that ran it. Its open questions are open decision 9.
- **2026-10-20, version 2 implemented and read locally.** Branch `paper/decomp-partition`, worktree `.claude/worktrees/decomp-v2`. The user asked for a local dev read before the sign-off; `EVAL_PLAN.md` discloses it.
  - Checkpoints: the local linear probes, `results/paper/smoke_checkpoints/{cifar10,ham10000}_resnet50_pt_lp_ep15_lr0.001_imagenet_seed0.pt`, via `--model-source checkpoint`. `--model-source smoke` is an ImageNet backbone with a random head. A first descriptive read used it by mistake, and was discarded.
  - Descriptive (R2): `PYTHONPATH=. python scripts/decomp_descriptive.py --dataset {cifar10,ham10000} --n 64 --model-source checkpoint --checkpoint <probe> --out results/paper/smoke/v2_local/descriptive`.
  - Cells: `PYTHONPATH=. python scripts/paper_run.py --game partition --dataset {cifar10,ham10000} --backbone resnet50 --seed 0 --split val --n 64 --methods core --candidates all --ablations DA3_gradient_partition,DA8_no_null,DA10_fixed_budgets --model-source checkpoint --checkpoint <probe> --out results/paper/smoke/v2_local/cells --image-batch 8`. Only CIFAR-10 finished and was read; it led to the fixes above.
  - Formulation section 13 lists the implementation's choices and its disagreements with the text. That includes the straight-through drift below step 0 that led to the polish.
  - Version tokens are banned in code and `docs/paper` (`tests/test_scope.py`). The game is therefore `--game partition`, and the draft documents say "version 1" and "version 2".

- **2026-10-20, version 2 re-scope (docs only).** `FORMULATION_DECOMPOSITION.md` was rewritten as version 2, a DRAFT; version 1 is at tag `decomp-declared`. `EVAL_PLAN.md` gained a dated re-scope entry with disclosures, and the draft sections 13.5–13.8; 13.1–13.4 are unchanged. Plan R and open decision 9 were added. No code changed, and no cell was run.
- **2026-10-20, B7 review of PR #40.** A fresh session reviewed the baseline changes. `extremal_shift` as deletion, Chefer's `start_layer` and the converged ROAD fill are sound, and count as reviewed. Fixed on branch `paper/review-pr40-fixes`; the method hash moved from `e910f251` to `d69b1ac6`.
  - **CVE `edit_share`.** The fill past the edits is the bilinear halo around them, then scattered tie-break pixels, not seeded untouched cells. `edit_share` is now the scored mask's share on edited cells (two edits at 5%: 0.74, where the old count gave 0.82), and the dossier and plan say what the fill is.
  - **Floor seed.** At seed 0 the shift floor, `random_decomp` and `random_pair` drew the null's own scores. They now draw from `floor_generator` (stream `1000 · seed + 7`).
  - **ROAD.** The neighbour sum is shifted float32 adds, so TF32 cannot reach the solve. The accuracy comment gives the measured range, and the direct-solve test covers 95%, 98% and 99.5% imputed.
  - **Not done.** The review's suggested deletion-variant tests in `tests/test_extremal.py` (not blocking), the ROAD clamp disclosure, and open decision 8 (the adapter floor).
- **2026-10-20, review fixes.** A verified review of `c3dee9b` was worked through on branch `paper/review-fixes`, one commit per item. The method hash moved from `11bdeb54` to `e910f251`, and the knobs hash to `38f5d7c0`.
  - **ROAD.** The fill was 24 Jacobi sweeps from black: a 32 px cell of a flat 0.5 image filled to 0.17, and a 25% block to 0.04. It now solves ROAD's own system (`road_evaluation` `e6b9987`) to convergence by conjugate gradient, within 1e-3 of scipy's direct solve in a test. `verify_roles` uses it too (user decision: converge).
  - **Shift.** The random floor drew one map for R and S and repeated it per chunk; now independent and whole-sample. `extremal_shift` runs deletion masks (user decision; needs the B7 review). The gate's "ahead" is defined: pooled paired ΔD@5% with its 95% bootstrap interval above 0 (user decision), and `gate_shift.py` prints the verdict.
  - **Records.** Fast-knob cells write to `results/paper/smoke/fast_knobs` and refuse other paths under `results/paper`. The gate and every degenerate check drop fast rows.
  - **Scores.** A missing S is NaN, not 0, so `margin_gradcam_ref`'s RR is over U_k and U_l. Constant maps are flagged (`constant_row`), and both read-outs print the failed and constant shares.
  - **Contrast game.** Chefer's `start_layer` reaches the rollout, so its two candidates differ. CVE keeps matched area, and rows record `edit_share` (user decision). K×K, contrast SV, DPC and TCS are descoped under open decision 3.
  - **Labels and docs.** CIFAR-10 and MNIST record their real val carve seed, 42; the indices are unchanged. The dossier, plans, `PAPER_NOTES` (section 10 rewritten) and the guide match the code.
  - **Checks.** Tests on the Mac in `marl`; the full-suite result is in the PR. No cell was run.
- **2026-10-14, launcher.** `scripts/launch_runs.py` runs the paper's phases for both games in one queue.
  - Phases: `dev`, `planted-val`, `natural-val`, `test`, `shift-gate` and `shift-val`.
  - It keeps two large cells (shift, and ViT decomposition) and two small cells (ResNet decomposition) running. When no large cell waits, small cells use the idle large slots.
  - It reuses a checkpoint or trains it under a lock, marks a cell failed on any method error except S6 and A8, and resumes from its markers.
  - ViT decomposition was added to the large class in `parallel_cells.MEM_SEED_HIGH`.
  - Checked on the Mac with unit tests, a `--dry-run` of every phase (40 val cells, 90 test cells), and one two-cell smoke launch. Real launches run on the GH200 only.
- **2026-10-14, input convention.** It is built in as `models.wrapper.INPUT_CONVENTION = "imagenet"`. `results/paper/input_convention.txt` and its reader were removed.
  - The old reader silently fell back to `raw` when the file was missing, which would have renamed every paper checkpoint.
  - `probe_input_convention.py` now reports its winner against the constant and writes nothing.
  - The checkpoint names for all 150 dataset × backbone × seed cells are unchanged.
  - The hash moved from `8486647c` to `11bdeb54`, and the decomposition rows are identical.
- **2026-10-14, later.** The shift-game changes were reviewed and committed, which moved the method hash from `a5deb9fe` to `8486647c`.
  - **Changes:**
    - binary draws, boundary offsets and the shared plan helpers in `cdea/shift.py`;
    - the `shift_assign` baseline;
    - the null-subtracted 2×2 shift scores (`delta_d`, `rn`, `pur_r`, `pur_s`, `role_sep`);
    - the soft/hard, overshoot and same-as-init stage payoffs;
    - the in-place `planted_patch` counterfactual;
    - shift D2/D4/D3/D9 in `check_degenerate.py`, and the `gate_shift.py --records` read-out.
  - **Review fixes:**
    - a junk line in `allocate_shift` that turned the device check into a `TypeError`;
    - `shift_assign`'s argmax now runs in eval mode;
    - `assign()` moves scores to the CPU before float64, so it runs on MPS.
  - **Checks:**
    - the tests pass, except `test_plan_refactor`'s Mac platform reference;
    - a fast smoke shift cell ran every core method and ablation except S6;
    - the decomposition rows are identical at both hashes.
- **2026-10-14.** Fresh start. PROGRESS.md was rewritten around the decomposition paper. The old file is at `62aecf6`. `SPARK_RUNBOOK.md` was removed, because it described the contrast-game launcher and the `results/paper/runs/` records.
