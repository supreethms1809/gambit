# Progress

This file tracks the decomposition paper from 2026-10-14 on. The method is `FORMULATION_DECOMPOSITION.md`. The evaluation is `EVAL_PLAN.md`, headed by section 13. The shift game is a separate paper (`SHIFT_EVAL_PLAN.md`) and is not tracked here.

History before 2026-10-14 is not carried forward. That covers the contrast-game route, gate G1, the D2 fix and the fractional-mask test. It is in git: `git show 62aecf6:docs/paper/PROGRESS.md`. The earlier method is on tag `framing-v1`.

**How to use this file.** At the end of every session:
- update "Next session";
- list every running job with its PID, command and log;
- tick the plan;
- add a dated line to the log, newest first.

Results are not typed here. Name the command and the records it read.

## Next session

On the GH200:
1. Pull `main` and run `PYTHONPATH=. python -m pytest tests/`. Every test should pass there. Two of them only mean something there:
   - `test_a_resnet_allocation_repeats_exactly_on_an_accelerator` checks CUDA determinism;
   - `test_plan_refactor` compares against a CPU reference that a Mac does not reproduce, even at the commit that wrote it.
2. Check the jobs under "Running".
3. Make the dev selection (plan A2), then run the degenerate checks (plan A3). Record the chosen labels in the log.

## Current state, 2026-10-14

- **Code.** `main`'s code is as of `62aecf6`, method hash `a5deb9fe`; later commits on 2026-10-14 changed docs only.
  - On `main`: the decomposition method, its runner (`paper_run.py --game decomposition`) and the planted-pair dataset with `verify_roles`.
  - Also on `main`: `scripts/decomp_readout.py` (RR, val-only selection, the candidate ranking and family D-T statistics) and D2/D8/D9 in `scripts/check_degenerate.py`.
- **Tags.** `framing-v1` and `decomp-declared` (on `80caf2a`) exist. `eval-plan-frozen`, `g0-baselines` and `final-runs` do not.
- **Plan.** `EVAL_PLAN.md` is not frozen, and the test split is locked. Section 13.4 reads "signed off" and is tagged, but the user asked to keep it a draft (open decision 1).
- **Checkpoints** (on the GH200).

  | Set | State |
  |---|---|
  | Natural datasets, seed 0 | 7 contrastive probes on both backbones; ImageNet uses torchvision weights |
  | Natural datasets, seed 1 | 7 contrastive probes on both backbones, from the seed-1 chain pinned at `cf07ac2` |
  | Natural datasets, seeds 2–4 | not trained (open decision 2) |
  | Planted pairs, ResNet-50, seeds 0–4 | trained 2026-10-14; best val balanced accuracy 0.986–0.987 |
  | Planted pairs, ViT-B/16, seeds 0–4 | training started 2026-10-14 01:13 UTC |

- **Records.** A record counts only when its hashes match the tree.

  | Where | What | Counts on `main` |
  |---|---|---|
  | `results/paper/decomp_dev/` | Dev cells: CIFAR-10 and HAM10000, ResNet-50, seed 0, val, n = 64, every method, DA1–DA9 and the 4-config grid. Planted-pair val cells: ResNet-50, seeds 0–4, n = 200 | yes, `a5deb9fe` |
  | `results/paper/cells/` | The contrast-game seed-0 chain, stopped at 10 of 24 units | no, `9d84e7f2` |
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

**A. Dev selection on val (GH200, nothing to build)**
- [ ] A1. Check that the dev and planted ResNet-50 val cells finished, with no method errors in their `summary.json`.
- [ ] A2. Read them:
  ```bash
  PYTHONPATH=. python scripts/decomp_readout.py --records results/paper/decomp_dev --split val --expect-hash a5deb9fe
  ```
  Record the chosen `cdea_decomp@lr…_t…` setting and the best min-split row per dataset and backbone (13.3).
- [ ] A3. Run the degenerate checks on each dev cell, for the chosen setting:
  ```bash
  PYTHONPATH=. python scripts/check_degenerate.py --records results/paper/decomp_dev/val/decomposition/<dataset>/resnet50/seed0 --method <chosen label>
  ```
  - D2 is gated. D8, D9 and the outside-prior share are reported.
  - If D9 (`same_as_init`) is high, CDEA stayed on its min-split start. Stop and decide before going on.
- [ ] A4. Once ViT-B/16 planted training is done, run its val cells and choose its min-split row by RR (13.3).

**B. Wiring (GH200)**
- [ ] B1. Add the decomposition game to `scripts/launch_paper_chain.py`, for the natural datasets and `planted_pair`.
- [ ] B2. Give `scripts/completion_check.py` the decomposition manifest. `FINAL_RUNS.md` and `tests/test_completion_check.py` pin today's contrastive manifest, so they change in the same commit.
- [ ] B3. Give `analysis/build_results.py` the D-T and D-C tables. `AUDIT.md` and `tests/test_audit_results.py` pin today's dataset list, so they change with it.
- [ ] B4. Add the decomposition units to `scripts/smoke_e2e.py`, and run the smoke on the GH200 (section 12, item 6).

**C. Natural-image val (family D-C selection)**
- [ ] C1. Run val cells for the 8 natural datasets, ResNet-50, seed 0. Choose the best min-split row per dataset (13.3).
- [ ] C2. Time a cell, then set n per seed and the test-sample seeds (2.4).

**D. Checkpoints**
- [ ] D1. Train natural-dataset seeds 2–4, after the user decides (open decision 2).

**E. Freeze**
- [ ] E1. Rewrite section 12 for the decomposition paper (open decision 4).
- [ ] E2. Record the selections, hashes and n in `EVAL_PLAN.md`. Add `frozen: true` and tag `eval-plan-frozen`.

**F. Confirmatory runs (test, `--final`, after E2 only)**
- [ ] F1. Family D-T: planted pairs, ResNet-50, seeds 0–4, n = 200 kept images per seed. ViT-B/16 is the robustness row.
- [ ] F2. Family D-C: the 8 natural datasets, ResNet-50, seeds 0–4.
- [ ] F3. Robustness and ablations: blur, the other areas, ViT-B/16, every image, DA1–DA9.
- [ ] F4. Read out and apply 13.4.

**G. Results and paper**
- [ ] G1. Generate `results/paper/RESULTS.md` with `analysis/build_results.py`, and check it with `analysis/audit_results.py`.
- [ ] G2. A fresh session, one that did not write the results, audits them and eye-checks a sample of masks on every dataset.
- [ ] G3. Write the paper from `FRAMING.md`. Use `PAPER_LESSONS_AND_WRITING_GUIDE.md` as the checklist.

## Open decisions

1. **Section 13.4: draft or signed off.** PR #34 recorded a sign-off and tagged `decomp-declared`. On 2026-10-14 the user asked to keep it a draft, because the plan and metrics may change after the dev results. If it stays a draft, remove the tag and restore the DRAFT heading. If it is signed off, a later change is a dated entry.
2. **Seeds 2–4** for the natural datasets: train them or not, and when.
3. **Contrast-game rows.** C1–C5 are descriptive (section 1). Either DA2 inside the decomposition cells is enough, or the contrast-game chain is rerun at the current hash.
4. **Section 12's freeze list** was written for the contrast-game paper: G0 tag, CVE reproduction, CUB part scorer, two-cue classifiers. Decide which items apply, and add the decomposition ones: the CDEA selection, the D2 and D9 checks, and the planted ViT.

## Parked

Not part of this plan unless reopened:
- the fractional-mask test (held 2026-10-14);
- ablation A8's map export from `framing-v1`;
- the shift paper.

## Log

- **2026-10-14.** Fresh start. PROGRESS.md was rewritten around the decomposition paper. The old file is at `62aecf6`. `SPARK_RUNBOOK.md` was removed, because it described the contrast-game launcher and the `results/paper/runs/` records.
