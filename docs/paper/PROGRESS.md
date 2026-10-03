# Progress

Source of truth for the stage sequence: `docs/paper/PLAN.md`.
Branch: `paper/p1-formulation`. The hygiene pull request is merged: https://github.com/supreethms1809/gambit/pull/1

## Stage status

| ID | Status | Commit | Notes |
|---|---|---|---|
| S01 | done | `1f322c3` through `a449b54` | Existing work split into commits. Plan, kickoff, `AGENTS.md`, and the workflow rule are in the repo. |
| S02 | done | `93aa355`, `49ffdc7` | Seeded subsets, three-way splits, test lock, interaction moved to the input device. |
| S03 | done | `46b1bfe` | Provenance in `save_json`, `NormalizedModel`, `--final` refuses a dirty tree. |
| S04 | done | `da589bb` | `evaluation/` scorer. Eval scripts import it. Throughput measured; see below. |
| S05 | done | `fd447a1`, `3b979ae` | One overlap weight, shared-sparse default 0.25, shift mass target, equation audit. |
| S06 | todo | | Degenerate-optimum checks D1–D7. Next session. |
| S07–S28 | todo | | Not started. |

## Decisions

- 2026-10-03. Pets and dogs: test is the complement of `random_split(seed=42)` at 80% train. Val is 10% of that train pool, carved with seed 43. The historical holdout is unchanged. Checked against `torch.utils.data.random_split`.
- 2026-10-03. HAM10000: the current `val/` folder is test. Val is a lesion-grouped 10% carve of `train/`. No lesion sits in both. All 7 classes are in both.
- 2026-10-03. Brain tumor: the current `Testing/` folder is test. Val is a patient-grouped 10% carve of `Training/`. No patient sits in both (168 train patients, 19 val patients).
- 2026-10-03. Checkpoint selection uses that val split for every paper dataset. Medical selection no longer reads the old holdout folders. Adam, cosine schedule, and cross-entropy are unchanged. Class weights for the medical sets are computed from the train subset.
- 2026-10-03. The throughput benchmark used randomly initialised ResNet-50 and ViT-B/16. Weights do not change the flop count, and ResNet-50 ImageNet weights were not downloaded.
- 2026-10-03. Provisional image budget, not frozen and not written into an eval plan: contrastive ResNet-50 `n=200`, contrastive ViT `n=64`, shift ResNet-50 `n=128`, shift ViT `n=32`. Revise after one timed CDEA cell. At 24.8 model steps/s, 200 images in batches of 4 is 50 batches; 50 allocator steps is about a minute per method before heavier baselines.
- 2026-10-03. CDEA means Contrastive Decomposition via Evidence Allocation. The method is Adam on the mask logits for a fixed number of steps, with the classifier frozen. Pairwise overlap has one weight, `lambda_overlap`. `lambda_disjoint` on the contrastive allocator is rejected. The shift game keeps its own product weight.
- 2026-10-03. Contrastive `kept_logit` is the raw logit \(z_k\). The returned key `suff` is an alias of that number. The shift quantity is the baseline-subtracted kept logit, \(z - z_0\).
- 2026-10-03. `lambda_shared_sparse` defaults to 0.25, including the script flags that used to pass 0. That closes degenerate route D1. `RobustShortcutObjective.lambda_mass` defaults to 0.1 with target mass `max(1, R / 49)` per mask.
- 2026-10-03. These changes invalidate the ablation `suff` and `overlap` columns, any contrastive run that applied both overlap weights, and any shared mask trained at `lambda_shared_sparse=0`. The map is `docs/paper/EQUATION_AUDIT.md`.

## Open issues

- Stored numbers under `results/` are pre-audit. The ablation tables are invalid: they used a class-ordered prefix, and pets/dogs eval included training images. Do not quote them.
- S05 invalidates stored ablation `suff` and `overlap` columns. Those files were already pre-audit.
- `docs/paper/EVAL_PLAN.md` does not exist. The test split stays locked.
- S07 datasets (CIFAR-100, Oxford-IIIT Pet-37, CUB-200, ImageNet-S) are not downloaded. Ask before downloading.
- No background jobs are running.

## Throughput (MPS, batch 4, 224, random init, 15 steps)

Log (local, gitignored): `results/paper/logs/throughput_mps.json`.

| Model | Forward+backward / s |
|---|---|
| ResNet-50 | 24.8 |
| ViT-B/16 | 10.1 |

## Exit check (S05)

```
55 passed, 8 warnings in 2.19s
```

`git log --oneline -5`:

```
3b979ae Record the equation-to-code map for the joint CDEA loss.
fd447a1 Count pairwise overlap once and close the shared-mask blanket.
301b259 Merge pull request #1 from supreethms1809/paper/p0-hygiene
08ed424 Record the hygiene pull request in the progress log.
2c500ba Record S01–S04 as done and point the next session at the equation audit.
```

## Next session

Stage **S06**. Degenerate-optimum checks D1–D7 on the val split of CIFAR-10 and HAM10000. Write `results/paper/degenerate/` and a short report. Every route is closed, or it is closed by a named constraint and recorded. Val only. Do not pass `--final`. Do not download a dataset. The formulation map is `docs/paper/EQUATION_AUDIT.md`.
