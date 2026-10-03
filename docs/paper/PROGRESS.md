# Progress

Source of truth for the stage sequence: `docs/paper/PLAN.md`.
Branch: `paper/p0-hygiene`.

## Stage status

| ID | Status | Commit | Notes |
|---|---|---|---|
| S01 | done | `1f322c3` through `a449b54` | Existing work split into commits. Plan, kickoff, `AGENTS.md`, and the workflow rule are in the repo. |
| S02 | done | `93aa355`, `49ffdc7` | Seeded subsets, three-way splits, test lock, interaction moved to the input device. |
| S03 | done | `46b1bfe` | Provenance in `save_json`, `NormalizedModel`, `--final` refuses a dirty tree. |
| S04 | done | `da589bb` | `evaluation/` scorer. Eval scripts import it. Throughput measured; see below. |
| S05 | todo | | Equation audit. Next session. |
| S06–S28 | todo | | Not started. |

## Decisions

- 2026-10-03. Pets and dogs: test is the complement of `random_split(seed=42)` at 80% train. Val is 10% of that train pool, carved with seed 43. The historical holdout is unchanged. Checked against `torch.utils.data.random_split`.
- 2026-10-03. HAM10000: the current `val/` folder is test. Val is a lesion-grouped 10% carve of `train/`. No lesion sits in both. All 7 classes are in both.
- 2026-10-03. Brain tumor: the current `Testing/` folder is test. Val is a patient-grouped 10% carve of `Training/`. No patient sits in both (168 train patients, 19 val patients).
- 2026-10-03. Checkpoint selection uses that val split for every paper dataset. Medical selection no longer reads the old holdout folders. Adam, cosine schedule, and cross-entropy are unchanged. Class weights for the medical sets are computed from the train subset.
- 2026-10-03. The throughput benchmark used randomly initialised ResNet-50 and ViT-B/16. Weights do not change the flop count, and ResNet-50 ImageNet weights were not downloaded.
- 2026-10-03. Provisional image budget, not frozen and not written into an eval plan: contrastive ResNet-50 `n=200`, contrastive ViT `n=64`, shift ResNet-50 `n=128`, shift ViT `n=32`. Revise after one timed CDEA cell. At 24.8 model steps/s, 200 images in batches of 4 is 50 batches; 50 allocator steps is about a minute per method before heavier baselines.

## Open issues

- Stored numbers under `results/` are pre-audit. The ablation tables are invalid: they used a class-ordered prefix, and pets/dogs eval included training images. Do not quote them.
- `lambda_shared_sparse` still defaults to 0. Phase 1 (S05) changes that.
- `docs/paper/EVAL_PLAN.md` does not exist. The test split stays locked.
- S07 datasets (CIFAR-100, Oxford-IIIT Pet-37, CUB-200, ImageNet-S) are not downloaded. Ask before downloading.
- No background jobs are running.

## Throughput (MPS, batch 4, 224, random init, 15 steps)

Log (local, gitignored): `results/paper/logs/throughput_mps.json`.

| Model | Forward+backward / s |
|---|---|
| ResNet-50 | 24.8 |
| ViT-B/16 | 10.1 |

## Exit check (S04)

```
49 passed, 8 warnings in 1.71s
```

`git log --oneline -5` at the scorer commit:

```
da589bb Score every method with one mask conversion, one removal operator, and one null.
49ffdc7 Give every dataset a train/val/test split and stop eval from reading a class prefix.
46b1bfe Record provenance on every result file and normalise images inside the model.
93aa355 Move the interaction module onto the input device before it runs.
a449b54 Add the paper plan and the session protocol so the next chat can find the next stage.
```

## Next session

Stage **S05**. First step: read `instantiations/contrastive/objective.py`, `instantiations/contrastive/allocator.py`, and `instantiations/shift/objective.py`, and write `docs/paper/EQUATION_AUDIT.md` with a file:line for each term. Do not change a default until that table exists. The overlap term and `_disjoint_penalty` are the same quantity and should become one term. `lambda_shared_sparse` should default above 0 only after the audit says which degenerate route that closes.

The test split is still locked. Do not pass `--final`. Do not download a dataset.
