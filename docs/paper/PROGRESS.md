# Progress

Source of truth for the stage sequence: `docs/paper/PLAN.md`.
Branch: `paper/p2-datasets`. Degenerate-check pull request: https://github.com/supreethms1809/gambit/pull/3

## Stage status

| ID | Status | Commit | Notes |
|---|---|---|---|
| S01 | done | `1f322c3` through `a449b54` | Existing work split into commits. Plan, kickoff, `AGENTS.md`, and the workflow rule are in the repo. |
| S02 | done | `93aa355`, `49ffdc7` | Seeded subsets, three-way splits, test lock, interaction moved to the input device. |
| S03 | done | `46b1bfe` | Provenance in `save_json`, `NormalizedModel`, `--final` refuses a dirty tree. |
| S04 | done | `da589bb` | `evaluation/` scorer. Eval scripts import it. Throughput measured; see below. |
| S05 | done | `fd447a1`, `3b979ae` | One overlap weight, shared-sparse default 0.25, shift mass target, equation audit. |
| S06 | done | `7724885` | D1–D7 on val. Open routes recorded in `results/paper/degenerate/REPORT.md`. |
| S07 | done | | CIFAR-100, Oxford-IIIT Pet (37), CUB-200. ImageNet-S is waiting on ImageNet-1k. |
| S08 | todo | | Shift datasets. Next session. |
| S09–S28 | todo | | Not started. |

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

## Open issues

- Stored numbers under `results/` are pre-audit. The ablation tables are invalid: they used a class-ordered prefix, and pets/dogs eval included training images. Do not quote them.
- S05 invalidates stored ablation `suff` and `overlap` columns. Those files were already pre-audit.
- `docs/paper/EVAL_PLAN.md` does not exist. The test split stays locked.
- ImageNet-S still needs a local ImageNet-1k copy. Do not download ImageNet-1k without being asked.
- No background jobs are running.

## Throughput (MPS, batch 4, 224, random init, 15 steps)

Log (local, gitignored): `results/paper/logs/throughput_mps.json`.

| Model | Forward+backward / s |
|---|---|
| ResNet-50 | 24.8 |
| ViT-B/16 | 10.1 |

## Exit check (S07)

```
64 passed, 8 warnings in 2.68s
```

Loaders open a raw `[0, 1]` batch for CIFAR-100, Oxford-IIIT Pet, and CUB-200. `tests/test_s07_datasets.py` checks that every class is in train, val, and test.

## Next session

Stage **S08**. Shift datasets: Waterbirds pairs, ImageNet-9 backgrounds, the two-patch planted variant, and a check of the existing ColoredMNIST loader. Val only for any smoke run. Do not pass `--final`. ImageNet-1k is still not on disk.
