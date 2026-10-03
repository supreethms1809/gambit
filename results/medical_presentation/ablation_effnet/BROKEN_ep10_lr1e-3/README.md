# Do not quote these numbers

Brain tumor / EfficientNetV2-S trained at `--epochs 10` with the default `lr 1e-3`,
against the 20 epochs at lr 1e-4 that `docs/MEDICAL_RESULTS.md` §14,
`docs/MEDICAL_DATASETS.md` and `run_stages_2to5.sh` all specify for these datasets.

The resulting model is insensitive to its input. Keeping one tile out of 49 gives
essentially the same logits as keeping the whole image:

| checkpoint | \|z(all tiles) − z(1 tile)\| on top-1 | logit spread | p(top1) |
| --- | --- | --- | --- |
| ham10000_efficientnet_v2_s | 3.441 | 10.29 | 0.937 |
| ham10000_resnet18 | 3.307 | 9.60 | 0.761 |
| brain_tumor_resnet18 | 3.078 | 10.05 | 0.967 |
| **brain_tumor_efficientnet_v2_s (this one)** | **0.188** | **2.48** | 0.713 |

Sufficiency is therefore constant under every mask, the allocator gets no gradient from
the sufficiency and margin terms, and only the overlap and sparsity penalties do any
work. That is exactly what the ablation shows — overlap falls 96%/76%/59%/39% while
sufficiency moves 0.749 → 0.755 and margin −0.82 → −0.81 in all four cells.

Balanced accuracy was 0.787, above `train_backbone`'s chance guard, so nothing warned.
An accuracy check is not sufficient to qualify a checkpoint for masked-intervention work.

The checkpoint is kept as
`examples/out/checkpoints/BROKEN_brain_tumor_efficientnet_v2_s_ep10_lr1e-3.pt`.
Superseded by the ep20 / lr1e-4 run.
