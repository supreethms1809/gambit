# Lightning talk — medical swap sheet

Source deck: `HPCAi_LightningTalkSupreeth_Suresh.pptx` (the ICAIIS slide set).
Rebuilt deck: `results/medical_presentation/deck/HPCAi_LightningTalk_medical.pptx`,
copied to the OneDrive folder as `HPCAi_LightningTalk_MEDICAL.pptx`.

Regenerate the pictures, then rebuild the deck:

```bash
PYTHONPATH=. python scripts/plot_talk_figures_medical.py \
  --checkpoint examples/out/checkpoints/ham10000_resnet18.pt \
  --prefer "melanoma,melanocytic nevus" --scan_images 512 --set_overlap "0.447,0.069"
python scripts/build_medical_deck.py <source.pptx> <out.pptx>
```

## What changed, slide by slide

| slide | was | now |
| --- | --- | --- |
| 1 Title | — | untouched |
| 2 Problem | corgi, Grad-CAM Pembroke vs Cardigan | `dropin/slide2_problem_ham10000.png` — melanoma 49% vs nevus 45%, cosine 0.99 |
| 3 Proposed solution | — | untouched (schematic is dataset-independent) |
| 4 Method | cat, keep-and-blur | `dropin/slide4_method_ham10000.png` — same lesion, melanoma mask |
| 5 Objective | — | untouched |
| 6 Results (qualitative) | corgi 5-panel | `dropin/slide6_hero_ham10000.png` |
| 7 Results (table) | MNIST/CIFAR/Pets/Dogs, 4 columns | 2 datasets × 2 backbones, 5 columns |
| 8 Limitations & summary | Stanford Dogs framing | medical framing + the localization caveat |

## The numbers on slide 7

Two datasets × two fine-tuned backbones, 7×7 tiles, top-5 hypotheses, n = 400 validation
images, in the configuration the submitted abstract quotes (no shared mask,
`lambda_disjoint` 0.5, 40 steps, lr 0.3). Grad-CAM:

| | HAM10000 · ResNet-18 | HAM10000 · EffNetV2-S | Brain MRI · ResNet-18 | Brain MRI · EffNetV2-S |
| --- | --- | --- | --- | --- |
| overlap | 0.447 → **0.010** (−98%) | 0.442 → **0.016** (−96%) | 0.067 → **0.002** (−97%) | 0.064 → **0.008** (−88%) |
| sufficiency | 0.672 → 0.992 | 0.268 → 1.491 | −0.001 → 1.508 | 0.212 → 1.544 |
| margin | −1.65 → −1.25 | −3.18 → −1.34 | −0.93 → **+1.50** | −0.61 → **+1.46** |
| mask budget | ×1.02 | ×1.12 | ×1.38 | ×1.21 |

The mask-budget row is **not on the slide** — it is the footnote. Four metric rows at the
deck's 0.94" row height is what fits; a fifth forced the rows down to 0.75", at which the
20pt headers wrapped to three lines and the table overflowed into the footnote. The table
is 15pt headers / 16pt body, worst cell at 91% of its column.

Integrated Gradients, same configuration:

| | HAM10000 · ResNet-18 | HAM10000 · EffNetV2-S | Brain MRI · ResNet-18 | Brain MRI · EffNetV2-S |
| --- | --- | --- | --- | --- |
| overlap | 0.495 → 0.030 | 0.543 → 0.029 | 0.326 → 0.010 | 0.328 → 0.014 |
| sufficiency | 0.660 → 0.993 | 0.199 → 1.432 | −0.107 → 1.599 | −0.012 → 1.551 |
| margin | −1.66 → −1.25 | −3.23 → −1.44 | −1.27 → +1.60 | −1.05 → +1.49 |
| mask budget | ×1.01 | ×1.07 | ×1.29 | ×1.16 |

Sources: `scripts/out/ablation_contrastive_*_metrics.json` (ResNet-18, Aug 7) and
`results/medical_presentation/ablation_effnet/abl_abstractcfg_*` (EfficientNetV2-S).

**The abstract's headline (0.447 → 0.0099, 0.672 → 0.993) is the HAM10000 / ResNet-18
column.** Reword "at a fixed mask budget": it holds there (×1.02) and nowhere else.
EfficientNetV2-S buys ×1.07–1.12 of extra highlight and brain tumor ×1.16–1.38, so on
three of the four combinations part of the sufficiency gain is bought rather than
relocated.

## The numbers behind the figures (shared mask on)

`game_mode=mixed`, `lambda_shared_sparse=0.25`, 50 steps, lr 0.2 — the configuration that
produces a shared mask, and therefore the one the pictures are drawn from.
Sources: `results/medical_presentation/ablation_sharedfix/` and `ablation_effnet/abl_sharedfix_*`.

| | overlap | sufficiency | margin | budget |
| --- | --- | --- | --- | --- |
| HAM10000 · ResNet-18 · Grad-CAM | 0.447 → 0.069 | 0.672 → 1.617 | −1.65 → −0.51 | ×1.03 |
| HAM10000 · EffNetV2-S · Grad-CAM | 0.442 → 0.091 | 0.268 → **3.252** | −3.18 → **+1.74** | ×1.34 |
| Brain MRI · ResNet-18 · Grad-CAM | 0.067 → 0.011 | −0.001 → 1.803 | −0.93 → +2.00 | ×1.50 |
| Brain MRI · EffNetV2-S · Grad-CAM | 0.064 → 0.026 | 0.212 → 2.114 | −0.61 → **+2.42** | ×1.30 |

Three training seeds (`seeds_sharedfix/`, ResNet-18): HAM10000 Grad-CAM overlap
**0.0497 ± 0.0066**, brain tumor Grad-CAM **0.0083 ± 0.0027**.

## Where the EfficientNet checkpoints came from

`ham10000_efficientnet_v2_s.pt` already existed. `brain_tumor_efficientnet_v2_s` did not,
and had to be trained:

```bash
PYTHONPATH=. python scripts/train_backbone.py --dataset brain_tumor \
  --model efficientnet_v2_s --epochs 20 --lr 1e-4 --batch_size 24 \
  --no-freeze-backbone --ckpt_dir examples/out/checkpoints
```

That is the recipe `docs/MEDICAL_RESULTS.md` §14, `docs/MEDICAL_DATASETS.md` and
`run_stages_2to5.sh` all specify. It reaches **0.9619** balanced accuracy, in line with
ResNet-18's 0.952–0.956. `scripts/train_backbone.py` did not offer `efficientnet_v2_s`
until this run; the builder was added to match `examples/contrastive_explanation.py`.

**A first attempt at `--epochs 10` with the default `lr 1e-3` produced a checkpoint that
passes an accuracy check and is still useless.** Balanced accuracy 0.787, above the chance
guard, but showing it one tile out of 49 gives the same logits as the whole image:

| checkpoint | \|z(all tiles) − z(1 tile)\| on top-1 | logit spread | p(top1) |
| --- | --- | --- | --- |
| ham10000_efficientnet_v2_s | 3.441 | 10.29 | 0.937 |
| ham10000_resnet18 | 3.307 | 9.60 | 0.761 |
| brain_tumor_resnet18 | 3.078 | 10.05 | 0.967 |
| brain_tumor_efficientnet_v2_s (ep20, lr 1e-4) | 2.631 | 9.05 | 0.983 |
| **the ep10 / lr 1e-3 attempt** | **0.188** | **2.48** | 0.713 |

Sufficiency is then constant under every mask, so the allocator gets no gradient from the
sufficiency and margin terms: overlap fell 96% while sufficiency moved 0.749 → 0.755. The
checkpoint and its four ablation cells are kept under
`ablation_effnet/BROKEN_ep10_lr1e-3/`. **Run this probe on any new checkpoint before
trusting an intervention result — an accuracy check does not catch it.**

## Backup slide material

- `dropin/slide7_separation_bars.png` (`figures/F2_separation.png`) — base / naive / CDEA
  overlap for all four cells with seed error bars. Use it instead of the table if you
  would rather show a chart.
- `dropin/backup_decomposition.png` (`figures/F4_decomposition.png`) — the interventional
  validation: probability spread collapses without unique evidence and returns with it,
  and the K×K deletion matrix. Removing class *j*'s unique mask costs class *j* (−0.67)
  and **helps its rivals** (+0.23, +0.17, +0.11); an equal-budget random deletion gives
  ≈ 0 (|·| ≤ 0.017). n = 2035, EfficientNetV2-S.
- `dropin/slide6_hero_brain_tumor_ALT.png` / `slide2_problem_brain_tumor_ALT.png` —
  the same two figures on brain MRI (glioma 52% vs meningioma 48%, cosine 0.88).
- `dropin/backup_hero_effnet.png` (`figures/F1_hero.png`) — the four-panel EfficientNetV2-S
  version with the shared/unique colour split.

## The one caveat worth pre-empting

Do not claim the masks localize the pathology. On HAM10000 a fixed centred rectangle with
no model in it scores 0.906 against CDEA's 0.693, because dermoscopy centres the lesion by
acquisition convention. What *is* defensible: the mask carries position — scrambling its
location while holding shape, budget and compactness drops it from 0.693 to 0.265, i.e. to
chance, on 89% of images. Full write-up in `results/medical_presentation/RESULTS.md`.
