# Correction: non-medical results re-run without the normalization mismatch

`scripts/train_backbone.py` applied ImageNet normalization while every evaluation
and explanation path consumes raw [0,1] tensors. Measured on the same eval:
the CIFAR-10 checkpoint scored **0.388** under the old (mismatched) pipeline versus
**0.809** retrained without normalization — and 0.816 for the old checkpoint
evaluated *with* normalization, i.e. removing it costs no accuracy, it only makes
training and evaluation agree.

Every number below feeds `docs/GAMBIT_PAPER.md` and `docs/CDEA_CONTRASTIVE_PAPER.md`.
3 seeds, mean ± std.

## Sufficiency and margin

| config | method | suff (old) | suff (corrected) | margin (old) | margin (corrected) |
| --- | --- | --- | --- | --- | --- |
| cifar10_gradcam | base_evidence | -0.868 | **0.254 ± 0.080** | -1.488 | **-2.018 ± 0.049** |
| cifar10_gradcam | naive_contrastive | -0.868 | **0.259 ± 0.080** | -1.488 | **-2.013 ± 0.048** |
| cifar10_gradcam | optimized | -0.685 | **0.786 ± 0.071** | -1.205 | **-1.253 ± 0.046** |
| cifar10_ig | base_evidence | -0.861 | **0.247 ± 0.080** | -1.399 | **-2.028 ± 0.051** |
| cifar10_ig | naive_contrastive | -0.864 | **0.249 ± 0.078** | -1.398 | **-2.024 ± 0.050** |
| cifar10_ig | optimized | -0.655 | **0.820 ± 0.069** | -1.099 | **-1.211 ± 0.051** |
| mnist_gradcam | base_evidence | -1.132 | **2.138 ± 0.014** | -2.119 | **-2.821 ± 0.020** |
| mnist_gradcam | naive_contrastive | -1.134 | **2.141 ± 0.014** | -2.119 | **-2.816 ± 0.020** |
| mnist_gradcam | optimized | -1.100 | **2.794 ± 0.019** | -2.063 | **-1.939 ± 0.028** |
| mnist_ig | base_evidence | -1.109 | **2.127 ± 0.014** | -2.140 | **-2.824 ± 0.022** |
| mnist_ig | naive_contrastive | -1.100 | **2.154 ± 0.011** | -2.127 | **-2.838 ± 0.019** |
| mnist_ig | optimized | -1.082 | **2.745 ± 0.020** | -2.077 | **-2.026 ± 0.027** |
| pets_gradcam | base_evidence | 0.471 | **-0.029 ± 0.242** | 0.014 | **0.053 ± 0.004** |
| pets_gradcam | naive_contrastive | 0.471 | **-0.028 ± 0.241** | 0.014 | **0.054 ± 0.003** |
| pets_gradcam | optimized | 0.683 | **1.116 ± 0.182** | 0.444 | **2.290 ± 0.072** |
| pets_ig | base_evidence | 0.471 | **-0.062 ± 0.248** | 0.000 | **-0.000 ± 0.001** |
| pets_ig | naive_contrastive | 0.464 | **-0.052 ± 0.248** | 0.005 | **0.005 ± 0.041** |
| pets_ig | optimized | 0.727 | **1.060 ± 0.178** | 0.537 | **2.163 ± 0.073** |
| stanford_dogs_gradcam | base_evidence | -4.731 | **-3.589 ± 0.097** | -1.814 | **-2.598 ± 0.133** |
| stanford_dogs_gradcam | naive_contrastive | -4.694 | **-3.509 ± 0.097** | -1.787 | **-2.533 ± 0.139** |
| stanford_dogs_gradcam | optimized | -4.344 | **-2.552 ± 0.088** | -1.318 | **-1.424 ± 0.122** |
| stanford_dogs_ig | base_evidence | -4.738 | **-3.560 ± 0.096** | -1.818 | **-2.604 ± 0.131** |
| stanford_dogs_ig | naive_contrastive | -4.713 | **-3.516 ± 0.095** | -1.803 | **-2.569 ± 0.137** |
| stanford_dogs_ig | optimized | -4.349 | **-2.580 ± 0.090** | -1.348 | **-1.453 ± 0.127** |

## What CDEA actually contributes (optimized − base_evidence)

The correction *widens* this gap everywhere: a properly trained model has sharper,
better-separated logits, so a diffuse raw-evidence mask leaves a larger deficit,
while the optimized mask holds. The bug was understating the method's benefit.

| config | Δsuff (old) | Δsuff (corrected) | Δmargin (old) | Δmargin (corrected) |
| --- | --- | --- | --- | --- |
| cifar10_gradcam | +0.183 | **+0.532** | +0.283 | **+0.765** |
| cifar10_ig | +0.206 | **+0.573** | +0.300 | **+0.817** |
| mnist_gradcam | +0.032 | **+0.656** | +0.057 | **+0.882** |
| mnist_ig | +0.027 | **+0.618** | +0.064 | **+0.798** |
| pets_gradcam | +0.212 | **+1.146** | +0.431 | **+2.237** |
| pets_ig | +0.255 | **+1.122** | +0.537 | **+2.163** |
| stanford_dogs_gradcam | +0.386 | **+1.036** | +0.497 | **+1.174** |
| stanford_dogs_ig | +0.389 | **+0.981** | +0.470 | **+1.151** |

## Consequence for the papers

- `GAMBIT_PAPER.md` §8.1 lists *negative sufficiency and margins with weak backbones*
  as a limitation. On CIFAR-10 and MNIST that reverses sign entirely once training
  and evaluation agree, so it was an artifact there. It **survives on Stanford Dogs**
  (-4.34 -> -2.55, still negative), so narrow the claim to fine-grained, many-class
  settings rather than deleting it.
- Every sufficiency and margin figure in both drafts needs replacing. Overlap is
  less affected, being a property of mask geometry rather than of logits.

---

## Shift experiments (robust vs shortcut)

The same normalization mismatch. `id_ood_gap` measures how much the model's
performance degrades from in-distribution to out-of-distribution — i.e. how much
it leans on the planted shortcut. 3 seeds, mean ± std.

| config | id_ood_gap (old) | id_ood_gap (corrected) |
| --- | --- | --- |
| colored_cifar10_competitive | 0.4843 | **3.3828 ± 0.158** |
| colored_cifar10_cooperative | 0.4052 | **2.8497 ± 0.065** |
| colored_cifar10_mixed | 0.4761 | **3.4432 ± 0.118** |
| colored_mnist_competitive | 0.0557 | **5.3310 ± 0.090** |
| colored_mnist_cooperative | 0.0509 | **5.3412 ± 0.084** |
| colored_mnist_mixed | 0.0376 | **5.3521 ± 0.148** |
| texture_mnist_competitive | 0.5786 | **5.4010 ± 0.507** |
| texture_mnist_cooperative | -0.2161 | **3.4170 ± 0.451** |
| texture_mnist_mixed | 0.5274 | **5.3915 ± 0.543** |

**This is the larger of the two corrections.** ColoredMNIST is built with
`correlation=0.9` precisely to force a colour shortcut, yet the old numbers (~0.04) said
the model barely used it — roughly 100x smaller than the corrected ~5.34. One config,
`texture_mnist_cooperative`, had a *negative* gap. The shift game was therefore being
evaluated on models that did not measurably exhibit the dependency it exists to
decompose, so every robust-vs-shortcut conclusion in the drafts needs re-deriving.

Two checks that the corrected numbers are the trustworthy ones: seed variance is tight
(±0.06-0.15 on MNIST/CIFAR), and `id_ood_gap` is now near-identical across game modes
(5.331 / 5.341 / 5.352 on colored_mnist), which is correct — it is a property of the
model, not of the allocation. The old values varied *more* across game modes than they
did in magnitude, which was itself a signal something was wrong.

## Status

All corrected results are under `results/paper_rerun/`; nothing in `scripts/out/` was
overwritten, so the original (buggy) numbers remain available for comparison. The
regenerated report is at `results/paper_rerun/shift/journal/JOURNAL_REPORT.md` and
`results/paper_rerun/contrastive/journal/JOURNAL_REPORT.md`.

