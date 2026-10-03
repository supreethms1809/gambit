# CDEA on Medical Images — Results and How to Read Them

Everything here was produced from real data on real trained models. Numbers were read
back out of the saved JSON in `scripts/out/`, not transcribed by hand.

**Contents**

1. [The problem in plain English](#1-the-problem-in-plain-english)
2. [What CDEA does about it](#2-what-cdea-does-about-it)
3. [The four measurements, in plain English](#3-the-four-measurements-in-plain-english)
4. [Experimental setup](#4-experimental-setup)
5. [Classification results](#5-classification-results)
6. [Test 1 — did CDEA separate the classes?](#6-test-1--did-cdea-separate-the-classes)
7. [Test 2 — did it cheat?](#7-test-2--did-it-cheat)
8. [Test 3 — did it move evidence somewhere better?](#8-test-3--did-it-move-evidence-somewhere-better)
9. [The shared/unique split](#9-the-sharedunique-split)
10. [Diminishing returns](#10-diminishing-returns)
11. [Foil masks](#11-foil-masks)
12. [Known problems and limitations](#12-known-problems-and-limitations)
13. [Bottom line](#13-bottom-line)
14. [Reproducing everything](#14-reproducing-everything)

---

## 1. The problem in plain English

A model looks at a skin photo and says **"melanoma, 80% confident."** You want to know why.

The standard tool is a heatmap (Grad-CAM): it colors in the parts of the image the model
used. The catch is that you can make a heatmap for *any* class. Make three — "why
melanoma?", "why a harmless mole?", "why a wart?" — and they come out looking nearly
identical, all glowing over the same blob.

That is useless for the question you actually care about. If the melanoma heatmap and the
mole heatmap are the same picture, they cannot tell you what separates the two diagnoses.

This is measurable. **Overlap between competing classes' raw heatmaps: 0.447.**

## 2. What CDEA does about it

CDEA splits the evidence into two buckets:

- **Shared** — what every candidate diagnosis relies on. "There's a dark spot here." True
  no matter which diagnosis is right, so it cannot help you choose.
- **Unique** — what supports only *one* candidate. "*This* edge is ragged." That is what
  tips melanoma over mole.

Three doctors examining the same photo will all circle the same mole. That agreement tells
you nothing about who is right. CDEA's job is to find the smaller regions where they
actually *disagree*.

Importantly, CDEA cannot add highlighting. It has a fixed budget and can only **move it
around** — enforced by `lambda_mass`, and confirmed by the measured `sparse` column below
(0.977 before, 1.001 after).

## 3. The four measurements, in plain English

| metric | the question it asks | direction |
| --- | --- | --- |
| `overlap` | Do different classes' highlights sit on top of each other? | lower is better |
| `suff` (sufficiency) | Show the model *only* the highlighted part — does it still recognize the class? | higher is better |
| `margin` | Does the highlighted part favor *this* class over the runner-up? | positive is good |
| lesion mass fraction | What share of the highlight lands on the actual lesion, as outlined by a dermatologist? | higher is better |
| `sparse` | How much highlight was spent in total (the budget). | should stay constant |

---

## 4. Experimental setup

| | |
| --- | --- |
| Datasets | HAM10000 (7 skin-lesion classes), Cheng et al. brain tumor MRI (3 classes) |
| Splits | HAM10000 lesion-grouped 7980/2035; brain tumor patient-grouped 2467/597 |
| Models | ResNet-18, EfficientNetV2-S (both ImageNet-pretrained, fully fine-tuned) |
| Base evidence | Grad-CAM, Integrated Gradients (16–24 steps) |
| Unit space | 7×7 spatial grid over 224×224 input (49 regions) |
| Training | 20 epochs, Adam lr 1e-4, class-weighted loss, model selected on balanced accuracy |
| Seeds | **1 (single seed — no variance estimates anywhere in this document)** |

### The two experiments do not share an allocator configuration

This matters for reading §6–§11 and is easy to miss. The ablation and the localization
evaluation instantiate `OptimizationAllocator` differently, because they were written
against different entry points:

| | ablation (§6, §7) | localization + figures (§8–§11) |
| --- | --- | --- |
| entry point | `scripts/ablation_contrastive.py` | `scripts/eval_localization.py`, `examples/contrastive_explanation.py` |
| `use_shared` | **False** | **True** (`game_mode=mixed`) |
| `lambda_partition` | 0.0 | 0.1 |
| steps / lr | 40 / 0.3 | 50 / 0.2 |

`OptimizationAllocator.__init__` defaults to `use_shared=False`, and
`ablation_contrastive.py` never passes the flag — so **there is no shared mask at all in the
ablation numbers**. The example and localization scripts pass `use_shared=game_cfg.use_shared`,
which is `True` for `mixed`.

Consequences:

- The overlap/sufficiency/margin figures in §6–§7 describe unique masks **only**, competing
  with no shared mask to absorb common evidence.
- `cdea_shared` in §8–§9 is a genuine optimized mask, which is why it can be scored at all —
  `eval_localization.py` reads `explanation.masks["shared"]`, a key that only exists when
  `use_shared=True`.
- **The two result sets are therefore not a single experiment**, and the overlap reduction in
  §6 should not be read as describing the same masks that §8 localizes. Re-running the
  ablation with `use_shared=True` (or the localization with `competitive`) is needed before
  they can be quoted side by side.

Both datasets use **grouped splits** so near-duplicate images never span train and
validation: by `lesion_id` for HAM10000 (many photos per lesion), by patient for brain
tumor (many slices per patient). A naive per-image split leaks near-duplicates and
inflates accuracy substantially. See [MEDICAL_DATASETS.md](MEDICAL_DATASETS.md).

## 5. Classification results

Balanced accuracy (mean per-class recall) is reported rather than top-1, because HAM10000
is 67% `nv` — a model that always predicts `nv` scores 0.67 top-1 and 0.14 balanced.

| model | dataset | balanced acc | best epoch |
| --- | --- | --- | --- |
| ResNet-18 | HAM10000 | 0.7379 | 5 |
| **EfficientNetV2-S** | HAM10000 | **0.7800** | 13 |
| ResNet-18 | brain tumor | 0.9693 | 11 |

Per-class recall, HAM10000:

| class | ResNet-18 | EfficientNetV2-S |
| --- | --- | --- |
| actinic keratosis | 0.7031 | 0.7812 |
| basal cell carcinoma | 0.7234 | 0.7553 |
| benign keratosis | 0.4762 | **0.7273** |
| dermatofibroma | 0.8333 | 0.7083 |
| melanoma | 0.7424 | **0.6638** |
| melanocytic nevus | 0.7834 | 0.8561 |
| vascular lesion | 0.9032 | 0.9677 |

Per-class recall, brain tumor (ResNet-18): glioma 0.9412, meningioma 0.9722, pituitary 0.9945.

> **Flag:** EfficientNetV2-S has higher balanced accuracy but **lower melanoma recall**
> (0.7424 → 0.6638). Melanoma is the clinically consequential class. On a
> sensitivity-first criterion it is not the better model.

---

## 6. Test 1 — did CDEA separate the classes?

**Yes, strongly.** This is CDEA's core job.

400 images per row, ResNet-18.

| dataset | evidence | overlap: base → **optimized** | reduction |
| --- | --- | --- | --- |
| HAM10000 | Grad-CAM | 0.4468 → **0.0099** | 98% |
| HAM10000 | IG | 0.4949 → **0.0298** | 94% |
| brain tumor | Grad-CAM | 0.0670 → **0.0023** | 97% |
| brain tumor | IG | 0.3260 → **0.0101** | 97% |

A simple `E_k − mean(E_foils)` baseline (`naive_contrastive`) only reaches 0.278 on
HAM10000/Grad-CAM, versus 0.0099 for the optimized allocator — so the gain is not just
"subtract the average."

## 7. Test 2 — did it cheat?

**No.** There is an obvious way to fake Test 1: shrink every highlight to nothing. Zero
overlap, zero information. So sufficiency and budget have to be checked alongside.

| dataset | evidence | suff: base → **opt** | margin: base → **opt** | sparse: base → opt |
| --- | --- | --- | --- | --- |
| HAM10000 | Grad-CAM | 0.6716 → **0.9925** | −1.6477 → −1.2476 | 0.977 → 1.001 |
| HAM10000 | IG | 0.6597 → **0.9926** | −1.6612 → −1.2474 | 1.000 → 1.012 |
| brain tumor | Grad-CAM | −0.0008 → **1.5079** | −0.9258 → **+1.4977** | 1.000 → 1.376 |
| brain tumor | IG | −0.1071 → **1.5988** | −1.2694 → **+1.5994** | 1.000 → 1.292 |

Sufficiency went **up** in every configuration while the budget stayed roughly constant.
CDEA **relocated** evidence rather than deleting it. On brain MRI the margin **flipped
sign** — before allocation the highlighted regions argued for the *wrong* diagnosis.

Tests 1 and 2 together are the solid result of this study.

> **Caveat on brain tumor's `sparse` column:** it rises to ~1.3–1.4, so the budget is not
> as tightly held there as on HAM10000. Some of the brain-tumor sufficiency gain may come
> from spending more highlight, not only from relocating it.

## 8. Test 3 — did it move evidence somewhere better?

**Partly — and this is the weakest part of the study.**

HAM10000 ships expert lesion segmentations for all 10,015 images, so mask placement can be
scored rather than eyeballed. Full validation set, n = 2035.

| configuration | raw heatmap | **after CDEA** | lift | paired test |
| --- | --- | --- | --- | --- |
| IG + ResNet-18 | 0.3398 | **0.5697** | +0.2299 | t=41.7, wins 78.5% |
| Grad-CAM + ResNet-18 | 0.4922 | **0.6257** | +0.1335 | t=27.3, wins 73.5% |
| Grad-CAM + EfficientNetV2-S | 0.6515 | **0.6929** | +0.0415 | t=11.1, wins 65.2% |

CDEA improves on the raw heatmap in every configuration, on ~65–79% of individual images
— so it is not a few outliers dragging an average.

### The problem: a centered rectangle beats it

Dermoscopy photos are framed with the lesion in the middle. Static, model-free masks
scored against the same 2035 lesion outlines:

| "method" (no model, no computation) | lesion mass fraction |
| --- | --- |
| uniform / all-ones ("chance") | 0.2779 |
| centered Gaussian σ = H/4 | 0.4535 |
| centered Gaussian σ = H/6 | 0.6072 |
| **fixed center 3×3 of the 7×7 grid** | **0.7074** |
| **fixed center 1 cell of the 7×7 grid** | **0.9058** |
| — | |
| best measured CDEA (EfficientNetV2-S) | 0.6929 |

**A rectangle drawn at the middle of the image, identical for every photo, beats every
CDEA result.** The mass comparison is direct: CDEA's unique mask carries ~1 grid cell of
total budget (`sparse` ≈ 1.0), and a same-budget mask at the center scores 0.9058.

This does **not** mean CDEA is broken. It means *this measurement* cannot support an
absolute claim, because the answer is baked into how the photos are framed.

**What still holds:** the CDEA-vs-raw-heatmap comparison. Both masks face the same center
bias, on the same images, with the same budget (`lambda_mass` matches them by
construction). That paired comparison is fair. What does not hold is any statement of the
form "CDEA localizes lesions well."

## 9. The shared/unique split

| mask | lands on lesion | vs chance (0.2779) |
| --- | --- | --- |
| CDEA **shared** (Grad-CAM, ResNet-18) | 0.2706 | at chance |
| CDEA **shared** (IG, ResNet-18) | 0.2706 | at chance |
| CDEA **shared** (Grad-CAM, EfficientNetV2-S) | 0.2674 | at chance |
| CDEA **unique** | 0.5697 – 0.6929 | well above |

> **Correction (Aug 11).** An earlier version of this section called the shared row "the
> most robust finding in the study" and read it as *shared evidence sits on the
> background, which is exactly what the decomposition claims*. **That reading was wrong,
> and the number is near-tautological.**
>
> The shared mask carried no penalty term at all: `lambda_sparse` applies only to the
> unique masks, `lambda_overlap` only to unique-unique pairs, and `lambda_mass` pins
> unique mass. Its only brake was the allocator's partition cap, which never binds on
> average (mean region occupancy 0.56 against a cap of 1.0). Measured over 96 HAM10000
> val images at 7×7, it inflates to **22.6 of 49 units of mass, 47.2% of regions above
> 0.5** — a blanket over roughly half the frame. Anything covering half the frame scores
> close to its own area fraction on a lesion-overlap metric *by construction*, whatever
> it is allocating.
>
> The decisive number: the λ=0 shared mask captures base evidence at **0.99× chance** —
> it is statistically uncorrelated with the very field it is supposed to be dividing up.
> It is not a mask that found the background; it is a mask that found nothing.
>
> The unique rows are unaffected. Unique masks sit at **3.35× chance** on base-evidence
> capture and are 2.04% of frame area — they were never the problem, and every claim in
> §8 and §11 that rests on `cdea_unique` stands.
>
> `lambda_shared_sparse=0.25` (`ContrastiveObjective`, `eval_decomposition.py`) fixes it:
> shared mass 22.56 → 2.76, regions above 0.5 47.2% → 3.5%, evidence capture 0.99× →
> 1.48× chance. See §9a for the full five-config re-run. On brain tumor the same penalty
> compacts the mask identically (19.49 → 2.71) but capture stays at chance (0.93× →
> 0.86×), so a compact shared mask is necessary but not sufficient — this remains open.

The Grad-CAM and IG shared means agree to 4 decimal places (0.27063 / 0.27055), but **zero**
of 2035 per-image scores match — independent computations converging on the same value, not
a bug or a shortcut. Under the correction above this agreement is expected rather than
striking: both are converging on the area fraction of a blanket.

## 9a. Constraining the shared mask — five-config re-run

Every config in §9 re-run with `--lambda_shared_sparse 0.25`, everything else identical
(`mixed`, 50 allocator steps, lr 0.2, seed 0, full val split). Outputs in
`results/medical_presentation/decomposition_sharedfix/`, λ=0 baseline in
`.../decomposition/`.

**What the penalty does to the mask** (96 val images, 7×7, Grad-CAM, ResNet-18):

| | HAM10000 λ=0 → 0.25 | brain_tumor λ=0 → 0.25 |
| --- | --- | --- |
| shared mass (of 49) | 22.56 → **2.76** (8.2× smaller) | 19.49 → **2.71** (7.2×) |
| shared regions above 0.5 | 47.2% → **3.5%** | 39.9% → **3.2%** |
| shared: base evidence captured / area | 0.99× → **1.48×** chance | 0.93× → 0.86× chance |
| unique: base evidence captured / area | 3.35× → 3.47× chance | 4.71× → 4.23× chance |
| unique mass | 1.01 → 1.03 | 1.20 → 1.47 |
| unique–unique overlap | 0.058 → 0.052 | 0.020 → 0.020 |
| mean region occupancy (cap 1.0) | 0.56 → 0.16 | 0.47 → 0.15 |

**What it does to the metrics:**

| config | n | K | full | shared-only λ=0 → 0.25 | +unique λ=0 → 0.25 | restore top-1 | diag−off | \|diag\|/off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAM10000 / EffNetV2-S / Grad-CAM | 2035 | 5 | 0.921 | 0.586 → **0.680** | 0.935 → **0.956** | 75.7% → 73.2% | −0.337 → −0.267 | 4.39 → **4.71** |
| HAM10000 / ResNet-18 / Grad-CAM | 2035 | 5 | 0.805 | 0.412 → **0.498** | 0.619 → **0.670** | 71.5% → 69.2% | −0.151 → −0.114 | 4.63 → **5.91** |
| HAM10000 / ResNet-18 / IG | 2035 | 5 | 0.805 | 0.420 → **0.508** | 0.615 → **0.669** | 67.7% → 66.9% | −0.160 → −0.111 | 4.56 → **6.04** |
| brain_tumor / ResNet-18 / Grad-CAM | 597 | 3 | 0.976 | 0.670 → **0.635** | 0.838 → **0.834** | 94.0% → 93.6% | −0.223 → −0.178 | 1.76 → **1.76** |
| brain_tumor / ResNet-18 / IG | 597 | 3 | 0.976 | 0.715 → **0.711** | 0.823 → **0.822** | 92.6% → 90.5% | −0.312 → −0.234 | 1.86 → **1.83** |

| config | t collapse | t recovery | recovery win rate | random-deletion control |
| --- | --- | --- | --- | --- |
| HAM / EffNet / Grad-CAM | 60.6 → 44.5 | 70.4 → 55.4 | 99.1% → 98.3% | +0.003 → +0.004 |
| HAM / ResNet-18 / Grad-CAM | 94.4 → 70.3 | 66.2 → 56.6 | 99.3% → 98.1% | −0.000 → +0.000 |
| HAM / ResNet-18 / IG | 91.7 → 68.1 | 65.2 → 52.2 | 99.2% → 97.7% | +0.000 → +0.001 |
| brain / ResNet-18 / Grad-CAM | 24.5 → 33.8 | 20.8 → 23.8 | 95.5% → 90.3% | +0.007 → +0.015 |
| brain / ResNet-18 / IG | 22.3 → 28.1 | 15.6 → 14.3 | 91.8% → 79.6% | +0.006 → +0.017 |

Reading this honestly, four things:

1. **Test A survives everywhere.** Collapse and recovery are both significant in all ten
   runs (t = 14–94, recovery win rate 79.6–98.3%). The decomposition's core claim does not
   depend on the shared mask being a blanket.

2. **On HAM10000 the λ=0 collapse was partly an artifact.** Shared-only spread rises
   0.09 in all three HAM configs once the blanket is removed. Keeping a soft mask spread
   over 46% of the frame is a *global degradation* of the image, and part of what looked
   like "the hypotheses become equally likely when you keep only shared evidence" was
   just a washed-out input. The real collapse is the smaller one: 0.805 → 0.508, not
   0.805 → 0.420. Recovery improves at the same time (0.615 → 0.669), so the corrected
   numbers are *better* evidence, from a narrower gap.

3. **Test B's absolute magnitude falls ~25%, but its specificity does not.** `diag−off`
   shrinks in all five configs — however the off-diagonal shrinks proportionally, so the
   |diagonal| / off-diagonal ratio is flat on brain (1.76 → 1.76, 1.86 → 1.83) and
   *improves* on all three HAM configs (up to 4.56 → 6.04). The random-deletion control
   stays at ≈0 (|·| ≤ 0.017) throughout, so neither setting is measuring generic image
   corruption. Class-specificity is intact; only the scale changed.

4. **Brain tumor behaves differently and it is not yet explained.** The penalty compacts
   its shared mask just as much (7.2×), but shared-only spread barely moves (−0.036,
   −0.005) and evidence capture stays at chance (0.93× → 0.86×). So compactness alone
   does not make a shared mask meaningful. Candidate explanations — K=3 vs 5, a
   near-saturated model (full spread 0.976), or genuinely shared anatomy that the
   class-conditioned Grad-CAM field does not mark — are untested.

**Recommendation:** make `lambda_shared_sparse=0.25` the default for any run that reports
a statistic *about the shared mask*. It stays 0.0 in `ContrastiveObjective.__init__` so
existing results reproduce exactly.

## 10. Diminishing returns

Three independent conditions line up monotonically:

| starting heatmap quality | CDEA lift | endpoint |
| --- | --- | --- |
| bad — IG + ResNet-18 (0.3398) | **+0.2299** | 0.5697 |
| medium — Grad-CAM + ResNet-18 (0.4922) | **+0.1335** | 0.6257 |
| good — Grad-CAM + EfficientNetV2-S (0.6515) | **+0.0415** | 0.6929 |

**The better the base evidence, the less allocation adds.** Like a proofreader: enormous
value on a rough draft, barely noticeable on a polished one.

Reads both ways:

- *For CDEA* — it is contributing real work, not passing good input through. A method that
  merely sharpened its input would show a lift that grows with input quality, not shrinks.
- *Against CDEA* — on a strong modern backbone the contribution is +0.0415, and much of the
  gain looks like *correcting weak attribution* rather than adding explanatory content a
  good attribution method would not already have.

Related: EfficientNetV2-S's **raw** heatmap (0.6515) beats ResNet-18's **fully
CDEA-processed** output (0.6257). Upgrading the backbone bought more localization than the
entire allocation machinery did on the weaker one.

## 11. Foil masks

The `rather than L` half of the contrastive claim. Rank 0 is the model's predicted class;
ranks 1–4 are the foils. n = 2035, Grad-CAM.

**ResNet-18**

| rank | raw heatmap | CDEA unique | CDEA lift |
| --- | --- | --- | --- |
| 0 (predicted) | 0.4922 | **0.6257** | +0.1335 |
| 1 | 0.4635 | 0.4771 | +0.0136 |
| 2 | 0.4144 | 0.4007 | **−0.0137** |
| 3 | 0.2803 | 0.2700 | **−0.0103** |
| 4 | 0.1768 | 0.1775 | +0.0007 |

**EfficientNetV2-S**

| rank | raw heatmap | CDEA unique | CDEA lift |
| --- | --- | --- | --- |
| 0 (predicted) | 0.6515 | **0.6929** | +0.0415 |
| 1 | 0.4739 | 0.4962 | +0.0223 |
| 2 | 0.2568 | 0.2979 | +0.0411 |
| 3 | 0.1545 | 0.1947 | +0.0402 |
| 4 | 0.1256 | 0.1658 | +0.0402 |

Two things to take from this:

1. **Foil degradation is model-specific, not intrinsic.** With ResNet-18, CDEA *hurts*
   ranks 2–3. With EfficientNetV2-S it helps every rank by a uniform ~+0.04. An earlier
   reading of this as a property of the method was an over-generalization from one model.
2. **The contrastive split is real.** Rank 0 is significantly more lesion-focused than
   rank 1: **+0.1486** (t=18.8, wins 67.4%) for ResNet-18, **+0.1967** (t=24.0, wins 71.7%)
   for EfficientNetV2-S. The better classifier gives the sharper separation.

Note that by ranks 3–4 both raw and CDEA masks fall **below chance** — those masks sit on
background. They are not meaningful explanations of "why not class L."

---

## 12. Known problems and limitations

**Measurement**

- **Center bias invalidates absolute localization claims** (§8). A fixed centered box
  scores 0.7074; the best CDEA result is 0.6929. Needs a null that controls for position —
  the cleanest is scoring each mask against *itself, randomly translated*, which holds
  shape, budget, and center-prior constant and scrambles only position.
- The `headroom` column emitted by `eval_localization.py` is normalized against the uniform
  baseline and inherits the same flaw. Do not quote it without the center-prior table.
- `ablation_contrastive.py` prints "at comparable sufficiency (diff=1.71)". 1.71 is not
  comparable. The conclusion survives because sufficiency *improved*, but the sentence is
  wrong and should be fixed before publication.

**Configuration**

- **The ablation and the localization runs used different allocator settings** (§4) — most
  importantly `use_shared=False` vs `True`. Do not present §6 and §8 as one experiment until
  one of them is re-run to match the other.
- The word "shared" is overloaded in this codebase. `masks["shared"]` is a real optimized
  mask that exists only when `use_shared=True`. Separately, `visualize_contrastive` *derives*
  a red "shared" region for display as `min(m_k, max(m_others))` — the intersection of the
  unique masks — whenever no explicit shared mask is present. The two look identical in a
  figure and are not the same object; panel titles read `(explicit shared)` only in the first
  case.

**Coverage**

- **Single seed throughout.** No variance estimates on any number in this document.
- Localization is HAM10000-only. Brain tumor cannot contribute (below).
- Only Grad-CAM was run for the EfficientNetV2-S localization; IG was not.
- The 200-image subset used for an earlier Grad-CAM/IG comparison was **not** representative
  for IG (base 0.436 on the subset vs 0.340 at full scale). Subsets of a few hundred are
  unreliable for this metric because lesion area varies so much.

**Brain tumor resolution**

Measured over all 3064 tumor masks: mean tumor area **1.7%** of frame, median 1.29%, while
one cell of the 7×7 grid is **2.04%**. **69.8% of tumors are smaller than a single grid
cell.** Brain-tumor classification results are sound; its *localization* would measure grid
quantization, so it was not run. A 14×14 grid (e.g. `vit_b_16`) is needed for a fair test.

**Scope**

- Nothing here is clinically validated. These are research artifacts.
- Lesion overlap is a proxy for explanation quality, not a measure of it. A mask can sit on
  the lesion and still be a poor explanation to a clinician. No human evaluation was done.
- Explanations are of a model with 0.78 balanced accuracy — often wrong.
- HAM10000 is CC BY-NC 4.0; derived figures inherit the non-commercial restriction.

## 13. Bottom line

**Demonstrated.** CDEA takes heatmaps that are ~45% redundant across competing diagnoses
and separates them to ~1% redundant, while *increasing* how much each one explains, without
spending more highlight. That converts "where did the model look?" into "what made it pick
A over B?" — a question raw heatmaps structurally cannot answer.

**Supported but weaker.** The separated evidence lands on real pathology more often than raw
attribution does, consistently across two attribution methods and two backbones, on 65–79%
of individual images.

**Not supported.** That the highlights are *good* in absolute terms. A centered rectangle
beats them on the only quantitative localization test available.

**Honest summary.** CDEA is a disentangling tool and an effective one. It is not a
localization tool, and the current evidence does not let you claim it is.

### Highest-value next steps

1. **Randomly-translated-mask null** (§12) — makes the localization claim defensible.
2. **Three seeds** — `run_experiments.py --datasets ham10000 --seeds 0 1 2`.
3. **Fix the ablation pass-condition wording.**
4. **`vit_b_16` at 14×14** so brain tumor can contribute a real localization number.
5. **Report the center-prior table alongside any localization result.**

## 14. Reproducing everything

Data setup is in [MEDICAL_DATASETS.md](MEDICAL_DATASETS.md). From the repo root with the
`marl` environment active:

```bash
# Train (HAM10000, modern backbone)
PYTHONPATH=. python examples/contrastive_explanation.py --dataset ham10000 \
  --model efficientnet_v2_s --train --epochs 20 --lr 1e-4 --batch_size 32 \
  --pretrained --num_alloc_steps 50 --num_viz_samples 5
```

```bash
# Ablation (overlap / sufficiency / margin)
PYTHONPATH=. python scripts/ablation_contrastive.py --dataset ham10000 --evidence gradcam \
  --model resnet18 --checkpoint examples/out/checkpoints/ham10000_resnet18.pt --num_images 400
```

```bash
# Localization incl. foil ranks
PYTHONPATH=. python scripts/eval_localization.py \
  --checkpoint examples/out/checkpoints/ham10000_efficientnet_v2_s.pt \
  --num_images 2035 --num_alloc_steps 50 --evidence gradcam \
  --export_prefix localization_foil_efficientnet_v2_s
```

```bash
# Grad-CAM vs IG comparison figure
PYTHONPATH=. python scripts/plot_evidence_comparison.py
```

### Result files

| file | contents |
| --- | --- |
| `scripts/out/ablation_contrastive_<dataset>_<evidence>_metrics.json` | overlap / suff / margin / sparse |
| `scripts/out/localization_ham10000_<evidence>.json` | full-set localization, Grad-CAM vs IG |
| `scripts/out/localization_foil_<model>.json` | per-rank localization incl. foils |
| `scripts/out/evidence_comparison.png` | Grad-CAM vs IG summary figure |
| `examples/out/contrastive_explanation_<dataset>_s1..s5.png` | per-sample explanation figures |
| `examples/out/checkpoints/<dataset>_<model>.pt` | trained checkpoints |
