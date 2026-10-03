# Medical presentation — run index

Everything here was produced for the 8-minute talk. Nothing under `scripts/out/`
or `examples/out/` was overwritten; the Aug 7 results stay intact as a cross-check.

Environment: `conda activate marl`, run from repo root with `PYTHONPATH=.`.
Device: MPS (Apple Silicon).

## Why this run set exists

The Aug 7 results validated the shared/unique decomposition by asking whether the
allocated masks land on the dermatologist's lesion outline. That test cannot
answer the question. The outline marks *where the lesion is*, and all seven
HAM10000 classes are lesions — so it is a ground truth for **shared** evidence,
not for class-**unique** evidence. On top of that, the metric is degenerate on
this dataset: a fixed centered rectangle beats every measured method.

These runs replace that validation with an interventional one (Stage 1), measure
exactly how degenerate the spatial metric is (center prior), and unify the
allocator configuration so the separation and validation results describe one
experiment rather than two (Stage 3).

---

## Code changes behind these runs

| ID | File | Change |
| --- | --- | --- |
| B1 | `scripts/eval_decomposition.py` *(new)* | Interventional test: probability-spread collapse/recovery, plus a K×K deletion matrix with an equal-budget random-mask control |
| B2 | `scripts/eval_localization.py` | `cdea_unique_translated` null (mask rolled to a random position — holds shape, budget and compactness, scrambles only location) and a fixed `center_cell` reference |
| B3 | `scripts/eval_localization.py` | Generalized past the `ham10000`-only guard; brain tumor masks now score, grid follows the backbone |
| B4/B5 | `scripts/ablation_contrastive.py`, `scripts/run_experiments.py` | `--game_mode` / `--num_steps` / `--lr` plumbed through; the allocator config is now recorded in the emitted JSON |
| B6 | `scripts/train_backbone.py` | Class-weighted loss + balanced-accuracy model selection, auto-on for medical datasets; checkpoints now metadata-wrapped so every eval script can load them |
| B7 | `scripts/eval_center_prior.py` *(new)* | The null ladder, swept over grid resolution |
| B8 | all entry points | `--out_dir` / `--ckpt_dir`, defaulting to the previous hardcoded paths |

### Two bugs fixed along the way

- **Subset sampling was a class-ordered prefix.** `eval_localization.py` built
  subsets with `range(N)` over an `ImageFolder`, which is sorted by class — so
  `--num_images 200` scored a few classes, not a sample of the val set. Measured:
  chance was 0.481 on a 32-image prefix versus 0.279 on a random sample of the
  same size (full set: 0.2779). This is the cause of the "200-image subset was
  not representative" anomaly recorded in `docs/MEDICAL_RESULTS.md` §12. Now a
  seeded `randperm`.
- **`--num_steps` was silently ignored** by `run_experiments.py`: the argument
  existed and was accepted but never forwarded to `run_ablation`.
- **The "at comparable sufficiency" line was wrong** and is now removed.
  `ablation_contrastive.py` asserted comparability while printing a sufficiency
  delta of 1.71. It now reports the sufficiency change and the mask-budget ratio
  and only says PASS when sufficiency did not fall and the budget did not grow.

---

## Results

### Center prior — is the spatial metric usable at all?

`scripts/eval_center_prior.py --num_masks 1000` → `figures/center_prior.{csv,json}`

Score of a *fixed centered mask*, no model and no computation, as the grid refines:

| dataset | chance | 7×7 | 14×14 | 28×28 | 56×56 | centroid spread (y, x) |
| --- | --- | --- | --- | --- | --- | --- |
| HAM10000 | 0.2652 | **0.9398** | 0.9343 | 0.9550 | **0.9611** | ±0.065, ±0.049 |
| brain tumor | 0.0166 | **0.1312** | 0.1103 | 0.1303 | **0.1376** | ±0.100, ±0.114 |

Two things follow, and both change what the talk can claim:

1. **A finer grid does not fix the metric — it strengthens the null.** The
   intuition that smaller masks make the degenerate baseline weaker is backwards:
   for a centered target, a smaller centered mask is *more* reliably inside it.
   HAM10000's centre-cell null rises from 0.940 to 0.961 across the sweep, and the
   centre-3×3 null rises from 0.729 to 0.958.
2. **HAM10000 is the wrong testbed; brain tumor is the right one.** Dermoscopy
   centres the lesion by acquisition convention, so the centre prior is
   overwhelming (null 0.94 against a best measured CDEA of 0.693). Tumor position
   genuinely varies across patients — twice the centroid spread — so the null sits
   at 0.13 against chance 0.017, leaving real headroom for a spatial claim.

### Stage 4 — the translation null at full scale

`eval_localization.py`, HAM10000, Grad-CAM, **n = 2035**:

| method | EfficientNetV2-S | ResNet-18 |
| --- | --- | --- |
| chance (flat mask) | 0.2779 | 0.2779 |
| **`cdea_unique_translated`** (same mask, scrambled position) | **0.2652** | **0.2647** |
| `cdea_shared` | 0.2674 | 0.2706 |
| `base_evidence` | 0.6515 | 0.4922 |
| `cdea_unique` | 0.6929 | 0.6257 |
| `center_cell` (fixed, no model) | 0.9058 | 0.9058 |

| paired comparison | EfficientNetV2-S | ResNet-18 |
| --- | --- | --- |
| unique vs **translated null** | **+0.4277** (t=52.3, wins 89.3%) | **+0.3609** (t=45.8, wins 84.3%) |
| unique vs raw evidence | +0.0415 (t=11.1, wins 65.2%) | +0.1335 (t=27.3, wins 73.5%) |
| unique vs `center_cell` | −0.2129 (t=−33.2, wins 7.2%) | −0.2802 (t=−39.3, wins 7.2%) |

**This is the first defensible spatial claim in the project, and it comes with its own
limit attached.** Scrambling a mask's position — holding its shape, budget and
compactness exactly — drops it from 0.693 to 0.265, i.e. to chance. So the mask really
does encode *where*, not merely *how compact and central*; it beats that null by +0.43
on 89% of images. It nonetheless loses to a fixed centred rectangle on 93% of images.

Both facts are true and the talk states both. What the rectangle demonstrates is that on
this dataset the metric is largely measuring acquisition geometry rather than
explanation quality — which is an argument about the benchmark, not about the method.

`center_cell` = 0.9058 here reproduces the 0.9058 in `docs/MEDICAL_RESULTS.md` §8 exactly,
from an independent code path, and 0.9398 from `eval_center_prior.py` on a different
sample — three routes to the same conclusion.

### Translation null — earlier smoke test

`eval_localization.py`, HAM10000, ResNet-18, Grad-CAM, n=64 random val images:

| method | score |
| --- | --- |
| chance (flat mask) | 0.2787 |
| **`cdea_unique_translated`** (same mask, random position) | 0.1957 |
| `cdea_unique` | 0.6489 |
| `center_cell` (fixed, no model) | 0.9394 |

`cdea_unique` vs its own translated copy: **+0.4531** (t=9.56, wins 87.5%).
`cdea_unique` vs `center_cell`: −0.2906 (t=−7.96).

So the mask does carry genuine positional information — it beats a
budget-matched, shape-matched copy of itself placed at random — while still
losing to a degenerate centred rectangle. Both facts belong on the slide.

The `center_cell` figure of 0.9394 here independently reproduces the 0.9398 from
`eval_center_prior.py`, which is computed from the segmentations alone with no
model in the loop.

### Stage 1 — does the decomposition mean what the objective claims? **Branch A**

> **Superseded numbers below.** This section reports the `lambda_shared_sparse=0.0` run,
> where the shared mask is unpenalized and blankets ~46% of the frame at **0.99× chance**
> on base-evidence capture. Keeping "only shared" therefore degrades the image globally,
> which inflates the collapse. The corrected five-config re-run is **Stage 6** below, and
> it is what the deck and figures now read. Branch A holds under both. Retained here
> because the λ=0 numbers appear in `docs/MEDICAL_RESULTS.md` §9 and in the earlier
> notebook, and both are cross-referenced against this section.

`scripts/eval_decomposition.py`, HAM10000, EfficientNetV2-S, Grad-CAM, **n = 2035** (full val split)

**Test A — the split carries the discrimination.** Top-1-minus-top-K probability gap:

| condition | gap |
| --- | --- |
| full image | 0.9214 |
| **shared evidence only** | **0.5861** |
| shared + unique | 0.9354 |

- collapse (full → shared only): **+0.3353**, t = 60.6, 95.6% of images
- recovery (shared+unique → shared only): **+0.3493**, t = 70.4, 99.1% of images
- shared+unique restores the model's original top-1 class on **75.7%** of images

**Test B — unique evidence is class-specific.** Δlogit when class *j*'s unique mask is removed:

| | remove u0 | remove u1 | remove u2 | remove u3 | remove u4 |
| --- | --- | --- | --- | --- | --- |
| class 0 | **−0.80** | +0.08 | +0.01 | +0.01 | +0.00 |
| class 1 | +0.32 | **−0.37** | +0.03 | +0.01 | +0.01 |
| class 2 | +0.21 | +0.08 | **−0.13** | +0.02 | +0.01 |
| class 3 | +0.13 | +0.08 | +0.03 | **−0.05** | +0.01 |
| class 4 | +0.09 | +0.07 | +0.03 | +0.01 | **−0.02** |

Equal-budget random deletion control: +0.013, −0.002, −0.003, +0.003, +0.002 — i.e. **nothing**.
Diagonal minus mean off-diagonal, per image: **−0.3368**.

Two things make this the validation the lesion-overlap test could not be:

1. **No annotation is involved anywhere.** The test interrogates the model directly, so
   it works on any dataset and cannot be confounded by a center prior.
2. **The off-diagonal is positive.** Removing the predicted class's unique evidence
   does not merely hurt that class — it actively *helps* its rivals (+0.32, +0.21,
   +0.13, +0.09 down column 0). That is the contrastive claim in its strongest form,
   and generic image corruption cannot produce it. The random control confirms the
   effect is not "deleting anything degrades the image."

The diagonal weakens monotonically with rank (−0.80 → −0.02), which is expected: lower-ranked
hypotheses carry less unique evidence to remove.

The 32-image smoke test predicted this almost exactly (0.920 / 0.607 / 0.950 versus the
full-set 0.921 / 0.586 / 0.935), so the effect is stable, not a large-sample artifact.

### Stage 3 — one configuration instead of two

All four cells re-run at `eval_localization.py`'s exact settings (`game_mode=mixed`,
`use_shared=True`, `lambda_partition=0.1`, 50 steps, lr 0.2) on **ResNet-18 throughout**, so
the separation result and the validation result now describe the same pipeline.

| config | overlap base | opt (original cfg) | **opt (unified)** | reduction | suff base → unified | budget |
| --- | --- | --- | --- | --- | --- | --- |
| HAM10000 / Grad-CAM | 0.4468 | 0.0099 | **0.0729** | 84% | 0.672 → **2.473** | 1.002 |
| HAM10000 / IG | 0.4949 | 0.0298 | **0.1478** | 70% | 0.660 → **2.504** | 1.022 |
| brain tumor / Grad-CAM | 0.0670 | 0.0023 | **0.0084** | 88% | −0.001 → **0.970** | 1.188 |
| brain tumor / IG | 0.3260 | 0.0101 | **0.0414** | 87% | −0.107 → **1.048** | 1.161 |

**The unified numbers are weaker on overlap and much stronger on sufficiency**, and the
unified ones are what the talk reports. Two reasons the overlap reduction drops from
94–98% to 70–88%: the `mixed` preset enables a shared mask that absorbs common evidence,
and it applies `lambda_disjoint=0.1` where the original ablation passed 0.5 — five times
less pressure to push the unique masks apart. Sufficiency roughly quadruples in exchange
(0.66 → 2.50 on HAM10000), and on brain tumor it crosses from negative to positive.

**Budget caveat, stated rather than buried.** The mask budget is genuinely held on
HAM10000 (×1.00, ×1.02) but drifts on brain tumor (**×1.19, ×1.16**). So on brain MRI part
of the sufficiency gain is *bought* with extra highlight rather than relocated. The clean
"same budget" claim belongs to HAM10000 only, and slide 6 says so.

### Resolution sweep — the grid was the bottleneck, and there is an optimum

ResNet-18 and Integrated Gradients throughout, mass budget fixed, same seeded images at
every grid. Only the grid changes. IG attributes at the pixel level and pools to any
grid, so resolution is a free parameter; Grad-CAM cannot do this (its resolution is the
backbone's feature map, and the `--grid` guard now refuses to fabricate detail there).

| dataset | grid | chance | CDEA | centre rect. | scrambled | CDEA − rect. | t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brain tumor | 7×7 | 0.0176 | 0.0984 | 0.1454 | 0.0175 | −0.0469 | −5.55 |
| brain tumor | 14×14 | 0.0176 | 0.1994 | 0.1549 | 0.0179 | **+0.0445** | 3.28 |
| brain tumor | 28×28 | 0.0176 | **0.2634** | 0.1805 | 0.0218 | **+0.0829** | 4.81 |
| HAM10000 | 7×7 | 0.2742 | **0.5847** | 0.9061 | 0.2611 | −0.3215 | −23.31 |
| HAM10000 | 14×14 | 0.2742 | 0.5755 | 0.8979 | 0.2649 | −0.3224 | −23.67 |
| HAM10000 | 28×28 | 0.2742 | 0.4462 | 0.9257 | 0.2761 | −0.4795 | −36.82 |

> ## RETRACTED — the reversal below does not exist. See Stage 7.
>
> These numbers came from an objective that predates the `mass_scale` mask-budget fix
> (F2). Under the current objective the same commands give **0.68× / 0.84× / 0.54×** the
> strongest null at 7/14/28 — CDEA never beats the centred rectangle at any resolution on
> brain tumor. The "reversal" was the pre-F2 unscaled budget: without `mass_scale` the
> unique mask held ~1 unit of mass at 28×28 instead of ~16, i.e. a tiny, highly
> concentrated blob, which scores well on an overlap metric against a small target.
> The 7×7 row is unaffected and still reproduces exactly, because `mass_scale` is 1.0
> there. Kept for the record; do not quote.

**On brain tumor the result reverses sign.** At 7×7 CDEA loses to a fixed centred
rectangle; at 14×14 and 28×28 it beats it, both significantly. Its absolute score nearly
triples (0.098 → 0.263). Every previous "a rectangle beats CDEA" conclusion on this
dataset was a statement about the grid, not about the method — at 7×7 one cell is 2.04%
of frame against a 1.76% mean tumor, so the mask could not express anything smaller than
the target it was meant to find.

**On HAM10000 finer resolution makes things worse** (0.585 → 0.446), and the gap to the
rectangle widens. Dermoscopy centres the lesion by acquisition convention, so the
rectangle is exploiting dataset geometry; no resolution fixes that, and the centre null
itself climbs with resolution (0.906 → 0.926).

#### The rule: match cell size to target size

Dividing mean target area by cell area explains both datasets at once:

| | 7×7 (cell 2.041%) | 14×14 (0.510%) | 28×28 (0.128%) |
| --- | --- | --- | --- |
| brain tumor (1.76% of frame) | 0.9 cells | 3.5 | **13.8** ← best |
| HAM10000 lesion (27.4%) | **13.4** ← best | 53.7 | 214.9 |

Both peak at **~13 cells per target**, approached from opposite directions. Too coarse
and the mask cannot represent a target smaller than one cell; too fine and the fixed mass
budget (`lambda_mass` pins total mass near 1.0 regardless of R) spreads so thin the mask
goes diffuse.

> **Retracted as a design rule.** An earlier version of this document proposed computing
> the grid from expected target size. The synthetic shortcut benchmark **contradicts it**:
> a 32px patch on a 224px frame is 2.04% of the area, so the rule predicts 28×28 — and
> 28×28 is the *worst* resolution there (10.2× chance) while 14×14 is the best (17.7×).
> Two datasets agreeing is a coincidence, not a law. Treat the table above as an
> observation about these two medical datasets and choose the grid empirically.

**The position-scrambled null improves monotonically on both datasets** (brain:
+0.081 → +0.182 → +0.242, winning 79% → 90% → 92% of images), so the mask encodes
location at every resolution. That comparison and the centre-rectangle comparison answer
different questions, and only the first is a statement about the method.

---

## Stage 7 — resolution sweep v2: both arms, one objective

`results/medical_presentation/resolution_v2/`, 18 cells, Aug 17. Prompted by a request to
verify that the "fine-grid claims fail" line was not itself a bug. It was.

**What was wrong.** The earlier comparison put an **Aug 9** λ=0 baseline against an
**Aug 16** λ=0.25 run. The `mass_scale` mask-budget fix (F2) landed between those dates,
so the two sides were different objectives. Re-running the identical command today at
λ=0 — same seed, same n=597 — gives recovery **−0.1068** against the stored **+0.1899**.
Because `mass_scale = max(1, R/49)` is 1.0 at 7×7, this contaminates *only* the 14×14 and
28×28 cells; every 7×7 track is unaffected. v2 re-runs **both** λ arms under one binary.

**Localization** — × strongest null (max of uniform / center_cell / translated):

| | 7×7 λ=0 → 0.25 | 14×14 λ=0 → 0.25 | 28×28 λ=0 → 0.25 |
| --- | --- | --- | --- |
| brain tumor | 0.68× → 0.64× | 0.84× → 0.85× | 0.54× → 0.56× |
| HAM10000 | 0.65× → 0.61× | 0.53× → 0.51× | 0.35× → 0.35× |

Three things follow:

1. **`lambda_shared_sparse` has essentially no effect on unique localization** — the two
   arms differ by ≤0.005 at every grid on both datasets. The earlier claim that
   constraining the shared mask "collapsed" unique from 1.46× to 0.56× was entirely the
   cross-objective artifact.
2. **CDEA never beats the strongest null**, on either dataset at any resolution. The best
   cell is 0.85×. This is consistent with the category error already documented in §9:
   the annotation marks the lesion/tumor, which every class shares, so it is ground truth
   for *shared* evidence and cannot validate a class-unique mask.
3. **The penalty works at every grid.** `cdea_shared` rises from ~chance to well above it
   — brain 0.013 → 0.035/0.037/0.039, HAM 0.265 → 0.567/0.503/0.453.

**Decomposition** (brain tumor; read *within* a grid only — see below):

| grid | λ | shared only | +unique | recovery | t | diag−off |
| --- | --- | --- | --- | --- | --- | --- |
| 7×7 | 0 | 0.7128 | 0.8235 | **+0.1107** | 15.88 | −0.312 |
| 7×7 | 0.25 | 0.7105 | 0.8233 | **+0.1128** | 14.52 | −0.234 |
| 14×14 | 0 | 0.5212 | 0.5980 | +0.0768 | 7.93 | −0.545 |
| 14×14 | 0.25 | 0.6062 | 0.5751 | −0.0310 | −2.87 | −0.434 |
| 28×28 | 0 | 0.3999 | 0.2930 | −0.1068 | −8.92 | −0.776 |
| 28×28 | 0.25 | 0.4943 | 0.2005 | −0.2938 | −25.74 | −0.740 |

**Test A is not comparable across grids, and this is a property of the metric.** `spread`
is `max − min` over per-class probabilities in which each element is measured on its own
`keep(x, m_shared + m_unique_k)` image. `mass_scale` makes the unique budget 16× larger at
28×28, every per-class condition keeps more of the image, and the between-class spread
compresses — far enough to flip the sign. Confirmed by varying only `mass_ref_regions` at
28×28, λ=0, n=96:

| `mass_ref_regions` | `mass_scale` | unique mass | recovery |
| --- | --- | --- | --- |
| 49 (current) | 16.0 | 15.98 | −0.1111 |
| 784 | 1.0 | 1.06 | **+0.1973** |

which reproduces the pre-F2 Aug 9 value (+0.1899). Note also that Test B *strengthens*
monotonically with resolution (−0.312 → −0.545 → −0.776) while Test A inverts: the two
tests disagree, which is the signature of Test A tracking the budget rather than the
decomposition. **Read Test A at the model's native grid**, where `mass_scale` is 1.0 and
recovery is solidly positive at both λ.

Three hypotheses were tested and killed before this one: under-convergence (more steps
makes it *worse* — 50/150/400 → 0.2995/0.0427/0.0234), `lambda_shared_sparse` (negative at
λ=0 too), and unclamped mask overflow (0.01–0.30% of entries, pixels already in range).

**Mask clamp.** `VisionGridUnitSpace._region_to_pixel_mask` now clamps to [0,1]; `keep`
blends `m*x + (1-m)*baseline`, so a sum above 1 extrapolated past `x`. Effect measured by
running one config twice in-process, clamped vs not: every metric moves in the 4th decimal
(recovery 0.1906 → 0.1900). It invalidates nothing. Regression test:
`tests/test_pass_conditions.py::test_keep_remove_clamp_out_of_range_masks`.

## Run log

| Stage | What | Output | Status |
| --- | --- | --- | --- |
| — | Center-prior ladder (annotation only) | `figures/center_prior.json` | done |
| 1 | Decomposition at scale, 5 configs | `decomposition/` | done |
| 2 | Fine grid | superseded by the resolution sweep | dropped |
| 3 | Unified-config ablations | `ablation/` | done |
| 4 | Translation-null localization, full val | `localization/` | done |
| 5 | Three-seed sweep (3rd attempt) | `seeds/` | done |
| — | Resolution sweep 7/14/28 | `resolution/` | done |
| — | Synthetic shortcut benchmark, 10 cells | `../shortcut/` | done |
| 6 | Decomposition re-run, `lambda_shared_sparse=0.25` | `decomposition_sharedfix/` | done |
| 7 | Resolution sweep v2, both λ arms, one objective | `resolution_v2/` | done |

Reproduce: `run_stage1.sh`, then `run_resolution_sweep.sh`, then `run_stage5_fixed.sh`.
Figures: `plot_hero_figure.py` and `plot_presentation_figures.py`. Deck: `node scripts/build_deck.js`.

### Stage 5 — three seeds, with error bars

Per-seed checkpoints trained at lr 1e-4 without normalization, all verified above chance
(brain tumor 0.9522–0.9555, HAM10000 0.7024–0.7082 balanced accuracy — spread under
0.006, so metric variance reflects allocation rather than model quality).

| config | overlap base | overlap optimized | reduction | sufficiency |
| --- | --- | --- | --- | --- |
| brain / Grad-CAM | 0.0610 | **0.0083 ± 0.0027** | 86% | 1.072 ± 0.226 |
| brain / IG | 0.2976 | **0.0501 ± 0.0043** | 83% | 1.118 ± 0.217 |
| HAM10000 / Grad-CAM | 0.3709 | **0.0497 ± 0.0066** | 87% | 2.131 ± 0.441 |
| HAM10000 / IG | 0.4891 | **0.1097 ± 0.0204** | 78% | 2.193 ± 0.465 |

Consistent with Stage 3's single-seed 70–88%.

### Stage 6 — constraining the shared mask

Prompted by a reading of the gallery figures: allocated evidence appeared in places the
base evidence never marked. It turned out not to be drift in the unique masks.

**The shared mask carried no penalty term at all.** `lambda_sparse` applies only to the
unique masks, `lambda_overlap` only to unique–unique pairs, `lambda_mass` pins unique
mass. Its only brake was the allocator's partition cap, which never binds on average
(mean region occupancy 0.56 against a cap of 1.0). Since the shared mask enters `m_tot`,
growing it always raises sufficiency and margin, so the optimizer grows it without limit.

Measured over 96 HAM10000 val images at 7×7 (`lambda_shared_sparse` 0.0 → 0.25):

| | λ=0 | λ=0.25 |
| --- | --- | --- |
| shared mass (of 49) | 22.56 | **2.76** |
| shared regions above 0.5 | 47.2% | **3.5%** |
| **shared: base evidence captured / area** | **0.99× chance** | **1.48× chance** |
| unique: base evidence captured / area | 3.35× chance | 3.47× chance |
| unique mass | 1.01 | 1.03 |
| unique–unique overlap | 0.058 | 0.052 |

The bolded row is the finding. At λ=0 the shared mask captures base evidence at *exactly
its own area fraction* — statistically uncorrelated with the field it is allocating. It
is not a mask that found the background; it is a mask that found nothing. **The unique
masks were never the problem** (3.35× chance either way), so every result resting on
`cdea_unique` — Stage 4, the resolution sweep, the shortcut benchmark — is unaffected.

All five configs re-run at λ=0.25, everything else identical; full table in
`docs/MEDICAL_RESULTS.md` §9a. Four conclusions:

1. **Test A survives everywhere** (t = 14–94 on collapse and recovery across all ten
   runs). Branch A does not depend on the blanket.
2. **On HAM10000 the λ=0 collapse was partly an artifact.** Shared-only spread rises
   ~0.09 in all three HAM configs once the blanket is gone — keeping a soft mask over 46%
   of the frame degrades the input globally. The honest collapse is 0.805 → 0.508, not
   0.805 → 0.420, and recovery improves at the same time (0.615 → 0.669).
3. **Test B's absolute magnitude falls ~25%, its specificity does not.** The off-diagonal
   shrinks proportionally, so |diagonal|/off-diagonal is flat on brain (1.76 → 1.76) and
   improves on all three HAM configs (up to 4.56 → 6.04). The equal-budget random control
   stays at ≈0 (|·| ≤ 0.017) in all ten runs.
4. **Brain tumor behaves differently and is unexplained.** Its shared mask compacts just
   as much (19.49 → 2.71) but capture stays at chance (0.93× → 0.86×) and shared-only
   spread barely moves. Compactness is necessary, not sufficient. Candidates — K=3 vs 5,
   a near-saturated model (full spread 0.976), genuinely shared anatomy the
   class-conditioned evidence field does not mark — untested.

`lambda_shared_sparse` defaults to 0.0 in `ContrastiveObjective.__init__` so prior runs
reproduce exactly. Use 0.25 for anything reporting a statistic *about the shared mask*.
The deck, `F4_decomposition`, and `results_full.ipynb` §2b now read the 0.25 run.

```bash
PYTHONPATH=. python scripts/eval_decomposition.py \
  --checkpoint examples/out/checkpoints/ham10000_efficientnet_v2_s.pt \
  --num_images 2035 --game_mode mixed --num_alloc_steps 50 \
  --lambda_shared_sparse 0.25 \
  --out_dir results/medical_presentation/decomposition_sharedfix \
  --export_prefix decomp_ham10000_effnet_gradcam
```

---

## Three attempts at Stage 5, and what each taught

Recorded because the failures were more instructive than the result, and because two of
the bugs reach beyond this run.

1. **DataLoader deadlock.** Both eval scripts hardcoded `num_workers=4`. PyTorch's
   multiprocessing loader deadlocks against MPS on long runs: workers go idle, the main
   process blocks on a queue read that never returns, and the job sits at 0% CPU
   indefinitely rather than failing. Cost ~7.5 h of overnight compute. Now
   `--num_workers`, default 0.
2. **Learning rate — a wrong diagnosis.** `get_or_train` defaults to 1e-3 and
   `run_experiments.py` could not override it. Plausible, and false: the re-run at 1e-4
   failed identically. Fixed anyway (`--train_lr`), and the checkpoint filename now
   encodes the lr, since without that a corrected re-run silently returns the broken
   cached checkpoint.
3. **The real cause: train/eval preprocessing mismatch.** `train_backbone` applied
   ImageNet normalization; *every* evaluation and explanation path in the repo consumes
   raw [0,1] tensors. The models were never broken — brain tumor seed 0 scored 0.337
   evaluated raw versus **0.948** normalized; HAM10000 seed 0, 0.161 versus **0.713**.

### This one is not confined to the medical work

The same function produced every checkpoint in `scripts/out/checkpoints/`. The CIFAR-10
linear probe scores **0.389 raw versus 0.816 normalized**, so `ablation_*`, `shift_*` and
`journal/JOURNAL_REPORT.md` — which feed both paper drafts — were computed on models
running at roughly half their true accuracy.

Worth re-examining before publication: `GAMBIT_PAPER.md` §8.1 lists "negative sufficiency
and margins with weak backbones" as a limitation. Sufficiency is a logit difference; those
backbones were being fed inputs at the wrong scale. The limitation may be an artifact.

`examples/` results are unaffected — `contrastive_explanation.py` never normalized, which
is why its checkpoints always worked.

Fixed by removing normalization from training rather than adding it to evaluation, because
the interventions the objective is built on are defined in [0,1] space: the blur/mean
baselines and IG's `baseline="zero"` all change meaning under standardized inputs.

### Guards added, so these fail loudly next time

| failure | guard |
| --- | --- |
| Grad-CAM returns an all-zero field on ViT (its non-negativity assumption does not hold for LayerNorm'd tokens) | `RuntimeWarning` naming the cause and pointing at IG |
| Training converges to chance | warning + banner at `balanced_acc <= 1.15 x chance`; Stage 5 verifies every checkpoint before reporting |
| `NaN` in result JSON silently breaks every non-Python reader | `save_json` refuses to emit it; `_json_safe` maps to `null` |
| A finer grid requested for Grad-CAM would fabricate resolution | `--grid` raises unless `--evidence ig` |
| Class-ordered subsets | seeded `randperm` instead of `range(N)` |
