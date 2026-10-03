# Getting to a conference paper: scoped plan

Written against the repo as of 2026-08-27. Effort figures are rough working estimates for
one person, and assume the compute we have (single Apple-silicon box, MPS).

## The one-line diagnosis

Every headline metric — overlap, sufficiency, margin, mask budget — is a term in
`ContrastiveObjective`'s loss, and the only published-method comparison in the repo is
`naive_contrastive`, which we wrote ourselves. So the results currently show that the
optimizer converges, against a strawman. The deletion matrix is the exception and is the
strongest asset we have.

Two gaps follow, in priority order: **the evaluation protocol is not the field's**, and
**there are no external baselines**. Fixing the first is cheap and makes the existing
results defensible; the second is the expensive part.

---

## Phase 1 — make the existing results defensible (~2–3 weeks)

Nothing here needs new science. All of it plugs into `unit_space.keep`/`remove` and the
`BaseEvidenceProvider` protocol.

### 1.1 Sanity checks (Adebayo et al., NeurIPS 2018) — **do this first**

Cascading model-parameter randomization: randomize layers top-down, recompute the CDEA
masks, and report rank correlation against the original. A method whose masks survive
randomization is measuring the image, not the model.

Highest leverage per hour in the whole plan, for two reasons. It is a standard reviewer
ask that we currently fail by default, and **we have direct evidence the pipeline is
vulnerable to exactly this failure**: the ep10/lr1e-3 brain-tumor checkpoint produced a
complete, plausible result set (overlap −96%, sufficiency flat) on a model whose logits
were unchanged between one tile and the whole image. See
`results/medical_presentation/ablation_effnet/BROKEN_ep10_lr1e-3/README.md`.

Ship alongside it the `|z(all regions) − z(1 region)|` probe already used in
`eval_robust_shortcut_dogs.py`, promoted into `train_backbone.py` next to the existing
chance-level warning, so no future checkpoint reaches an experiment unscreened.

*New file:* `scripts/eval_sanity_checks.py`. **~2 days.**

### 1.2 Deletion / Insertion AUC (RISE protocol)

Order regions by mask value, sweep, integrate. This is the currency reviewers read
faithfulness in, and we report none of it.

*New file:* `scripts/eval_faithfulness.py`. **~2 days.**

### 1.3 ROAD (Rong et al., ICML 2022)

Noisy linear imputation instead of naive masking. Defends against the standing objection
that `keep()` produces off-distribution inputs — an objection that applies to *every*
number in the project, since every metric is a masked forward pass.

Same file as 1.2. **~3 days.**

### 1.4 Evidence-backend breadth

`pip install grad-cam captum`. One adapter class pooling any of their outputs to the grid
gives Score-CAM, Layer-CAM, XGrad-CAM, Ablation-CAM, HiResCAM, plus Captum's DeepLift and
GradientShap. Our central claim is that CDEA is agnostic to the base evidence; right now
we assert it from two backends and demonstrate it from none.

*New file:* `base_evidence/library_adapters.py`. **~2 days.**

**Phase 1 exit criterion:** every existing claim has an out-of-objective number attached,
and the backend-agnostic claim is demonstrated across six providers rather than two.

---

## Phase 2 — external baselines (~3–4 weeks)

This is what makes it a paper rather than a report.

### 2.1 Extremal Perturbations (Fong et al., ICCV 2019) — the critical one

Mask optimization under an explicit area constraint: the same method family as CDEA, so
without it "we optimize masks" has no prior-art anchor. It is per-class rather than
contrastive, which is exactly the point — run it per hypothesis and show what joint
allocation buys over independent optimization at matched budget.

Implementable directly against `unit_space` and `OptimizationAllocator`'s structure.
**~1 week.**

### 2.2 Counterfactual Visual Explanations (Goyal et al., ICML 2019)

Region-swap counterfactuals: the closest published analogue to what CDEA-Contrastive
claims. This is the true prior art for the contrastive framing and its absence is the
first thing a reviewer familiar with the area will notice. **~1 week.**

### 2.3 CEM (Dhurandhar et al., 2018)

Already cited in `GAMBIT_PAPER.md` §2 and not run. Citing a method as related work and not
comparing to it is worse than not citing it. **~4 days.**

**Phase 2 exit criterion:** the ablation table's three arms become six, half of them
published, and `naive_contrastive` is demoted to a sanity row.

---

## Phase 3 — scale and the shift game (~3–4 weeks)

### 3.1 ImageNet-scale run

An ImageNet-S / 100-class subset with segmentations. n=400 on 7-class HAM10000 will read
as small at a top venue. **~1 week including compute.**

### 3.2 Waterbirds, with a group objective

The shift game's weakest point is that environments are *paired transformations* of one
image, because `RobustShortcutObjective` applies one spatial mask across all environments
and reads labels from `xs[0]`. Waterbirds has real group labels and is the canonical
benchmark, so this needs the unpaired variant: group-level statistics (mean sufficiency
per group, variance across group means) instead of per-example cross-environment variance.

That is a genuine change to the instantiation, not a script. Pairs with comparisons to
GroupDRO and DFR, which are the methods this literature actually reports against.
**~2 weeks.**

---

## What to cut

- **The three synthetic shift datasets.** Superseded by the real ones
  (`results/shift_real_dogs/`, `results/medical_presentation/shift_real/`). Keep as a
  smoke test, not a result.
- **Every spatial-localization claim for the contrastive game.** CDEA never beats the
  strongest null at any resolution on either medical dataset (best cell 0.85×). Do not
  argue it; report the centre prior as a finding instead — see below.
- **"At a fixed mask budget."** True for one of four dataset×backbone cells (×1.02);
  the others spend ×1.12–1.38.

---

## The cheaper alternative worth considering first

A **benchmark-critique paper**, buildable from material largely in hand:

> Spatial overlap metrics cannot validate contrastive explanations on medical imaging. A
> fixed centred rectangle with no model in it scores 0.906 against every method measured;
> the centre null *strengthens* with grid resolution (0.940 → 0.961) rather than weakening;
> and the annotation marks the lesion, which every class shares, making it ground truth
> for shared rather than class-unique evidence. We propose the deletion-matrix protocol
> instead, and show it separates methods where overlap metrics cannot.

Needs Phase 1 (sanity checks and faithfulness give it teeth) but little of Phase 2, since
the contribution is the protocol rather than beating prior methods. Roughly a month rather
than three, and the negative results become the contribution instead of a limitations
section.

---

## Recommended sequence

Phase 1 regardless — it is cheap, it is required either way, and it decides which paper is
viable. Then choose: the critique paper (~1 more month) or the full method paper
(Phases 2–3, ~2 more months).

Do not start Phase 3 before Phase 2. Scale without baselines reads as a bigger report.
