# Phase 1 results — out-of-objective evaluation

`results/run_phase1.sh`. ResNet-18, 7×7 grid, `mixed` preset, 50 allocation steps, lr 0.2.
Sanity checks n=128, faithfulness n=200 stride 2.

Everything here is **outside `ContrastiveObjective`'s loss**. Overlap, sufficiency, margin
and mask budget are all loss terms, so the headline table measures convergence; these
measure whether the explanation is faithful and whether it depends on the model at all.

---

## 1.1 Model-randomization sanity check (Adebayo et al., NeurIPS 2018)

Cascading top-down randomization, Spearman ρ against the trained model's masks.
**Hypotheses are frozen to the trained model's top-K** — otherwise randomization changes
which classes are ranked and we would be comparing masks for different classes.
Pass threshold |ρ| < 0.30.

| cell | ref acc | rand acc | chance | **CDEA ρ** | base ρ | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| brain tumor / Grad-CAM | 0.953 | 0.336 | 0.333 | **−0.002** | 0.035 | PASS |
| HAM10000 / Grad-CAM | 0.641 | 0.156 | 0.143 | **0.060** | 0.096 | PASS |
| HAM10000 / IG | 0.641 | 0.156 | 0.143 | **0.059** | 0.307 | PASS |
| brain tumor / IG | 0.953 | 0.336 | 0.333 | **0.568** | 0.865 | **FAIL** |

### Two findings, one good and one bad

**Good: allocation makes the explanation model-dependent.** On HAM10000, Integrated
Gradients itself largely fails — ρ = 0.689 with only the classifier head randomized, and
still 0.307 with the whole network destroyed. CDEA over that same evidence sits at 0.059.
The masks are optimized through the frozen classifier's responses to interventions, so
destroying the classifier leaves them nothing to lock onto even when the underlying
attribution field is unchanged. That is a positive argument for the method, not a
box-tick.

**Bad: IG on brain tumor is essentially model-independent, and CDEA only half-rescues it.**
Base ρ = 0.865 after full randomization; CDEA ρ = 0.568, well past the threshold. The
likely cause is the interaction of IG's `baseline="zero"` with MRI intensity structure:
IG's attribution carries a factor of (x − baseline) = x, and a brain MRI is a bright object
on a black field, so the attribution tracks pixel intensity almost regardless of the
model's weights. Dermoscopy has no comparable structure, which is consistent with
HAM10000/IG passing.

**Consequence: every brain-tumor + IG number in the project is suspect** — including the
0.326 → 0.010 overlap figure quoted in the deck's slide-7 footnote. Not necessarily wrong,
but not currently supported. Re-run with a blur or mean baseline before quoting it, and
re-run this check on the result.

Note on protocol: on 3-class brain tumor, randomizing `fc` alone only drops accuracy from
0.953 to 0.781 and CDEA ρ sits at 0.201. The model is not destroyed until the cascade
reaches `layer4`. Reading stage 1 alone would be misleading on few-class problems.

---

## 1.2 / 1.3 Deletion, Insertion (RISE) and ROAD (Rong et al., ICML 2022)

Target-class probability, regions ordered by mask value. Deletion and MoRF lower is
better; insertion and LeRF higher is better. `random` is a random region ordering.

| cell | method | del ↓ | ins ↑ | ROAD MoRF ↓ | ROAD LeRF ↑ |
| --- | --- | --- | --- | --- | --- |
| brain / Grad-CAM | base | 0.7937 | 0.9512 | 0.2705 | 0.5087 |
| | **CDEA** | **0.7573** | **0.9700** | 0.2727 | 0.4885 |
| | random | 0.9039 | 0.9041 | 0.3027 | 0.2954 |
| brain / IG | base | 0.8133 | 0.9438 | 0.2917 | 0.4698 |
| | **CDEA** | **0.7462** | **0.9677** | 0.2928 | 0.4560 |
| | random | 0.9039 | 0.9041 | 0.3027 | 0.2954 |
| HAM / Grad-CAM | base | 0.5603 | 0.7206 | 0.5526 | 0.5856 |
| | **CDEA** | **0.4957** | **0.7674** | 0.5522 | 0.5876 |
| | random | 0.6321 | 0.6296 | 0.5643 | 0.5642 |
| HAM / IG | base | 0.5722 | 0.7018 | 0.5583 | 0.5806 |
| | **CDEA** | **0.4247** | **0.7868** | 0.5610 | 0.5750 |
| | random | 0.6321 | 0.6296 | 0.5643 | 0.5642 |

**CDEA wins deletion and insertion in all four cells**, and beats random comfortably. The
largest margin is HAM10000/IG: deletion 0.572 → 0.425, insertion 0.702 → 0.787.

**CDEA does not win ROAD in any cell.** Under noisy linear imputation the three methods
collapse to within ~0.01 of each other on MoRF, and base evidence edges CDEA on LeRF in
three of four. Random is still clearly worst, so ROAD is not measuring nothing — it just
does not separate CDEA from its own base evidence.

That gap between protocols is itself the finding. Deletion and insertion replace regions
with a blur, which leaves the mask's *shape* in the image; ROAD imputes from neighbours so
the hole is not visually recoverable. CDEA's advantage surviving one and not the other is
consistent with part of it coming from mask shape rather than mask content. **Report both.
Quoting deletion/insertion alone would overstate the result.**

---

## What this changes

1. **Slide-7 footnote.** Drop or re-derive the brain-tumor IG numbers.
2. **The IG baseline is a real variable**, not a detail. `baseline="zero"` should be
   re-examined on any dataset with strong global intensity structure.
3. **ROAD belongs in the paper.** Leaving it out and reporting only deletion/insertion
   would be the kind of selective reporting a reviewer looks for.
