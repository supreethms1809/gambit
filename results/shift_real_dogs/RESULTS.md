# Robust/shortcut on real data: Stanford Dogs backgrounds

`scripts/eval_robust_shortcut_dogs.py`, ResNet-18, 7×7 grid, 120 breeds, `mixed` preset,
40 allocation steps, lr 0.3. Runner: `results/run_shift_dogs.sh`.

Companion to `results/medical_presentation/shift_real/` (HAM10000 acquisition sites).
Both exist because `eval_robust_shortcut.py` runs only on synthetic sets where the
shortcut is stamped in and the environments are built by re-stamping it.

## The shortcut is in the data already

Breeds are photographed in correlated settings. A nearest-centroid classifier reading
*only* six background colour statistics — mean and std of RGB outside the bounding box —
recovers the breed at **11.1% against 5.0% chance** over 20 breeds. And the model uses it:
re-rendering only the background to a different measured style flips **12.0%** of
predictions.

Environments are three background *styles* from k-means over measured background
statistics (bright 0.71, dark 0.30, mid 0.49 mean RGB), applied outside the box only,
foreground untouched. Paired, as the objective requires.

## Result — n = 400, `lambda_sparse` 1.0

| | in-box mask mass |
| --- | --- |
| box area fraction (chance) | 0.5558 |
| shortcut mask | **0.5561** |
| base evidence (Grad-CAM) | 0.7061 |
| robust mask | **0.7259** |

**robust − shortcut = +0.1698, t = 25.8, on 91.5% of images.**

Three things make this the cleanest version of this experiment in the repo:

1. **The shortcut mask sits at exactly chance on the object** (0.5561 against 0.5558). It
   is not merely "less on the dog" — it is indistinguishable from a mask placed without
   regard to where the dog is, which is what a mask covering only background should be.
2. **The robust mask beats the evidence it was built from** (0.726 against Grad-CAM's
   0.706). On HAM10000 the robust mask lands *below* raw evidence (0.324 against 0.490),
   so this does not hold in general.
3. **91.5% of images**, against 53.3% on HAM10000. There the positive mean came from a
   subset with large differences while the median sat near zero; here the effect is
   consistent image to image.

## Stability across the sparsity weight

`RobustShortcutObjective` has no mass target, unlike `ContrastiveObjective`'s
`lambda_mass`, so `lambda_sparse` is the only brake on mask size. The preset ships 0.05.

| `lambda_sparse` | n | sparse | disjoint | robust | shortcut | Δ | t | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 (preset) | 200 | 14.77 | 2.764 | 0.6534 | 0.5139 | +0.1396 | 15.1 | 87.0% |
| 1.0 | 400 | 5.41 | 0.511 | 0.7259 | 0.5561 | **+0.1698** | 25.8 | 91.5% |
| 2.0 | 200 | 3.58 | 0.249 | 0.6973 | 0.5653 | +0.1319 | 15.7 | 82.5% |

At the preset the two masks together hold 14.8 of 49 regions and overlap at 2.76, and the
separation is weakest. The result survives the setting but is best with the masks
constrained. **On HAM10000 the preset was worse than weak — at 0.05 both masks blanketed
the frame and every spatial statistic degenerated into a measure of area.**

## What this does not show

The nuisance is applied outside the box, so "the shortcut mask avoids the box" is the
hypothesis under test, not an independent discovery. What the allocator is not given is
*where* the box is: it recovers that structure from environment variation alone. That is
the same thing ColoredCIFAR10's corner patch tests, with a real photograph and a real
correlated background instead of a stamped square.

A claim about generalizing across genuinely distinct environments — rather than across
transformations measured from them — needs environments that are not constructed from the
same images. The objective is paired (one spatial mask across all environments, labels
from `xs[0]`), so that requires a group-statistics variant it does not currently have.

## Checkpoint health

Both runs report `|z(all regions) − z(1 region)|` before any result, after a brain-tumor
EfficientNet checkpoint passed an accuracy check while being insensitive to its input
(see `results/medical_presentation/ablation_effnet/BROKEN_ep10_lr1e-3/README.md`).
Stanford Dogs ResNet-18 measures **13.4**. The 3.1–3.4 figures quoted in that README are
from 3- and 7-class models and are not directly comparable across a 120-class logit range;
what matters is that it is nowhere near the 0.19 of the broken checkpoint.
