# Robust/shortcut on real data: HAM10000 acquisition sites

`scripts/eval_robust_shortcut_medical.py`, ResNet-18, 7×7 grid, `mixed` preset, 40 steps,
lr 0.3, `lambda_sparse` 1.0, n = 400 val images (all 2035 have expert segmentations).
Environments: three acquisition sites — `vidir_molemax`, `vidir_modern`, `rosendahl`.
Runner: `results/medical_presentation/run_shift_real.sh`.

## The shortcut is in the data already

The release records an acquisition site per image, and site is confounded with diagnosis:
`vidir_molemax` is 94% nevus (3720/3954) with essentially no akiec or bcc, `rosendahl` is
broadly balanced. Re-rendering a val image to another site's measured colour statistics
flips **34.4%** of predictions and moves p(top-1) by 0.279.

Two nuisances were run, both measured from the real site groups:

- **vignette** — per-site radial luminance profile. The spread across sites is 0.000 at
  the centre and 0.955 at the rim, so it is *spatially localized* at the periphery.
- **colour** — per-site channel mean/std. A global cast, with no spatial signature.

## Result — n = 400

| | vignette | colour |
| --- | --- | --- |
| lesion area fraction (chance) | 0.2746 | 0.2746 |
| shortcut mask | 0.2820 | 0.3000 |
| **robust mask** | **0.3237** | **0.3275** |
| base evidence (Grad-CAM) | 0.4904 | 0.4904 |
| robust − shortcut | **+0.0417**, t = 9.94 | **+0.0274**, t = 5.38 |
| win rate | 53.3% | 43.3% |
| rob_var (lower = more stable) | 0.0319 | 0.0377 |
| shortcut gap | 0.070 | 0.410 |

The localized nuisance separates better than the global one (t 9.9 against 5.4, shortcut
gap 0.07 against 0.41), which is the expected ordering: a region mask has something to
grab only when the nuisance lives in particular regions.

**A 32-image pilot put the colour arm at t = 0.2 and was read as a null. At n = 400 it is
t = 5.38.** The pilot was underpowered; the correct statement is that the global cast
separates *less well*, not that it fails.

## Two results that cut against the method

1. **Both masks localize the lesion worse than the evidence they were built from.** The
   robust mask reaches 1.18× chance (0.324 against 0.275) where raw Grad-CAM reaches
   1.79× (0.490). The split is statistically solid; the spatial quality is not an
   improvement over the input.
2. **Win rates sit at or below half** — 53.3% and 43.3% — despite reliably positive means.
   The effect comes from a subset of images with large positive differences, not a
   consistent shift; the median is near zero. "Robust beats shortcut on most images" would
   be false here. It *is* true on Stanford Dogs (91.5%), where the same experiment runs
   much cleaner: see `results/shift_real_dogs/RESULTS.md`.
