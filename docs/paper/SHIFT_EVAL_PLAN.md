# Shift evaluation plan

Status: implemented and ready for val gate and selection runs (`scripts/paper_run.py --game shift`, `scripts/gate_shift.py`, `scripts/check_degenerate.py`). Not frozen; the test split stays locked. This file has no version name.

## Datasets

Waterbirds, ImageNet-9, Stanford Dogs restyled outside the box, planted-patch CIFAR-10 (`planted_patch`), ColoredMNIST (non-spatial), and a sixth dataset (COCO-on-Places, still to be chosen). The five implemented datasets run through `evaluation/run_data.py:shift_sample`.

On `planted_patch` (`evaluation/planted_cues.py:PlantedPatchCIFAR`), `present` (`x_id`) stamps the class-tied hue patch `A` at `a` and the class-neutral checker `B` at `b`. The counterfactual view `moved` changes both patches in-place at `(a, b)` (`A` to foil hue `(label + 5) % 10`, `B` to the inverted checker `1 - B`), and `removed` removes both patches. Because `PlantedPatchClassifier` trains on `present` only, existing `planted_patch` checkpoints are unchanged, and fixed-mask transplant and fixed-mask ROAD deletion act at the exact coordinates `(a, b)` across all views.

Image counts per backbone are the contrastive minimums until this plan is frozen. The gate uses `n = 64`.

## Scorer null and the 2×2 shift matrix (`evaluation/scores.py`)

For each image and area `a`, the scorer forms `N = 8` seeded cell-structured random masks through the shared adapter (`null_pixel_masks`, seed 0). Under ROAD removal (the converged linear fill, noise 0.01; `EVAL_PLAN.md` section 5), each mask `M ∈ {M_R, M_S}` is evaluated on two raw intervention drops (`shift_drop_matrix`):
- **Worst-environment robust drop:** `d_rob(M) = min_{e ∈ E} [ m(x_e) - m(x_e ⊖ M) ]`
- **Cross-environment disagreement drop:** `d_gap(M) = D(x) - D(x ⊖ M)`, where `D(x) = mean_{e ∈ E'} |m(x_id) - m(x_e)|`

Subtracting the `N = 8` random-mask mean null `(ν_rob(a), ν_gap(a))` (`null_shift_drops`) yields the excess drops `d̃_rob(M)` and `d̃_gap(M)`.

## Metrics (`shift_scores`)

| Angle | Metric | Definition | Optimised by |
|---|---|---|---|
| Shortcut necessity, primary | `ΔD@a` (`delta_d`) | `d̃_gap(M_S) = d_gap(M_S) - ν_gap(a)` under ROAD deletion in every environment (with probability companion `delta_d_prob`) | Nothing directly. `cdea_shift` and `shift_assign` transplant; they do not delete |
| Robust necessity | `RN@a` (`rn`) | `d̃_rob(M_R) = d_rob(M_R) - ν_rob(a)` under ROAD deletion | The robust player (under blur), disclosed |
| Robust purity | `PUR_R@a` (`pur_r`) | `d̃_rob(M_R) - |d̃_gap(M_R)|`: `M_R` drops worst-case log-odds without altering the cross-environment gap | Reported alongside `RN@a` |
| Shortcut purity | `PUR_S@a` (`pur_s`) | `d̃_gap(M_S) - max(0, d̃_rob(M_S))`: `M_S` closes the cross-environment gap without damaging worst-case robust evidence | Reported alongside `ΔD@a` |
| 2×2 role separation | `RoleSep@a` (`role_sep`) | `(d̃_rob(M_R) - d̃_rob(M_S)) + (d̃_gap(M_S) - d̃_gap(M_R))` | Descriptive summary of the 2×2 matrix |
| Transplant closure | `TC@a` (`tc`) | The shortcut transplant payoff of the binary `M_S` | The shortcut player, disclosed. Descriptive |
| Ground truth, family T-shift | `SPC@a` (`spc`) | On `planted_patch`: `mass(M_S on A) - mass(M_S on B)`, where `A` is the class-tied patch and `B` is the neutral checker | Nothing |
| Construction masses | Foreground / background | `mass_shortcut_background`, `mass_robust_foreground`, `mass_random_background` | Descriptive (`M_S` on background is near-tautological for a transplant, stated in text) |

## Methods

Core (`SHIFT_CORE`):
- `cdea_shift`: joint Sinkhorn-Adam shift game with binary draws and boundary offsets (`cdea/shift.py`)
- `shift_assign`: exact HiGHS LP assignment of single-cell worst-environment blur deletion `g_{R, r}` and single-cell transplant closure `g_{S, r}` (`baselines/shift_maps.py`)
- `gap_attribution`: path-integrated gradient of `g_e` along `x_e → x_id` for `S` and min over environments of blur-baseline IG for `R` (`baselines/gap_attribution.py`)
- `attribution_difference`: elementwise min and max-gap of per-environment Grad-CAM / IG maps (`baselines/shift_maps.py`)
- `spray`: Zennit `EpsilonPlus` LRP + CoRelAy spectral clustering (`baselines/spray.py`)
- `extremal_shift`: per-environment TorchRay `extremal_perturbation`, deletion variant, + min/max-gap reduction (`baselines/shift_maps.py`)
- `random_floor`: seeded random cell masks, drawn independently for R and S over the whole sample

Extended (`SHIFT_EXTENDED`): `r2r`, fetched by `scripts/fetch_r2r.sh`, unlicensed, research use, never vendored.

## Selection

Per dataset, at most four configurations per method, maximising val `ΔD@a`. The `cdea_shift` grid is initialisation `{Grad-CAM, IG}` times learning rate `{0.05, 0.2}`, with `T = 100`. The area pilot is `{5, 10, 25}` percent. The family S comparator is fixed on val before `cdea_shift`'s val score is read.

## Statistics

- **Family S:** ResNet-50 across the shift datasets, two-sided exact Wilcoxon signed-rank (`analysis/stats.py:family_summary`).
- **Family T-shift:** `SPC@a` on `planted_patch` against the val-chosen comparator, pooling seeds with the within-seed sign-flip permutation test and two-level bootstrap (`analysis/stats.py:seed_pooled_summary`).
- **Framing rule (`cdea_shift` vs `shift_assign`):** Mirrors the decomposition game. If `shift_assign` matches `cdea_shift`, the contribution is the 2×2 shift intervention formulation, the transplant-vs-deletion payoff definitions, and the additive assignment solver; if `cdea_shift` beats `shift_assign`, joint multi-cell search is the headline method.

## Gate (`scripts/gate_shift.py`)

Default configuration on Waterbirds and `planted_patch` val, ResNet-50, seeds 0 and 1, `n = 64`:

```bash
# Run the four gate cells
PYTHONPATH=. python scripts/gate_shift.py --run

# Read out all scores, paired bootstrap intervals and the gate verdict
PYTHONPATH=. python scripts/gate_shift.py --records results/paper/gate_shift --split val

# Check D2, D4, D3 and D9 on each shift cell
PYTHONPATH=. python scripts/check_degenerate.py --records results/paper/gate_shift/val/shift/waterbirds/resnet50/seed0
```

Stop rule: continue as a method paper only if `cdea_shift` (or `shift_assign` under the framing rule) is ahead of both `gap_attribution` and `attribution_difference` on both datasets. Otherwise take the evaluation-paper route, with no further formulation revision. Fast knobs cannot satisfy the gate.

"Ahead" (defined 2026-10-20): on that dataset, the paired difference in ΔD@5% (`cdea_shift` minus the comparator, ResNet-50), with images paired by index within each seed and the differences of seeds 0 and 1 pooled, has its 95% percentile bootstrap interval (4000 resamples over images) above 0. A dataset with a seed or a method missing leaves the gate incomplete. `gate_shift.py --records` prints this verdict last.
