# Shift formulation

Status: implemented in `cdea/shift.py` and ready for experiments. Reuses the shared plan machinery in `cdea/_plan.py`. This file has no version name.

## Question

Which evidence does the prediction need in every environment (robust), and which evidence explains why the prediction changes between environments (shortcut)?

## Notation

`y = argmax z(x_id)`. Log-odds of `y`:

`m(x) = z_y(x) - LSE_{j ≠ y} z_j(x)`.

Non-saturating, and invariant to a shift shared by every logit.

`E` is every environment, including id. `E'` is `E` without id. `g_e = m(x_id) - m(x_e)`.

Units, the hard cell indicator `φ_r`, the deletion baseline `b(x)`, and boundary-robust pixel offsets `T_δ` (`s = ⌊½ unit side⌋`) are `core/grid.py`. The same blur is the scorer's blur.

## Players and payoffs

Both payoffs are in nats. There are no weights.

Robust `R`, deletion, worst case over environments:

`u_R(A_R) = min_{e ∈ E} [ m(x_e) - m(x_e ⊖_δ A_R) ]`.

Shortcut `S`, transplant. The gap closed by copying units from `x_e` into `x_id`:

`x_id^{A_S ← e, δ} = (1 - T_δ Ã_S) ⊙ x_id + T_δ Ã_S ⊙ x_e`,

with `Ã_S` built from the hard cell indicators, and

`u_S(A_S) = mean_{e ∈ E'} [ |g_e| - |m(x_id^{A_S ← e, δ}) - m(x_e)| ]`.

Units that are identical across environments earn exactly 0, so `S` cannot win by deleting the object. Background that supports `y` in id and opposes it in `e` cannot be robust, because the min over environments rejects it.

## Constraints and objective

Rows are `{R, S, ∅}`. Each unit's column sums to 1. Each player's row sums to `a · R`, so `a ≤ 1/2`. The empty row receives the rest.

Same log-domain Sinkhorn (`I = 20` unrolled after detached catch-up), same prior initialisation, same straight-through binary draws (`_draw`), same boundary-robust pixel offsets (`_offsets`, one generator seeded by `draw_seed = 0`, never by the batch), same Adam, same per-image summed loss, and the same per-area solve as the decomposition and contrastive games (`cdea/_plan.py`). The objective is to maximise `u_R + u_S`.

The prior is sum-normalised class evidence. Robust evidence is the minimum over environments. Shortcut evidence is the mean absolute difference between the id map and each other environment. The log-initialisation uses `ε = 1e-6`. The default evidence is Grad-CAM on ResNet-50 and integrated gradients (blur baseline) on ViT-B/16. Uniform initialisation is an ablation.

## First-order and additive per-cell solutions

- **Gap attribution** (`baselines/gap_attribution.py`). Integrated gradients with baseline `x_e`, on the path `x_e → x_id`, attributes `g_e` completely. Pooled to units and signed by `g_e`, it is the path-integrated first-order solution of `u_S`. Its robust companion is the min over environments of the blur-baseline path integral.
- **Additive per-cell assignment (`shift_assign`, `baselines/shift_maps.py`).** Each cell `r` is scored on its own by `g_{R, r} = u_R(e_r)` (worst-environment blur deletion of cell `r` alone) and `g_{S, r} = u_S(e_r)` (mean transplant closure of cell `r` alone). The exact two-player transportation linear programme (`baselines.decomposition.assign`, HiGHS) maximises `Σ_r (A_{R, r} g_{R, r} + A_{S, r} g_{S, r})` under row budgets `a · R` and column exclusivity `A_{R, r} + A_{S, r} ≤ 1`. Comparing `cdea_shift` against `shift_assign` isolates what joint multi-cell optimisation adds over additive per-cell shift scores.
- **First-order gradient assignment** (ablation `S5_first_order`). The unit gradients of `u_R` and `u_S` at the empty mask, filled under the two budgets without giving a unit to both players.

## Known limit

On ColoredMNIST the colour lives on the digit's pixels, so `R` and `S` compete for the same units. The paper states this by construction. ColoredMNIST stays in the study and is labelled non-spatial.

## Ablations

- `S1_deletion` replaces the transplant payoff with deletion of the same units in every environment.
- `S2_independent` drops the column constraint. Each player has its own empty row.
- `S3_mean` replaces the min over environments with the mean.
- `S4_uniform` starts from a uniform prior.
- `S5_first_order` is the first-order gradient solution above.
- `S6_precomputed` scores maps written by the earlier objective on tag `framing-v1`.
- `S7_fixed_boundary` turns off the boundary-robust pixel offsets (`boundary_shift=False`).
- `S8_fractional` evaluates payoffs on the continuous Sinkhorn plan (`binary_masks=False`).

## Degenerate routes (`scripts/check_degenerate.py`)

- **D1 and D5** (gate): marginals within `1e-3` (asserted inside Sinkhorn).
- **D2** (gate): the soft plan and the hard top-`a` mask differ by at most 0.5 nats on average for both `R` and `S` (`payoff_soft_{r,s} - payoff_hard_{r,s}`).
- **D4** (gate): `S` beats a translated copy of itself on val `ΔD` (`delta_d > delta_d_translated`).
- **D3** (reported): the share of `u_S` that comes from the transplant crossing past `m(x_e)` (`overshoot_share`).
- **D9** (reported): the share of images whose final top cells equal the step-0 plan's (`same_as_init`).
