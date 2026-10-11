# Shift formulation

Status: implemented in `cdea/shift.py`. The contrastive solver is unchanged. This file has no version name.

## Question

Which evidence does the prediction need in every environment (robust), and which evidence explains why the prediction changes between environments (shortcut)?

## Notation

`y = argmax z(x_id)`. Log-odds of `y`:

`m(x) = z_y(x) - LSE_{j ≠ y} z_j(x)`.

Non-saturating, and invariant to a shift shared by every logit.

`E` is every environment, including id. `E'` is `E` without id. `g_e = m(x_id) - m(x_e)`.

Units, the hard cell indicator `φ_r`, and the deletion baseline `b(x)` are `core/grid.py`. The same blur is the scorer's blur.

## Players and payoffs

Both payoffs are in nats. There are no weights.

Robust `R`, deletion, worst case over environments:

`u_R(A) = min_{e ∈ E} [ m(x_e) - m(x_e ⊖ A) ]`.

Shortcut `S`, transplant. The gap closed by copying units from `x_e` into `x_id`:

`x_id^{A ← e} = (1 - Ã) ⊙ x_id + Ã ⊙ x_e`,

with `Ã` built from the hard cell indicators, and

`u_S(A) = mean_{e ∈ E'} [ |g_e| - |m(x_id^{A ← e}) - m(x_e)| ]`.

Units that are identical across environments earn exactly 0, so `S` cannot win by deleting the object. Background that supports `y` in id and opposes it in `e` cannot be robust, because the min over environments rejects it.

## Constraints and objective

Rows are `{R, S, ∅}`. Each unit's column sums to 1. Each player's row sums to `a · R`, so `a ≤ 1/2`. The empty row receives the rest.

Same Sinkhorn, same prior initialisation, same Adam, same per-image summed loss, and the same per-area solve as contrastive. The objective is to maximise `u_R + u_S`.

The prior is sum-normalised class evidence. Robust evidence is the minimum over environments. Shortcut evidence is the mean absolute difference between the id map and each other environment. The log-initialisation uses `ε = 1e-6`. The default evidence is Grad-CAM on ResNet-50. On ViT-B/16 the default is integrated gradients with the blur baseline, because Grad-CAM on the last block is zero. Uniform initialisation is an ablation.

## First-order solution

Integrated gradients with baseline `x_e`, on the path `x_e → x_id`, attributes `g_e` completely. Pooled to units and signed by `g_e`, it is the path-integrated first-order solution of `u_S`. That map is the core baseline called gap attribution. Ablation S5 reports this allocation's own first-order solution: the unit gradients of `u_R` and `u_S` at the empty mask, filled under the two budgets without giving a unit to both players.

## Known limit

On ColoredMNIST the colour lives on the digit's pixels, so `R` and `S` compete for the same units. The paper states this by construction. ColoredMNIST stays in the study and is labelled non-spatial.

## Ablations

- S1 replaces the transplant payoff with deletion of the same units in every environment.
- S2 drops the column constraint. Each player has its own empty row.
- S3 replaces the min over environments with the mean.
- S4 starts from a uniform prior.
- S5 is the first-order solution above.
- S6 scores maps written by the earlier objective on tag `framing-v1`. Those maps are read. The earlier objective is not in this tree.
- S7 compares step counts with per-environment Extremal Perturbations at a matched number of passes.

## Degenerate routes

- D1 and D5, gate: marginals within `1e-3`.
- D2, gate: the soft plan and the hard mask differ by at most 0.5 nats.
- D4, gate: `S` beats a translated copy of itself on val `ΔD`.
- D3, reported: the share of `u_S` that comes from the transplant crossing past `m(x_e)`.
