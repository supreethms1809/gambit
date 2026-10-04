# Equation audit

CDEA means **Contrastive Decomposition via Evidence Allocation**. The method minimises one joint loss by Adam on the mask logits for a fixed number of steps. The classifier weights stay frozen. "Game" names that identical-interest loss. The procedure does not solve for an equilibrium and does not prove that overlap falls.

`docs/GAMBIT_PAPER.md` and `docs/CDEA_CONTRASTIVE_PAPER.md` are old drafts. They are not edited here. They still use a different expansion of CDEA and the older method language. This file is the formulation source.

## Contrastive loss

One scalar, minimised. File: `instantiations/contrastive/objective.py`.

| Quantity | Definition | Where |
|---|---|---|
| Kept logit | Raw logit \(z_k\) of class \(k\) on `keep(x, m_unique_k + m_shared)`. Not baseline-subtracted. | line 150 |
| `suff` | Alias of the batch-mean kept logit. Equal to `kept_logit`. | lines 270–271 |
| Margin | Kept logit of \(k\) minus the strongest other valid hypothesis under the same keep. Rows with no valid foil report 0, not \(z_k - (-\infty)\). | lines 154–170 |
| Overlap | Each unordered pair of valid unique masks once: \(\tfrac{1}{2}(\sum_{k,l} m_k\cdot m_l - \sum_k m_k\cdot m_k)\), then a batch mean. Invalid rows are zeroed first. | `pairwise_overlap`, lines 22–29; used at line 176; weighted at line 262 |
| Unique sparsity | Mean L1 of the valid unique masks, averaged over valid K. | line 179, weighted at line 263 |
| Shared sparsity | L1 of the shared mask, divided by `mass_scale`. Weight default `0.25`. | lines 19, 42, 212–214, weighted at line 264 |
| Mass | Valid unique-mask mass against `evidence.sum(-1) * mass_scale`. `mass_scale = max(1, R / 49)`. | lines 218–221, weighted at line 265 |
| Loss | \(-(λ_{suff}\,\text{kept logit} + λ_{margin}\,\text{margin}) + λ_{overlap}\,\text{overlap} + λ_{sparse}\,\text{sparse} + λ_{shared}\,\text{shared sparse} + λ_{mass}\,\text{mass}\). | lines 260–266 |

The allocator does not add overlap a second time. `OptimizationAllocator` rejects a non-zero `lambda_disjoint` (`instantiations/contrastive/allocator.py` lines 64–69). The only extra term it can add is the partition penalty, `relu(sum of masks − 1)`, averaged over batch and regions — a different quantity (`allocator.py` lines 14–21 and 179). Contrastive presets set `lambda_disjoint` to 0 (`core/game_modes.py` lines 35, 44, 53). Manual mode defaults an omitted value to 0 and rejects any other value (lines 127–137). Competitive and mixed still differ through `lambda_overlap` (0.35 and 0.2), the margin weight, and whether a shared mask is used.

The allocation loop freezes the classifier, holds it in eval mode (`core/eval_mode.py`, applied at `allocator.py` line 154), runs Adam on the mask logits, adds the partition term when its weight is positive, steps, clamps the logits, and restores `requires_grad` (`allocator.py` lines 146–186). There is no learning-rate schedule on the masks. Interaction attention defaults to ignored on both sides (`attn_mix=0.0`, `attn_weight_blend=0.0`): the stock modules are untrained random projections, so nonzero blends are opt-in only.

## Shift loss

One scalar, minimised. File: `instantiations/shift/objective.py`. Overlap is the single product \(m_{robust}\cdot m_{shortcut}\) (line 41), weighted by `lambda_disjoint` (line 90). The shift allocator checks that its weight matches the objective and does not add the product again (`instantiations/shift/allocator.py` lines 57–60 and 98–100). The shortcut init carries a deterministic ±0.25-logit seeded jitter so the two masks do not start identical. Shift presets keep a non-zero disjoint weight (`core/game_modes.py` lines 67 and 87).

| Quantity | Definition | Where |
|---|---|---|
| Baseline-subtracted kept logit | \(z(\text{keep}) - z(\text{keep of an empty mask})\) for the chosen class. | lines 59–63 |
| Robust mean / variance | Mean and variance of that quantity across environments. | lines 73–74 |
| Shortcut gap | In-distribution shortcut score minus the mean of the other environments. | lines 76–78 |
| Mass | Each mask is pulled toward `mass_scale = max(1, R / 49)`. On a 7×7 grid the target is 1.0 per mask. `lambda_mass` defaults to 0.1. | lines 18, 81–86, 92 |
| Loss | Reward terms are negated. Disjoint, sparse, and mass are added, then the batch is averaged. | lines 88–96 |

This mass target is an absolute budget of one evidence-unit, scaled with the grid. The contrastive mass target is the base-evidence mass, scaled the same way. They share `lambda_mass` and `mass_ref_regions`. They do not share a formula.

## Unpaired group statistics

`GroupStatisticsObjective` in the same file. Images are not paired across environments. `group` gives each image one group id. The kept-logit quantity is the paired one: baseline-subtracted, for the chosen class.

| Quantity | Definition | Where |
|---|---|---|
| Per-group mean | Mean of the baseline-subtracted kept logit inside each group. | lines 174–175, 201–206 |
| Variance across group means | Population variance of those means. Robust variance is penalised. Shortcut variance is rewarded through `lambda_gap`. | lines 177–178, 187 |
| Loss | The reward is negated. Disjoint, sparse, and mass use the paired formulas, then the batch is averaged. | lines 180–190 |

`lambda_gap` here is the weight on shortcut variance. It is not the paired in-distribution gap. The mass target is unchanged.

## Reporting

`scripts/ablation_contrastive.py:compute_metrics` (lines 263–314) uses the same `pairwise_overlap`. It reports `kept_logit` and stores the same number under `suff`. `baseline_subtracted_kept_logit` is the shift quantity and is not called suff.

Script defaults that used to pass `lambda_shared_sparse=0` now pass `0.25`, so a run that omits the flag uses the class default. Contrastive `--lambda_disjoint` defaults to 0. A non-zero value is rejected.

## What this invalidates

- Contrastive runs that applied both `lambda_overlap` and the allocator's disjoint penalty. That penalty was the same pair sum, accumulated over the batch rather than averaged, so it was about one batch larger than the objective term.
- The ablation `suff` column. It was \(z_k - z(\text{zeros})\). It is now the raw kept logit. The subtracted value is `baseline_subtracted_kept_logit`.
- The ablation `overlap` column. It was the full off-diagonal sum. It is now the once-per-pair sum.
- Any shared mask trained at `lambda_shared_sparse=0`. At 0 the shared mask blankets the frame (degenerate route D1). The measured closure, recorded in the comment at `objective.py` lines 168–184, is weight 0.25.

Pre-audit files under `results/` stay archived. Do not quote them.
