# Equation audit

CDEA means **Contrastive Decomposition via Evidence Allocation**. The method minimises one joint loss by Adam on the mask logits for a fixed number of steps. The classifier weights stay frozen. "Game" names that identical-interest loss. The procedure does not solve for an equilibrium and does not prove that overlap falls.

`docs/GAMBIT_PAPER.md` and `docs/CDEA_CONTRASTIVE_PAPER.md` are old drafts. They are not edited here. They still use a different expansion of CDEA and the older method language. This file is the formulation source.

## Contrastive loss

One scalar, minimised. File: `instantiations/contrastive/objective.py`.

| Quantity | Definition | Where |
|---|---|---|
| Kept logit | Raw logit \(z_k\) of class \(k\) on `keep(x, m_unique_k + m_shared)`. Not baseline-subtracted. | line 144 |
| `suff` | Alias of the batch-mean kept logit. Equal to `kept_logit`. | lines 250–251 |
| Margin | Kept logit of \(k\) minus the strongest other valid hypothesis under the same keep. | lines 148–157 |
| Overlap | Each unordered pair of unique masks once: \(\tfrac{1}{2}(\sum_{k,l} m_k\cdot m_l - \sum_k m_k\cdot m_k)\), then a batch mean. | `pairwise_overlap`, lines 22–29; used at line 159; weighted at line 242 |
| Unique sparsity | Mean L1 of the unique masks. | line 162, weighted at line 243 |
| Shared sparsity | L1 of the shared mask, divided by `mass_scale`. Weight default `0.25`. | lines 19, 42, 195–197, 244 |
| Mass | Unique-mask mass against `evidence.sum(-1) * mass_scale`. `mass_scale = max(1, R / 49)`. | lines 194–201, weighted at line 245 |
| Loss | \(-(λ_{suff}\,\text{kept logit} + λ_{margin}\,\text{margin}) + λ_{overlap}\,\text{overlap} + λ_{sparse}\,\text{sparse} + λ_{shared}\,\text{shared sparse} + λ_{mass}\,\text{mass}\). | lines 240–246 |

The allocator does not add overlap a second time. `OptimizationAllocator` rejects a non-zero `lambda_disjoint` (`instantiations/contrastive/allocator.py` lines 48–53). The only extra term it can add is the partition penalty, `relu(sum of masks − 1)`, which is a different quantity (`allocator.py` lines 13–18 and 157–159). Contrastive presets set `lambda_disjoint` to 0 (`core/game_modes.py` lines 35, 44, 53). Manual mode rejects any other value (line 134). Competitive and mixed still differ through `lambda_overlap` (0.35 and 0.2), the margin weight, and whether a shared mask is used.

The allocation loop freezes the classifier, runs Adam on the mask logits, adds the partition term when its weight is positive, steps, clamps the logits, and restores `requires_grad` (`allocator.py` lines 128–168). There is no learning-rate schedule on the masks.

## Shift loss

One scalar, minimised. File: `instantiations/shift/objective.py`. Overlap is the single product \(m_{robust}\cdot m_{shortcut}\) (line 41), weighted by `lambda_disjoint` (line 90). The shift allocator checks that its weight matches the objective and does not add the product again (`instantiations/shift/allocator.py` lines 51–54 and 98–100). Shift presets keep a non-zero disjoint weight (`core/game_modes.py` lines 67 and 87).

| Quantity | Definition | Where |
|---|---|---|
| Baseline-subtracted kept logit | \(z(\text{keep}) - z(\text{keep of an empty mask})\) for the chosen class. | lines 59–63 |
| Robust mean / variance | Mean and variance of that quantity across environments. | lines 73–74 |
| Shortcut gap | In-distribution shortcut score minus the mean of the other environments. | lines 76–78 |
| Mass | Each mask is pulled toward `mass_scale = max(1, R / 49)`. On a 7×7 grid the target is 1.0 per mask. `lambda_mass` defaults to 0.1. | lines 18, 81–86, 92 |
| Loss | Reward terms are negated. Disjoint, sparse, and mass are added, then the batch is averaged. | lines 88–96 |

This mass target is an absolute budget of one evidence-unit, scaled with the grid. The contrastive mass target is the base-evidence mass, scaled the same way. They share `lambda_mass` and `mass_ref_regions`. They do not share a formula.

## Reporting

`scripts/ablation_contrastive.py:compute_metrics` (lines 263–314) uses the same `pairwise_overlap`. It reports `kept_logit` and stores the same number under `suff`. `baseline_subtracted_kept_logit` is the shift quantity and is not called suff.

Script defaults that used to pass `lambda_shared_sparse=0` now pass `0.25`, so a run that omits the flag uses the class default. Contrastive `--lambda_disjoint` defaults to 0. A non-zero value is rejected.

## What this invalidates

- Contrastive runs that applied both `lambda_overlap` and the allocator's disjoint penalty. That penalty was the same pair sum, accumulated over the batch rather than averaged, so it was about one batch larger than the objective term.
- The ablation `suff` column. It was \(z_k - z(\text{zeros})\). It is now the raw kept logit. The subtracted value is `baseline_subtracted_kept_logit`.
- The ablation `overlap` column. It was the full off-diagonal sum. It is now the once-per-pair sum.
- Any shared mask trained at `lambda_shared_sparse=0`. At 0 the shared mask blankets the frame (degenerate route D1). The measured closure, recorded in the comment at `objective.py` lines 168–184, is weight 0.25.

Pre-audit files under `results/` stay archived. Do not quote them.
