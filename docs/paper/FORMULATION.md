# CDEA formulation

**Status.** Implemented in `cdea/`. It replaces the earlier formulation (tag `framing-v1`). Ablation A8 scores precomputed maps from that tag. `EVAL_PLAN.md` evaluates this formulation. `EQUATION_AUDIT.md` maps each equation to the code.

## 1. Why the earlier formulation (tag `framing-v1`) is replaced

The earlier-formulation val read: seed 0, all 8 contrastive datasets, both backbones, n = 64 per cell, read 2026-10-09.

- **Result.** The earlier formulation (tag `framing-v1`) trailed the better margin-attribution variant on CD@5% in all 16 cells. The 95% interval was below zero in 13 of them.
- **Measurement caveat.** 15 of those 16 CDEA rows carry the chunk-scale bug fixed in PR #28. The diagnosis below does not rest on those rows alone.

| Cause | Measurement |
|---|---|
| The sufficiency term rewards evidence the classes share | Removing CDEA's foil mask drops z_k by about as much as removing per-class Grad-CAM's does. CIFAR-100 ResNet-50: CDEA 5.68, per-class Grad-CAM 4.52, margin Grad-CAM 3.63. ImageNet: 2.03, 2.04, 0.45. These numbers come from the seed-0 records. |
| The masks cannot leave their initial support | The hard projection `(logits − τ).clamp(0, 1)` uses a detached τ, so it gives zero gradient to any unit pinned at 0. 96% of the final rank-0 support, and 81% of the rank-1 support, lay inside the support of the initial evidence. Measured on 16 Oxford Pets images with the ImageNet ResNet-50, 50 steps, on `main`, without chunking. |
| The split between shared and unique evidence is defined only by penalties | The earlier formulation (tag `framing-v1`) has seven loss weights: λ_suff, λ_margin, λ_sparse, λ_overlap, λ_mass, λ_shared_sparse and λ_partition. Each was added to close a degenerate route. No single quantity defines "shared" or "unique". |
| A mask depends on how many images share the batch | The loss is a batch mean, and Adam's ε makes its scale visible. |

this formulation follows four design rules:
1. Each quantity has one definition, derived from the question.
2. Constraints replace penalties.
3. There are no loss weights.
4. The objective is not the scoring operator. CDEA deletes with blur. The paper scores with ROAD.

## 2. Setting

- **Classifier.** The classifier f gives logits z(x) ∈ ℝ^C for an input x ∈ [0, 1]^{3×H×W}.
- **Units.** There are R grid cells: 7×7 on ResNet-50 and 14×14 on ViT-B/16, from `grid_of`. φ_r is unit r's pixel indicator, upsampled by the existing unit space.
- **Hypotheses.** H is the model's top K = min(5, C) classes on the full image. The pair scored by CD is k = rank 0 and l = rank 1.
- **Deletion.** x ⊖ M = (1 − M̃) ⊙ x + M̃ ⊙ b(x), where M̃ = Σ_r M_r φ_r and b is the Gaussian blur of `VisionGridUnitSpace.remove`.

## 3. The question and its decomposition

"Why k rather than the alternatives" has a standard reading: Lipton's difference condition, and Miller (2019). The evidence is what the preference **counterfactually depends on**: removing it removes the preference. That is necessity, so this formulation uses deletion.

For k ∈ H, the chain rule splits the model's belief in two:

  log p(k | x) = log p(H | x) + log p(k | H, x)

Each factor has a log-odds that does not saturate when the model is confident:

- **Shared.** s_H(x) = LSE_{j∈H} z_j − LSE_{j∉H} z_j. This is the log-odds that the answer lies in H at all: why one of these K rather than any other class.
- **Unique.** c_k(x) = z_k − LSE_{j∈H∖{k}} z_j. This is the log-odds of k within H: why k rather than the other candidates.

Both are in nats. With K = 2, c_k = z_k − z_l, the exact quantity CD scores.

When H covers every class (C ≤ 5, which here is only brain tumor, C = 3), s_H is undefined, and there is no shared player.

## 4. Players and payoffs

Each player p owns a soft unit set A_p ∈ [0, 1]^R. Its payoff is the log-odds it loses when its units are deleted:

  u_k(A_k) = c_k(x) − c_k(x ⊖ A_k), for k ∈ H
  u_S(A_S) = s_H(x) − s_H(x ⊖ A_S)

Consider a unit that supports k and its rivals equally. Deleting it lowers their logits together, so c_k barely moves: it is worth nothing to player k. It does lower s_H, so it is worth something to the shared player. **The decomposition routes common evidence to S and discriminative evidence to the unique players by itself.** No overlap penalty does this.

A unit whose deletion raises a rival also counts toward u_k. That unit was suppressing the rival, which is legitimate evidence for k over that rival. The share of u_k that comes this way is reported (route D3).

## 5. Constraints

The allocation A ∈ ℝ_{≥0}^{(P+1)×R} has one row per player and one "unallocated" row ∅. P = K + 1 when the shared player exists, and P = K otherwise.

- **Each unit is fully distributed:** Σ_rows A_{·,r} = 1. Entries are therefore at most 1. A unit cannot be both shared and unique, or unique to two hypotheses.
- **Budget:** Σ_r A_{p,r} = a·R for every player, where a is the scored area. ∅ receives R(1 − P·a).
- **Feasibility:** P·a ≤ 1. With K = 5 and a shared player, a ≤ 1/6, which covers the scored areas {2.5%, 5%, 10%}.

## 6. The problem

  maximise over A:  Σ_{k∈H} u_k(A_k) + u_S(A_S),  subject to A satisfying section 5

There are no weights. Every payoff is a log-odds in the same unit.

The word "game" means what it means here and nothing more: players with their own payoffs share a constrained resource, and the welfare sum is maximised. No equilibrium or convergence property is claimed (`FRAMING.md`).

## 7. Optimisation

- **Parameterisation.** A = Sinkhorn(exp θ), with column marginals 1 and the row marginals of section 5. Sinkhorn runs in the log domain for I = 20 iterations, unrolled for backpropagation. exp θ is positive everywhere, so **every unit receives gradient**, which removes the earlier formulation (tag `framing-v1`)'s trapped support. Any temperature is absorbed into θ.
- **Initialisation from base evidence E,** normalised per hypothesis (Grad-CAM on ResNet-50, IG on ViT):
  - θ_k = log(E_k + ε);
  - θ_S = log(mean_{j∈H} E_j + ε);
  - θ_∅ = log(1/R);
  - ε = 1e-6.

  The evidence is a prior. Ablation A5 measures how much the result depends on it.
- **Steps.** Adam on θ, T steps, learning rate η.
- **Loss.** The loss is −Σ_images Σ_players u, summed over images, not averaged. An image's allocation therefore does not depend on n or on chunking.
- **Area.** The problem is solved separately at each scored area, as Extremal Perturbations is.
- **Cost.** One base-evidence pass, then T × P forward and backward passes per image per area: one deleted input per player per step.
- **Output.** A_k for every k ∈ H, and A_S. They go through the same top-a conversion as every baseline (`baselines.adapter.adapt_scores`).

## 8. Hyperparameters

| Name | Value | Status |
|---|---|---|
| K | min(5, C) | Fixed (shared rule) |
| a | The scored area | Fixed by the evaluation |
| T | {50, 100} | Selected on val (ViT) |
| η | {0.05, 0.2} | Selected on val |
| Evidence backend for initialisation | {Grad-CAM, IG} on ResNet-50, IG on ViT | Selected on val |
| Sinkhorn iterations I | 20 | Fixed. Marginal error ≤ 1e-3 is asserted |
| Blur in b(x) | The existing unit-space default | Fixed |
| ε | 1e-6 | Fixed |

The default for the G1 gate is: Grad-CAM initialisation on ResNet-50, T = 100, η = 0.1.

## 9. Relation to the baselines

- **Margin attribution is the first-order solution.** Linearise c_k around no deletion:

    c_k(x ⊖ A) ≈ c_k(x) − Σ_r A_r g_{k,r},  with  g_{k,r} = ⟨∇_x c_k(x), (x − b(x)) ⊙ φ_r⟩

  So u_k ≈ Σ_r A_{k,r} g_{k,r}. Under the budget, the first-order optimum gives player k its top units by g_k.

  With K = 2, g_k is gradient × (input − blur baseline) of z_k − z_l, pooled to units: a one-step margin attribution with a blur baseline. With K = 2, c_l = −c_k, so g_l = −g_k, and the two top sets are disjoint without exclusivity.

  **So CDEA and margin attribution agree wherever the margin is additive over units.** They can differ only through interactions (units that are redundant, or necessary only jointly), through exclusivity, and through K > 2. Ablation A9 reports this first-order solution, so the gain from optimisation is measured rather than assumed.
- **Contrastive Extremal Perturbations, deletion variant.** One player, one pair, smooth pixel-level masks, no exclusivity and no shared evidence. It uses the same family of blur deletions. CDEA against deletion EP isolates what allocation adds: the players, exclusivity, and the units.
- **Extremal Perturbations, preservation variant, and the earlier formulation (tag `framing-v1`).** These answer the sufficiency question. They are the comparators for the sufficiency score, not for CD.
- **CVE and SC-CVE.** They replace query cells with distractor cells in feature space, which is a counterfactual edit rather than a deletion. Their maps are one-sided.
- **SCOUT.** Not run. The reason is in `BASELINES.md`.

## 10. What earlier terms became

| earlier term | this formulation |
|---|---|
| λ_suff · z_k under keep | Removed. It answers "why k", not "rather than", and it is the measured source of the leak of shared evidence |
| λ_margin · keep margin against the strongest foil | Payoff u_k: the deletion log-odds within H |
| λ_overlap, λ_partition | The unit marginal (exclusivity) |
| λ_sparse, λ_mass, the hard 1-cell budget | The budget marginal at the scored area |
| Shared mask + λ_shared_sparse | Shared player S, with payoff from s_H |
| Hard top-mass projection | Sinkhorn projection (A6 keeps the old projection as an ablation) |
| Presets (cooperative / mixed / competitive), interaction modules | Retired. They were weight presets, and this formulation has no weights |
| Batch-mean loss | Per-image sum |

## 11. Degenerate routes

These redefine D1–D5 for the contrastive game. D6 and D7 belong to the shift plan.

- **D1** (gate). The shared row's marginal equals its budget within 1e-3. It is closed by construction, and the check asserts it.
- **D2** (gate). The gap between the soft and hard payoffs: payoff with the soft A_k, minus payoff with its top-a binarisation, under blur. Closed when at most 0.5 nats on average, for ranks 0 and 1.
- **D3** (reported, not a gate). The share of u_k that comes from rivals rising.
- **D4** (gate). Arbitrary cells. Closed when the unique masks beat a translated copy of themselves on val CD@5% (mean paired difference > 0).
- **D5** (gate). Every row and column marginal is within 1e-3.

The final allocation entropy is reported, to show how sharp the allocations are.

## 12. What this formulation supports

- **Supports:**
  - necessity of the contrast (claim C1);
  - alignment of that evidence with annotated discriminative structure and planted cues (C2, C3), tested against methods that do not optimise those metrics;
  - a decomposition into shared and unique evidence whose parts are defined by the chain rule and checked on the model (C4).
- **Does not support:**
  - sufficiency (not optimised; measured);
  - localisation;
  - equilibrium or convergence properties;
  - usefulness to people.

## Inherited defaults

Decided before the code. Nothing in this list is inherited silently from the earlier formulation (tag `framing-v1`).

| Default | Decision |
|---|---|
| Deletion baseline b(x) | Gaussian blur, σ = ½ unit side (16 px at 7×7/224, 8 px at 14×14), kernel 2⌈3σ⌉+1, reflect padding. One function in `core/grid.py`, also used by the scorer's blur operator |
| Unit indicator φ_r inside the method | Hard (cell-exact, nearest). Scoring of every method keeps the shared bilinear-plus-top-a conversion (`baselines.adapter.adapt_scores`) |
| Grad-CAM layer | Standard ResNet `layer4` output, for CDEA initialisation, base evidence, margin Grad-CAM, and contrastive Grad-CAM |
| IG for CDEA initialisation | Blur baseline b(x); the positive part is the prior. Margin IG keeps its dossier definition (zero baseline, signed) |
| Evidence prior | Per-hypothesis sum-normalised, used only as log-initialisation, ε = 1e-6 |
| Budget | a·R may be fractional |
| Adam | Default betas, ε = 1e-8 |
| Model during optimisation | Parameters frozen, `eval_mode`, explicit device |
| Hypotheses | K = min(5, C), no padded slots |

## 13. Implementation

- **Code:** `cdea/payoffs.py`, `cdea/sinkhorn.py`, `cdea/allocation.py`, `cdea/first_order.py`. Records go to `results/paper/cells/`. Ablation A8 reads precomputed maps.
- **Tests:** `tests/test_method.py` covers the list above.
- **Docs:** `EQUATION_AUDIT.md` maps each equation to that code.
