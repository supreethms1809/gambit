# CDEA formulation

**Status.** Implemented in `cdea/`. This is the contrast game. Since 2026-10-14 it is ablation DA2 of the decomposition game (`FORMULATION_DECOMPOSITION.md`), which is the method. The decomposition reuses this file's setting (section 2), deletion (section 4.1), constraints (section 5) and optimisation machinery (section 7). Ablation A8 scores precomputed maps from the earlier formulation (tag `framing-v1`).

## 1. Design rules

This formulation and the decomposition game follow four rules:
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

### 4.1 Boundary-robust payoff (adopted 2026-10-11)

**Why.** In the seed-0 val diagnosis on CIFAR-10 and HAM10000 (ResNet-50), CDEA's optimised payoff did not survive moving the mask boundary by less than a cell. On hard top cells under blur, CDEA's CD beat margin Grad-CAM's on the same cells. It fell behind on the mask the scorer uses: bilinear top-a, which covers the same cells give or take half a cell. Margin Grad-CAM gained under the same change. The evidence a player claims should not depend on where the cell border falls.

**Definition.** Let T_δ move the upsampled mask M̃ by an integer pixel offset δ, with zero fill at the image border, so nothing outside the image is deleted:

  x ⊖_δ A = (1 − T_δ M̃) ⊙ x + T_δ M̃ ⊙ b(x)

Each payoff of section 4 becomes its expectation over offsets:

  ū_p(A_p) = E_{δ ~ U(Δ)} [ payoff of p under x ⊖_δ A_p ],  Δ = {−s, …, s}²

- s = ⌊½ unit side⌋: 16 px at 7×7/224, 8 px at 14×14. The scored mask differs from the cell mask by at most about half a cell, which is why s takes this value. It is not tuned.
- δ = 0 is in Δ.
- The constraints, the problem (section 6) and the output are unchanged. The output is the unshifted plan.

**Optimisation.** One offset per player per step, drawn from a generator with a fixed seed that does not see the batch. Every image in a step uses the same offsets, so an image's allocation still does not depend on n or on chunking. Cost per step is unchanged.

**Status.** Adopted by user decision on 2026-10-11. The contrast game optimises the section-4 payoffs in this boundary-robust form. The decomposition game deletes with the same offsets (`FORMULATION_DECOMPOSITION.md` section 3.1). The fixed-boundary payoff is ablation A10, and the draw seed is ablation A11.

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
- **Binary draws** (adopted 2026-10-11; fractional masks are ablation A12). At each step, the payoffs are evaluated on a binary allocation drawn from the plan.
  - Unit r goes to player p with probability A_{p,r}, or to no player with the rest of its column. Players are exclusive in every draw.
  - One draw serves every image in the step, and the gradient is straight-through.
  - The objective is then the expected payoff of binary masks, the kind that is scored.
  - Why: on the fractional plan, the optimiser rewarded spreading partial deletions over many cells, which no binary mask reproduces. D2 failed on CIFAR-10, and more steps made it worse.
  - The boundary offsets of section 4.1 and the draws come from one generator, seeded by the draw seed and never by the batch.
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
| Boundary offset s (section 4.1) | ⌊½ unit side⌋ | Fixed |
| Draw seed (offsets of section 4.1 and binary draws of section 7) | 0 | Fixed. Seeds 1 and 2 are ablation A11 |

The default configuration is: Grad-CAM initialisation on ResNet-50, T = 100, η = 0.1, with the boundary-robust payoff of section 4.1.

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

## 10. Relation to the earlier formulation

The earlier formulation (tag `framing-v1`) had seven loss weights, each added to close a degenerate route. This formulation replaces them with the payoffs of section 4 and the constraints of section 5. Ablation A8 scores the earlier formulation's maps, and ablation A6 keeps its hard top-mass projection.

## 11. Degenerate routes

These redefine D1–D5 for the contrastive game. D6 and D7 belong to the shift plan.

- **D1** (gate). The shared row's marginal equals its budget within 1e-3. It is closed by construction, and the check asserts it.
- **D2** (gate). The gap between the soft and hard payoffs: payoff with the soft A_k, minus payoff with its top-a binarisation, under blur. Closed when at most 0.5 nats on average, for ranks 0 and 1.
- **D3** (reported, not a gate). The share of u_k that comes from rivals rising.
- **D4** (gate). Arbitrary cells. Closed when the unique masks beat a translated copy of themselves on val CD@5% (mean paired difference > 0).
- **D5** (gate). Every row and column marginal is within 1e-3.

The final allocation entropy is reported, to show how sharp the allocations are.

## 12. What this formulation supports

As ablation DA2, the contrast game is reported, not tested:
- CD (its claim C1) is reported as SPEC_k + SPEC_l for every method (`EVAL_PLAN.md` 13.1);
- its claims C2–C5 are descriptive rows (`EVAL_PLAN.md` section 1).

It does not support sufficiency (not optimised; measured), localisation, equilibrium or convergence properties, or usefulness to people.

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
