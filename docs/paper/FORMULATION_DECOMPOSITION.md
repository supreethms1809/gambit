# CDEA decomposition formulation

**Status (2026-10-20).** Version 2, **DRAFT**. Implemented on branch `paper/decomp-partition` (section 13). Not signed off.
- On `main`, `cdea/decompose.py` implements version 1 (`decompose`). Its text is at tag `decomp-declared`: `git show decomp-declared:docs/paper/FORMULATION_DECOMPOSITION.md`. Version 1 records are read under `EVAL_PLAN.md` 13.1–13.4. The branch adds version 2 (`partition`) beside it.
- Version 2's claims and tests are `EVAL_PLAN.md` 13.5–13.8, also a draft. No version 2 number is read before the user signs off both. On 2026-10-20 the user asked for a local dev read before the sign-off; `EVAL_PLAN.md` discloses it.
- Section 11 lists what changed from version 1 and why. Section 12 lists the open decisions.

The contrast game (`FORMULATION.md`) stays in the code unchanged. Its role in version 2 is open decision 4.

## 1. Why

The goal is the original one. When the evidence for each class is computed separately, the two maps often overlap. In those images, CDEA separates the overlapping evidence into evidence **unique** to each class and evidence **shared** by both, and every label is checked against the model.

Version 1 drifted from this goal in three ways (review of 2026-10-20):
- Each player owned a fixed 5% of the image, so "how much is shared?" was an input, not an answer. An image with no shared evidence still got an S of 5%.
- Players could take any cell of the image, so the output was not a division of the evidence that overlapped.
- No measure of the overlap existed, before or after.

Two facts shape version 2.

**Removing overlap is trivial, so it cannot be the success measure.**
- Any rule that divides two maps gives unique parts that do not overlap. The min-split rule does it at K = 2: at every cell one of E_k − m and E_l − m is 0.
- Subtracting the strongest rival does it at any K.
- The archived first formulation (tag `framing-v1`) reported a large overlap drop. It came from two things:
  - a baseline that subtracted the *mean* of four rivals over the top 5;
  - an objective that penalised the same overlap formula it was scored on.

**What a rule cannot do is check its labels.** Subtraction calls every overlapping cell shared. An overlapping cell can be:
1. evidence that both classes use;
2. an artefact of the attribution map;
3. redundant evidence that shows only when cells are deleted together;
4. a k-feature and an l-feature sitting in the same cell.

Deleting the cell separates case 1 from case 2. A joint search is needed only for case 3. No method that labels image cells can resolve case 4 (section 10).

## 2. Scope: one pair (K = 2)

- k = rank 0, l = rank 1.
- "Unique to k" means unique **against l**, not against every class.
- Rank 2, written m, is scored but is not a player (section 6.6). It answers two questions:
  - Is S specific to the pair, or generic evidence that raises other classes too?
  - How often is a third class close enough that a pair is not the whole story?
- K > 2 is future work (section 10).

## 3. The question

### 3.1 Reference evidence and the overlap

- E_k and E_l are the per-class evidence of `_prior`: Grad-CAM on ResNet-50, and IG with the blur baseline on ViT-B/16. They live on the grid of R cells (7×7 or 14×14).
- Ê_k and Ê_l are those maps clamped at 0 and sum-normalised.
- **Overlap per cell:** o(r) = min(Ê_k(r), Ê_l(r)).
- **Image overlap:** O = Σ_r o(r), in [0, 1].
  - O = 0: the maps share nothing.
  - O = 1: the maps are identical.
  - O is the mass of the min-split's S.

### 3.2 The evidence region Ω

- C_j(ρ) is the smallest set of top cells of Ê_j that holds at least ρ of its mass. Ties go to the lower cell index.
- Ω = C_k(ρ) ∪ C_l(ρ), with ρ = 0.5 (open decision 1).
- Every method divides Ω and only Ω. Cells outside Ω are never deleted and never labelled.

### 3.3 The output

A partition of Ω into four roles: U_k, U_l, S and ∅ (no evidence for either class). Role sizes are free. Any role may be empty, including S.

### 3.4 Drops and the null

- **Drop.** d_j(M) = z_j(x) − z_j(x ⊖ M), in logits. n = |M| is the number of cells in M.
- **Null.** ν_j(n) is the mean of d_j over N seeded random n-cell masks drawn from the **whole grid**, not from Ω. One set of masks serves every image, and ν_j(0) = 0.
- **Excess drop.** d̃_j(M) = d_j(M) − ν_j(|M|).

The null comes from the whole grid so that it keeps its version 1 meaning: the common-mode drop of deleting that many cells. A null drawn from Ω would measure "random evidence". A set with no effect on l would then read as d̃_l < 0, and the payoff would penalise it.

### 3.5 Roles, defined by deletion

| Role | Deleting it … |
|---|---|
| U_k, unique to k | lowers z_k and leaves z_l unchanged, in either direction |
| U_l, unique to l | the same, with k and l swapped |
| S, shared | lowers both z_k and z_l |
| ∅, neither | lowers neither |

Suppressing l is still not unique evidence for k (D3, settled in version 1). A region whose deletion raises z_l is penalised in u_k.

## 4. Method

### 4.1 Payoffs (`cdea/payoffs.py`)

The payoffs are version 1's, with the size-matched null:
- u_k = d̃_k(U_k) − |d̃_l(U_k)|_s
- u_l = d̃_l(U_l) − |d̃_k(U_l)|_s
- u_S = min_s(d̃_k(S), d̃_l(S))
- An empty role pays 0, and ∅ always pays 0.
- The welfare is W = u_k + u_l + u_S, with no weights.

**Entry threshold.** ν grows with n. So adding a cell raises u_k only if its marginal drop beats a random cell's. The size-matched null is the entry threshold, and no sparsity weight is needed.

**Known exploit (reported; open decision 3).** The set-level min and |·| reward mixing and cancellation:
- S = {a unique-k cell, a unique-l cell} gives min(Σ) > Σ min, so S looks shared.
- A U_k whose cells push z_l in opposite directions looks pure.

Fixed budgets made mixing bite in version 1, because S had to be filled. Free sizes remove that pressure. Section 6.3 measures what is left.

### 4.2 Plan and constraints

- θ ∈ R^{4 × |Ω|}. The plan is a softmax over the four roles in every cell, so exclusivity holds by construction.
- There are no row budgets, so there is no Sinkhorn step.
- Binary draws: one categorical draw per cell, with a straight-through gradient. The boundary offsets are version 1's (`FORMULATION.md` section 4.1).

### 4.3 Initialisation: the subtraction rule

With m = o(r) on Ω:
- θ_k = log(Ê_k − m + ε)
- θ_l = log(Ê_l − m + ε)
- θ_S = log(m + ε)
- θ_∅ = log ε
- ε = 1e-6

The hard plan at step 0 is the subtraction partition (section 5). So the gain over subtraction is measured directly.

### 4.4 Optimisation (`cdea/decompose.py`)

The machinery is unchanged:
- Adam;
- a per-image sum loss, so results do not depend on the chunk;
- one generator seeded by the draw seed, never by the batch;
- 100 steps, lr 0.1, and the version 1 selection grid;
- deletion by CDEA's blur.

The null is computed once, before optimisation, and detached, for n = 0 … |Ω|. That costs N·|Ω| forward passes per image. On ViT-B/16 it is computed on a size grid and interpolated (open decision 9).

### 4.5 Output

The hard partition: each Ω cell takes the role with the largest final plan entry.

### 4.6 Ablations

| Label | Change |
|---|---|
| DA1 | Independent roles: a Bernoulli per role and cell, so roles may overlap |
| DA3 | `gradient_partition` (section 5) |
| DA4 | Uniform initialisation |
| DA5 | Mean initialisation |
| DA6 | Fractional plan, scored at its argmax |
| DA7 | Fixed boundary |
| DA8 | No null |
| DA9 | Steps 25, 50, 200 |
| DA10 | **Version 1**: fixed 5% budgets anywhere in the image. Shows what the change of question costs or gains |
| DA2 | Open decision 4 |

## 5. Comparators (`baselines/decomposition.py`)

Every comparator divides the same Ω.

**Rule rows.**
1. A per-class method runs unmodified.
2. Its two maps are sum-pooled to the grid, clamped at 0, sum-normalised, and split by min-split.
3. Each Ω cell takes the largest of the three parts: U_k from Ê_k − m, U_l from Ê_l − m, S from m.

Rules never label a cell ∅.

| Row | Source | Status |
|---|---|---|
| `subtract` | The reference evidence itself, through the rule. CDEA's step 0 | Core. The "before → after by subtraction" answer |
| `ig_minsplit` | Captum IntegratedGradients, as version 1 | Core |
| `occlusion_minsplit` | Captum `Occlusion`, black, one cell, as version 1 | Core |
| `rise_minsplit`, `extremal_minsplit` | As version 1 | Extended |
| `occlusion_partition` | Ours. The single-cell blur drops g_{j,r} (excess over ν(1)) give cell scores s_k = g_k − \|g_l\|, s_l = g_l − \|g_k\|, s_S = min(g_k, g_l). A cell takes its best role if that score is > 0, else ∅. With free sizes this per-cell rule is the exact optimum of the additive objective, so no linear programme is needed | Core. Decides the framing |
| `gradient_partition` | The same, with the first-order g of version 1's `gradient_assign` | Ablation DA3 |
| `margin_gradcam_ref` | The margin map on Ω: positive cells go to U_k, negative cells to U_l, with no S | Reference |
| `random_partition` | Each Ω cell takes a seeded uniform role | Floor |

Post-2021 rule rows are open decision 8.

**Framing rule (unchanged in spirit).** If `occlusion_partition` matches CDEA within the declared margin (`EVAL_PLAN.md` 13.8), the contribution is the question, the scores and the ground truth, and the method is the per-cell partition.

## 6. Scores (`evaluation/scores.py`)

### 6.1 Masks and removal

- A role's mask is its cells, whole: nearest upsampling to pixels. The shared top-a adapter is not used.
- Removal is ROAD (primary) or blur (robustness), with version 1's seeds and offsets.
- **The scorer's null.** For every n that occurs in an image, the scorer computes ν^sc_j(n) and its 95th percentile q_j(n) over 32 seeded random n-cell masks from the whole grid. They are computed once per image and shared by every method.
- Drops are recorded for three classes: k, l and m.

### 6.2 Before

Per image: O, |Ω|, and O_Ω = Σ_{r∈Ω} o(r), the overlap inside the region every method divides.

### 6.3 After: scores per role

| Score | Definition | Reads as |
|---|---|---|
| PUR_k, PUR_l | d̃_k(U_k) − \|d̃_l(U_k)\|, and the mirror | Unique purity |
| SV | min(d̃_k(S), d̃_l(S)) | Shared validity |
| BAL | \|d̃_k(S) − d̃_l(S)\| | Shared balance, lower is better |
| PUR^cell, SV^cell | The same formulas applied to each cell of the role deleted alone, with ν(1), and summed. That needs \|Ω\| single-cell ROAD deletions per image, shared by every method | Versions that mixing and cancellation cannot inflate |
| MIX | SV − SV^cell | Large when S is a mixture of unique cells, or holds redundant evidence. Section 6.4 tells the two apart |
| Sizes | \|U_k\|, \|U_l\|, \|S\|, \|∅\| in cells | How much is shared |

A score on an empty role is NaN, and empty roles are counted.

### 6.4 Confirmation of each role

- count_j(M) means d_j(M) > q_j(|M|).
- U_k is confirmed when count_k holds and count_l does not. U_l is the mirror.
- S is confirmed when count_k and count_l both hold.
- ∅ is confirmed when neither holds.

This is `verify_roles`' rule applied to the method's own roles.

**Mixture or redundancy.** For an S with MIX > 0, split its cells into two seeded halves and delete each half.
- A mixture gives a half that counts for one class only.
- Redundancy gives halves that count for neither class.

### 6.5 The headline: how the overlap was resolved

Of the overlap inside Ω, the share that ends in each verdict:
- **shared, confirmed:** Σ_{r∈S} o(r) / O_Ω, when S is confirmed;
- **unique, confirmed:** the same sum over U_k and U_l, each counted when it is confirmed;
- **neither, confirmed:** the same sum over ∅, when ∅ is confirmed;
- **unconfirmed:** the rest.

**VR (verified resolution) = 1 − unconfirmed.**

The before/after figure shows, per method and per overlap tertile, O before and these four shares after. For the `subtract` row it reads: "the rule called this share of the overlap shared, and this part of it was confirmed".

### 6.6 Pair-specificity and the third class

| Quantity | Definition | Reads as |
|---|---|---|
| GEN_S | d̃_m(S) | Small: the shared evidence is specific to the pair. Large: it is generic, for example "dog" |
| PSV | SV − max(0, d̃_m(S)) | Shared validity, counting only the pair-specific part |
| d̃_m(U_k), d̃_m(U_l) | Drop of the third class | Unique against l; also unique against m? |
| K-sufficiency | Share of images with z_l − z_m below 0.5, 1 and 2 logits | A property of the data, not of a method. How often a pair is not the whole story |

### 6.7 Planted pairs

- **Ω coverage.** The share of planted patch cells inside Ω. It is a property of the reference evidence and is the same for every method.
- **Role accuracy RA.** For each planted patch (A_k, A_l, C), the share of its cells labelled with its role. RA is the mean over the three patches.
  - In "RA, all", patch cells outside Ω count as wrong.
  - In "RA, in Ω", they are excluded.
- **Confusion.** A 3 × 5 table: patch × {U_k, U_l, S, ∅, outside Ω}.

### 6.8 Overlap strata

O tertiles per dataset. The cut points come from val and are frozen with the plan. "High overlap" is the top tertile.

## 7. Ground truth: planted pairs (`evaluation/planted_cues.py`)

Unchanged from version 1: `PlantedPairCIFAR`, the three pairs, the training rule, grid placement, the sample rule and `verify_roles`. The checkpoints are version 1's.

One addition: report Ω coverage (6.7). A patch that the reference evidence misses cannot be labelled by any method, so it is a limit of the evidence, not of a method.

## 8. Degenerate routes (`scripts/check_degenerate.py`)

- **D2 (gate).** The soft-plan payoff minus the hard-partition payoff, under blur, is at most 0.5 on average.
- **D8 (reported).** The share of images whose S is empty, and the share whose S is non-empty but not confirmed.
- **D9 (reported).** The share of images whose final partition equals step 0, the subtraction rule.
- **MIX (reported).** Its distribution per method, with the split-half verdicts (6.4).
- **Trapped roles (reported).** The share of final labels whose initial logit was log ε, for example S where m = 0.
- D1 and D5, the Sinkhorn marginals, no longer apply: exclusivity holds by construction.

## 9. Tests (written with the implementation)

- Ω construction: ρ and ties.
- **Toy 1, additive.** Sizes are free. `subtract`, `occlusion_partition` and CDEA all return U_k = {1}, U_l = {2}, S = {3}, ∅ = {4}.
- **Toy 2, suppression.** As version 1.
- **Toy 3, redundancy.** The pair {a, b} belongs in U_k. `occlusion_partition` misses it, because each cell alone is below a random cell. CDEA finds it. This is the key test, as in version 1.
- **Toy 4, mixing (new).** There is no shared cell, so S must stay empty, or MIX and the split halves must flag it.
- **Toy 5, co-location (new).** One cell carries a k-only and an l-only feature. Every method that labels image cells calls it S. The test documents the limit in section 10.
- The size-matched null, chunk invariance, exclusive categorical draws, the score definitions, and the confirmation rule against `verify_roles`.

## 10. Limits and what lies outside the method

- **Co-located features (case 4).** No partition of image cells can split them. At ResNet-50's last layer an exact split exists by channel: unit (r, c) contributes w_kc·A_rc/49 to z_k and w_lc·A_rc/49 to z_l. The **co-location share** compares two totals:
  - shared evidence computed per position, Σ_r min(c_k(r), c_l(r));
  - the same computed per position and channel, Σ_{r,c} min(·).

  The share is the gap between them divided by the first. It measures how much "shared" evidence is really unique evidence in the same place. It is a planned descriptive measurement (open decision 10), not part of the method.
- **Redundancy is the only reason for a search.** At ResNet-50's head, contributions add up exactly, so non-additivity can only come from the convolutional trunk. Measure it with the pairwise interaction I(a, b) = d({a, b}) − d(a) − d(b) on the top Ω cells (open decision 10).
- **K = 2** (section 2). A K-class version would give each class subset T a role, with u_T = min_{j∈T} d̃_j − Σ_{j∉T} |d̃_j|. Unique evidence is the case |T| = 1. The version 2 S is u_{k,l} without the outside penalty, which section 6.6 measures instead.
- **The answer is relative to the reference evidence.** Ω comes from Grad-CAM (ResNet-50) or IG (ViT-B/16).

## 11. Changes from version 1

| Aspect | Version 1 (tag `decomp-declared`) | Version 2 | Why |
|---|---|---|---|
| Question | Three fixed-size regions anywhere in the image | A partition of the evidence region Ω | The original goal: divide the overlapping evidence |
| Sizes | a·R cells per player | Free; S may be empty | "How much is shared" becomes an answer |
| Roles | 3 | 4 (∅ added) | Evidence cells may serve neither class |
| Null | ν(a) at one area | ν(n) at each role's size, from the whole grid | Free sizes |
| Plan | Sinkhorn with row budgets | A softmax per cell | No budgets |
| Scored masks | The top-a adapter at matched area | Whole cells, no adapter | A partition is already a set of masks |
| Scores | SPEC, PUR, SV, BAL, SEP, RR@2.5% | O and O_Ω before; per-role scores with the size-matched null; cell-level scores; MIX; confirmations; VR and the resolution shares; the third class; RA | Measure overlap before and verified separation after, and expose mixing |
| SPEC and CD | CD reported as SPEC_k + SPEC_l | Not reported | They compare masks of different sizes |
| Third class | Not scored | m is scored | Pair-specific or generic S; how often K = 2 falls short |
| Additive comparator | `occlusion_assign` (a budgeted linear programme) | `occlusion_partition` (best role per cell) | No budgets |
| Version 1 itself | The method | Ablation DA10 | Shows what the change costs or gains |

Carried over from version 1's decisions (2026-10-14): logits as units (3), the suppression penalty (2), binary draws by default (6), and the planted pairs (7). Version 1's text, including its decision table and its implementation notes, is at tag `decomp-declared`.

## 12. Open decisions

| # | Decision | Proposed |
|---|---|---|
| 1 | ρ for Ω | 0.5, with 0.3 and 0.7 reported |
| 2 | Reference evidence | Grad-CAM on ResNet-50 and IG on ViT-B/16. IG on ResNet-50 as a robustness row |
| 3 | Mixing in S | Report MIX and the split halves. Add a penalty only if the dev read shows mixing |
| 4 | DA2, the contrast game | Drop it from version 2. It keeps fixed budgets, so it cannot divide Ω |
| 5 | Primary natural-image metric | VR on the top overlap tertile. PUR + SV with the size-matched null as the secondary |
| 6 | Strata | O tertiles per dataset, cut on val |
| 7 | DA10 (version 1 as an ablation) | Keep |
| 8 | Post-2021 rule rows | Chefer min-split on ViT-B/16 (its class maps exist). AttnLRP if time allows |
| 9 | ViT-B/16 null cost | A size grid with linear interpolation |
| 10 | Descriptive measurements outside the method | Run the co-location share and the pairwise interaction on dev val first. Both bear on the framing |
| 11 | Sign-off | Version 2 and `EVAL_PLAN.md` 13.5–13.8 are signed off together, before any version 2 number is read |

## 13. Implementation notes (2026-10-20, branch `paper/decomp-partition`)

The code follows sections 3 to 9 except where noted here.
- **Method.** `cdea/region.py` (Ω and the rule), `cdea/null.py` (`size_null`) and `cdea/decompose.py` (`partition`).
- **Comparators.** `baselines/decomposition.py`.
- **Scores.** `evaluation/scores.py` (`partition_scores`).
- **Runner and readouts.** `evaluation/run_partition.py` (`paper_run.py --game partition`), `scripts/partition_readout.py` and `scripts/decomp_descriptive.py`.
- **Tests.** `tests/test_partition.py`.

Choices the text leaves open:
- **Null masks.** Row i's n-cell mask is the top n cells of seeded score row i, so the masks of one row are nested in n. CDEA's null (N = 8) is measured under blur. The scorer's (32 masks) is measured under each scoring operator, for each image only at the sizes its roles and halves need.
- **Entry threshold.** ν is linear between whole sizes. Its gradient in a role's size is the step to the next size.
- **An empty role** pays −ε_s (1e-3) in the smoothed payoff, not 0, so the gradient through the deletion survives.
- **Split halves.** One seeded permutation of the role's cells, the first half the larger. The verdicts are:
  - mixture: a half counts for one class only;
  - redundant: neither half counts;
  - shared: both halves count for both;
  - unclear: anything else.
- **Overlapping roles (DA1).** VR counts a cell once, in the first confirmed role of S, U_k, U_l, ∅.
- **DA10.** Each version 1 player takes its round(a·R) cells of largest plan mass (2 at 5% on 7×7), with no cell taken twice. Read by argmax instead, version 1's plan leaves almost every cell unallocated.
- **`margin_gradcam_ref`.** pytorch-grad-cam clamps and rescales each map, so the sign is read from the two margin maps, of z_k − z_l and of z_l − z_k.
- **Rule rows** pool the raw class maps to the grid before they clamp (section 5, step 2).

Where the text and the implementation disagree:
- **Toy 1** asks `subtract` for ∅ = {4}. Rules never label ∅, and a cell with no evidence lies outside Ω. The test sets ρ = 1, so Ω holds the three evidence cells.
  - An added artefact toy covers ∅: evidence on a cell whose deletion changes nothing.
  - `occlusion_partition` and CDEA label that cell ∅; `subtract` labels it U_k.
- **The straight-through search can end below its own step 0.** Two strict xfails record it.
  - With the whole-grid null, a pure U_k has d̃_l = −ν_l < 0. There |d̃_l| has slope −1, so to first order l-evidence looks like a gain for U_k.
  - On toy 1 at lr 0.1 the shared cell moves to U_k, and the hard welfare falls from 2.625 at step 0 to 2.0.
  - On toy 5 the co-located cell moves to U_l.
  - The cells record the blur welfare of the hard partition at step 0 and at the end (`payoff_init_*`, `payoff_hard_*`).
- **Toy 4.** With no shared cell, CDEA parks a weak k cell in S.
  - S and ∅ both start at log ε, and Adam moves both alike.
  - That S has one cell, so neither MIX nor the halves can flag it. The scorer does not confirm it (D8).
- **Co-location share.** As written, P nets each position's channels before the min. On the dev models it falls far below 0, because Q is several times P. A positive-part version, in [0, 1], is reported beside it (`evaluation/descriptive.py`).

