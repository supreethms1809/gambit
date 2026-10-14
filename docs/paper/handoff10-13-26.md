# CDEA decomposition — implementation handoff (draft, 2026-10-13)

Status: draft for the user's sign-off. Nothing here is implemented. Read with `docs/paper/FORMULATION.md`, `docs/paper/PROGRESS.md`, and `OPT_DIAGNOSTIC_2026-10-13.md` (untracked, on the GH200).

Repo grounding: written against `main` at `9705348`. Files cited: `cdea/allocation.py`, `cdea/payoffs.py`, `cdea/first_order.py`, `cdea/sinkhorn.py`, `evaluation/scores.py`, `evaluation/run_methods.py`, `evaluation/run_cell.py`, `evaluation/planted_cues.py`, `baselines/adapter.py`, `baselines/margin.py`.

## 0. Purpose

Return CDEA to its original goal: per-class evidence overlaps, and the method must split it into evidence **unique** to each class and evidence **shared** between them, in a form that can be **measured** and **visualised**.

The current formulation answers a contrast question instead. Its payoffs are log-odds differences, so they mix "evidence for k" with "evidence against l", and its shared player measures "evidence for the shortlist against the other classes", not "evidence common to the members". The current CDEA stays in the code unchanged and becomes an ablation (DA2 below).

Rules carried over: no test-split reads before the freeze tag, no hand-typed numbers, baselines run unmodified, every record carries `method_code_hash` and `knobs_hash`.

## 1. The question, defined by intervention

Phase 1 uses one pair: k = rank 0, l = rank 1 (K = 2). K > 2 is out of scope (section 13).

For a mask M and a class j:

- **Drop.** d_j(M) = z_j(x) − z_j(x ⊖ M), in logits.
- **Null.** ν_j(a) = mean of d_j(R) over N = 8 seeded random masks R of the same area a. The seed never depends on the batch.
- **Excess drop.** d̃_j(M) = d_j(M) − ν_j(a). This removes the common-mode drop: deleting almost anything lowers a confident logit.

Three maps per image:

| Map | Definition |
|---|---|
| U_k, unique to k | Deleting it lowers z_k and leaves z_l unchanged, in either direction |
| U_l, unique to l | The same with k and l swapped |
| S, shared | Deleting it lowers both z_k and z_l |

The ideal 3×2 matrix of excess drops:

```
          d̃_k     d̃_l
U_k      high      ≈ 0
U_l      ≈ 0       high
S        high      high
```

**Decision D3 is settled here: suppressing l does not count as unique evidence for k.** A region whose deletion raises z_l is penalised in u_k, not rewarded.

## 2. What changes and why

| Element | Current CDEA (`cdea/allocation.py`) | Decomposition CDEA |
|---|---|---|
| Players | K = 5 unique + 1 shared + ∅ | U_k, U_l, S + ∅ |
| Unique payoff | Drop in c_k = z_k − LSE(rivals). Rival rising counts as k's evidence | d̃_k(A_k) − \|d̃_l(A_k)\|. Neutral to l |
| Shared payoff | Drop in s_H = LSE(H) − LSE(rest), dominated by z_k | min(d̃_k(A_S), d̃_l(A_S)). Both must drop |
| Null | None | ν_j subtracted inside every payoff |
| Pairwise sharing | No player owns evidence shared by k and l only | S owns it by definition |
| Init | log of per-class Grad-CAM; shared = log of the mean | log of the min-split (section 3.5), so step 0 equals the naive baseline |
| Headline score | CD@5% | Decomposition matrix scores and planted-cue recovery (section 5). CD survives as the unique specificity, SPEC_k + SPEC_l |

## 3. Method specification

### 3.1 Payoffs

All in logit units. Add to `cdea/payoffs.py` without changing the existing functions:

- smooth absolute value: |t|_s = sqrt(t² + ε²), with ε = 1e-3 fixed;
- smooth minimum: min_s(a, b) = (a + b)/2 − |a − b|_s / 2;
- u_k = d̃_k(A_k) − |d̃_l(A_k)|_s;
- u_l = d̃_l(A_l) − |d̃_k(A_l)|_s;
- u_S = min_s(d̃_k(A_S), d̃_l(A_S)).

Welfare W = u_k + u_l + u_S. There are no weights: every term is a logit drop.

### 3.2 Null estimate

New file `cdea/null.py`:

- `null_drops(model, x, ids, grid_h, grid_w, area, n=8, seed=DRAW_SEED, offset_fn=None) -> (B, 2)`.
- Masks are seeded random cell masks at budget a·R. Use `hard_top_mass` on seeded uniform scores, so a fractional last cell is handled the same way as the plan.
- The same N masks serve every image, so the result is chunk-invariant.
- It is computed once before optimisation under blur deletion, and detached. Cost: N forward passes per image per area.
- The scorer computes its own null under ROAD with the same seeded masks (section 5).

### 3.3 Constraints

Unchanged from `FORMULATION.md` section 5, with P = 3:

- each column sums to 1 (exclusivity);
- each player owns a·R;
- ∅ takes the rest;
- feasibility 3a ≤ 1, which covers {2.5, 5, 10}%.

Fixed budgets force S to take a·R even in an image with no shared evidence. That is accepted for phase 1 so the masks can be compared with baselines at matched area. Route D8 (section 8) reports how often it happens.

### 3.4 Optimisation

Reuse the machinery of `allocate()`: Sinkhorn plan (I = 20, log domain), binary draws with a straight-through gradient, boundary offsets, per-image sum loss, Adam, a draw seed that never depends on the batch.

New file `cdea/decompose.py`:

- `DecomposeConfig`: `steps=100`, `lr=0.1`, `init="minsplit"`, `backend` (`gradcam` on ResNet-50, `ig` on ViT), `binary_masks=True`, `boundary_shift=True`, `draw_seed=0`, `null=True`, `n_null=8`, `independent=False`, `payoff="decomposition"`.
- `decompose(model, x, hypotheses, grid_h, grid_w, area, cfg) -> Decomposition(unique_k, unique_l, shared, transport, row_target, null)`.
- Each step makes 3 deleted inputs per image: one per player.

Move `_draw`, `_offsets`, `_edited` and `_project` from `allocation.py` into `cdea/_plan.py` and import them in both files. The refactor must leave `allocate()` byte-identical in behaviour; a test asserts this (section 9). Keeping `allocate()` untouched keeps existing records and hashes valid.

### 3.5 Initialisation (min-split prior)

E = per-class evidence from `_prior()`: clamped at 0 and sum-normalised per class. Let m = min(E_k, E_l) per cell. Then:

- θ_k = log(E_k − m + ε);
- θ_l = log(E_l − m + ε);
- θ_S = log(m + ε);
- θ_∅ = log(1/R);
- ε = 1e-6, as now. See decision 5 on trapped support.

Step 0 is then the min-split baseline. The gain from optimisation is measured directly.

### 3.6 Output

U_k, U_l and S as 7×7 soft maps (14×14 on ViT). They go through `baselines.adapter.adapt_scores` like every other method.

## 4. Baselines

Every per-class method runs unmodified. One declared split rule is applied identically to all of them.

**Split rule** (new file `baselines/decomposition.py`):

- `min_split(E_k, E_l)`: clamp at 0, sum-normalise each map, S = min(E_k, E_l), U_k = E_k − S, U_l = E_l − S.
- The split runs at the method's native resolution (pixel or region). Each of the three maps then goes through `adapt_scores`.

| Row | Per-class source | Status |
|---|---|---|
| gradcam_minsplit | pytorch-grad-cam, target z_j | Core. On a GAP + linear head this is CAM, an exact per-location class contribution, so it is the strongest naive split |
| ig_minsplit | Captum IntegratedGradients, author default baseline | Core |
| occlusion_minsplit | Captum `Occlusion`, window and stride = one cell, author default baseline | Core. Zeiler & Fergus occlusion, unmodified |
| rise_minsplit | vendored official RISE (`class_maps`) | Extended |
| extremal_minsplit | TorchRay class masks (`extremal_class_pair`) | Extended |
| margin_gradcam_ref | existing `margin_gradcam_pair` | Reference row for U_k and U_l only, no S. Labelled as a construction, not a published method |
| random | seeded | Floor |

### First-order and occlusion assignment (most important comparison)

These decide whether the joint search adds anything over additive per-cell scores.

- **occlusion_assign.** Per cell r, g_{j,r} = d̃_j(cell r deleted alone), using CDEA's own blur operator. That is R + 1 forward passes, and every class comes from the same forwards. Cell scores:
  - s_{k,r} = g_{k,r} − |g_{l,r}|;
  - s_{l,r} = g_{l,r} − |g_{k,r}|;
  - s_{S,r} = min(g_{k,r}, g_{l,r}).

  Then solve the exact assignment: maximise Σ A_{p,r} s_{p,r}, each row summing to a·R, each column at most 1. This is a 3 × R transportation LP, small enough for `scipy.optimize.linprog`.
- **gradient_assign** (ablation DA3). The same, with g from gradient × (x − blur), as in `cdea/first_order.py`.

If occlusion_assign matches CDEA, the contribution is the definition, the scores and the ground truth, and the method is the assignment. That framing must be fixed before results are read (section 11).

## 5. Scores

Changes in `evaluation/scores.py`:

1. **`drop_matrix(model, x, masks (B, P, H, W), class_ids (B, K)) -> (B, P, K)`.** This generalises `deletion_matrix`, which assumes P = K. Keep `deletion_matrix` as a wrapper. ROAD removal, same seeds and offsets as now.
2. **`null_drops_road(model, x, class_ids, area, n=8, seed)`.** Uses the same seeded random masks as section 3.2, converted through `random_floor` at area a.
3. **`decomposition_scores(D_raw, null)`** returns per image:

| Score | Definition | Reads as |
|---|---|---|
| SPEC_k, SPEC_l | D[U_k,k] − D[U_k,l], and the mirror (raw) | Unique specificity. SPEC_k + SPEC_l = CD; asserted in a test |
| PUR_k, PUR_l | d̃_k(U_k) − \|d̃_l(U_k)\|, and the mirror | Unique purity, the decomposition's unique criterion |
| SV | min(d̃_k(S), d̃_l(S)) | Shared validity |
| BAL | \|d̃_k(S) − d̃_l(S)\| | Shared balance, lower is better |
| SEP | SV − max(min(d̃(U_k)), min(d̃(U_l))) | Is S more shared than the unique masks? |

4. **`mask_patch_confusion(masks (B, 3, H, W), patches (B, 3, H, W))`** gives a 3×4 matrix of mask mass on {A_k, A_l, C, rest}. Report both precision (share of the mask on its patch) and recall (share of the patch covered).

ROAD is primary and blur is the robustness check. Raw and null-corrected values are both written to records. Keep `contrastive_deletion` and `shared_validity` unchanged as legacy scores.

## 6. Ground truth: planted pair dataset

New class `PlantedPairCIFAR` in `evaluation/planted_cues.py`. Reuse `_stamp`, `_solid` and `patch_layout`, extended to three disjoint boxes.

**Training rule** (decision 7 sets the class pairs):

| Patch | Appears in training images of | With probability |
|---|---|---|
| A_k (solid hue of k) | class k only | 0.9 |
| A_l (solid hue of l) | class l only | 0.9 |
| C (new texture, e.g. diagonal stripes, not the existing checker) | classes k and l only | 0.9 |

Other classes get no patches. C is then evidence for {k, l} against the rest and carries no information between k and l.

**Test images.** A class-k or class-l val image with A_k, A_l and C stamped at disjoint positions. Keep only images where the model's top 2 are {k, l}.

**Role check by intervention.** New function `verify_roles(model, item)`:

1. ROAD-delete each patch alone.
2. A role is confirmed when the patch's excess drop for the expected class(es) exceeds the 95th percentile of 32 random patch-sized masks, and its effect on the other class does not.
3. Excluded images are counted and reported, never dropped silently.

**Placement.** Phase 1 uses grid-aligned 32-pixel patches, exactly one 7×7 cell at 224. Random placement is the robustness condition.

**Checkpoints.** Train with the same recipe and script as the existing `planted_patch` checkpoints (`scripts/train_backbone.py`, `launch_paper_training.py`), ResNet-50 first and ViT-B/16 second. **Training needs the user's approval** (`AGENTS.md`).

## 7. Natural images

Hypotheses are the top 2. Development uses CIFAR-10 and HAM10000 val, seed 0, n = 64. Confirmatory runs use the eight contrastive datasets on held-out data after the freeze.

Fine-grained sets (CUB, Pets) are where overlap is expected, and they make the qualitative figure. Natural images have no ground truth, so only the matrix scores apply.

## 8. Degenerate routes for the decomposition

Extend `scripts/check_degenerate.py`:

- **D1, D5.** Marginals within 1e-3, as now.
- **D2.** Soft-plan payoff minus hard top-a payoff, under blur, for all three players. Gate at 0.5 on average, as now.
- **D8 (new, reported).** Share of images whose S has SV ≤ 0 under blur, meaning S found no shared evidence.
- **D9 (new, reported).** Share of images where CDEA's final masks equal the min-split init, meaning the optimiser did nothing.

## 9. Tests

New file `tests/test_decomposition.py`:

1. **Additive toy** (GAP + linear head, 4 cells; deletion zeroes a cell's feature). Cell 1 feeds k only, cell 2 l only, cell 3 both equally, cell 4 nothing. `decompose()`, `occlusion_assign` and `gradcam_minsplit` must all return U_k = 1, U_l = 2, S = 3.
2. **Suppression toy.** One cell raises z_k and lowers z_l. Expected: the decomposition does not put it in U_k with full weight, while `margin_gradcam_ref` ranks it first. This documents the D3 decision.
3. **Interaction toy.** Two cells are needed jointly for k (an AND through a nonlinearity). The joint search should find both, and single-cell occlusion should miss them. If no toy separates the two, the method claim has no basis. This is the key test.
4. **Payoff units.** `smooth_abs`, min_s within ε of min, null subtraction, an empty mask's payoff equals −ν.
5. **Refactor guard.** `allocate()` output is identical before and after the `_plan.py` move, on a fixed seed and fixed images.
6. **Determinism.** Chunk invariance; repeated runs identical on MPS, and on CUDA on the GH200.
7. **LP assignment.** Budgets and exclusivity hold; the result equals brute force on R = 6.
8. **Scores.** SPEC_k + SPEC_l equals `contrastive_deletion` on the same masks and seeds to 1e-5. `drop_matrix` equals `deletion_matrix` when P = K.
9. **Dataset.** Patches are disjoint, roles stamp as specified, and the training label/patch frequencies match section 6.

## 10. Runner and records

- **`evaluation/run_methods.py`:**
  - add `DecompMaps(k, l, shared, grid, extra)`;
  - add `cdea_decomp`, the `*_minsplit` rows, `occlusion_assign`, `gradient_assign`, `margin_gradcam_ref`, `random_decomp`;
  - registries `DECOMP_CORE`, `DECOMP_EXTENDED`, `DECOMP_ABLATIONS`.
- **Ablations:**

| Name | Change |
|---|---|
| DA1_independent | No column constraint, so masks may overlap |
| DA2_contrast | The current CDEA with `pair_only=True` and its s_H shared player, scored on the new matrix |
| DA3_gradient_assign | First-order solution |
| DA4_uniform_init | |
| DA5_mean_init | The current prior |
| DA6_fractional | `binary_masks=False` |
| DA7_fixed_boundary | |
| DA8_no_null | |
| DA9_steps | T ∈ {25, 50, 200} |

- **`evaluation/run_cell.py`:** new game `decomposition`. Areas {2.5, 5, 10}%. Operators ROAD (primary) and blur. Per image × method × area, the record holds: the raw 3×2 matrix, the null, every score in section 5, and the patch confusion matrix on planted data. Done markers carry the hashes, as now.
- **Scripts:** `scripts/decomp_readout.py` prints the tables from records. `scripts/launch_paper_eval.py` gets `--game decomposition`.

## 11. Pre-declaration (commit before any run)

New doc `docs/paper/FORMULATION_DECOMPOSITION.md`: sections 1–3 of this handoff in final form. New section in `EVAL_PLAN.md`:

- **Primary, ground truth.** On planted pairs, the mass of S on C and of U_k, U_l on A_k, A_l. Compare CDEA with the best min-split row and with occlusion_assign. Paired bootstrap over images, per seed.
- **Primary, natural images.** PUR_k + PUR_l and SV as co-primary, per dataset. The confirmatory test is at dataset level (Wilcoxon + Holm), as in the existing plan.
- **Win conditions.** Written before the first run, including a non-inferiority margin for the unique masses if one is used.
- **Framing rule.** If occlusion_assign is within the margin of CDEA, the paper presents the definition, scores and ground truth as the contribution and the assignment as the method.
- **Tuning.** Only `steps` and `lr` (the existing grid), on dev val. Nothing tuned on planted test images.

Tag `decomp-declared` after the commit. Log in `PROGRESS.md`.

## 12. Order of work

Effort estimates are rough.

1. Pre-declaration docs and decisions. 0.5 day.
2. `cdea/_plan.py` refactor with the guard test, then `payoffs.py`, `null.py`, `decompose.py` with toys 1–4. About 2 days.
3. `baselines/decomposition.py`: min-split, Captum occlusion, LP assignment, with tests. About 1.5 days.
4. Scores, runner, readout, degenerate checks. About 2 days.
5. Planted pair dataset, role check, and training (after approval). About 2–3 days including GH200 time.
6. Development runs: planted val, CIFAR-10 and HAM10000 val. About 1 day.
7. Freeze, then confirmatory runs. 1–2 days.

## 13. Out of scope

- K > 2, which needs a shared player per subset of H. Phase 2.
- The shift game, which goes to the separate paper.
- A user study.

## 14. Decisions for the user (before step 1)

1. **K = 2 for phase 1.** Recommended.
2. **Unique excludes suppression of the rival** (|d̃_l| penalty). Recommended. This settles D3.
3. **Units: logits.** Recommended over log-probabilities, which couple classes through the normaliser.
4. **Fixed per-player budget at matched area.** Recommended for phase 1. A pooled budget, where S may be empty, is phase 2.
5. **Trapped support.** Keep ε = 1e-6 and report D9 plus the share of final cells outside the prior's support, or mix the prior with a uniform floor (one new fixed constant).
6. **Binary draws (default) or fractional plans.** The diagnostic showed straight-through reaches about 36% less of its own objective. Recommended: binary as default, DA6 reported.
7. **Planted pairs.** Which CIFAR-10 class pairs (e.g. cat/dog, automobile/truck, deer/horse), placement, and approval to train ResNet-50 and ViT-B/16 checkpoints.
8. **Win conditions and non-inferiority margin** (section 11).
9. **Old CDEA and CD.** Recommended: keep the old CDEA as ablation DA2 and CD as SPEC_k + SPEC_l. Neither is the headline.

## 15. Risks

- **Additivity.** occlusion_assign may match CDEA, as margin attribution did on CD. Interaction toy 3 and the planted data are where a difference has to show. Framing is fixed in advance (section 11).
- **Strong naive split on ResNet.** With a GAP + linear head, per-class Grad-CAM is exact in feature space, so gradcam_minsplit is a strong baseline. CDEA's possible edge is in pixel-deletion effects and interactions only.
- **Plan to pixel mask.** Near-binary cell plans upsampled bilinearly can tie at the top-a threshold. Record the number of tied pixels at the threshold for every region-map method. Report cell-exact scoring as a secondary.
- **Common-mode drops.** Null-corrected and raw values must both be reported. The ROAD negative floor on the l side applies here too.
- **Planted shortcuts.** The model may not learn C as shared. The role check measures this, and the exclusion count is reported.
- **Time.** Section 12 adds up to about two weeks of focused work before confirmatory runs.