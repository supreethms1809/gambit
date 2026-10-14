# CDEA decomposition formulation

**Status.** This is the method. Implemented in `cdea/decompose.py`, `cdea/null.py` and `cdea/_plan.py`, and merged to `main` on 2026-10-14. Development runs on val are in progress (`PROGRESS.md`). The test split has not been read. The evaluation is `EVAL_PLAN.md` section 13. Its win conditions were signed off on 2026-10-14 (tag `decomp-declared`).

The contrast game (`FORMULATION.md`) stays in the code unchanged and becomes ablation DA2.

## 1. Why

The goal is the original one. Per-class evidence overlaps. The method splits it into evidence **unique** to each class and evidence **shared** between them, in a form that can be measured and shown.

The contrast game answers a different question:
- Its payoffs are log-odds differences, so they mix "evidence for k" with "evidence against l".
- Its shared player measures "evidence for the shortlist against the other classes", not "evidence common to the members".
- A seed-0 val diagnostic of its optimiser (2026-10-13) showed:
  - its objective does not track CD;
  - CD itself is answered better by a direct search on the margin.

## 2. The question, defined by intervention

Phase 1 uses one pair: k = rank 0, l = rank 1 (K = 2).

For a mask M and a class j:
- **Drop.** d_j(M) = z_j(x) − z_j(x ⊖ M), in logits.
- **Null.** ν_j(a) is the mean of d_j(R) over N = 8 seeded random masks R of area a. The seed never depends on the batch.
- **Excess drop.** d̃_j(M) = d_j(M) − ν_j(a). This removes the common-mode drop: deleting almost anything lowers a confident logit.

Three maps per image:

| Map | Definition |
|---|---|
| U_k, unique to k | Deleting it lowers z_k and leaves z_l unchanged, in either direction |
| U_l, unique to l | The same, with k and l swapped |
| S, shared | Deleting it lowers both z_k and z_l |

The ideal 3×2 matrix of excess drops:

```
          d̃_k     d̃_l
U_k      high      ≈ 0
U_l      ≈ 0       high
S        high      high
```

**Suppressing l is not unique evidence for k (D3, settled).** A region whose deletion raises z_l is penalised in u_k, not rewarded.

## 3. Method

### 3.1 Payoffs (`cdea/payoffs.py`)

All payoffs are in logit units.
- |t|_s = sqrt(t² + ε²) and min_s(a, b) = (a + b)/2 − |a − b|_s/2, with ε = 1e-3 fixed.
- u_k = d̃_k(A_k) − |d̃_l(A_k)|_s
- u_l = d̃_l(A_l) − |d̃_k(A_l)|_s
- u_S = min_s(d̃_k(A_S), d̃_l(A_S))

The welfare is W = u_k + u_l + u_S. It has no weights: every term is a logit drop.

Deletion is CDEA's blur, x ⊖ M = (1 − M̃) x + M̃ b(x). It uses the boundary-robust offsets of `FORMULATION.md` section 4.1 (s = ⌊½ unit side⌋).

### 3.2 Null (`cdea/null.py`)

- **Masks.** `null_unit_scores` draws N = 8 rows of seeded uniform cell scores (seed 0). `null_unit_masks` keeps the top a·R cells of each row, whole cells with the last one fractional (`top_mass`).
- **One set for every image.** The same masks serve every image, so the null is chunk-invariant.
- **When it is computed.** Once, before optimisation, under blur, and detached. It costs N forward passes per image per area.
- **The scorer's null.** The scorer ranks the same cell scores through the shared adapter (bilinear upsampling, then top-a) and removes them with its own operator (section 5).

### 3.3 Constraints

These are the constraints of `FORMULATION.md` section 5, with P = 3 players:
- **Exclusivity:** every column sums to 1.
- **Budgets:** every player owns a·R, and ∅ takes the rest.
- **Feasibility:** 3a ≤ 1, which covers {2.5, 5, 10}%.

The budgets are fixed, so S takes a·R even in an image with no shared evidence. That is accepted for phase 1, so that masks are compared with baselines at matched area. D8 reports how often it happens.

### 3.4 Optimisation (`cdea/decompose.py`)

- **Machinery.** It reuses `allocate()`'s machinery from `cdea/_plan.py`:
  - a Sinkhorn plan (I = 20, log domain);
  - binary draws with a straight-through gradient;
  - boundary offsets;
  - a per-image sum loss and Adam;
  - one generator seeded by the draw seed, never by the batch.
- **Cost.** Each step deletes three inputs per image, one per player.
- **Defaults** (`DecomposeConfig`):

| Setting | Default | Ablation |
|---|---|---|
| steps | 100 | DA9 |
| lr | 0.1 | |
| init | `minsplit` | DA4 uniform, DA5 mean |
| backend | Grad-CAM on ResNet-50, IG (blur baseline) on ViT | |
| binary masks | on | DA6 off |
| boundary shift | on | DA7 off |
| draw seed | 0 | |
| null | on | DA8 off |
| independent | off | DA1 on |

### 3.5 Initialisation: the min-split prior

E is the per-class evidence of `_prior`, clamped at 0 and sum-normalised per class. With m = min(E_k, E_l) per cell:
- θ_k = log(E_k − m + ε)
- θ_l = log(E_l − m + ε)
- θ_S = log(m + ε)
- θ_∅ = log(1/R)
- ε = 1e-6

Step 0 is therefore the gradcam_minsplit row on ResNet-50, so the gain from optimisation is measured directly. The projected step-0 plan is kept (`Decomposition.initial`) for D9.

### 3.6 Output

U_k, U_l and S are soft maps on the grid: 7×7 on ResNet-50, 14×14 on ViT. They are scored through `baselines.adapter.budget_or_floor`, like every other method.

## 4. Comparators (`baselines/decomposition.py`)

**Min-split rule.** Every per-class method runs unmodified, and one split rule turns its two class maps into three. Clamp at 0, sum-normalise each map, then S = min(E_k, E_l), U_k = E_k − S, U_l = E_l − S. The split is applied at the method's own resolution.

| Row | Per-class source | Status |
|---|---|---|
| gradcam_minsplit | The repo's Grad-CAM region provider, cross-checked against pytorch-grad-cam in G0. It is CDEA's own prior, so step 0 equals this row | Core |
| ig_minsplit | Captum IntegratedGradients, zero baseline, right Riemann sum, channel sum, pixels | Core |
| occlusion_minsplit | Captum `Occlusion`, window and stride of one cell, zero baseline | Core |
| occlusion_assign | Exact assignment of single-cell occlusion scores (below) | Core |
| margin_gradcam_ref | `margin_gradcam_pair`: U_k and U_l only, with no S. A construction, not a published method | Core (reference) |
| random_decomp | Seeded cell scores | Core (floor) |
| rise_minsplit | Vendored official RISE `class_maps` | Extended |
| extremal_minsplit | TorchRay class masks at the scored area | Extended |

**Assignment comparators.** These decide whether the joint search adds anything over additive per-cell scores.
- **Cell scores.** Each cell gets scores from g_{j,r}:
  - s_k = g_k − |g_l|
  - s_l = g_l − |g_k|
  - s_S = min(g_k, g_l)
- **The assignment.** Maximise Σ A_{p,r} s_{p,r}, with every row summing to a·R and every column at most 1. This is solved exactly as a linear programme (HiGHS, per image).
- **occlusion_assign.** g_{j,r} is the excess drop when cell r alone is blur-deleted, with the null at one cell. That costs R + 1 forward passes, plus the null's.
- **gradient_assign** (ablation DA3). g_{j,r} = ⟨∇z_j, (x − b(x)) ⊙ φ_r⟩. Its null is the mean over cells, which is the first-order expectation of a one-cell deletion.

**Framing rule.** If occlusion_assign matches CDEA within the declared margin (`EVAL_PLAN.md` section 13), the contribution is the definition, the scores and the ground truth, and the method is the assignment.

## 5. Scores (`evaluation/scores.py`)

**The drop matrix.** `drop_matrix(model, x, masks (B, P, H, W), class_ids) → (B, P, K)` generalises `deletion_matrix`, which is now its transpose. Removal is ROAD (primary) or blur (robustness), with the seeds and offsets of the contrastive game.

| Score | Definition | Reads as |
|---|---|---|
| SPEC_k, SPEC_l | D[U_k, k] − D[U_k, l], and the mirror (raw drops) | Unique specificity. SPEC_k + SPEC_l = CD (tested) |
| PUR_k, PUR_l | d̃_k(U_k) − \|d̃_l(U_k)\|, and the mirror | Unique purity, the decomposition's unique criterion |
| SV | min(d̃_k(S), d̃_l(S)) | Shared validity |
| BAL | \|d̃_k(S) − d̃_l(S)\| | Shared balance, lower is better |
| SEP | SV − max(min_j d̃_j(U_k), min_j d̃_j(U_l)) | Whether S is more shared than the unique masks |

On planted pairs, `mask_patch_confusion` gives each mask's mass on {A_k, A_l, C, rest} (precision) and the share of each patch covered (recall).

Records hold the raw 3×2 matrix, the scorer's null, every score, and the patch confusion. CDEA rows also hold the D2/D9 stage payoffs.

## 6. Ground truth: planted pairs (`evaluation/planted_cues.py`)

- **The pairs.** `PlantedPairCIFAR` uses the CIFAR-10 pairs cat/dog, automobile/truck and deer/horse.
- **Training rule.**
  - A_k is a solid patch in the hue of class k, and appears in class-k training images only.
  - A_l is the same for class l.
  - C is the pair's stripe texture (diagonal, anti-diagonal or horizontal, period 8 px). It appears in both classes.
  - Each patch appears with probability 0.9. Other classes get no patches.
- **Patches.** They are 32 px, exactly one 7×7 cell at 224, placed on the grid in cells that do not touch. Random placement is the robustness condition.
- **Test images.** A pair-class val image carries all three patches. The sample keeps, in seeded draw order, the images whose model top 2 is the pair. It orders A_k and A_l by the model's rank and counts every image tried.
- **Role check** (`verify_roles`).
  - Each patch is ROAD-deleted alone.
  - A drop counts when it exceeds the 95th percentile of 32 random patch-sized squares.
  - A_k is confirmed when its k drop counts and its l drop does not. A_l is the mirror. C is confirmed when both drops count.
  - Every image stays in the records with its role flags. The read-out reports all images and the confirmed ones.
- **Checkpoints.** They use the planted-patch recipe: a fine-tune, `scripts/train_backbone.py --dataset planted_pair`. **They are not trained.** Training needs the user's approval.

## 7. Degenerate routes (`scripts/check_degenerate.py`)

- **D1, D5.** The marginals are within 1e-3, as before (asserted inside Sinkhorn).
- **D2** (gate). The soft-plan payoff minus the hard top-a payoff, under blur, for all three players, is at most 0.5 on average.
- **D8** (reported). The share of images with SV ≤ 0 under blur, meaning S found no shared evidence.
- **D9** (reported). The share of images whose final top cells equal the step-0 plan's, meaning the optimiser did nothing.
- **Trapped support** (reported, decision 5). The share of final top cells where the min-split prior was 0. ε stays 1e-6, and no new constant is added.

## 8. Decisions (user, 2026-10-14: "go with the recommendations")

| # | Decision | Taken |
|---|---|---|
| 1 | K | 2 for phase 1 |
| 2 | Unique evidence | Excludes suppression of the rival (\|d̃_l\| penalty). Settles D3 |
| 3 | Units | Logits |
| 4 | Budget | Fixed per player, at matched area. A pooled budget is phase 2 |
| 5 | Trapped support | No recommendation was given. Taken: keep ε = 1e-6, and report D9 and the outside-prior share. It adds no constant, and the 2026-10-13 diagnostic found trapping minor (CIFAR-10) to absent (ImageNet) |
| 6 | Masks | Binary draws by default. DA6 (fractional) is reported |
| 7 | Planted pairs | No recommendation was given. Three pairs are implemented (cat/dog, automobile/truck, deer/horse), with grid placement. Training was approved on 2026-10-14 (`PROGRESS.md`) |
| 8 | Win conditions and non-inferiority margin | No recommendation was given. Drafted in `EVAL_PLAN.md` section 13, and signed off by the user on 2026-10-14 as drafted |
| 9 | The old CDEA and CD | The old CDEA is ablation DA2. CD is reported as SPEC_k + SPEC_l. Neither is the headline |

## 9. Implementation choices, and properties found while implementing

- **Null masks.** The null uses `top_mass` on seeded uniform scores: whole top cells, with the last one fractional, which is the shape of a hard plan. `hard_top_mass` is not used, because it gives a diffuse fractional mask, (s − τ) clamped to [0, 1] spread over many cells.
- **The scorer's null.** The scorer ranks the null cell scores through the shared adapter, so its masks look like the scored masks. `random_floor` is not used, because it ranks a constant field and its tie-break scatters single pixels.
- **The helper move.** `_row_targets`, `_independent_masks` and `_player_logits` moved to `cdea/_plan.py` along with the four named helpers. `allocate()` is unchanged (`tests/test_plan_refactor.py`).
- **The null on a model without deletion artefacts.**
  - ν then also removes the expected evidence that a random mask hits.
  - On a 4-cell toy with three evidence cells, this made the unique and shared cells tie. With 16 cells the toys behave as specified.
  - On real models, ν is mostly the common-mode drop it is meant to remove, but the readout reports DA8 (no null) beside the default.
- **Method code hash.** `method_code_hash` covers every file under `CODE_ROOTS`, so the merge changed it. The seed-0 chain's records (hash `9d84e7f2`) no longer match the tree. Comparing them needs the branch-point hash given explicitly (`--expect-hash`), or a rerun.

## 10. Tests

- `tests/test_decomposition.py`:
  - toys 1–3 (additive, suppression, interaction);
  - payoff units and the null;
  - chunk invariance and exact CUDA repeats;
  - the assignment LP against brute force.
- `tests/test_decomposition_scores.py`:
  - SPEC_k + SPEC_l = CD;
  - `drop_matrix` against `deletion_matrix`;
  - score definitions and patch confusion;
  - runner rows;
  - planted-pair geometry and frequencies, and the role check.
- `tests/test_plan_refactor.py`: the `allocate()` guard.
- **The key test** is interaction toy 3. z_k needs either of two cells, so deleting one alone does nothing. Single-cell occlusion assignment misses the pair, and the joint search finds it. Without this toy, the method claim would have no basis.
