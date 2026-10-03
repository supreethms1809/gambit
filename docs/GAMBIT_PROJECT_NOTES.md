# GAMBIT — Full Project Notes

**Game-theoretic Allocation for Model-Based Interpretability and Trust**

Single self-contained record of the project: what it is, how it works, every experiment
run, every result, every bug found, every claim retracted, and what is and is not
established. Written 2026-08-24.

This file is intended to be readable on its own — method, datasets, results, and
reproduction instructions are inlined rather than cross-referenced. The separate
documents it consolidates (`MEDICAL_RESULTS.md`, `MEDICAL_DATASETS.md`,
`GAMBIT_PAPER.md`, `CDEA_CONTRASTIVE_PAPER.md`, `CDEA_BLOCK_DIAGRAM.md`,
`NEW_INSTANTIATION_GAME_GUIDE.md`, the `PRELIMS_*` starters,
`results/medical_presentation/RESULTS.md`, `results/paper_rerun/CORRECTION.md`) remain on
disk as the detailed run logs and paper drafts, but nothing in them is required to read
this one.

> **Reading warning for the paper drafts.** `GAMBIT_PAPER.md` §5 and
> `CDEA_CONTRASTIVE_PAPER.md` §6 still carry results computed before the
> training-normalization bug was found (§14.1). Their **method** sections are current;
> their **results** sections are stale. Take empirical claims from this file.

---

## Contents

**Part I — What this is**
1. [The problem](#1-the-problem)
2. [The idea](#2-the-idea)
3. [Architecture](#3-architecture)

**Part II — Method**
4. [The CDEA kernel](#4-the-cdea-kernel)
5. [Instantiation I — contrastive shared/unique game](#5-instantiation-i--contrastive-sharedunique-game)
6. [Instantiation II — shift-aware robust/shortcut game](#6-instantiation-ii--shift-aware-robustshortcut-game)
7. [How the games are solved](#7-how-the-games-are-solved)
8. [Evidence providers, unit space, interventions](#8-evidence-providers-unit-space-interventions)

**Part III — Setup**
9. [Datasets](#9-datasets)
10. [Metrics, and what each one can and cannot show](#10-metrics-and-what-each-one-can-and-cannot-show)
11. [Experiment map](#11-experiment-map)

**Part IV — Results**
12. [Instantiation I — contrastive results](#12-instantiation-i--contrastive-results)
13. [Instantiation II — robust/shortcut results](#13-instantiation-ii--robustshortcut-results)
14. [Medical classification](#14-medical-classification)
15. [Medical separation and budget](#15-medical-separation-and-budget)
16. [Interventional validation of the decomposition](#16-interventional-validation-of-the-decomposition)
17. [Localization, and why it fails on medical data](#17-localization-and-why-it-fails-on-medical-data)
18. [The planted-shortcut benchmark](#18-the-planted-shortcut-benchmark)
19. [Resolution](#19-resolution)
20. [The shared-mask fix](#20-the-shared-mask-fix)

**Part V — Corrections**
21. [Bugs found, and what each invalidated](#21-bugs-found-and-what-each-invalidated)
22. [The normalization correction in full](#22-the-normalization-correction-in-full)
23. [Retractions](#23-retractions)

**Part VI — Status**
24. [What is established, what is not](#24-what-is-established-what-is-not)
25. [Open questions and next steps](#25-open-questions-and-next-steps)

**Part VII — Practical**
26. [Reproduction](#26-reproduction)
27. [Artifact map](#27-artifact-map)
28. [Adding a new game instantiation](#28-adding-a-new-game-instantiation)

---
---

# Part I — What this is

## 1. The problem

A model looks at a skin photo and says **"melanoma, 80% confident."** You want to know why.

The standard tool is a heatmap — Grad-CAM, Integrated Gradients — which colors in the
parts of the image the model used. The catch is that you can make a heatmap for *any*
class. Make three — "why melanoma?", "why a harmless mole?", "why a wart?" — and they
come out looking nearly identical, all glowing over the same blob.

That is useless for the question you actually care about. If the melanoma heatmap and
the mole heatmap are the same picture, they cannot tell you what separates the two
diagnoses.

This is measurable, not rhetorical. Pairwise overlap between competing classes' raw
heatmaps:

| dataset | evidence | base overlap |
| --- | --- | ---: |
| HAM10000 | Grad-CAM | **0.447** |
| HAM10000 | IG | **0.495** |
| CIFAR-10 | Grad-CAM | 0.398 |
| CIFAR-10 | IG | 0.526 |
| MNIST | IG | 1.748 |
| brain tumor | IG | 0.326 |

The same structural gap appears under distribution shift: an attribution map cannot tell
you whether the model is using a genuine feature or a spurious shortcut that happens to
correlate with the label in the training distribution.

## 2. The idea

GAMBIT reframes explanation as an **allocation game** over a fixed grid of evidence
units. Competing hypotheses are players. A differentiable allocator distributes a fixed
evidence budget among them under an objective that rewards sufficiency and contrastive
margin and penalizes overlap, excess mass, and mass drift.

The contrastive instantiation splits evidence into two buckets:

- **Shared** — what every candidate hypothesis relies on. *"There's a dark spot here."*
  True no matter which diagnosis is right, so it cannot help you choose.
- **Unique** — what supports only *one* candidate. *"This* edge is ragged." That is what
  tips melanoma over mole.

Three doctors examining the same photo will all circle the same mole. That agreement
tells you nothing about who is right. The job is to find the smaller regions where they
actually *disagree*.

**Crucially, the allocator cannot add highlighting.** It has a fixed budget and can only
**move it around** — enforced by `lambda_mass`, and confirmed empirically by the measured
`sparse` column staying at ×1.00–1.02 on the datasets where the budget holds (§15).

The game-theoretic content is in the **objective structure** — agents, competing and
cooperative payoff terms, a cooperative-to-competitive spectrum controlled by the λ
weights — not in the solution concept. Nothing here computes Shapley values or Nash
equilibria; §7 explains why.

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CDEA Core Kernel                           │
│                                                                 │
│  Input x ──► Forward Pass ──► Hypothesis Selection (Top-K)      │
│                                    │                            │
│                           Base Evidence E(B,K,R)                │
│                                    │                            │
│                        [Optional Interaction]                   │
│                                    │                            │
│                     Allocation Game Solver                      │
│                        (Gradient-based)                         │
│                                    │                            │
│                   Intervention-Based Evaluation                 │
│                                    │                            │
│                   Explanation(masks, metrics)                   │
└───────────────┬──────────────────────────┬──────────────────────┘
                │                          │
    ┌───────────▼──────────┐   ┌───────────▼──────────────┐
    │ Instantiation I:     │   │ Instantiation II:         │
    │ Contrastive          │   │ Shift-Aware               │
    │ Shared-Unique Game   │   │ Robust-Shortcut Game      │
    │                      │   │                           │
    │ Agents: class hyps   │   │ Agents: robust, shortcut  │
    │ Output: unique +     │   │ Output: robust mask +     │
    │   shared masks,      │   │   shortcut mask,          │
    │   pairwise margins   │   │   stability diagnostics   │
    └──────────────────────┘   └───────────────────────────┘
```

**Code inventory.**

| area | files | notes |
| --- | --- | --- |
| Core kernel | `core/runner.py` (`CDEAExplainer`), `hypotheses.py`, `base_evidence.py`, `interaction.py`, `allocator.py`, `objective.py`, `unit_space.py`, `game_modes.py`, `types.py`, `reporting.py`, `visualization.py`, `device.py` | all extension points are Python `Protocol`s — zero-inheritance extensibility |
| Contrastive | `instantiations/contrastive/{allocator,objective}.py` | `OptimizationAllocator`, `ContrastiveObjective` |
| Shift | `instantiations/shift/{allocator,objective,env,biased_data}.py` | `RobustShortcutObjective`, bias-injection dataset builders |
| Evidence | `base_evidence/{gradcam,integrated_gradients,occlusion}_regions.py` | occlusion added for the shortcut benchmark |
| Modality | `modality/grid_regions.py` (`VisionGridUnitSpace`), `tokens.py`, `graph_nodes.py` | text/graph are stubs |
| Tests | `tests/test_{contrastive_allocator_objective,game_modes,interaction,shift_instantiation,pass_conditions}.py` | includes the mask-clamp regression |

**Key data types** (`core/types.py`): `HypothesisSet(ids, mask)`,
`Explanation(hypotheses, masks, metrics, extras)`, `EnvBatch(xs, env_ids)`.

**Backbones exercised.** ResNet-18, EfficientNetV2-S, ViT-B/16. Grad-CAM emits a
`RuntimeWarning` and an all-zero field on ViT — its non-negativity assumption does not
hold for LayerNorm'd tokens — so use IG there.

**Notebooks** (`examples/notebooks/`): `tutorial_pipeline`,
`cdea_contrastive_quickstart`, `mask_visualization`, `results_contrastive`,
`results_shift`, `results_full` (the consolidated one; reads the λ=0.25 runs),
`explainable_boosting_machines_tutorial`.

**High-level API:**

```python
import gambit

explainer = gambit.ContrastiveExplainer(model, game_mode="mixed", evidence="gradcam")
explanation = explainer.explain(x)
# explanation.masks["unique"]: (B, K, H, W)
# explanation.masks["shared"]: (B, H, W)

shift_exp = gambit.ShiftExplainer(model, game_mode="competitive")
explanation = shift_exp.explain(x, env=EnvBatch(xs=[x_id, x_ood], env_ids=[0, 1]))
# explanation.masks["robust"], explanation.masks["shortcut"]: (B, H, W)
```

---
---

# Part II — Method

## 4. The CDEA kernel

Given an input $x \in \mathbb{R}^{C \times H \times W}$, a trained classifier $f$, and an
evidence-unit space $\mathcal{U}$ with $R$ spatial units, `CDEAExplainer.explain`
proceeds in five stages:

**1. Hypothesis selection.** Take the top-$m$ classes from $f(x)$ as competing
hypotheses $H(x) = \{h_1, \ldots, h_m\}$ via `TopMSelector`.

**2. Base evidence extraction.** For each hypothesis $k$, compute nonnegative evidence
$E_k(u) \geq 0$ per unit, giving $E \in \mathbb{R}_{\geq 0}^{B \times K \times R}$.
Optionally normalized per $(B,K)$ across $R$ so each hypothesis's evidence sums to one.

**3. Hypothesis interaction (optional).** Build tokens
$t_k = \sum_u E_k(u)\,\phi(u)$ from 2D sinusoidal positional embeddings $\phi(u)$, then
condition them with `none`, attention-only, or a single Transformer layer. Yields
attention weights that can reweight each hypothesis's contribution to the averaged
objective and mix evidence for allocator initialization. **Optional throughout** — the
objective definitions do not change when `interaction=None`.

**4. Mask allocation.** An instantiation-specific allocator optimizes continuous masks
$m \in [0,1]^R$ by gradient descent over logit-parameterized variables.

**5. Intervention-based evaluation.** Masks are scored through *keep* and *remove*
interventions:

$$x^{\text{keep}}(m) = m \odot x + (1 - m) \odot \bar{x}$$

where $\bar{x}$ is a baseline (channel-wise global mean, or an adaptive blur). All
sufficiency and margin terms are computed on $f(x^{\text{keep}}(m))$ — **metrics come
from interventions, never from raw evidence**.

## 5. Instantiation I — contrastive shared/unique game

**Mask variables.** Per hypothesis $k$, a unique mask
$m_k^{\text{unique}} \in [0,1]^R$, plus an optional shared mask
$m^{\text{shared}} \in [0,1]^R$. The effective mask is

$$m_k^{\text{tot}} = m_k^{\text{unique}} + m^{\text{shared}}$$

when shared is enabled, otherwise $m_k^{\text{tot}} = m_k^{\text{unique}}$.

**Objective.**

$$\mathcal{L}_{\text{ctr}} = -\left(\lambda_{\text{suff}} \overline{\text{suff}} + \lambda_{\text{margin}} \overline{\text{margin}}\right) + \lambda_{\text{overlap}} \overline{\text{overlap}} + \lambda_{\text{sparse}} \overline{\text{sparse}} + \lambda_{\text{mass}} \overline{\text{mass\_dev}}$$

- **Sufficiency** $\text{suff}_k = f(x_k^{\text{keep}})_k$ — the target logit under the
  keep intervention. (Several scripts report this **baseline-subtracted** against an
  all-zeros input; each result table below says which.)
- **Contrastive margin** $\text{margin}_k = f(x_k^{\text{keep}})_k - \max_{l \neq k, l \in H(x)} f(x_k^{\text{keep}})_l$
  — discrimination against the strongest competing hypothesis.
- **Overlap** $\sum_{k<l} \langle m_k^{\text{unique}}, m_l^{\text{unique}} \rangle$ —
  redundant mass across unique masks.
- **Sparsity** $\frac{1}{K}\sum_k \lVert m_k^{\text{unique}} \rVert_1$.
- **Mass deviation** $\left|\sum_r m_k^{\text{unique}}(r) - \sum_r E_k(r)\right|$ —
  pins the budget, preventing the trivial "highlight everything" solution.

**Allocator-level penalties**, added on top:

| penalty | formula | purpose |
| --- | --- | --- |
| Disjointness | $\sum_{k<l} m_k^{\text{unique}} \cdot m_l^{\text{unique}}$ | unique masks should not overlap |
| Partition | $\text{ReLU}\left(\sum_k m_k(u) + m^{\text{shared}}(u) - 1\right)$ | total assigned mass per unit ≤ 1 |

**Game modes** (`core/game_modes.py`):

| mode | shared mask | margin | overlap | effect |
| --- | :---: | :---: | :---: | --- |
| Cooperative | yes | off | off | emphasizes shared evidence |
| Mixed | yes | moderate | moderate | balanced decomposition — **default** |
| Competitive | no | strong | strong | maximally disjoint unique masks |
| Manual | — | — | — | explicit user weights |

**Reporting.** Beyond scalars, the objective records per hypothesis: $p_k^{\text{shared}}$
(probability under `keep(x, m_shared)` alone), $p_k^{\text{total}}$ (under
`keep(x, m_k_tot)`), and $\Delta_k$ — the discriminative contribution of unique evidence.
It also emits a full $K \times K$ **pairwise margin matrix** for every "why $k$ not $l$"
pair, under both shared-only and shared+unique interventions, plus the delta.

> **`shared` is an overloaded word in this codebase.** `masks["shared"]` is a real
> optimized mask that exists **only** when `use_shared=True`. Separately,
> `visualize_contrastive` *derives* a red "shared" region for display as
> $\min(m_k, \max_{l\neq k} m_l)$ — the intersection of the unique masks — whenever no
> explicit shared mask is present. The two look identical in a figure and are not the
> same object. Panel titles read `(explicit shared)` only in the first case.

## 6. Instantiation II — shift-aware robust/shortcut game

**Setup.** Given an `EnvBatch` holding views of the same instance across environments
(in-distribution $x_{\text{id}}$ plus one or more OOD views), optimize two masks:
$m_{\text{rob}}$ (evidence that should be *stable* across environments) and
$m_{\text{sho}}$ (evidence that is *environment-specific*).

**Sufficiency**, baseline-subtracted:

$$\text{suff}(m, x_e) = f(\text{keep}(x_e, m))_y - f(\text{keep}(x_e, \mathbf{0}))_y$$

**Objective.**

$$\mathcal{L}_{\text{shift}} = -\left(\lambda_m \text{rob\_mean} - \lambda_v \text{rob\_var} + \lambda_g \text{sho\_gap} + \lambda_{sh} \text{sho\_mean}\right) + \lambda_d \text{disjoint} + \lambda_s \text{sparse}$$

| term | definition | intent |
| --- | --- | --- |
| `rob_mean` | $\mathbb{E}_e[\text{suff}(m_{\text{rob}}, x_e)]$ | robust evidence is sufficient in every environment |
| `rob_var` | $\text{Var}_e[\text{suff}(m_{\text{rob}}, x_e)]$ | ...and *stably* so |
| `sho_gap` | $\text{suff}(m_{\text{sho}}, x_{\text{id}}) - \mathbb{E}_{e \neq \text{id}}[\text{suff}(m_{\text{sho}}, x_e)]$ | shortcut evidence is ID-specific |
| `sho_mean` | $\mathbb{E}_e[\text{suff}(m_{\text{sho}}, x_e)]$ | shortcut utility (cooperative mode) |
| `disjoint` | $m_{\text{rob}} \cdot m_{\text{sho}}$ | the two masks separate |
| `sparse` | $\lVert m_{\text{rob}} \rVert_1 + \lVert m_{\text{sho}} \rVert_1$ | compactness |

Disjointness lives in the **objective** here, not the allocator (the reverse of the
contrastive case). The shift allocator explicitly checks for double-counting and raises
if both apply it.

**Bias-injection datasets** — synthetic, with the shortcut planted so ground truth is
known:

| dataset | shortcut signal | robust signal | mechanism |
| --- | --- | --- | --- |
| ColoredMNIST | class-correlated hue tint | digit shape | digit colorized by class ID |
| ColoredCIFAR10 | class-correlated colour patch (top-left 25%) | object content | solid colour patch added |
| TextureBiasedMNIST | class-correlated stripe angle | digit shape | sinusoidal background texture |

OOD views reassign the shortcut signal (different hue, patch colour, stripe angle) while
preserving the robust signal.

## 7. How the games are solved

**Logit parameterization.** Masks are not optimized directly in $[0,1]$. Unconstrained
logits $\ell \in \mathbb{R}^R$ map through a sigmoid, $m = \sigma(\ell)$. Three reasons:
unconstrained optimization suits Adam's adaptive moments better than box constraints;
sigmoid gradients are well-behaved everywhere, unlike projected-gradient boundaries; and
warm-starting from evidence is natural via $\ell_k = \sigma^{-1}(E_k)$.

Logits are clamped to $[-C, C]$ after each step. Contrastive uses $C = 12$ (masks in
~$[6\times10^{-6}, 1-6\times10^{-6}]$); shift uses a tighter $C = 3$ (~$[0.05, 0.95]$) to
discourage degenerate all-on/all-off solutions.

**The loop.**

```
Input:  x, f (frozen), E (base evidence), U (unit space), H (hypotheses), env (optional)
Output: optimized masks {m_1..m_P} for P players

1.  Initialize logits:
      if init_from_evidence:  ℓ_p ← σ⁻¹(clamp(E_p, ε, 1−ε))     # warm start
      else:                   ℓ_p ← 0                            # cold start, m ≈ 0.5
2.  (Contrastive) if attention available:
        E_mixed ← (1−α)·E + α·(A_norm · E)     # α = 0.35
        ℓ_p ← σ⁻¹(E_mixed)
3.  Freeze all model parameters
4.  Adam optimizer over {ℓ_1..ℓ_P}
5.  for t = 1..N_steps:
      a. m_p ← σ(ℓ_p)
      b. L ← game-specific loss on keep-intervened forwards
      c. P ← allocator structural penalties (disjoint, partition)
      d. backprop ∇_ℓ (L + P)
      e. Adam step
      f. ℓ ← clamp(ℓ, −C, C)
6.  Restore model parameter grads; return m_p ← σ(ℓ_p), detached
```

Per step: $K$ forward passes (contrastive) or $2E$ (shift, $E$ environments × 2 masks).
The model is frozen throughout — only mask logits receive gradients.

**Why Adam.** Different spatial regions carry wildly different evidence magnitudes and
gradient scales; per-element moments handle that without per-region tuning. The loss
combines competing objectives (sufficiency vs sparsity, margin vs overlap) and momentum
smooths conflicting gradients. The loop runs 25–50 steps, not hundreds of epochs, and
bias-corrected estimates converge faster than SGD in that regime — which matters because
each step is an expensive model forward. Empirically lr 0.2–0.5 converges across every
tested dataset, evidence provider, and game mode without per-configuration tuning; SGD
needed substantially more.

Second-order methods (L-BFGS, natural gradient) were considered and rejected: per-step
Hessian/Fisher cost across $K$ forwards, and a non-convex landscape where curvature
estimates mislead.

**Why not exact equilibria.** Four reasons. (a) Continuous action spaces — discretizing
$[0,1]^{49}$ or $[0,1]^{196}$ gives $2^{49}$ or $2^{196}$ pure strategies per player.
(b) Non-linear payoffs — sufficiency and margin involve full network forwards, while
standard solvers (support enumeration, Lemke-Howson, LCP) need bilinear or polynomial
payoffs. (c) Coupled constraints — disjointness and partition couple all players.
(d) The allocations need not be exact equilibria to be useful explanations; they need
only satisfy the interpretability desiderata encoded in the loss. A 50-step loop with
$K=5$ completes in under 2 s on GPU for a batch of 8.

The approach is closest to **gradient-based best-response dynamics**: every step updates
all players simultaneously to reduce the joint loss, approximating a
cooperative-competitive equilibrium. The λ weights *are* the cooperative-competitive
dial.

**Soft constraints, not hard ones.** Penalties rather than projections, because they keep
gradients smooth everywhere, let the optimizer transiently violate a constraint when
that helps the primary objective, and give smooth tradeoff curves under λ.

**Convergence.** Fixed step count, no early stopping — every explanation gets the same
budget regardless of input difficulty, which keeps runs reproducible. Loss stabilizes
within 20–30 steps for most configurations. Adaptive stopping on gradient norm or mask
entropy is future work.

> **A convergence result worth recording.** On the resolution investigation (§19),
> *more* optimization made Test A recovery strictly worse: 50 / 150 / 400 steps →
> 0.2995 / 0.0427 / 0.0234. Under-convergence was not the explanation for that anomaly;
> the metric was.

## 8. Evidence providers, unit space, interventions

**Providers** (all return nonnegative $(B, K, R)$):

1. **Grad-CAM** (`gradcam_regions.py`) — gradient-weighted activations at the target
   layer (last Conv2d for CNNs, last encoder block for ViTs), pooled to the grid. Fast,
   spatially smooth. Resolution is fixed by the backbone's feature map, so it **cannot**
   be pooled to a finer grid — `--grid` raises unless `--evidence ig`.
2. **Integrated Gradients** (`integrated_gradients_regions.py`) — path integrals from a
   zero or mean baseline, pixel-level then pooled. Finer-grained, slower
   ($n_{\text{steps}} = 24$ default, 16–24 in medical runs). Because it attributes at
   pixel level, **grid resolution is a free parameter**.
3. **Occlusion** (`occlusion_regions.py`) — model-agnostic, no gradient assumptions.
   Added for the shortcut benchmark; the only provider that works where Grad-CAM's
   non-negativity assumption breaks.

**Unit space.** `VisionGridUnitSpace` divides the input into $R = G_h \times G_w$ regions
(7×7 default for CNNs, 14×14 for ViTs; 28×28 and 56×56 used in sweeps). Provides
`num_units()`, `keep(x, m)`, `remove(x, m)`, and optional `embed_units(x)` returning
deterministic 2D sinusoidal embeddings $(B, R, D)$ when `embed_dim > 0`.

**Baselines.** `mean` (channel-wise global mean) or `blur` (adaptive-pooled image,
reducing boundary artifacts). **The choice changes metric behavior** — it is part of the
configuration, not an implementation detail.

**Mask clamping.** `_region_to_pixel_mask` clamps to $[0,1]$. Without it, a mask sum
above 1 makes `keep` extrapolate past $x$ rather than interpolate. Measured effect on
real runs: 4th decimal (recovery 0.1906 → 0.1900). Regression test:
`tests/test_pass_conditions.py::test_keep_remove_clamp_out_of_range_masks`.

**Budget scaling.** `mass_scale = max(1, R / mass_ref_regions)` keeps the mask a constant
*fraction* of the frame as the grid refines. `mass_ref_regions` defaults to 49, so at
28×28 the unique budget is **16×** what it is at 7×7. This is deliberate — and it makes
one metric non-comparable across grids (§19).

---
---

# Part III — Setup

## 9. Datasets

### 9.1 Standard vision

| dataset | classes | notes |
| --- | --- | --- |
| MNIST | 10 | 28×28 grayscale → RGB |
| CIFAR-10 | 10 | 32×32 → 224×224 |
| Oxford-IIIT Pets | 2 | cat vs dog, 224×224; top-$m$ = 2 |
| Stanford Dogs | 120 | fine-grained breeds — the hard case |

Data lives in `data/` (git-ignored). If a dataset is missing, the pipeline falls back to
a random batch, which is useful for plumbing and useless for results — check the config
block in the emitted JSON.

### 9.2 Bias-injection (synthetic, known ground truth)

ColoredMNIST, ColoredCIFAR10, TextureBiasedMNIST — see §6. ColoredMNIST is built with
`correlation=0.9` specifically to force a colour shortcut.

### 9.3 Medical

Two datasets, both using a pre-split `<root>/<split>/<class>/` layout with **grouped
splits** so near-duplicates never span train and validation. Split roots are declared in
`MEDICAL_SPLIT_ROOTS` (`examples/contrastive_explanation.py`); mirrored constants in
`scripts/train_backbone.py` and `scripts/ablation_contrastive.py` must stay in sync.

**HAM10000 — 7 skin lesion classes, 10,015 dermoscopic images.** The headline
contrastive question is *"why melanoma rather than a benign nevus?"* — a genuine clinical
distinction rather than a toy one.

| folder | diagnosis | ~count |
| --- | --- | ---: |
| `nv` | melanocytic nevus | 6705 |
| `mel` | melanoma | 1113 |
| `bkl` | benign keratosis | 1099 |
| `bcc` | basal cell carcinoma | 514 |
| `akiec` | actinic keratosis | 327 |
| `vasc` | vascular lesion | 142 |
| `df` | dermatofibroma | 115 |

Download from Harvard Dataverse (`doi:10.7910/DVN/DBW86T`) or the Kaggle mirror, arranged
as `data/ham10000_raw/HAM10000_metadata.csv` plus
`HAM10000_images_part_{1,2}/*.jpg`. Then:

```bash
PYTHONPATH=. python scripts/prepare_ham10000.py
```

This writes `data/ham10000/{train,val}/<dx>/` and handles two things a naive split does
not: **lesion grouping** (multiple photos exist per lesion; whole `lesion_id` groups go
to one split, since per-image splitting leaks near-duplicates and inflates accuracy) and
**class stratification** (the split runs per diagnosis, so all 7 classes appear in both
sides despite the imbalance). Flags: `--max_per_class N` (worth using — `nv` is ~67% of
the data and dominates every top-K hypothesis set, making explanations repetitive),
`--link` (symlink, saves ~2.5 GB), `--force`.

Split sizes used throughout: **7980 train / 2035 val.** HAM10000 also ships expert lesion
segmentations for all 10,015 images, which is what made the localization experiments
(§17) possible — and, ultimately, what made them uninformative.

**Brain Tumor MRI — 3 classes, 3,064 T1-weighted contrast-enhanced slices from 233
patients** (Cheng et al.): `glioma` (1426), `pituitary` (930), `meningioma` (708). There
is **no "no tumor" class** — this is a 3-way tumor-type problem. Use the figshare
original (`doi:10.6084/m9.figshare.1512427`, CC BY 4.0, credential-free), **not** the
Kaggle repackaging, which strips the patient IDs and tumor masks this project needs.
Unzip all four archives into `data/brain_tumor_raw/mats/`, then:

```bash
PYTHONPATH=. python scripts/prepare_brain_tumor.py --resize 224
```

Needs `h5py` (MATLAB v7.3 files). Writes `data/brain_tumor/{Training,Testing}/<class>/`
plus tumor masks to `data/brain_tumor_raw/masks/`. Handles **patient-grouped splits**
(3,064 slices from only 233 patients; adjacent slices of one tumor are near-identical, so
a per-slice split puts them on both sides and inflates accuracy sharply) and **per-image
intensity scaling** (int16 with per-scan ranges, not windowed; each normalized to 0–255
independently, without which darker scans wash out).

Split sizes: **2467 train / 597 val.**

> **The 7×7 grid is too coarse for these tumors.** Measured over all 3,064 masks: mean
> tumor area **1.7%** of frame, median 1.29%, against a 7×7 cell of **2.04%**. **69.8% of
> tumors are smaller than a single grid cell.** Classification is sound; localization at
> 7×7 measures grid quantization. HAM10000 does not have this problem — its lesions
> average ~27–37% of the frame.

### 9.4 Dataset caveats worth stating in any writeup

- **Not clinical evidence.** These are research artifacts, not validated for diagnostic
  use. Explanations are of a model with 0.78 balanced accuracy — often wrong.
- **Licensing.** HAM10000 is CC BY-NC 4.0 — non-commercial, and derived figures inherit
  the restriction.
- **HAM10000 has known shortcut artifacts** — ruler markings, surgical ink, dark corner
  vignetting — that correlate with malignancy because images came from different
  acquisition sites. A liability for a straight accuracy claim, but an **asset** for the
  shift game: the same dataset could show a shortcut mask locking onto rulers while the
  robust mask stays on the lesion. Not yet run.
- **Brain tumor provenance.** The Kaggle aggregation has duplicate and leakage concerns;
  accuracy on it is not publication-grade. The figshare original used here is clean, but
  treat cross-paper comparisons carefully.

### 9.5 Models and training

ImageNet-pretrained backbones, fully fine-tuned (linear probing for the earliest
non-medical runs). Medical: 20 epochs, Adam lr 1e-4, **class-weighted loss**, model
selected on **balanced accuracy**, auto-on for medical datasets. Checkpoints are
metadata-wrapped so every eval script can load them, and the filename encodes lr and
seed — a guard added after a corrected re-run silently returned a stale cached
checkpoint (§21, B1).

**Training applies no ImageNet normalization.** This is deliberate and load-bearing: every
evaluation and explanation path consumes raw $[0,1]$ tensors, and the interventions the
objective is built on — blur/mean baselines, IG's `baseline="zero"` — are defined in
$[0,1]$ space. See §21/§22 for the bug this fixed.

## 10. Metrics, and what each one can and cannot show

| metric | the question it asks | direction |
| --- | --- | --- |
| `overlap` | do different hypotheses' highlights sit on top of each other? | lower better |
| `suff` | show the model *only* the highlighted part — does it still recognize the class? | higher better |
| `margin` | does the highlighted part favour *this* hypothesis over the runner-up? | positive good |
| `sparse` | how much highlight was spent in total (the budget) | should stay constant |
| lesion/tumor mass fraction | what share of the highlight lands on the annotated pathology? | higher better — **but see below** |
| spread collapse / recovery | does shared-only equalize the hypotheses, and does unique restore one? | collapse and recovery both positive |
| K×K deletion matrix | does removing hypothesis $j$'s unique evidence hurt $j$ specifically? | dominant diagonal |
| `sho_gap`, `id_ood_gap` | is the shortcut mask in-distribution-specific? | higher better |

**Two metric warnings that shaped the whole project.**

1. **`overlap` is an unnormalized pairwise dot-product sum.** Its scale depends on
   evidence magnitude and mask mass, so absolute values are comparable only within a
   configuration. Percent reductions against the same run's base evidence are the
   meaningful quantity.
2. **Annotation overlap cannot validate a contrastive decomposition.** A lesion outline
   marks *where the lesion is*, and all seven HAM10000 classes **are** lesions — so the
   outline is ground truth for **shared** evidence, not class-**unique** evidence.
   Nothing in either medical dataset annotates what makes melanoma melanoma rather than a
   nevus. This is a **category error, not a tuning problem**, and it is why §16 and §18
   exist. Details in §17.

**The null family** used wherever a spatial claim is made:

| null | what it holds fixed | what it scrambles |
| --- | --- | --- |
| `uniform` / all-ones | nothing — pure chance | — |
| `center_cell`, `center_3x3` | fixed position, no model, no computation | — |
| Gaussian σ = H/4, H/6 | centred, smooth | — |
| **`cdea_unique_translated`** | **shape, budget, compactness, centre prior** | **position only** |

The translated null is the informative one: it is the mask's own copy, rolled to a random
position. Beating it means the mask encodes *where*, not merely *how compact and central*.

## 11. Experiment map

| # | experiment | script | data | status |
| --- | --- | --- | --- | --- |
| E1 | Contrastive ablation: base / naive / CDEA, 3 seeds | `ablation_contrastive.py`, `run_experiments.py` | MNIST, CIFAR-10, Pets, Stanford Dogs | done, **re-run corrected** |
| E2 | Robust/shortcut, 3 game modes, 3 seeds | `eval_robust_shortcut.py` | ColoredMNIST, ColoredCIFAR10, TextureMNIST | done, **re-run corrected** |
| E3 | Medical classification | `train_backbone.py` | HAM10000, brain tumor | done |
| E4 | Medical contrastive ablation, unified config | `ablation_contrastive.py` | HAM10000, brain tumor | done |
| E5 | Medical 3-seed sweep | `run_experiments.py` | HAM10000, brain tumor | done (3rd attempt — §21) |
| E6 | Interventional decomposition, Tests A + B | `eval_decomposition.py` | HAM10000, brain tumor | done, 5 configs × 2 λ arms |
| E7 | Localization + full null ladder, incl. foil ranks | `eval_localization.py` | HAM10000, brain tumor | done — **conclusion is negative** |
| E8 | Centre-prior null ladder, annotation only | `eval_center_prior.py` | HAM10000, brain tumor | done |
| E9 | Resolution sweep 7/14/28, both λ arms | `eval_localization.py`, `eval_decomposition.py` | HAM10000, brain tumor | done (v2 supersedes v1) |
| E10 | **Planted-shortcut benchmark** | `eval_shortcut.py` | CIFAR-10 + planted patch | done, 10 cells × 2 λ arms |
| E11 | Shared-mask penalty sweep | all of the above at λ=0.25 | all | done |

---
---

# Part IV — Results

## 12. Instantiation I — contrastive results

**Setup.** ResNet-18, ImageNet-pretrained, fine-tuned per dataset; 7×7 grid; top-5
hypotheses (top-2 on Pets); Grad-CAM and IG; **3 seeds, mean ± std**. Sufficiency here is
**baseline-subtracted**: $f(\text{keep}(x,m))_k - f(\mathbf{0})_k$, so positive means the
kept regions carry more signal than seeing nothing.

Source: `results/paper_rerun/contrastive/journal/JOURNAL_REPORT.md`. **These replace
`GAMBIT_PAPER.md` §5.1 and `CDEA_CONTRASTIVE_PAPER.md` §6** — see §22.

| dataset | evidence | suff ↑ | margin ↑ | overlap ↓ | overlap vs base | sparse |
| --- | --- | ---: | ---: | ---: | :---: | ---: |
| CIFAR-10 | Grad-CAM | 0.787 ± 0.071 | −1.253 ± 0.046 | 0.0085 ± 0.0001 | **−97.9%** | 1.038 |
| CIFAR-10 | IG | 0.820 ± 0.069 | −1.211 ± 0.051 | 0.0141 ± 0.0002 | **−97.3%** | 1.034 |
| MNIST | Grad-CAM | 2.794 ± 0.019 | −1.939 ± 0.028 | 0.0096 ± 0.0003 | **−97.6%** | 1.044 |
| MNIST | IG | 2.745 ± 0.020 | −2.026 ± 0.027 | 0.0259 ± 0.0003 | **−98.5%** | 1.017 |
| Pets | Grad-CAM | 1.116 ± 0.182 | **+2.290 ± 0.072** | 0.0004 ± 0.0001 | −68.0% | 1.327 |
| Pets | IG | 1.060 ± 0.178 | **+2.163 ± 0.074** | 0.0032 ± 0.0002 | −94.7% | 1.224 |
| Stanford Dogs | Grad-CAM | −2.553 ± 0.089 | −1.424 ± 0.122 | 0.0217 ± 0.0002 | −95.8% | 1.073 |
| Stanford Dogs | IG | −2.580 ± 0.091 | −1.454 ± 0.127 | 0.0257 ± 0.0008 | −96.2% | 1.058 |

**Mean overlap reduction across the 8 cells: 93.2%.** Mean margin change vs base:
**+1.248**.

**Full breakdown, CIFAR-10** (the pattern is representative):

| evidence | method | suff ↑ | margin ↑ | overlap ↓ | sparse |
| --- | --- | ---: | ---: | ---: | ---: |
| Grad-CAM | base | 0.254 ± 0.080 | −2.018 ± 0.049 | 0.398 ± 0.003 | 1.000 |
| Grad-CAM | naive | 0.259 ± 0.080 | −2.013 ± 0.048 | 0.233 ± 0.001 | 1.000 |
| Grad-CAM | **CDEA** | **0.787 ± 0.071** | **−1.253 ± 0.046** | **0.0085 ± 0.0001** | 1.038 |
| IG | base | 0.247 ± 0.080 | −2.028 ± 0.051 | 0.526 ± 0.001 | 1.000 |
| IG | naive | 0.249 ± 0.078 | −2.024 ± 0.050 | 0.296 ± 0.001 | 1.000 |
| IG | **CDEA** | **0.820 ± 0.069** | **−1.211 ± 0.051** | **0.0141 ± 0.0002** | 1.034 |

**The naive baseline is not the story.** `naive_contrastive` ($E_k - \text{mean}(E_{\text{foils}})$)
sits essentially on top of base evidence on sufficiency and margin, and halves overlap at
best (0.398 → 0.233) where the optimizer reaches 0.0085. The gain is not "subtract the
average." The same holds on medical data: naive reaches 0.278 on HAM10000/Grad-CAM
against 0.0099 optimized under the same configuration.

**What CDEA contributes** (optimized − base_evidence):

| config | Δ suff | Δ margin |
| --- | ---: | ---: |
| cifar10 / Grad-CAM | +0.532 | +0.765 |
| cifar10 / IG | +0.573 | +0.817 |
| mnist / Grad-CAM | +0.656 | +0.882 |
| mnist / IG | +0.618 | +0.798 |
| pets / Grad-CAM | +1.146 | +2.237 |
| pets / IG | +1.122 | +2.163 |
| stanford_dogs / Grad-CAM | +1.036 | +1.174 |
| stanford_dogs / IG | +0.981 | +1.151 |

**Where it still fails.** Stanford Dogs (120 fine-grained classes) keeps negative
sufficiency and margin even after the correction (−3.59 → −2.55). Overlap still collapses
(−96%) and margins still improve (+1.17), but the absolute logits stay negative under
keep interventions. **That is the one place the "negative sufficiency" limitation
survives** — on MNIST and CIFAR-10 it was purely the normalization bug.

**Qualitative artifacts.** Every run also emits `contrastive_*_split.csv` (shared-only vs
shared+unique probabilities) and `contrastive_*_pairwise.csv` (the $K \times K$ "why $k$
not $\ell$" matrix). The gallery figures overlay masks with a colorblind-safe scheme:
**blue** = unique, **orange** = shared, **purple** = claimed by both. On Pets, unique
masks for "cat" and "dog" attend to distinct facial features while the shared mask covers
the body common to both hypotheses. In the pairwise matrix, green cells mark successful
discrimination and red cells flag explanation failures — the delta panel isolates the
unique mask's marginal contribution.

## 13. Instantiation II — robust/shortcut results

3 seeds, mean ± std. Source: `results/paper_rerun/shift/journal/JOURNAL_REPORT.md`.

| dataset | mode | rob mean ↑ | rob var ↓ | sho gap ↑ | disjoint ↓ | sparse ↓ | ID−OOD gap ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| colored_mnist | cooperative | 1.405 ± 0.100 | 0.021 | 0.134 ± 0.063 | 12.02 | 14.46 | 5.341 ± 0.084 |
| colored_mnist | mixed | 1.329 ± 0.093 | 0.015 | 0.109 ± 0.024 | 0.72 | 8.09 | 5.352 ± 0.148 |
| colored_mnist | competitive | 1.239 ± 0.088 | **0.012** | 0.141 ± 0.031 | **0.62** | **7.58** | 5.331 ± 0.090 |
| colored_cifar10 | cooperative | **6.870 ± 0.216** | 0.074 | 0.438 ± 0.104 | 18.55 | 22.13 | 2.850 ± 0.065 |
| colored_cifar10 | mixed | 6.762 ± 0.225 | 0.036 | **1.000 ± 0.100** | 1.21 | 12.83 | **3.443 ± 0.118** |
| colored_cifar10 | competitive | 6.662 ± 0.243 | **0.023** | **1.001 ± 0.092** | **1.10** | **12.41** | 3.383 ± 0.158 |
| texture_mnist | cooperative | **5.030 ± 0.370** | 0.259 | 0.687 ± 0.084 | 13.29 | 16.59 | 3.417 ± 0.451 |
| texture_mnist | mixed | 4.876 ± 0.360 | 0.229 | 2.647 ± 0.254 | 1.80 | 14.04 | 5.392 ± 0.543 |
| texture_mnist | competitive | 4.742 ± 0.330 | **0.197** | **2.665 ± 0.260** | **1.52** | **13.82** | **5.401 ± 0.507** |

**The game-mode tradeoff is clean and monotone in all three datasets.**

- **Cooperative** maximizes robust sufficiency but leaves the masks entangled — disjoint
  12–19, i.e. the robust and shortcut masks are largely the *same mask*. Not a usable
  decomposition, which is what the mode is for: it is the no-separation control.
- **Competitive** minimizes overlap, variance and mass and maximizes the shortcut gap, at
  a modest robust-mean cost: −12% on ColoredMNIST, −3% on ColoredCIFAR10, −6% on
  TextureMNIST.
- **Mixed** sits between and is the sensible default.

This is exactly the designed behavior: increasing competitive pressure improves mask
separation at the cost of total evidence retention.

**Decomposition works where the shortcut is strong.** On TextureBiasedMNIST competitive,
`sho_gap` = 2.665 and `id_ood_gap` = 5.401 — the framework identifies the background
stripe texture as the shortcut. On ColoredCIFAR10 mixed, `sho_gap` = 1.000 with robust
`rob_var` = 0.036, confirming the shortcut mask captures evidence useful *only* where the
colour patch is class-correlated. Qualitatively, the shortcut mask concentrates on the
top-left colour patch while the robust mask attends to object content; on
TextureBiasedMNIST the shortcut mask highlights background stripes while the robust mask
holds the digit.

**A sanity check that the corrected numbers are the right ones.** `id_ood_gap` is a
property of the **model**, not of the allocation, so it should be near-identical across
game modes. After the fix it is: **5.331 / 5.341 / 5.352** on ColoredMNIST. Before the
fix it was 0.056 / 0.051 / 0.038 — varying *more* across modes than it did in magnitude,
which was itself the tell that something was broken. See §22.

## 14. Medical classification

Balanced accuracy (mean per-class recall) is reported, not top-1: HAM10000 is 67% `nv`,
so an always-`nv` model scores 0.67 top-1 and 0.14 balanced.

| model | dataset | balanced acc | best epoch |
| --- | --- | ---: | ---: |
| ResNet-18 | HAM10000 | 0.7379 | 5 |
| **EfficientNetV2-S** | HAM10000 | **0.7800** | 13 |
| ResNet-18 | brain tumor | 0.9693 | 11 |

Per-class recall, HAM10000:

| class | ResNet-18 | EfficientNetV2-S |
| --- | ---: | ---: |
| actinic keratosis | 0.7031 | 0.7812 |
| basal cell carcinoma | 0.7234 | 0.7553 |
| benign keratosis | 0.4762 | **0.7273** |
| dermatofibroma | 0.8333 | 0.7083 |
| melanoma | **0.7424** | 0.6638 |
| melanocytic nevus | 0.7834 | 0.8561 |
| vascular lesion | 0.9032 | 0.9677 |

Brain tumor (ResNet-18): glioma 0.9412, meningioma 0.9722, pituitary 0.9945.

Three-seed checkpoints, retrained without normalization: brain tumor 0.9522–0.9555,
HAM10000 0.7024–0.7082 — spread under 0.006, so downstream metric variance reflects
**allocation**, not model quality.

> **Flag.** EfficientNetV2-S has higher balanced accuracy but **lower melanoma recall**
> (0.742 → 0.664). Melanoma is the clinically consequential class; on a
> sensitivity-first criterion it is not the better model. Both are used below as
> explanation targets, not as clinical recommendations.

## 15. Medical separation and budget

**Q: does CDEA actually separate competing classes' evidence, and does it cheat by
shrinking the masks?**

There is an obvious way to fake a low overlap: shrink every highlight to nothing. Zero
overlap, zero information. So sufficiency and budget have to be checked alongside.

Unified allocator configuration — `game_mode=mixed`, `use_shared=True`,
`lambda_partition=0.1`, 50 steps, lr 0.2 — ResNet-18 throughout, so this and the
localization/decomposition results describe **one pipeline** rather than two.

| config | overlap base | **overlap opt** | reduction | suff base → opt | budget ratio |
| --- | ---: | ---: | :---: | ---: | ---: |
| HAM10000 / Grad-CAM | 0.4468 | **0.0729** | 84% | 0.672 → **2.473** | ×1.00 |
| HAM10000 / IG | 0.4949 | **0.1478** | 70% | 0.660 → **2.504** | ×1.02 |
| brain tumor / Grad-CAM | 0.0670 | **0.0084** | 88% | −0.001 → **0.970** | ×1.19 |
| brain tumor / IG | 0.3260 | **0.0414** | 87% | −0.107 → **1.048** | ×1.16 |

Three seeds, same configuration:

| config | overlap base | overlap optimized | reduction | sufficiency |
| --- | ---: | ---: | :---: | ---: |
| brain / Grad-CAM | 0.0610 | **0.0083 ± 0.0027** | 86% | 1.072 ± 0.226 |
| brain / IG | 0.2976 | **0.0501 ± 0.0043** | 83% | 1.118 ± 0.217 |
| HAM10000 / Grad-CAM | 0.3709 | **0.0497 ± 0.0066** | 87% | 2.131 ± 0.441 |
| HAM10000 / IG | 0.4891 | **0.1097 ± 0.0204** | 78% | 2.193 ± 0.465 |

**This is the solid empirical result of the medical work.** Overlap falls 70–88% while
sufficiency roughly *quadruples* on HAM10000 and crosses from negative to positive on
brain MRI. Evidence is **relocated, not deleted** — and on brain MRI the margin **flips
sign**, meaning that before allocation the highlighted regions argued for the *wrong*
diagnosis.

**Two caveats, stated rather than buried.**

1. **The budget is genuinely held on HAM10000 only** (×1.00, ×1.02). On brain tumor it
   drifts to ×1.16–1.19, so part of that dataset's sufficiency gain is *bought* with
   extra highlight rather than relocated. The clean "same budget" claim belongs to
   HAM10000.
2. **An earlier, non-unified configuration reported much larger reductions** — 94–98%,
   overlap down to 0.0099 on HAM10000/Grad-CAM. Those numbers are real but describe a
   *different allocator*: `use_shared=False` (no shared mask to absorb common evidence),
   `lambda_disjoint=0.5` (five times more separation pressure), 40 steps at lr 0.3. The
   ablation and localization scripts had been written against different entry points and
   silently disagreed — `OptimizationAllocator.__init__` defaults to `use_shared=False`
   and `ablation_contrastive.py` never passed the flag, so **there was no shared mask at
   all in the original ablation numbers**. Sufficiency is much better under the unified
   config (0.66 → 2.50 rather than 0.66 → 0.99). **Quote the unified 70–88%.**

## 16. Interventional validation of the decomposition

**Q: does the shared/unique split mean what the objective claims?**

Expert segmentation cannot answer this — see §10 and §17. `scripts/eval_decomposition.py`
tests the claim **against the model directly, with no annotation**, so it cannot be
confounded by a centre prior and it works on any dataset.

### Test A — does the split carry the discrimination?

`shared` is defined as evidence every candidate relies on, so keeping only the shared
mask should leave the top-K hypotheses closer to equally likely; adding hypothesis $k$'s
unique mask back should restore $k$. Reported as the top-1-minus-top-K probability spread
under each condition. Values below are the corrected **λ=0.25** runs.

| config | n | K | full | shared only | +unique | restores top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HAM / EffNetV2-S / Grad-CAM | 2035 | 5 | 0.921 | **0.680** | 0.956 | 73.2% |
| HAM / ResNet-18 / Grad-CAM | 2035 | 5 | 0.805 | **0.498** | 0.670 | 69.2% |
| HAM / ResNet-18 / IG | 2035 | 5 | 0.805 | **0.508** | 0.669 | 66.9% |
| brain / ResNet-18 / Grad-CAM | 597 | 3 | 0.976 | **0.635** | 0.834 | 93.6% |
| brain / ResNet-18 / IG | 597 | 3 | 0.976 | **0.711** | 0.822 | 90.5% |

| config | t collapse | t recovery | recovery win rate | random-deletion control |
| --- | --- | --- | --- | --- |
| HAM / EffNet / Grad-CAM | 60.6 → 44.5 | 70.4 → 55.4 | 99.1% → 98.3% | +0.003 → +0.004 |
| HAM / ResNet-18 / Grad-CAM | 94.4 → 70.3 | 66.2 → 56.6 | 99.3% → 98.1% | −0.000 → +0.000 |
| HAM / ResNet-18 / IG | 91.7 → 68.1 | 65.2 → 52.2 | 99.2% → 97.7% | +0.000 → +0.001 |
| brain / ResNet-18 / Grad-CAM | 24.5 → 33.8 | 20.8 → 23.8 | 95.5% → 90.3% | +0.007 → +0.015 |
| brain / ResNet-18 / IG | 22.3 → 28.1 | 15.6 → 14.3 | 91.8% → 79.6% | +0.006 → +0.017 |

*(arrows are λ=0 → λ=0.25)*

**Test A survives everywhere.** Collapse and recovery are both significant in **all ten
runs**: t = 14–94, recovery win rate 79.6–98.3%. The core claim does not depend on the
shared-mask configuration.

A 32-image smoke test predicted the full-set result almost exactly (0.920 / 0.607 / 0.950
versus full-set 0.921 / 0.586 / 0.935), so the effect is stable rather than a large-sample
artifact.

### Test B — is unique evidence class-specific?

Remove `unique_j` and record the change in every hypothesis's logit. If the decomposition
is real, the matrix $D[i][j]$ has a dominant diagonal. **A control matters here** —
deleting *any* region degrades the image, so a uniformly negative matrix would prove
nothing. An equal-budget random mask is deleted alongside.

HAM10000 / EfficientNetV2-S / Grad-CAM, n = 2035, λ=0 arm:

| | remove u0 | remove u1 | remove u2 | remove u3 | remove u4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| class 0 | **−0.80** | +0.08 | +0.01 | +0.01 | +0.00 |
| class 1 | +0.32 | **−0.37** | +0.03 | +0.01 | +0.01 |
| class 2 | +0.21 | +0.08 | **−0.13** | +0.02 | +0.01 |
| class 3 | +0.13 | +0.08 | +0.03 | **−0.05** | +0.01 |
| class 4 | +0.09 | +0.07 | +0.03 | +0.01 | **−0.02** |

Equal-budget random deletion control: **+0.013, −0.002, −0.003, +0.003, +0.002** — i.e.
nothing. The control stays at ≈0 (|·| ≤ 0.017) in all ten runs, so neither λ arm is
measuring generic image corruption.

**The off-diagonal is positive, and that is the contrastive claim in its strongest form.**
Removing the predicted class's unique evidence does not merely hurt that class — it
actively **helps its rivals** (+0.32, +0.21, +0.13, +0.09 down column 0). Generic
degradation cannot produce that pattern.

Class-specificity ratio |diagonal| / off-diagonal, λ=0 → 0.25:

| config | diag−off | \|diag\|/off |
| --- | --- | --- |
| HAM / EffNetV2-S / Grad-CAM | −0.337 → −0.267 | 4.39 → **4.71** |
| HAM / ResNet-18 / Grad-CAM | −0.151 → −0.114 | 4.63 → **5.91** |
| HAM / ResNet-18 / IG | −0.160 → −0.111 | 4.56 → **6.04** |
| brain / ResNet-18 / Grad-CAM | −0.223 → −0.178 | 1.76 → 1.76 |
| brain / ResNet-18 / IG | −0.312 → −0.234 | 1.86 → 1.83 |

The absolute magnitude falls ~25% under the shared-mask penalty, but the off-diagonal
shrinks proportionally — **specificity is intact; only the scale changed.**

The diagonal weakens monotonically with rank (−0.80 → −0.02), as expected: lower-ranked
hypotheses carry less unique evidence to remove.

### Why this is the validation that annotation overlap could not be

1. **No annotation is involved anywhere.** The test interrogates the model directly, so
   it works on any dataset and cannot be confounded by acquisition geometry.
2. **The positive off-diagonal is a signature no confound reproduces.** A centre prior,
   a shared blanket, or generic corruption all fail to make removing $k$'s evidence
   *help* $l$.

> **Test A is not comparable across grid resolutions.** See §19. Read it at the model's
> native grid.

## 17. Localization, and why it fails on medical data

**Q: do the allocated masks land on the actual pathology?**

HAM10000 ships expert segmentations for all 10,015 images; brain tumor ships tumor masks.
`scripts/eval_localization.py` scores mask mass inside the annotation against the full
null family. **Full validation set, n = 2035, Grad-CAM.**

| method | EfficientNetV2-S | ResNet-18 |
| --- | ---: | ---: |
| chance (flat mask) | 0.2779 | 0.2779 |
| `cdea_unique_translated` (same mask, scrambled position) | 0.2652 | 0.2647 |
| `cdea_shared` (λ=0) | 0.2674 | 0.2706 |
| `base_evidence` | 0.6515 | 0.4922 |
| **`cdea_unique`** | **0.6929** | **0.6257** |
| `center_cell` (fixed, no model) | **0.9058** | **0.9058** |

| paired comparison | EfficientNetV2-S | ResNet-18 |
| --- | --- | --- |
| unique vs **translated null** | **+0.4277** (t=52.3, wins 89.3%) | **+0.3609** (t=45.8, wins 84.3%) |
| unique vs raw evidence | +0.0415 (t=11.1, wins 65.2%) | +0.1335 (t=27.3, wins 73.5%) |
| unique vs `center_cell` | −0.2129 (t=−33.2, wins 7.2%) | −0.2802 (t=−39.3, wins 7.2%) |

**Two facts, both true, both of which belong on any slide showing this.**

1. **The mask encodes *where*.** Scrambling its position while holding shape, budget and
   compactness *exactly* drops it to chance (0.693 → 0.265). It beats that null by +0.43
   on 89% of images. So the mask carries genuine positional information.
2. **It loses to a fixed centred rectangle on 93% of images.**

### The metric is measuring acquisition geometry

`scripts/eval_center_prior.py` scores a fixed centred mask with **no model in the loop**,
swept over grid resolution (n = 1000 masks):

| dataset | chance | 7×7 | 14×14 | 28×28 | 56×56 | centroid spread (y, x) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HAM10000 | 0.2652 | **0.9398** | 0.9343 | 0.9550 | **0.9611** | ±0.065, ±0.049 |
| brain tumor | 0.0166 | **0.1312** | 0.1103 | 0.1303 | **0.1376** | ±0.100, ±0.114 |

Static, model-free masks scored against the same 2035 lesion outlines:

| "method" (no model, no computation) | lesion mass fraction |
| --- | ---: |
| uniform / all-ones ("chance") | 0.2779 |
| centred Gaussian σ = H/4 | 0.4535 |
| centred Gaussian σ = H/6 | 0.6072 |
| **fixed centre 3×3 of the 7×7 grid** | **0.7074** |
| **fixed centre 1 cell of the 7×7 grid** | **0.9058** |
| best measured CDEA (EfficientNetV2-S) | 0.6929 |

**A rectangle drawn at the middle of the image, identical for every photo, beats every
CDEA result.** The mass comparison is direct: CDEA's unique mask carries ~1 grid cell of
budget (`sparse` ≈ 1.0), and a same-budget mask at the centre scores 0.9058.

**A finer grid strengthens the null rather than weakening it.** The intuition that
smaller masks make the degenerate baseline weaker is backwards: for a centred target, a
smaller centred mask is *more* reliably inside it. HAM10000's centre-cell null rises
0.940 → 0.961 across the sweep; centre-3×3 rises 0.729 → 0.958.

Three independent routes converge on `center_cell` = 0.9058 / 0.9394 / 0.9398 — the
full-scale localization script, its 64-image smoke test, and the annotation-only
centre-prior script — so the degenerate baseline is not an artifact of one code path.

### Two separate problems, and only one is about dermoscopy

**Problem 1: the centre prior.** Dermoscopy centres the lesion by acquisition convention.
Tumor position genuinely varies across patients (twice the centroid spread), so brain
tumor has real headroom — null 0.13 against chance 0.017.

**Problem 2, the deeper one: the annotation is ground truth for the wrong thing.** A
lesion outline marks where the lesion is, and all seven classes *are* lesions. It is a
ground truth for **shared** evidence and cannot validate a class-**unique** mask. Under
one consistent objective (Stage 7, §19), `cdea_unique` never beats the strongest null on
either dataset at any resolution — best cell 0.85×.

**Verdict.** The paired CDEA-vs-raw-evidence and CDEA-vs-translated-null comparisons are
fair — same images, same budget, same centre bias, `lambda_mass` matching them by
construction — and both favour CDEA. **Any absolute statement of the form "CDEA localizes
lesions well" does not hold.** The `headroom` column emitted by `eval_localization.py`
normalizes against the uniform baseline and inherits the same flaw; do not quote it
without the centre-prior table.

### Diminishing returns

Three independent conditions line up monotonically:

| starting heatmap quality | CDEA lift | endpoint |
| --- | ---: | ---: |
| bad — IG + ResNet-18 (0.3398) | **+0.2299** | 0.5697 |
| medium — Grad-CAM + ResNet-18 (0.4922) | **+0.1335** | 0.6257 |
| good — Grad-CAM + EffNetV2-S (0.6515) | **+0.0415** | 0.6929 |

**The better the base evidence, the less allocation adds.** Like a proofreader: enormous
value on a rough draft, barely noticeable on a polished one.

Reads both ways. *For* CDEA — it is contributing real work, not passing good input
through; a method that merely sharpened its input would show a lift that *grows* with
input quality, not shrinks. *Against* — on a strong modern backbone the contribution is
+0.0415, and much of the gain looks like correcting weak attribution rather than adding
explanatory content a good attribution method would not already have. Relatedly,
EfficientNetV2-S's **raw** heatmap (0.6515) beats ResNet-18's **fully CDEA-processed**
output (0.6257): upgrading the backbone bought more localization than the entire
allocation machinery did on the weaker one.

### Foil masks — the "rather than L" half

Rank 0 is the predicted class; ranks 1–4 are foils. n = 2035, Grad-CAM.

**ResNet-18**

| rank | raw heatmap | CDEA unique | lift |
| ---: | ---: | ---: | ---: |
| 0 (predicted) | 0.4922 | **0.6257** | +0.1335 |
| 1 | 0.4635 | 0.4771 | +0.0136 |
| 2 | 0.4144 | 0.4007 | **−0.0137** |
| 3 | 0.2803 | 0.2700 | **−0.0103** |
| 4 | 0.1768 | 0.1775 | +0.0007 |

**EfficientNetV2-S**

| rank | raw heatmap | CDEA unique | lift |
| ---: | ---: | ---: | ---: |
| 0 (predicted) | 0.6515 | **0.6929** | +0.0415 |
| 1 | 0.4739 | 0.4962 | +0.0223 |
| 2 | 0.2568 | 0.2979 | +0.0411 |
| 3 | 0.1545 | 0.1947 | +0.0402 |
| 4 | 0.1256 | 0.1658 | +0.0402 |

1. **Foil degradation is model-specific, not intrinsic.** With ResNet-18, CDEA *hurts*
   ranks 2–3; with EfficientNetV2-S it helps every rank by a uniform ~+0.04. An earlier
   reading of this as a property of the method was an over-generalization from one model.
2. **The contrastive split is real.** Rank 0 is significantly more lesion-focused than
   rank 1: **+0.1486** (t=18.8, wins 67.4%) for ResNet-18, **+0.1967** (t=24.0, wins
   71.7%) for EfficientNetV2-S. The better classifier gives the sharper separation.
3. By ranks 3–4 both raw and CDEA masks fall **below chance** — those masks sit on
   background and are not meaningful explanations of "why not class L."

## 18. The planted-shortcut benchmark

**This is the one experiment with exact, per-pixel ground truth for class-*unique*
evidence, and it is the strongest positive result in the project.**

Every other evaluation lacked one, and the reason was structural (§10, §17).
`scripts/eval_shortcut.py` manufactures a ground truth instead: a small distinctive patch
— 32px on a 224px frame, **2.04% of area, exactly one 7×7 cell** — is pasted into a
fraction of one CIFAR-10 class's training images **at randomized positions**. The model
learns to use it (verified: attack success rate 92.3% at rate 1.0), and then "does
`unique_k` land on the patch?" has an exact answer.

Two properties make this clean where lesion overlap was not:

- **Ground truth is known per image, to the pixel**, and it is genuinely *unique* to one
  class rather than shared across all of them.
- **Position is randomized**, so the centre prior that makes HAM10000 unmeasurable is
  **absent by construction**. The degenerate baselines have nothing to exploit.

n = 105 per cell, mixed mode, 50 allocator steps, ResNet-18, 4 epochs. Chance = 0.0204.

| cell | evidence | grid | base evidence | **cdea_unique** | translated null | centre cell | × chance |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| planted | Grad-CAM | 7×7 | 0.0897 | **0.3568** | 0.0134 | 0.0297 | **17.5×** |
| planted | IG | 7×7 | 0.0582 | **0.3562** | 0.0138 | 0.0297 | **17.5×** |
| planted | IG | 14×14 | 0.0649 | **0.3617** | 0.0160 | 0.0278 | **17.7×** |
| planted | IG | 28×28 | 0.0666 | **0.2086** | 0.0245 | 0.0202 | 10.2× |
| planted | occlusion | 7×7 | 0.1252 | **0.3280** | 0.0127 | 0.0297 | 16.1× |
| planted | occlusion | 14×14 | 0.1076 | **0.2959** | 0.0175 | 0.0278 | 14.5× |
| planted | occlusion | 28×28 | 0.0725 | **0.1540** | 0.0245 | 0.0202 | 7.5× |
| **control** (no patch) | IG | 28×28 | 0.0144 | **0.0183** | 0.0200 | 0.0202 | **0.90×** |

Paired tests, planted / IG / 7×7:

| comparison | delta | t | win rate |
| --- | ---: | ---: | ---: |
| unique vs base evidence | **+0.2981** | 38.1 | **100%** |
| unique vs uniform (chance) | **+0.3358** | 43.5 | **100%** |
| unique vs translated null | **+0.3425** | 37.7 | 98.1% |
| unique vs centre cell | **+0.3266** | 26.8 | 97.1% |

**Dose-response.** Vary the fraction of the class's images carrying the patch, holding
everything else fixed (IG, 28×28):

| shortcut rate | attack success | **cdea_unique** | vs base | × chance |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 32.2% | 0.0588 | +0.0338 (t=14.4) | 2.9× |
| 0.50 | 62.1% | 0.0940 | +0.0609 (t=17.5) | 4.6× |
| 1.00 | 92.3% | 0.2086 | +0.1420 (t=33.8) | 10.2× |

The mask tracks **how much the model actually relies on the shortcut** — monotone in the
planting rate, with the same monotonicity in attack success.

**The control run is the decisive check.** With no patch planted, `cdea_unique` sits at
0.90× chance and *loses* to the uniform null (t = −3.03, wins 31%). The method finds the
planted evidence when it exists and **finds nothing when it does not**.

**Four things this establishes that no other experiment here does.**

1. `cdea_unique` beats **every** null — uniform, centre cell, and its own
   position-scrambled copy — on the same images at the same budget, on **97–100%** of them.
2. It beats **raw attribution** by 3–6× on all three evidence providers.
3. **`cdea_shared` correctly finds nothing** on the planted patch (0.003–0.008 at λ=0,
   0.014–0.032 at λ=0.25). The patch is unique to one class, and the shared mask declines
   to claim it — the decomposition is behaving as specified, not just producing a
   low number somewhere.
4. It holds across **three independent evidence providers**, including occlusion, which
   shares no assumptions with the two gradient-based ones.

**It also killed a design rule** — see §23, R2.

## 19. Resolution

`mass_scale = max(1, R / mass_ref_regions)` keeps the mask a constant *fraction* of the
frame as the grid refines, so at 28×28 the unique budget is **16×** what it is at 7×7.
That is deliberate, and it has consequences.

### Localization across resolution

Stage 7 v2, both λ arms under **one binary** (the v1 sweep compared two different
objectives — §21, B3). Scores are × the **strongest** null (max of uniform / centre cell /
translated):

| | 7×7 λ=0 → 0.25 | 14×14 λ=0 → 0.25 | 28×28 λ=0 → 0.25 |
| --- | --- | --- | --- |
| brain tumor | 0.68× → 0.64× | 0.84× → 0.85× | 0.54× → 0.56× |
| HAM10000 | 0.65× → 0.61× | 0.53× → 0.51× | 0.35× → 0.35× |

Three findings:

1. **`lambda_shared_sparse` has essentially no effect on unique localization** — the two
   arms differ by ≤0.005 at every grid on both datasets.
2. **CDEA never beats the strongest null** on either medical dataset at any resolution;
   the best cell is 0.85×. Consistent with the category error in §17 — and in direct
   contrast to the planted benchmark (§18), where it beats every null at every resolution.
3. **The shared-mask penalty works at every grid.** `cdea_shared` rises from ~chance to
   well above it: brain 0.013 → 0.035/0.037/0.039, HAM 0.265 → 0.567/0.503/0.453.

The **position-scrambled null improves monotonically** on both datasets (brain:
+0.081 → +0.182 → +0.242, winning 79% → 90% → 92% of images), so the mask encodes location
at every resolution. That comparison and the centre-rectangle comparison answer different
questions, and only the first is a statement about the method.

### Test A is not comparable across grids — a property of the metric

`spread` is `max − min` over per-class probabilities where **each element is measured on
its own image**: entry $k$ is $P(k \mid \text{keep}(x, m_{\text{shared}} + m_{\text{unique},k}))$.
Larger unique budget → every per-class condition keeps more image → between-class spread
compresses. Far enough to flip sign:

| grid | λ | shared only | +unique | recovery | t | diag−off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7×7 | 0 | 0.7128 | 0.8235 | **+0.1107** | 15.88 | −0.312 |
| 7×7 | 0.25 | 0.7105 | 0.8233 | **+0.1128** | 14.52 | −0.234 |
| 14×14 | 0 | 0.5212 | 0.5980 | +0.0768 | 7.93 | −0.545 |
| 14×14 | 0.25 | 0.6062 | 0.5751 | −0.0310 | −2.87 | −0.434 |
| 28×28 | 0 | 0.3999 | 0.2930 | −0.1068 | −8.92 | −0.776 |
| 28×28 | 0.25 | 0.4943 | 0.2005 | −0.2938 | −25.74 | −0.740 |

Confirmed by varying **only** `mass_ref_regions` at 28×28 (λ=0, n=96):

| `mass_ref_regions` | `mass_scale` | unique mass | recovery |
| ---: | ---: | ---: | ---: |
| 49 (current default) | 16.0 | 15.98 | −0.1111 |
| 784 | 1.0 | 1.06 | **+0.1973** |

which reproduces the pre-fix value (+0.1899) exactly. Note also that **Test B strengthens
monotonically with resolution** (−0.312 → −0.545 → −0.776) while Test A inverts. **The two
tests disagreeing is the signature of Test A tracking the budget rather than the
decomposition.**

**Rule: read Test A at the model's native grid**, where `mass_scale` = 1.0 and recovery is
solidly positive at both λ.

Three alternative hypotheses were tested and killed before this one: **under-convergence**
(more steps makes it *worse* — 50/150/400 → 0.2995/0.0427/0.0234), **`lambda_shared_sparse`**
(negative at λ=0 too), and **unclamped mask overflow** (0.01–0.30% of entries; clamping
moves every metric only in the 4th decimal).

### The cells-per-target observation

Dividing mean target area by cell area:

| | 7×7 (cell 2.041%) | 14×14 (0.510%) | 28×28 (0.128%) |
| --- | ---: | ---: | ---: |
| brain tumor (1.76% of frame) | 0.9 cells | 3.5 | **13.8** ← best |
| HAM10000 lesion (27.4%) | **13.4** ← best | 53.7 | 214.9 |

Both peak at ~13 cells per target, approached from opposite directions — too coarse and
the mask cannot represent a target smaller than one cell; too fine and the fixed mass
budget spreads so thin the mask goes diffuse. **This is an observation about these two
datasets, not a design rule** — see §23, R2.

## 20. The shared-mask fix

**Found by looking at gallery figures**: allocated evidence appeared in places the base
evidence never marked. It turned out not to be drift in the unique masks.

**The shared mask carried no penalty term at all.** `lambda_sparse` applies only to the
unique masks, `lambda_overlap` only to unique–unique pairs, `lambda_mass` pins unique
mass. Its only brake was the allocator's partition cap, which **never binds on average**
(mean region occupancy 0.56 against a cap of 1.0). Since the shared mask enters
$m_k^{\text{tot}}$, growing it always raises sufficiency and margin — so the optimizer grew
it without limit.

Measured over 96 HAM10000 val images at 7×7, Grad-CAM, ResNet-18:

| | HAM10000 λ=0 → 0.25 | brain_tumor λ=0 → 0.25 |
| --- | --- | --- |
| shared mass (of 49) | 22.56 → **2.76** (8.2× smaller) | 19.49 → **2.71** (7.2×) |
| shared regions above 0.5 | 47.2% → **3.5%** | 39.9% → **3.2%** |
| **shared: evidence captured / area** | **0.99× → 1.48× chance** | 0.93× → 0.86× chance |
| unique: evidence captured / area | 3.35× → 3.47× chance | 4.71× → 4.23× chance |
| unique mass | 1.01 → 1.03 | 1.20 → 1.47 |
| unique–unique overlap | 0.058 → 0.052 | 0.020 → 0.020 |
| mean region occupancy (cap 1.0) | 0.56 → 0.16 | 0.47 → 0.15 |

**The bolded row is the finding.** At λ=0 the shared mask captures base evidence at
*exactly its own area fraction* — statistically uncorrelated with the very field it is
supposed to be dividing up. **It is not a mask that found the background; it is a mask
that found nothing** — a blanket over roughly half the frame.

**The unique masks were never the problem** (3.35× chance either way, 2.04% of frame
area), so every result resting on `cdea_unique` — the separation numbers, Stage 4
localization, the resolution sweep, the shortcut benchmark — is unaffected. Verified
directly: the non-medical contrastive journal reports at λ=0 and λ=0.25 are
**byte-identical** on the main results table.

**What changed downstream.** On HAM10000 the λ=0 Test A collapse was **partly an
artifact**: keeping a soft mask spread over 46% of the frame is a *global degradation* of
the input, so part of what looked like "the hypotheses become equally likely when you keep
only shared evidence" was just a washed-out image. The honest collapse is 0.805 → 0.508,
not 0.805 → 0.420 — and recovery *improves* at the same time (0.615 → 0.669). **Better
evidence, from a narrower gap.**

**Brain tumor behaves differently and it is not explained.** Its shared mask compacts just
as much (7.2×) but shared-only spread barely moves (−0.036, −0.005) and evidence capture
stays at chance (0.93× → 0.86×). **A compact shared mask is necessary but not
sufficient.** Untested candidates: K=3 vs 5, a near-saturated model (full spread 0.976),
or genuinely shared anatomy that the class-conditioned evidence field does not mark.

**Recommendation.** `lambda_shared_sparse` defaults to **0.0** in
`ContrastiveObjective.__init__` so every prior run reproduces exactly. Use **0.25 for
anything that reports a statistic *about the shared mask***. The deck, `F4_decomposition`,
and `results_full.ipynb` §2b all read the 0.25 run.

One incidental note: the Grad-CAM and IG shared means agree to four decimal places
(0.27063 / 0.27055) while **zero** of 2035 per-image scores match — independent
computations converging, not a bug. Under the correction above the agreement is *expected*
rather than striking: both are converging on the area fraction of a blanket.

---
---

# Part V — Corrections

## 21. Bugs found, and what each invalidated

| # | bug | blast radius | fix |
| --- | --- | --- | --- |
| **B1** | **Train/eval preprocessing mismatch** — `train_backbone.py` applied ImageNet normalization; *every* eval and explanation path consumes raw [0,1] tensors | **Every checkpoint in `scripts/out/checkpoints/`.** CIFAR-10 probe 0.389 raw vs 0.816 normalized; brain tumor seed 0 **0.337 vs 0.948**; HAM10000 seed 0 **0.161 vs 0.713**. All `ablation_*`, `shift_*` and journal numbers feeding both paper drafts were computed on models running at ~half their true accuracy | Normalization **removed from training** rather than added to eval — the interventions the objective is built on (blur/mean baselines, IG `baseline="zero"`) are defined in [0,1] space. Full re-run in `results/paper_rerun/`. See §22 |
| **B2** | Unpenalized shared mask (§20) | Every statistic *about the shared mask*; unique-mask results unaffected | `lambda_shared_sparse`, default 0.0, use 0.25 |
| **B3** | `mass_scale` mask-budget fix landed mid-project; an Aug-9 λ=0 baseline was then compared against an Aug-16 λ=0.25 run | The whole v1 resolution sweep at 14×14 and 28×28 (7×7 unaffected — `mass_scale` = 1.0 there). Produced a **fake sign reversal**; see §23 R1 | Resolution sweep v2 re-runs **both** λ arms under one binary |
| **B4** | **Subset sampling was a class-ordered prefix** — `range(N)` over a class-sorted `ImageFolder`, so `--num_images 200` scored a few classes rather than a sample | Every subset result. Chance was **0.481 on a 32-image prefix vs 0.279 on a random sample of the same size** (full set 0.2779). This is the cause of the "200-image subset was not representative for IG" anomaly (base 0.436 on subset vs 0.340 at full scale) | Seeded `randperm` |
| **B5** | `--num_steps` accepted but never forwarded by `run_experiments.py` | Any run that believed it was varying allocator steps | Plumbed through; allocator config now recorded in the emitted JSON |
| **B6** | **DataLoader deadlock** — hardcoded `num_workers=4` deadlocks against MPS on long runs: workers idle, main process blocks on a queue read that never returns, job sits at 0% CPU indefinitely rather than failing | ~7.5 h of overnight compute lost | `--num_workers`, default 0 |
| **B7** | `ablation_contrastive.py` printed "at comparable sufficiency (diff=1.71)" as a pass condition. 1.71 is not comparable | Wording only — the conclusion survived because sufficiency *improved* | Now reports the sufficiency change and mask-budget ratio, and says PASS only when sufficiency did not fall and the budget did not grow |
| **B8** | `VisionGridUnitSpace._region_to_pixel_mask` unclamped; `keep` blends `m*x + (1−m)*baseline`, so a mask sum above 1 extrapolated *past* `x` | 0.01–0.30% of entries; measured by running one config twice in-process, every metric moves in the 4th decimal (recovery 0.1906 → 0.1900). **Invalidates nothing** | Clamp to [0,1]; regression test `test_keep_remove_clamp_out_of_range_masks` |

**A wrong diagnosis worth recording.** Stage 5 (the medical 3-seed sweep) failed twice
before succeeding. The second attempt blamed learning rate — `get_or_train` defaults to
1e-3 and `run_experiments.py` could not override it. Plausible, and **false**: the re-run
at 1e-4 failed identically. It was B1. The lr override was added anyway (`--train_lr`),
and **the checkpoint filename now encodes the lr**, because without that a corrected
re-run silently returns the broken cached checkpoint.

**Guards added so these fail loudly next time:**

| failure | guard |
| --- | --- |
| Grad-CAM returns an all-zero field on ViT (non-negativity does not hold for LayerNorm'd tokens) | `RuntimeWarning` naming the cause, pointing at IG |
| Training converges to chance | warning + banner at `balanced_acc ≤ 1.15 × chance`; sweeps verify every checkpoint before reporting |
| `NaN` in result JSON silently breaks every non-Python reader | `save_json` refuses to emit it; `_json_safe` maps to `null` |
| A finer grid requested for Grad-CAM would fabricate resolution | `--grid` raises unless `--evidence ig` |
| Class-ordered subsets | seeded `randperm` instead of `range(N)` |
| A corrected re-run silently returns a stale cached checkpoint | checkpoint filename encodes lr and seed |

## 22. The normalization correction in full

Because B1 touched everything feeding both paper drafts, the old and corrected numbers
are recorded side by side. 3 seeds, mean ± std. Source:
`results/paper_rerun/CORRECTION.md`.

Measured on the same eval: the CIFAR-10 checkpoint scored **0.388** under the mismatched
pipeline versus **0.809** retrained without normalization — and **0.816** for the old
checkpoint evaluated *with* normalization. **Removing normalization costs no accuracy; it
only makes training and evaluation agree.**

### Sufficiency and margin

| config | method | suff (old) | suff (corrected) | margin (old) | margin (corrected) |
| --- | --- | ---: | ---: | ---: | ---: |
| cifar10_gradcam | base | −0.868 | **0.254 ± 0.080** | −1.488 | **−2.018 ± 0.049** |
| cifar10_gradcam | naive | −0.868 | **0.259 ± 0.080** | −1.488 | **−2.013 ± 0.048** |
| cifar10_gradcam | optimized | −0.685 | **0.786 ± 0.071** | −1.205 | **−1.253 ± 0.046** |
| cifar10_ig | base | −0.861 | **0.247 ± 0.080** | −1.399 | **−2.028 ± 0.051** |
| cifar10_ig | naive | −0.864 | **0.249 ± 0.078** | −1.398 | **−2.024 ± 0.050** |
| cifar10_ig | optimized | −0.655 | **0.820 ± 0.069** | −1.099 | **−1.211 ± 0.051** |
| mnist_gradcam | base | −1.132 | **2.138 ± 0.014** | −2.119 | **−2.821 ± 0.020** |
| mnist_gradcam | naive | −1.134 | **2.141 ± 0.014** | −2.119 | **−2.816 ± 0.020** |
| mnist_gradcam | optimized | −1.100 | **2.794 ± 0.019** | −2.063 | **−1.939 ± 0.028** |
| mnist_ig | base | −1.109 | **2.127 ± 0.014** | −2.140 | **−2.824 ± 0.022** |
| mnist_ig | naive | −1.100 | **2.154 ± 0.011** | −2.127 | **−2.838 ± 0.019** |
| mnist_ig | optimized | −1.082 | **2.745 ± 0.020** | −2.077 | **−2.026 ± 0.027** |
| pets_gradcam | base | 0.471 | **−0.029 ± 0.242** | 0.014 | **0.053 ± 0.004** |
| pets_gradcam | naive | 0.471 | **−0.028 ± 0.241** | 0.014 | **0.054 ± 0.003** |
| pets_gradcam | optimized | 0.683 | **1.116 ± 0.182** | 0.444 | **2.290 ± 0.072** |
| pets_ig | base | 0.471 | **−0.062 ± 0.248** | 0.000 | **−0.000 ± 0.001** |
| pets_ig | naive | 0.464 | **−0.052 ± 0.248** | 0.005 | **0.005 ± 0.041** |
| pets_ig | optimized | 0.727 | **1.060 ± 0.178** | 0.537 | **2.163 ± 0.073** |
| stanford_dogs_gradcam | base | −4.731 | **−3.589 ± 0.097** | −1.814 | **−2.598 ± 0.133** |
| stanford_dogs_gradcam | naive | −4.694 | **−3.509 ± 0.097** | −1.787 | **−2.533 ± 0.139** |
| stanford_dogs_gradcam | optimized | −4.344 | **−2.552 ± 0.088** | −1.318 | **−1.424 ± 0.122** |
| stanford_dogs_ig | base | −4.738 | **−3.560 ± 0.096** | −1.818 | **−2.604 ± 0.131** |
| stanford_dogs_ig | naive | −4.713 | **−3.516 ± 0.095** | −1.803 | **−2.569 ± 0.137** |
| stanford_dogs_ig | optimized | −4.349 | **−2.580 ± 0.090** | −1.348 | **−1.453 ± 0.127** |

### The correction *widens* CDEA's measured benefit everywhere

A properly trained model has sharper, better-separated logits, so a diffuse raw-evidence
mask leaves a larger deficit while the optimized mask holds. **The bug was understating
the method.**

| config | Δsuff (old) | Δsuff (corrected) | Δmargin (old) | Δmargin (corrected) |
| --- | ---: | ---: | ---: | ---: |
| cifar10_gradcam | +0.183 | **+0.532** | +0.283 | **+0.765** |
| cifar10_ig | +0.206 | **+0.573** | +0.300 | **+0.817** |
| mnist_gradcam | +0.032 | **+0.656** | +0.057 | **+0.882** |
| mnist_ig | +0.027 | **+0.618** | +0.064 | **+0.798** |
| pets_gradcam | +0.212 | **+1.146** | +0.431 | **+2.237** |
| pets_ig | +0.255 | **+1.122** | +0.537 | **+2.163** |
| stanford_dogs_gradcam | +0.386 | **+1.036** | +0.497 | **+1.174** |
| stanford_dogs_ig | +0.389 | **+0.981** | +0.470 | **+1.151** |

### The shift experiments — the larger of the two corrections

| config | id_ood_gap (old) | id_ood_gap (corrected) |
| --- | ---: | ---: |
| colored_cifar10_competitive | 0.4843 | **3.3828 ± 0.158** |
| colored_cifar10_cooperative | 0.4052 | **2.8497 ± 0.065** |
| colored_cifar10_mixed | 0.4761 | **3.4432 ± 0.118** |
| colored_mnist_competitive | 0.0557 | **5.3310 ± 0.090** |
| colored_mnist_cooperative | 0.0509 | **5.3412 ± 0.084** |
| colored_mnist_mixed | 0.0376 | **5.3521 ± 0.148** |
| texture_mnist_competitive | 0.5786 | **5.4010 ± 0.507** |
| texture_mnist_cooperative | **−0.2161** | **3.4170 ± 0.451** |
| texture_mnist_mixed | 0.5274 | **5.3915 ± 0.543** |

ColoredMNIST is built with `correlation=0.9` **precisely to force a colour shortcut**, yet
the old numbers (~0.04) said the model barely used it — roughly **100× smaller** than the
corrected ~5.34. One config had a *negative* gap. **The shift game was being evaluated on
models that did not measurably exhibit the dependency it exists to decompose**, so every
robust-vs-shortcut conclusion in the drafts needed re-deriving.

Two checks that the corrected numbers are trustworthy: seed variance is tight
(±0.06–0.15 on MNIST/CIFAR), and `id_ood_gap` is now near-identical across game modes
(5.331 / 5.341 / 5.352 on colored_mnist), which is **correct** — it is a property of the
model, not the allocation.

### Consequence for the paper drafts

- `GAMBIT_PAPER.md` §8.1 lists "negative sufficiency and margins with weak backbones" as
  a limitation. On CIFAR-10 and MNIST that **reverses sign entirely** once training and
  evaluation agree — it was an artifact there. It **survives on Stanford Dogs**
  (−4.34 → −2.55, still negative), so **narrow the claim to fine-grained many-class
  settings rather than deleting it**.
- Every sufficiency and margin figure in both drafts needs replacing. Overlap is less
  affected, being a property of mask geometry rather than of logits.
- `examples/` results are **unaffected** — `contrastive_explanation.py` never normalized,
  which is why its checkpoints always worked.
- Nothing in `scripts/out/` was overwritten, so the original (buggy) numbers remain
  available for comparison — and must not be quoted.

## 23. Retractions

Three claims were made internally and then withdrawn. Recorded here so they are not
re-quoted from older files.

**R1 — "CDEA beats a centred rectangle on brain tumor at 14×14 and 28×28; the result
reverses sign with resolution."** *Retracted.* The comparison put a pre-fix λ=0 baseline
against a post-fix λ=0.25 run (§21, B3). Re-running the identical command at λ=0 — same
seed, same n=597 — gives recovery **−0.1068** against the stored **+0.1899**. Under one
consistent objective the same commands give **0.68× / 0.84× / 0.54×** the strongest null
at 7/14/28: **CDEA never beats the centred rectangle at any resolution on brain tumor.**
The 7×7 row was unaffected and still reproduces exactly, because `mass_scale` is 1.0 there.

**R2 — "Choose the grid by dividing expected target area by cell area (~13 cells per
target)."** *Retracted as a design rule.* It fit both medical datasets (§19) and is
directly contradicted by the planted-shortcut benchmark: a 32px patch on a 224px frame is
2.04% of the area, so the rule predicts 28×28 — and **28×28 is the worst resolution there**
(10.2× chance) while 14×14 is the best (17.7×). Two datasets agreeing is a coincidence,
not a law. **Choose the grid empirically.**

**R3 — "The shared mask sits on the background, which is exactly what the decomposition
claims — the most robust finding in the study."** *Retracted;* see §20. The λ=0 shared
mask captured base evidence at **0.99× chance** — statistically uncorrelated with the field
it was supposed to be dividing. It found nothing, and the number was near-tautological:
anything covering half the frame scores close to its own area fraction on an overlap
metric by construction. The unique rows are unaffected.

---
---

# Part VI — Status

## 24. What is established, what is not

### Established

1. **Separation.** CDEA takes attribution maps that are 45–50% redundant across competing
   hypotheses and separates them to ~1–15% redundant: 8 non-medical cells at **93.2% mean
   reduction** (3 seeds) and 4 medical cells at **70–88%** (3 seeds).
2. **Without cheating.** Sufficiency *increases* in every configuration while the budget
   stays roughly constant — evidence is relocated, not deleted. The budget is genuinely
   held on HAM10000 (×1.00–1.02) and on all non-medical datasets; it drifts on brain tumor
   (×1.16–1.19), and that is stated wherever brain-tumor sufficiency is quoted.
3. **The decomposition means what the objective claims.** Interventional Tests A and B
   across ten runs, t = 14–94, with an equal-budget random-deletion control at ≈0. The
   **positive off-diagonal** — removing class $k$'s unique evidence *helps* its rivals —
   is the contrastive claim in its strongest form and cannot be produced by generic
   corruption.
4. **On a benchmark with real ground truth, the unique mask finds the right thing.**
   Planted-shortcut: **17.5× chance**, beating uniform, centre-cell and position-scrambled
   nulls on **97–100%** of images, across three independent evidence providers, with a
   monotone dose-response and a control run that correctly finds nothing (0.90× chance).
   The shared mask correctly declines to claim the class-unique patch.
5. **Game modes behave as designed.** Competitive → maximum separation and shortcut gap at
   modest robust-mean cost; cooperative → maximum robust sufficiency with entangled masks;
   mixed between. Monotone across all three shift datasets.
6. **The masks encode position.** Scrambling location while holding shape, budget and
   compactness exactly drops a mask to chance: +0.36 to +0.43 on HAM10000 (84–89% of
   images), +0.34 on the planted benchmark (98%).

### Not established

1. **That the masks are *good* in absolute terms on medical data.** A fixed centred
   rectangle beats every measured CDEA result on HAM10000 (0.906 vs 0.693, winning 93% of
   images), and under one consistent objective CDEA never beats the strongest null at any
   resolution on either medical dataset.
2. **That annotation overlap is the right test at all.** A lesion outline is ground truth
   for *shared* evidence — all seven classes are lesions. **A category error, not a tuning
   problem**, and the reason §18 exists.
3. **Why brain tumor's shared mask stays at chance** even when compacted 7.2×.
4. **Clinical validity.** Nothing here is clinically validated. Lesion overlap is a proxy
   for explanation quality, not a measure of it — a mask can sit on the lesion and still be
   a poor explanation to a clinician. **No human evaluation was done.** The explained model
   is often wrong (0.78 balanced accuracy).
5. **That CDEA adds much on top of a strong backbone.** On EfficientNetV2-S the
   localization lift is +0.0415, and that backbone's *raw* heatmap already beats
   ResNet-18's *fully processed* output.
6. **Robust/shortcut on real (non-synthetic) shift.** Every shift result is on
   bias-injection datasets with a planted signal. HAM10000's ruler/ink artifacts are the
   obvious real-data test and have not been run.

### Honest one-paragraph summary

CDEA is a **disentangling** tool and an effective one. It converts "where did the model
look?" into "what made it pick A over B?" — a question raw attribution structurally cannot
answer — and the planted-shortcut benchmark shows the separated unique evidence lands on
genuinely class-unique signal when such signal exists and on nothing when it does not. It
is **not** a localization tool, and on medical images the available spatial metric is
measuring acquisition geometry and shared anatomy rather than explanation quality.

## 25. Open questions and next steps

**High value**

1. **Extend the planted-shortcut design to a medical dataset.** It is the only evaluation
   here with real ground truth for unique evidence and the only one immune to the centre
   prior. Planting a patch into HAM10000 would give the medical arm a defensible positive
   result.
2. **Run the shift game on HAM10000's natural artifacts** (rulers, surgical ink, corner
   vignetting). It is the only available *real* distribution-shift test, the dataset is
   already prepared, and it would move Instantiation II off synthetic data.
3. **Explain brain tumor's shared mask.** Three untested candidates (K=3 vs 5,
   near-saturated model, genuinely shared anatomy). Cheapest test: re-run HAM10000 with
   K=3 and check whether its shared capture drops toward chance.
4. **Re-derive both paper drafts against the corrected numbers.** `GAMBIT_PAPER.md` §5 and
   `CDEA_CONTRASTIVE_PAPER.md` §6 still carry pre-B1 values throughout, and §8.1's
   negative-sufficiency limitation needs narrowing to fine-grained settings.
5. **Set `mass_ref_regions = R`** (or report Test A only at the native grid) so resolution
   comparisons stop tracking the budget rather than the decomposition.

**Medium**

6. Human evaluation, or at minimum a clinician read of a curated case set. The gap between
   "mask sits on pathology" and "explanation is useful" is currently unmeasured.
7. IG localization on EfficientNetV2-S — only Grad-CAM was run there.
8. **ViT-B/16 at 14×14 with IG evidence** — the only path to a fair brain-tumor spatial
   test at a resolution its 1.7%-of-frame tumors can actually be expressed in.
9. Confidence intervals and significance testing as pipeline defaults rather than
   per-script additions.
10. Interaction ablation at scale (`none` vs `attention` vs `transformer`) — implemented
    and tested, but never swept as an experiment.

**Lower / architectural**

11. Amortized allocation — a learned mask predictor — to remove the O(T·K) forward-pass
    cost per explanation.
12. Adaptive stopping on gradient norm or mask entropy instead of fixed steps.
13. Exercise the text-token and graph-node unit space stubs.
14. New instantiations: fairness-aware allocation, temporal consistency for video,
    multi-modal evidence allocation.

---
---

# Part VII — Practical

## 26. Reproduction

Environment: the `marl` conda env; all commands from the repo root with `PYTHONPATH=.`.
Device used throughout: **MPS (Apple Silicon)**.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate marl
```

**Non-medical contrastive + shift, 3 seeds** (the corrected run):

```bash
PYTHONPATH=. python scripts/run_experiments.py --datasets cifar10 mnist pets stanford_dogs --seeds 0 1 2 --out_dir results/paper_rerun/contrastive
```

**Medical training** (modern backbone):

```bash
PYTHONPATH=. python examples/contrastive_explanation.py --dataset ham10000 --model efficientnet_v2_s --train --epochs 20 --lr 1e-4 --batch_size 32 --pretrained --num_alloc_steps 50 --num_viz_samples 5
```

**Unified-config ablation** (overlap / sufficiency / margin):

```bash
PYTHONPATH=. python scripts/ablation_contrastive.py --dataset ham10000 --evidence gradcam --model resnet18 --game_mode mixed --num_steps 50 --lr 0.2 --num_images 400
```

**Interventional decomposition** — the validation that actually works on medical data:

```bash
PYTHONPATH=. python scripts/eval_decomposition.py --checkpoint examples/out/checkpoints/ham10000_efficientnet_v2_s.pt --num_images 2035 --game_mode mixed --num_alloc_steps 50 --lambda_shared_sparse 0.25 --out_dir results/medical_presentation/decomposition_sharedfix --export_prefix decomp_ham10000_effnet_gradcam
```

**Planted-shortcut benchmark** — the clean ground truth:

```bash
PYTHONPATH=. python scripts/eval_shortcut.py --grid 28 --evidence ig --out_dir results/shortcut
```

**Centre-prior null ladder** (annotation only, no model):

```bash
PYTHONPATH=. python scripts/eval_center_prior.py --num_masks 1000
```

**Localization with the full null family**, incl. foil ranks:

```bash
PYTHONPATH=. python scripts/eval_localization.py --checkpoint examples/out/checkpoints/ham10000_efficientnet_v2_s.pt --num_images 2035 --num_alloc_steps 50 --evidence gradcam --export_prefix localization_foil_efficientnet_v2_s
```

**Shift-aware game standalone:**

```bash
PYTHONPATH=. python scripts/eval_robust_shortcut.py --game_mode mixed
```

**Tests:**

```bash
PYTHONPATH=. python -m pytest tests/
```

**Batch drivers** live beside their outputs: `results/medical_presentation/run_stage1.sh`,
`run_resolution_sweep.sh`, `run_stage5_fixed.sh`,
`results/shortcut/run_shortcut_sweep.sh`, `results/paper_rerun/run_nonmedical.sh`.
**Figures:** `scripts/plot_hero_figure.py`, `plot_presentation_figures.py`,
`plot_evidence_comparison.py`, `plot_shortcut_figure.py`. **Deck:**
`node scripts/build_deck.js`.

**Dataset prep:** `scripts/prepare_ham10000.py`, `scripts/prepare_brain_tumor.py` — see §9.3.

## 27. Artifact map

### Result trees

| path | contents | trust |
| --- | --- | --- |
| `results/paper_rerun/` | non-medical contrastive + shift, 3 seeds, post-B1 | **current** |
| `results/shortcut/`, `results/shortcut_sharedfix/` | planted-shortcut benchmark, 10 cells each | **current** |
| `results/medical_presentation/decomposition_sharedfix/` | Tests A+B, 5 configs, λ=0.25 | **current** |
| `results/medical_presentation/resolution_v2/` | resolution sweep, both λ arms, one objective | **current** |
| `results/medical_presentation/localization*/`, `seeds*/`, `ablation*/` | Stages 3–5 | current |
| `results/medical_presentation/figures/` | F1_hero … F7_shortcut_hero (png + pdf), center_prior | current |
| `results/medical_presentation/deck/` | pptx + recorded talk | current |
| `results/medical_presentation/decomposition/`, `resolution/` | λ=0 arms, kept for cross-reference | superseded |
| `scripts/out/` | pre-B1 runs, **never overwritten** | **stale — do not quote** |
| `examples/out/` | example-pipeline figures and metrics | unaffected by B1 (that path never normalized) |

### File conventions

| pattern | contents |
| --- | --- |
| `ablation_*_metrics.json` | overlap / suff / margin / sparse, with allocator config block |
| `ablation_*_per_batch.csv` | per-batch distributions for stability analysis |
| `localization_*.json` | per-method means, per-image scores, paired t-tests vs each null |
| `decomp_*.json` | Test A spreads, Test B K×K matrix, random-deletion control |
| `sc_*_g<grid>.json` | shortcut benchmark: per-method rows, paired tests, attack success rate |
| `contrastive_*_split.csv` | shared-only vs shared+unique probabilities |
| `contrastive_*_pairwise.csv` | K×K "why k not ℓ" margin matrix |
| `summary_*.{json,csv}`, `journal/JOURNAL_REPORT.md` | multi-seed rollups |
| `examples/out/checkpoints/<dataset>_<model>.pt` | example-path checkpoints |
| `results/*/checkpoints/<dataset>_<model>_pt_ft_ep20_lr<lr>_seed<n>.pt` | metadata-wrapped; lr and seed in the filename (B1 guard) |

### Figures for the talk / paper

`F1_hero` (the pitch) · `F2_separation` (overlap reduction) · `F3_budget` (relocation not
deletion) · `F4_decomposition` (Tests A+B, λ=0.25) · `F5_center_prior` and `F5_resolution`
(why the spatial metric fails) · `F6_foil_ranks` (the "rather than L" half) ·
`F7_shortcut_hero` (the planted benchmark).

## 28. Adding a new game instantiation

**Decide the game formulation before writing code.** Players/masks, objective terms, data
context (single batch `x` or multi-env `env`), target signal, and metrics to report. If
this is not fixed, implementation destabilizes quickly.

**Layout.** Create `instantiations/<your_game>/` with `objective.py`, `allocator.py`, and
optionally `env.py`. Add a runner in `scripts/eval_<your_game>.py` and tests in
`tests/test_<your_game>.py`.

**Objective first.** Follow the `AllocationObjective` protocol in `core/objective.py`:

```python
def compute(self, x, model, unit_space, hypotheses, masks, evidence,
            tokens=None, attn=None, env=None, **kwargs):
    ...
    return {"loss": loss, ...metrics...}
```

Rules: return a scalar tensor as `loss`; keep metric names stable and explicit (they
become CSV/JSON columns); compute faithfulness via `unit_space.keep(...)` /
`.remove(...)`, **never from raw evidence**; make behavior explicit when `env is None`.

**Allocator second.** Follow `core/allocator.py`. Typical pattern: init mask logits from
`evidence` (or zeros) → optimize with Adam for `num_steps` → sigmoid to $[0,1]$ → call the
objective each step → add structural penalties → return a mask dict with stable keys, e.g.
`{"unique": ...}`, `{"unique": ..., "shared": ...}`, `{"robust": ..., "shortcut": ...}`.

**Reuse the kernel.** Wire through `CDEAExplainer` with a `unit_space`, `selector`,
`base_evidence` provider, your allocator and objective, and optional `interaction` — do
not rewrite orchestration.

**Practical design considerations, most of them learned the hard way here:**

1. **Intervention baseline choice** (`mean` vs `blur`) changes metric behavior.
2. **Mask collapse risk** — all-zero or all-one masks happen without balanced penalties.
3. **Metric leakage** — always compute metrics via interventions, not raw evidence.
4. **Mass control** — use mass/partition penalties if masks drift. **And penalize every
   mask you optimize**: the shared-mask bug (§20) was exactly a mask that entered the
   objective's reward term with no penalty term of its own.
5. **Top-K dependence** — objective behavior changes with hypothesis count $K$.
6. **Compute cost** — per-step objective with model forwards is expensive; profile early.
7. **Interaction optionality** — stay valid when `interaction=None`.
8. **Reproducibility** — lock seeds and log the **full config** in the saved JSON. Several
   of the bugs in §21 were only diagnosable because the config block was there, and B5
   existed because part of it was not.
9. **Check your metric is comparable across the axis you plan to sweep.** §19 is a
   cautionary tale: an entire resolution sweep was built on a metric that tracked the mask
   budget rather than the quantity of interest.
10. **Build a null family, not a single baseline.** Uniform, fixed-position, and
    self-translated. §17 is what happens when the degenerate baseline is only discovered
    after the conclusions are written.

**Definition of done:** objective and allocator implemented; example script runs
end-to-end and saves artifacts; tests pass in `marl`; metrics are interpretable (not
degenerate or all near-zero); docs contain the run command and metric definitions.

**Immediately after the first successful run, do one ablation** — interaction on/off,
evidence provider A vs B, or two or three regularization settings. That quickly tells you
whether the new game is structurally valid or just numerically fitting noise.
