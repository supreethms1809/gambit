# CDEA Prelims: Method and Algorithm Details

This file is a thesis-writing draft tied directly to the current implementation.

## 1) Problem Setup

Given an input \(x\), a model \(f\), and an evidence-unit space \(\mathcal{U}\) with \(R\) units, we first define a hypothesis set \(H(x)\) as the top-\(m\) classes selected from \(f(x)\). For each hypothesis \(k \in H(x)\), base evidence provider \(g\) produces nonnegative evidence scores \(E_k(u)\) over units \(u \in \mathcal{U}\), yielding tensor \(E \in \mathbb{R}_{\ge 0}^{B \times K \times R}\).

## 2) Kernel Pipeline

The implemented `CDEAExplainer.explain` pipeline (`/Users/ssuresh/gambit/core/runner.py`) is:

1. Forward pass: `logits = f(x)`, `probs = softmax(logits)`.
2. Hypothesis selection: `H = TopMSelector(m)`.
3. Base evidence extraction: `E = provider.explain(x, f, H)`.
4. Evidence normalization (optional): per `(B,K)` normalization over `R`.
5. Optional token construction:
   - If `phi(u)` exists from `embed_units`, tokens are
   - \(t_k = \sum_u E_k(u)\phi(u)\), implemented by `einsum("bkr,brd->bkd", ...)`.
6. Optional interaction over tokens: `none`, `attention`, or `transformer-1layer`.
7. Allocation optimization to produce explanation masks.
8. Objective computation via intervention tests (`keep/remove`).
9. Return structured `Explanation`.

## 3) Interaction Ablation

Implemented in `/Users/ssuresh/gambit/core/interaction.py`.

Modes:

1. `none`: no interaction, `attn = None`.
2. `attention`: attention-only residual update over hypothesis tokens.
3. `transformer`: one-layer Transformer (self-attn + FFN + residual/LN).

Interaction is optional and does not change the core objective definitions.

## 4) Instantiation I: Contrastive Shared-Unique

Implementation:

- Objective: `/Users/ssuresh/gambit/instantiations/contrastive/objective.py`
- Allocator: `/Users/ssuresh/gambit/instantiations/contrastive/allocator.py`

### 4.1 Mask variables

1. Unique masks: \(m^{\text{unique}} \in [0,1]^{B \times K \times R}\)
2. Optional shared mask: \(m^{\text{shared}} \in [0,1]^{B \times R}\)
3. Effective mask per hypothesis:
   - \(m_k^{\text{tot}} = m_k^{\text{unique}} + m^{\text{shared}}\) when shared is enabled
   - \(m_k^{\text{tot}} = m_k^{\text{unique}}\) otherwise

### 4.2 Contrastive objective terms (implemented)

For each sample and hypothesis \(k\):

1. Keep intervention: \(x_k^{\text{keep}} = \text{keep}(x, m_k^{\text{tot}})\)
2. Sufficiency signal: \(z_k = f(x_k^{\text{keep}})[k]\)
3. Contrastive margin:
   - \(z_k - \max_{l \ne k,\, l \in H(x)} z_l\)

Batch-level penalties:

1. Overlap among unique masks: pairwise dot-product sum.
2. Sparsity: mean L1 mass of unique masks.
3. Mass deviation: deviation from base evidence mass.

Implemented optimization target:

\[
\mathcal{L}_{\text{contrastive}} =
-\left(\lambda_{\text{suff}}\cdot \overline{\text{suff}} + \lambda_{\text{margin}}\cdot \overline{\text{margin}}\right)
+ \lambda_{\text{overlap}}\cdot \overline{\text{overlap}}
+ \lambda_{\text{sparse}}\cdot \overline{\text{sparse}}
+ \lambda_{\text{mass}}\cdot \overline{\text{mass\_dev}}
\]

Game-mode presets control these weights:

1. `cooperative`: sets `lambda_margin=0` and removes separation penalties.
2. `competitive`: increases margin/separation emphasis and disables shared mask.
3. `mixed`: balanced default.
4. `manual`: explicit user-provided weights.

### 4.3 Probability split reporting

Current code also computes:

1. Shared-only probabilities (`keep(m_shared)`)
2. Shared+unique probabilities (`keep(m_shared + m_unique[k])`)
3. Pairwise margins for "why \(k\) rather than \(l\)" under both settings
4. Delta between shared+unique and shared-only margins

This is implemented as metric tensors in `ContrastiveObjective.compute`.

### 4.4 Allocation solver

`OptimizationAllocator` optimizes mask logits with Adam:

1. Initialize from evidence logits.
2. Optionally blend evidence by attention (`attn_mix`).
3. Apply sigmoid to logits each step.
4. Add disjointness and optional partition penalties.
5. Clamp logits for numerical stability.

## 5) Instantiation II: Shift-aware Robust-Shortcut

Implementation:

- Objective: `/Users/ssuresh/gambit/instantiations/shift/objective.py`
- Allocator: `/Users/ssuresh/gambit/instantiations/shift/allocator.py`
- Environment generation: `/Users/ssuresh/gambit/instantiations/shift/env.py`

### 5.1 Inputs

`EnvBatch` stores multiple views of the same instance:

1. `x_id` (in-distribution view)
2. `x_ood1`, `x_ood2` (shifted views via recoloring/augmentations)

### 5.2 Mask variables

1. \(m_{\text{rob}} \in [0,1]^{B \times R}\)
2. \(m_{\text{sho}} \in [0,1]^{B \times R}\)

### 5.3 Objective terms (implemented)

For each env \(e\), sufficiency is defined as baseline-subtracted target logit:

\[
\text{suff}(m, x_e) = f(\text{keep}(x_e,m))_y - f(\text{keep}(x_e,0))_y
\]

Then:

1. `rob_mean`: mean robust sufficiency across environments
2. `rob_var`: variance of robust sufficiency across environments
3. `sho_gap`: ID shortcut sufficiency minus OOD shortcut sufficiency mean
4. `sho_mean`: mean shortcut sufficiency across environments (used for cooperative utility)
5. `disjoint`: overlap penalty \(m_{\text{rob}} \odot m_{\text{sho}}\)
6. `sparse`: average mask mass penalty

Implemented loss:

\[
\mathcal{L}_{\text{shift}} =
-\left(\lambda_m\cdot \text{rob\_mean} - \lambda_v\cdot \text{rob\_var} + \lambda_g\cdot \text{sho\_gap} + \lambda_{sh}\cdot \text{sho\_mean}\right)
+ \lambda_d\cdot \text{disjoint}
+ \lambda_s\cdot \text{sparse}
\]

Game-mode presets control these weights:

1. `cooperative`: `lambda_gap=0`, `lambda_disjoint=0`, positive `lambda_shortcut`.
2. `competitive`: larger `lambda_gap` and `lambda_disjoint`.
3. `mixed`: balanced default.
4. `manual`: explicit user-provided weights.

### 5.4 Solver

`RobustShortcutOptimizationAllocator` optimizes two mask logits jointly with Adam and sigmoid-constrained masks. Disjointness is applied in the objective term (single source), avoiding double counting.

## 6) Current Claims Supported by Implementation

Statements you can safely make in prelims text:

1. The proposed framework is modular and supports multiple evidence providers with a shared kernel.
2. The contrastive instantiation supports explicit shared and unique masks plus pairwise "why k not l" reporting.
3. The shift-aware instantiation supports robust-shortcut decomposition across environment batches.
4. Interaction is optional and can be ablated without changing the core kernel API.
5. The implementation already generates both visual artifacts and machine-readable metrics.

## 7) Suggested Thesis Paragraph (Method)

We formulate explanation as evidence allocation over a fixed evidence-unit space. For each input, we select top hypotheses and compute nonnegative base evidence over units using attribution methods (e.g., Grad-CAM or Integrated Gradients). Optional hypothesis interaction conditions top-hypothesis token representations, while the central allocation remains an optimization problem over continuous masks constrained to \([0,1]\). In the contrastive setting, we optimize shared and unique masks to balance sufficiency, pairwise margin, overlap, and sparsity. In the shift-aware setting, we optimize robust and shortcut masks across environment variants to maximize robust mean sufficiency, minimize robust variance, and maximize shortcut ID-OOD gap under disjointness and sparsity regularization.

```
PYTHONPATH=. python scripts/run_experiments.py --datasets mnist cifar10 pets stanford_dogs --shift_datasets colored_mnist colored_cifar10 texture_mnist --evidence gradcam ig --seeds 0 1 2 --num_steps 40 --train_epochs 100
```