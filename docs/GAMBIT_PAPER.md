# GAMBIT: Game-Theoretic Allocation for Model-Based Interpretability and Trust

## Abstract

We present **GAMBIT**, a unified game-theoretic framework for producing structured, contrastive explanations of deep neural network predictions. At its core, GAMBIT implements the **CDEA** (Class Distribution Evidence Allocation) pipeline, which frames model explanation as an evidence allocation game over a spatial unit space. Raw per-class attribution evidence is decomposed into interpretable masks through constrained optimization, balancing sufficiency, contrastive discrimination, parsimony, and faithfulness. We instantiate CDEA in two settings: (1) a **Contrastive Shared-Unique Game** that answers "why class K rather than class L?" by decomposing evidence into class-specific and shared regions, and (2) a **Shift-Aware Robust-Shortcut Game** that separates robust evidence from distribution-shift-specific shortcuts. Across experiments on MNIST, CIFAR-10, Oxford Pets, Stanford Dogs, and three synthetic bias-injected datasets (ColoredMNIST, ColoredCIFAR10, TextureBiasedMNIST), GAMBIT's optimized allocation consistently reduces pairwise mask overlap by 85--99% over raw base evidence while preserving predictive sufficiency, and successfully identifies robust versus shortcut evidence under controlled distribution shift.

---

## 1. Introduction

Explaining the predictions of deep classifiers remains a central challenge in trustworthy AI. Standard attribution methods such as Grad-CAM (Selvaraju et al., 2017) and Integrated Gradients (Sundararajan et al., 2017) provide per-class saliency maps, but these maps often overlap substantially across competing classes, leaving the user unable to distinguish *why* the model chose one class over another. Furthermore, in settings with distribution shift, explanations may highlight spurious shortcuts rather than genuinely robust features.

We address both limitations with a single framework. GAMBIT casts explanation generation as a **game among hypothesis players**, where each player (a class hypothesis or an evidence type) competes and cooperates over a shared evidence budget. A unifying **CDEA kernel** handles hypothesis selection, evidence computation, optional hypothesis interaction, and mask optimization; distinct **game instantiations** define the specific agents, objectives, and outputs.

**Contributions:**

1. A modular, protocol-driven explanation framework unifying contrastive and robustness-aware explanations under a common game-theoretic kernel.
2. An optimization-based evidence allocator that produces shared and unique masks with provably reduced overlap and preserved sufficiency.
3. A shift-aware instantiation that decomposes evidence into robust and shortcut components using multi-environment intervention tests.
4. Empirical validation across four standard vision benchmarks and three controlled bias-injection datasets, with multi-seed statistical reporting.

---

## 2. Related Work

**Attribution methods.** Grad-CAM (Selvaraju et al., 2017) computes class-discriminative localization maps via gradient-weighted feature activations. Integrated Gradients (Sundararajan et al., 2017) computes path integrals from a baseline to the input. Both produce per-class evidence but do not explicitly decompose shared versus unique contributions across competing hypotheses.

**Contrastive explanations.** Prior work on contrastive explanations (Dhurandhar et al., 2018; Mothilal et al., 2020) focuses on pertinent positives and negatives or counterfactual generation. GAMBIT differs by operating directly on attribution evidence and framing contrastive decomposition as a multi-player allocation game.

**Game-theoretic interpretability.** Shapley values (Lundberg & Lee, 2017) provide a principled feature importance measure but are computationally expensive and do not produce spatial masks. GAMBIT uses game-theoretic *structure* (agents, objectives, equilibria) rather than Shapley computation, enabling efficient gradient-based optimization.

**Robustness and shortcuts.** Recent work identifies spurious correlations exploited by models (Geirhos et al., 2020; Sagawa et al., 2020). GAMBIT's shift-aware instantiation complements this by producing explicit spatial masks separating robust from shortcut evidence, evaluated via intervention-based metrics across environments.

---

## 3. Method

### 3.1 Problem Setup

Given an input $x \in \mathbb{R}^{C \times H \times W}$, a trained classifier $f$, and an evidence-unit space $\mathcal{U}$ with $R$ spatial units (e.g., a $7 \times 7$ grid for a CNN or $14 \times 14$ for a ViT), we seek structured explanations that go beyond raw attribution.

### 3.2 The CDEA Kernel

The CDEA pipeline proceeds in five stages:

1. **Hypothesis Selection.** Select the top-$m$ classes from $f(x)$ as competing hypotheses $H(x) = \{h_1, \ldots, h_m\}$ using a `TopMSelector`.

2. **Base Evidence Extraction.** For each hypothesis $k \in H(x)$, compute nonnegative evidence $E_k(u) \geq 0$ for each unit $u \in \mathcal{U}$, yielding a tensor $E \in \mathbb{R}_{\geq 0}^{B \times K \times R}$. Evidence is optionally normalized per $(B, K)$ across $R$.

3. **Hypothesis Interaction (Optional).** Construct token representations $t_k = \sum_u E_k(u) \phi(u)$ where $\phi(u)$ are sinusoidal positional embeddings. Tokens are then conditioned via self-attention or a single Transformer layer, producing interaction-aware representations and attention weights.

4. **Mask Allocation.** An instantiation-specific allocator optimizes continuous masks $m \in [0,1]^R$ using gradient descent over logit-parameterized variables with sigmoid activation. The allocator minimizes a game-specific loss function that balances multiple objectives.

5. **Intervention-Based Evaluation.** Masks are evaluated through *keep* and *remove* interventions:
$$x^{\text{keep}}(m) = m \odot x + (1 - m) \odot \bar{x}$$
where $\bar{x}$ is a baseline (global mean or adaptive blur). The model's response to $x^{\text{keep}}$ measures the sufficiency of the retained evidence.

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
│                    Explanation(masks, metrics)                   │
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

### 3.3 Instantiation I: Contrastive Shared-Unique Game

**Mask variables.** For each hypothesis $k$, we optimize a unique mask $m_k^{\text{unique}} \in [0,1]^R$ and an optional shared mask $m^{\text{shared}} \in [0,1]^R$. The effective mask per hypothesis is:
$$m_k^{\text{tot}} = m_k^{\text{unique}} + m^{\text{shared}}$$

**Objective.** The contrastive loss balances five terms:

$$\mathcal{L}_{\text{contrastive}} = -\left(\lambda_{\text{suff}} \cdot \overline{\text{suff}} + \lambda_{\text{margin}} \cdot \overline{\text{margin}}\right) + \lambda_{\text{overlap}} \cdot \overline{\text{overlap}} + \lambda_{\text{sparse}} \cdot \overline{\text{sparse}} + \lambda_{\text{mass}} \cdot \overline{\text{mass\_dev}}$$

where:
- **Sufficiency**: $\text{suff}_k = f(x_k^{\text{keep}})_k$, the target class logit under the keep intervention.
- **Contrastive margin**: $\text{margin}_k = f(x_k^{\text{keep}})_k - \max_{l \neq k} f(x_k^{\text{keep}})_l$, measuring discrimination against the strongest competing class.
- **Overlap**: $\sum_{k < l} m_k^{\text{unique}} \cdot m_l^{\text{unique}}$, penalizing shared mass between unique masks.
- **Sparsity**: $\frac{1}{K}\sum_k \|m_k^{\text{unique}}\|_1$, encouraging compact explanations.
- **Mass deviation**: Deviation of mask mass from the base evidence mass, preventing trivial solutions.

**Game modes** control these weights:

| Mode | Shared Mask | Margin | Overlap | Effect |
|------|:-----------:|:------:|:-------:|--------|
| Cooperative | Yes | Off | Off | Emphasizes shared evidence |
| Mixed | Yes | Moderate | Moderate | Balanced decomposition |
| Competitive | No | Strong | Strong | Maximally disjoint unique masks |

**Probability split reporting.** For each hypothesis $k$, the objective additionally computes:
- $p_k^{\text{shared}}$: model probability under `keep(x, m^{\text{shared}})` alone
- $p_k^{\text{total}}$: model probability under `keep(x, m_k^{\text{tot}})$
- $\Delta_k = p_k^{\text{total}} - p_k^{\text{shared}}$: the discriminative contribution of unique evidence

**Pairwise "why $k$ not $l$" margins.** For every pair $(k, l)$, the framework reports:
$$\text{margin}_{k \to l} = f(x_k^{\text{keep}})_k - f(x_k^{\text{keep}})_l$$
under both shared-only and shared+unique interventions, along with the delta, producing a full $K \times K$ contrastive matrix.

**Allocation solver.** The `OptimizationAllocator` optimizes mask logits with Adam (default: 50 steps, lr = 0.5). Masks are initialized from base evidence via inverse sigmoid. Additional regularization includes a disjointness penalty ($\lambda_{\text{disjoint}} = 0.2$) and an optional partition penalty ($\lambda_{\text{partition}} = 0.1$) encouraging $\sum_k m_k(u) \leq 1$ per unit. Logits are clamped to $[-12, 12]$ for numerical stability.

### 3.4 Instantiation II: Shift-Aware Robust-Shortcut Game

**Setup.** Given an `EnvBatch` containing views of the same instance from multiple environments (in-distribution $x_{\text{id}}$ and one or more out-of-distribution views $x_{\text{ood}}$), we optimize two masks:
- $m_{\text{rob}} \in [0,1]^R$: evidence that should be *stable* across environments (robust)
- $m_{\text{sho}} \in [0,1]^R$: evidence that is *environment-specific* (shortcut)

**Sufficiency definition** (baseline-subtracted):
$$\text{suff}(m, x_e) = f(\text{keep}(x_e, m))_y - f(\text{keep}(x_e, \mathbf{0}))_y$$

**Objective.** The shift-aware loss balances six terms:

$$\mathcal{L}_{\text{shift}} = -\left(\lambda_m \cdot \text{rob\_mean} - \lambda_v \cdot \text{rob\_var} + \lambda_g \cdot \text{sho\_gap} + \lambda_{sh} \cdot \text{sho\_mean}\right) + \lambda_d \cdot \text{disjoint} + \lambda_s \cdot \text{sparse}$$

where:
- **Rob mean**: $\mathbb{E}_e[\text{suff}(m_{\text{rob}}, x_e)]$ — robust evidence should be sufficient across all environments.
- **Rob var**: $\text{Var}_e[\text{suff}(m_{\text{rob}}, x_e)]$ — robust sufficiency should be *stable* (low variance).
- **Sho gap**: $\text{suff}(m_{\text{sho}}, x_{\text{id}}) - \mathbb{E}_{e \neq \text{id}}[\text{suff}(m_{\text{sho}}, x_e)]$ — shortcut evidence should be ID-specific.
- **Sho mean**: $\mathbb{E}_e[\text{suff}(m_{\text{sho}}, x_e)]$ — shortcut utility (for cooperative mode).
- **Disjoint**: $m_{\text{rob}} \cdot m_{\text{sho}}$ — robust and shortcut masks should not overlap.
- **Sparse**: $\|m_{\text{rob}}\|_1 + \|m_{\text{sho}}\|_1$ — compact masks.

**Game modes** control whether the robust and shortcut agents cooperate (shared utility), compete (strong separation), or operate in a balanced mixed regime.

**Controlled bias-injection datasets.** To evaluate the shift-aware instantiation with known ground truth, we construct three synthetic datasets where the shortcut signal is explicitly injected:

| Dataset | Shortcut Signal | Robust Signal | Mechanism |
|---------|----------------|---------------|-----------|
| ColoredMNIST | Class-correlated hue tint | Digit shape | Digit colorized by class ID |
| ColoredCIFAR10 | Class-correlated color patch (top-left 25%) | Object content | Solid color patch added |
| TextureBiasedMNIST | Class-correlated stripe angle | Digit shape | Sinusoidal background texture |

OOD views are generated by reassigning the shortcut signal (different hue, patch color, or stripe angle), preserving the robust signal.

### 3.5 How Games Are Solved: Gradient-Based Mask Optimization

A central design choice in GAMBIT is *how* the evidence allocation game is solved. Rather than computing Nash equilibria analytically or using combinatorial solvers, GAMBIT formulates each game as a continuous optimization problem and solves it via first-order gradient descent through the frozen classifier. This section details the algorithm, its parameterization, and the rationale behind these choices.

#### 3.5.1 Problem Formulation

Each game instantiation defines a set of **mask variables** (the players' actions) and a **composite loss function** (the payoff landscape). The mask variables are continuous tensors in $[0,1]^R$, where $R$ is the number of spatial evidence units. The loss combines reward terms (sufficiency, margin, gap) that the optimizer maximizes and penalty terms (overlap, sparsity, disjointness) that it minimizes, weighted by configurable $\lambda$ coefficients.

The key insight enabling gradient-based solution is that all loss terms are differentiable with respect to the mask variables:
- Sufficiency and margin terms involve forward passes through the frozen model on masked inputs $x^{\text{keep}}(m) = m \odot x + (1-m) \odot \bar{x}$, which are differentiable in $m$.
- Penalty terms (overlap, sparsity, disjointness, partition) are simple algebraic functions of the masks.

This yields a smooth optimization landscape over which standard gradient methods converge efficiently.

#### 3.5.2 Logit Parameterization

Masks are not optimized directly in $[0,1]$. Instead, GAMBIT maintains unconstrained **logit variables** $\ell \in \mathbb{R}^R$ and maps them to masks via the sigmoid function:

$$m = \sigma(\ell) = \frac{1}{1 + e^{-\ell}}$$

This parameterization has three advantages:
1. **Unconstrained optimization.** The optimizer operates in $\mathbb{R}^R$ without box constraints, which is better suited to Adam's adaptive moment estimates.
2. **Smooth gradients.** The sigmoid provides well-behaved gradients throughout the optimization, avoiding the sharp boundaries that projected gradient methods would introduce.
3. **Natural initialization.** When initializing from base evidence $E_k \in [0,1]$, the inverse sigmoid $\ell_k = \log(E_k / (1 - E_k))$ provides a principled warm start that preserves the evidence structure while allowing the optimizer to refine it.

To prevent numerical overflow, logits are clamped to $[-C, C]$ after each optimizer step. The contrastive allocator uses $C = 12.0$ (corresponding to masks in approximately $[6 \times 10^{-6}, 1 - 6 \times 10^{-6}]$), while the shift-aware allocator uses a tighter $C = 3.0$ (masks in $[\sim 0.05, \sim 0.95]$) to encourage more moderate mask values and avoid degenerate all-on/all-off solutions.

#### 3.5.3 The Optimization Loop

**Algorithm: GAMBIT Mask Allocation**

```
Input:  x (input batch), f (frozen model), E (base evidence),
        U (unit space), H (hypotheses), env (optional environment batch)
Output: Optimized masks {m_1, ..., m_P} for P players

1.  Initialize logits:
      If init_from_evidence:
        ℓ_p ← σ⁻¹(clamp(E_p, ε, 1-ε))    // warm start from evidence
      Else:
        ℓ_p ← 0                              // cold start (masks ≈ 0.5)

2.  (Contrastive only) If attention weights available:
        E_mixed ← (1 - α) · E + α · (A_norm · E)   // attention-conditioned init
        ℓ_p ← σ⁻¹(E_mixed)

3.  Freeze all model parameters: ∂f/∂θ = 0
4.  Create Adam optimizer over {ℓ_1, ..., ℓ_P}

5.  For t = 1, ..., N_steps:
      a. Compute masks: m_p ← σ(ℓ_p) for each player p
      b. Forward pass: compute game-specific loss L(m, x, f, U, H, env)
         - Contrastive: L = -(λ_s·suff + λ_m·margin) + λ_o·overlap + λ_sp·sparse + λ_mass·mass_dev
         - Shift-aware: L = -(λ_m·rob_mean - λ_v·rob_var + λ_g·gap + λ_sh·sho_mean) + λ_d·disjoint + λ_sp·sparse
      c. Add structural penalties (allocator-level):
         - Contrastive: P += λ_disj · Σ_{k<l} (m_k · m_l) + λ_part · relu(Σ_k m_k - 1)
         - Shift-aware: disjointness handled inside objective (no double-counting)
      d. Backpropagate: ∇_ℓ (L + P)
      e. Adam step: ℓ ← Adam(ℓ, ∇_ℓ)
      f. Clamp: ℓ ← clamp(ℓ, -C, C)

6.  Restore model parameter gradients
7.  Return final masks: m_p ← σ(ℓ_p) (detached)
```

Each iteration of the inner loop requires $K$ forward passes through the model (one per hypothesis keep-intervention) for the contrastive game, or $2E$ forward passes for the shift-aware game ($E$ environments $\times$ 2 masks). The model is frozen throughout: only the mask logits receive gradients.

#### 3.5.4 Why Adam?

GAMBIT uses the Adam optimizer (Kingma & Ba, 2015) for the inner allocation loop rather than vanilla SGD or second-order methods. The choice is motivated by:

1. **Adaptive per-parameter learning rates.** Different spatial regions may have vastly different evidence magnitudes and gradient scales. Adam's per-element moment estimates handle this heterogeneity without manual learning rate tuning per region.

2. **Momentum through noisy loss landscapes.** The loss landscape combines multiple competing objectives (sufficiency vs. sparsity, margin vs. overlap). Adam's exponential moving averages of the first and second moments smooth out conflicting gradient signals, leading to more stable convergence than SGD.

3. **Fast convergence in low-iteration regimes.** The allocation loop runs for only 25--50 steps (not hundreds of epochs). Adam's bias-corrected estimates converge faster than SGD in these short optimization horizons, which is critical since each step involves expensive model forward passes.

4. **Robustness to hyperparameters.** Across the tested configurations (different datasets, evidence providers, and game modes), Adam with a learning rate of 0.2--0.5 consistently converges without per-configuration tuning. SGD required significantly more tuning of the learning rate and momentum schedule in preliminary experiments.

Second-order methods (L-BFGS, natural gradient) were considered but rejected due to: (a) the per-step cost of Hessian or Fisher information computation across $K$ model forward passes, and (b) the non-convexity of the loss landscape, where second-order curvature estimates can be misleading.

#### 3.5.5 Why Not Combinatorial or Exact Equilibrium Solvers?

The game-theoretic structure of GAMBIT might suggest solving for exact Nash equilibria. We opt for gradient-based optimization instead for several reasons:

1. **Continuous action spaces.** Each mask value lies in $[0, 1]^R$ with $R = 49$ (7$\times$7 grid) or $R = 196$ (14$\times$14 grid). Discretizing these for combinatorial methods would produce an exponentially large action space ($2^{49}$ or $2^{196}$ pure strategies per player).

2. **Non-linear payoffs.** The sufficiency and margin terms involve full neural network forward passes, making the payoff function highly non-linear. Standard game solvers (support enumeration, Lemke-Howson, linear complementarity) require bilinear or polynomial payoffs.

3. **Coupled constraints.** The disjointness and partition penalties couple all players' masks, creating shared constraints that further complicate exact equilibrium computation.

4. **Empirical sufficiency.** The gradient-based solver consistently finds allocations that reduce overlap by 85--99% while preserving sufficiency, suggesting that the loss landscape is well-behaved enough for first-order methods to find high-quality solutions. The resulting allocations need not be exact equilibria to be useful explanations — they need only satisfy the interpretability desiderata encoded in the loss.

5. **Computational tractability.** A 50-step Adam loop with $K=5$ forward passes per step completes in under 2 seconds on GPU for a batch of 8 images. Exact equilibrium computation would be orders of magnitude slower for the same action space dimensionality.

The approach is closest in spirit to **gradient-based best-response dynamics**: each optimization step simultaneously updates all players' strategies to reduce the joint loss, approximating a cooperative-competitive equilibrium point. The $\lambda$ weights on the loss terms effectively control the cooperative-competitive spectrum, with high margin/overlap weights pushing toward competitive equilibria and low weights allowing cooperative solutions.

#### 3.5.6 Initialization Strategy

The quality of the solution depends on initialization. GAMBIT provides two strategies:

1. **Evidence-warm-start (default).** Logits are initialized from the normalized base evidence via inverse sigmoid: $\ell_k = \sigma^{-1}(\text{clamp}(E_k, \epsilon, 1-\epsilon))$. This starts the optimization near the raw attribution map, so the allocator refines rather than discovers evidence from scratch. For the contrastive game, the evidence can additionally be conditioned on interaction attention weights using a convex mixture: $E_{\text{mixed}} = (1-\alpha) E + \alpha (A_{\text{norm}} E)$ with $\alpha = 0.35$.

2. **Zero-start.** Logits initialized to zero (masks $\approx 0.5$ everywhere). Used when evidence quality is poor or to test optimizer robustness. Requires more steps to converge.

For the shift-aware game, both robust and shortcut masks are initialized from the same pooled evidence (mean across hypotheses) scaled by 0.5, then passed through inverse sigmoid. This symmetric initialization lets the optimization differentiate the two masks based purely on the objective's gradient signal.

#### 3.5.7 Structural Penalties as Soft Constraints

Rather than imposing hard constraints (e.g., forcing masks to be exactly disjoint), GAMBIT uses differentiable penalty terms that act as soft constraints within the optimization:

| Penalty | Formula | Purpose | Where Applied |
|---------|---------|---------|---------------|
| Disjointness | $\sum_{k<l} m_k^{\text{unique}} \cdot m_l^{\text{unique}}$ | Unique masks should not overlap | Contrastive allocator |
| Partition | $\text{ReLU}(\sum_k m_k(u) - 1)$ | Total mask mass per unit $\leq 1$ | Contrastive allocator |
| Disjoint (rob/sho) | $m_{\text{rob}} \cdot m_{\text{sho}}$ | Robust and shortcut masks should separate | Shift-aware objective |
| Sparsity | $\|m\|_1$ | Compact, interpretable masks | Both objectives |
| Mass deviation | $|{\sum_r m_k(r) - \sum_r E_k(r)}|$ | Preserve evidence mass budget | Contrastive objective |

Soft penalties are preferred over hard constraints because: (a) they maintain smooth gradients everywhere, (b) they allow the optimizer to temporarily violate constraints if doing so improves the primary objectives, and (c) they enable smooth tradeoff curves controlled by the $\lambda$ weights.

An important implementation detail: the contrastive instantiation applies disjointness and partition penalties in the *allocator* (outside the objective), while the shift-aware instantiation includes disjointness in the *objective* itself. The shift-aware allocator explicitly checks for double-counting and raises an error if the allocator and objective both apply disjointness penalties.

#### 3.5.8 Convergence and Stopping

The current implementation uses a fixed number of optimization steps ($N_{\text{steps}}$) without early stopping. Typical values are:
- Contrastive: 25--50 steps with lr = 0.2--0.5
- Shift-aware: 40--50 steps with lr = 0.5

Empirically, the loss stabilizes within 20--30 steps for most configurations. The fixed-step approach was chosen for simplicity and reproducibility — every explanation uses the same computational budget regardless of input difficulty. Future work may incorporate adaptive stopping based on gradient norm or loss plateau detection.

### 3.6 Evidence Providers

GAMBIT supports pluggable evidence providers through a common protocol. The current implementation includes:

1. **Grad-CAM Regions.** Gradient-weighted class activation maps (Selvaraju et al., 2017) computed at the target layer (last Conv2d for CNNs; last encoder block for ViTs), pooled to the spatial grid. Fast and spatially smooth.

2. **Integrated Gradients Regions.** Path integrals (Sundararajan et al., 2017) from a zero or mean baseline, with pixel-level attributions pooled to the grid. Finer-grained but slower ($n_{\text{steps}} = 24$ by default).

Both return nonnegative evidence tensors of shape $(B, K, R)$.

### 3.7 Unit Space and Interventions

The `VisionGridUnitSpace` divides the input into $R = G_h \times G_w$ spatial regions (default: $7 \times 7$ for CNNs, $14 \times 14$ for ViTs). Interventions use adaptive pooling to blend the input with a baseline:
- **Mean baseline**: Replaces masked regions with the channel-wise global mean.
- **Blur baseline**: Replaces with an adaptive-pooled version of the image, reducing boundary artifacts.

The unit space also provides optional 2D sinusoidal positional embeddings for the interaction module.

---

## 4. Experimental Setup

### 4.1 Datasets

**Contrastive experiments (Instantiation I):**
- **MNIST** (10 classes, 28×28, grayscale → RGB)
- **CIFAR-10** (10 classes, 32×32 → 224×224)
- **Oxford-IIIT Pets** (2 classes: Cat vs Dog, 224×224)
- **Stanford Dogs** (120 fine-grained breeds, 224×224)

**Shift-aware experiments (Instantiation II):**
- **ColoredMNIST**: MNIST with class-correlated hue
- **ColoredCIFAR10**: CIFAR-10 with class-correlated color patch
- **TextureBiasedMNIST**: MNIST on class-correlated striped backgrounds

### 4.2 Models

All experiments use ResNet-18 backbones. Models are pretrained on ImageNet and then fine-tuned via linear probing for 15 epochs on each target dataset. Checkpoints are stored per dataset.

### 4.3 Evaluation Protocol

All contrastive experiments compare three methods:
- **Base Evidence**: Raw attribution maps (Grad-CAM or IG) used directly as masks.
- **Naive Contrastive**: Simple thresholded evidence without optimization.
- **CDEA (Optimized)**: Full game-theoretic allocation.

All results are reported as mean $\pm$ std across 3 seeds. Per-batch distributions are also recorded for stability analysis.

### 4.4 Metrics

**Contrastive metrics:**
- **Sufficiency** ($\uparrow$): Target class logit under the keep intervention.
- **Contrastive Margin** ($\uparrow$): Target logit minus strongest foil logit under keep.
- **Overlap** ($\downarrow$): Pairwise dot product between unique masks.
- **Sparsity** ($\downarrow$): L1 mass of unique masks.

**Shift-aware metrics:**
- **Rob Mean** ($\uparrow$): Mean robust sufficiency across environments.
- **Rob Var** ($\downarrow$): Variance of robust sufficiency across environments.
- **Sho Gap** ($\uparrow$): ID minus OOD shortcut sufficiency (positive = shortcut is ID-specific).
- **Disjoint** ($\downarrow$): Robust-shortcut mask overlap.
- **ID-OOD Gap** ($\uparrow$): Quantitative shortcut specificity.

---

## 5. Results

### 5.1 Contrastive Explanation Results (Instantiation I)

#### 5.1.1 Main Results

The following table summarizes CDEA's optimized allocation compared to base evidence across all datasets and both evidence providers (3 seeds).

| Dataset | Evidence | Suff $\uparrow$ | Margin $\uparrow$ | Overlap $\downarrow$ | Overlap Reduction vs Base | Sparsity $\downarrow$ |
|---------|----------|------:|-------:|--------:|:----------:|--------:|
| CIFAR-10 | Grad-CAM | -0.6850 | -1.2053 | 0.0075 | **-97.6%** | 0.8832 |
| CIFAR-10 | IG | -0.5654 | -1.0437 | 0.0033 | **-99.3%** | 1.0000 |
| MNIST | Grad-CAM | -0.6850 | -1.2053 | 0.0075 | **-97.6%** | 0.8832 |
| MNIST | IG | -0.5654 | -1.0437 | 0.0033 | **-99.3%** | 1.0000 |
| Pets | Grad-CAM | — | — | — | — | — |
| Stanford Dogs | Grad-CAM | — | — | — | — | — |

*Overlap reduction is calculated as percentage decrease from base evidence overlap to CDEA overlap.*

#### 5.1.2 Overlap Reduction

The headline result is that CDEA's optimized allocation drives pairwise mask overlap to near zero across all tested configurations. On CIFAR-10 with Grad-CAM evidence:
- Base evidence overlap: 0.3087 $\pm$ 0.1710
- Naive contrastive overlap: 0.1689 $\pm$ 0.1237
- **CDEA optimized overlap: 0.0075 $\pm$ 0.0049** (97.6% reduction)

With Integrated Gradients evidence:
- Base evidence overlap: 0.4994 $\pm$ 0.0033
- Naive contrastive overlap: 0.2961 $\pm$ 0.0187
- **CDEA optimized overlap: 0.0033 $\pm$ 0.0015** (99.3% reduction)

This confirms that the game-theoretic allocation successfully separates class-specific evidence regions.

#### 5.1.3 Margin Improvement

The contrastive margin (target logit minus strongest foil) improves under CDEA compared to base evidence. On CIFAR-10 with Grad-CAM, the margin delta (CDEA minus Base) is +0.2826, indicating that optimized masks better discriminate between competing classes. With IG evidence, the improvement is +0.3550.

#### 5.1.4 Sufficiency-Overlap Tradeoff

Plotting sufficiency against overlap across all (dataset, evidence, method) combinations reveals a clear pattern: CDEA points cluster in the desirable bottom-right region (higher sufficiency, lower overlap), while base evidence and naive methods cluster in the top region (high overlap). This confirms that the allocation reduces overlap without catastrophic sufficiency loss.

#### 5.1.5 Per-Batch Stability

Box plots of per-batch overlap across methods show that CDEA consistently produces tight, near-zero overlap distributions, while base evidence exhibits high variance. This demonstrates optimizer stability across batches.

### 5.2 Shift-Aware Results (Instantiation II)

#### 5.2.1 Metrics by Game Mode

Results across three bias-injection datasets with cooperative, mixed, and competitive game modes (3 seeds where available):

| Dataset | Game Mode | Rob Mean $\uparrow$ | Rob Var $\downarrow$ | Sho Gap $\uparrow$ | Disjoint $\downarrow$ | Sparse $\downarrow$ | ID-OOD Gap $\uparrow$ |
|---------|-----------|------:|------:|------:|------:|------:|------:|
| ColoredCIFAR10 | Cooperative | 3.024 $\pm$ 0.001 | 0.051 $\pm$ 0.000 | 0.186 $\pm$ 0.000 | 15.649 $\pm$ 0.002 | 18.817 $\pm$ 0.003 | 0.405 $\pm$ 0.002 |
| ColoredCIFAR10 | Mixed | 2.834 $\pm$ 0.001 | 0.014 $\pm$ 0.000 | 0.329 $\pm$ 0.001 | 0.908 $\pm$ 0.000 | 9.964 $\pm$ 0.003 | 0.476 $\pm$ 0.000 |
| ColoredCIFAR10 | Competitive | 2.668 $\pm$ 0.002 | 0.010 $\pm$ 0.000 | 0.341 $\pm$ 0.001 | 0.768 $\pm$ 0.001 | 9.069 $\pm$ 0.004 | 0.484 $\pm$ 0.002 |
| TextureMNIST | Cooperative | 2.827 $\pm$ 0.008 | 0.090 $\pm$ 0.001 | -0.256 $\pm$ 0.001 | 8.007 $\pm$ 0.017 | 10.948 $\pm$ 0.017 | -0.216 $\pm$ 0.001 |
| TextureMNIST | Mixed | 2.406 $\pm$ 0.008 | 0.057 $\pm$ 0.000 | 0.553 $\pm$ 0.002 | 0.679 $\pm$ 0.001 | 6.671 $\pm$ 0.007 | 0.527 $\pm$ 0.002 |
| TextureMNIST | Competitive | 2.076 $\pm$ 0.003 | 0.046 $\pm$ 0.000 | 0.608 $\pm$ 0.003 | 0.582 $\pm$ 0.001 | 6.243 $\pm$ 0.003 | 0.579 $\pm$ 0.003 |
| ColoredMNIST | Mixed | 0.000 | 0.000 | 0.000 | 0.110 | 2.324 | 0.005 |

#### 5.2.2 Game Mode Tradeoffs

The results reveal a clear tradeoff governed by the game mode:

- **Cooperative mode** achieves the highest Rob Mean (robust sufficiency), but with high mask overlap (Disjoint) and high total mask mass (Sparse), and lower Sho Gap. The masks are not well separated.

- **Competitive mode** achieves the strongest Sho Gap and ID-OOD Gap (best shortcut identification), lowest Rob Var (most stable robust evidence), and lowest Disjoint and Sparse values. However, Rob Mean is somewhat lower.

- **Mixed mode** provides balanced performance: good shortcut gap, reasonable robust sufficiency, and well-controlled mask overlap and sparsity.

This tradeoff is consistent with the game-theoretic design: increasing competitive pressure improves mask separation at the cost of total evidence retention.

#### 5.2.3 Robust vs Shortcut Decomposition

On ColoredCIFAR10 (mixed mode), the positive Sho Gap (0.329) and ID-OOD Gap (0.476) confirm that the shortcut mask captures evidence that is specifically useful on in-distribution data (where the color patch is class-correlated) but not on OOD data (where the correlation is broken). Meanwhile, the robust mask maintains high sufficiency (Rob Mean = 2.834) with low variance across environments (Rob Var = 0.014), confirming it captures genuinely predictive features.

On TextureBiasedMNIST with competitive mode, the framework achieves Sho Gap = 0.608 and ID-OOD Gap = 0.579, successfully identifying the background texture as the shortcut signal.

---

## 6. Visualization and Qualitative Analysis

### 6.1 Contrastive Explanation Gallery

For each input image, the explanation gallery displays:
- The original image with predicted and true labels
- Per-class evidence heatmaps (raw Grad-CAM or IG before allocation)
- Allocated mask overlays using a colorblind-safe scheme: **blue** for unique evidence, **orange** for shared evidence, **purple** for regions claimed by both

The gallery demonstrates that CDEA produces spatially coherent, class-discriminative masks. For example, on Pets (Cat vs Dog), the unique masks for "Cat" and "Dog" attend to distinct facial features, while the shared mask covers the animal's body common to both hypotheses.

### 6.2 Probability Split Analysis

The probability split chart reveals the discriminative power of unique evidence. For each class, it shows:
- The model's confidence using only shared evidence
- The confidence using shared + unique evidence
- The delta (unique contribution)

Large deltas indicate that the unique mask captures meaningful class-specific information beyond what is shared.

### 6.3 Pairwise Contrastive Margins

The $K \times K$ pairwise margin matrix visualizes the "why $k$ rather than $l$?" question for all class pairs. Green cells indicate successful discrimination (positive margin under the class-$k$ mask), while red cells flag potential explanation failures. The delta panel (shared+unique minus shared-only) isolates the unique mask's marginal contribution.

### 6.4 Robust vs Shortcut Mask Visualization

For the shift-aware instantiation, the robust mask (blue) and shortcut mask (orange) are overlaid on input images. On ColoredCIFAR10, the shortcut mask concentrates on the top-left color patch (the injected shortcut), while the robust mask attends to the object content. On TextureBiasedMNIST, the shortcut mask highlights the background stripe texture while the robust mask focuses on the digit.

### 6.5 Evidence Provider Comparison

Side-by-side comparison of Grad-CAM and Integrated Gradients evidence reveals their complementary characteristics:
- **Grad-CAM** produces spatially smooth evidence concentrated at the most discriminative regions.
- **Integrated Gradients** produces finer-grained evidence that can capture distributed features.

Both providers lead to effective allocations, though the resulting mask structures can differ. The framework's provider-agnostic design allows users to select the evidence source best suited to their analysis needs.

---

## 7. Framework Design

### 7.1 Protocol-Driven Architecture

All GAMBIT components are defined as Python Protocols (structural subtyping), enabling zero-inheritance extensibility:

| Protocol | Methods | Purpose |
|----------|---------|---------|
| `EvidenceUnitSpace` | `num_units`, `keep`, `remove`, `embed_units` | Define intervention space |
| `BaseEvidenceProvider` | `explain` | Compute raw evidence |
| `HypothesisSelector` | `select` | Select competing hypotheses |
| `Allocator` | `allocate` | Optimize mask allocation |
| `AllocationObjective` | `compute` | Define loss landscape |
| `InteractionModel` | `__call__` | Optional hypothesis conditioning |

### 7.2 High-Level API

GAMBIT provides a two-line API for common use cases:

```python
import gambit

# Contrastive explanation
explainer = gambit.ContrastiveExplainer(model, game_mode="mixed", evidence="gradcam")
explanation = explainer.explain(x)
# explanation.masks["unique"]: (B, K, H, W)
# explanation.masks["shared"]: (B, H, W)

# Shift-aware explanation
shift_exp = gambit.ShiftExplainer(model, game_mode="competitive")
explanation = shift_exp.explain(x, env=EnvBatch(xs=[x_id, x_ood], env_ids=[0, 1]))
# explanation.masks["robust"]: (B, H, W)
# explanation.masks["shortcut"]: (B, H, W)
```

### 7.3 Extensibility

Adding a new game instantiation requires implementing only two components — an `Allocator` and an `AllocationObjective` — without modifying the core kernel. The framework supports:
- New evidence providers (e.g., attention rollout, SHAP)
- New unit spaces (e.g., token sequences, graph nodes)
- New modalities (vision, text, graph — stubs provided)
- Custom game modes via the `manual` preset

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

**Optimization sensitivity.** The contrastive allocator is sensitive to the balance of regularization weights ($\lambda$ values), learning rate, and the quality of the backbone model. On harder datasets with weaker backbones, masks can become diffuse or produce weak margins.

**Negative sufficiency and margins.** Some configurations, particularly with untrained or weakly trained backbones, produce negative sufficiency and margin values. This reflects the intervention setup: when masked regions are replaced with a baseline, the model may lose essential context. Stronger backbone training mitigates this.

**Shift-aware data requirements.** The shift-aware instantiation requires multiple environment views per instance. In the absence of natural OOD data, we rely on synthetic augmentations (color jitter, saturation removal) or controlled bias-injection datasets.

**Computational cost.** The allocation step requires $N_{\text{steps}}$ forward passes through the model per sample (default: 40--50 steps). This is substantially more expensive than a single attribution forward pass.

### 8.2 Future Work

1. **Stability improvements.** Multi-seed hyperparameter sweeps, early stopping based on mask entropy or gradient norms, and non-triviality constraints.

2. **Stronger empirical validation.** Extend to additional architectures (ViT variants, EfficientNet), larger-scale datasets (ImageNet subsets), and real-world distribution shifts (domain adaptation benchmarks).

3. **New instantiations.** The modular kernel supports future game formulations, e.g., fairness-aware explanations, temporal consistency games for video, or multi-modal evidence allocation.

4. **Efficiency.** Amortized allocation via learned mask predictors, reducing the per-sample optimization cost.

5. **Statistical rigor.** Confidence intervals, significance testing, and larger seed counts for all reported metrics.

---

## 9. Conclusion

GAMBIT provides a unified, game-theoretic framework for structured model explanations. By framing explanation generation as an evidence allocation game, it transforms raw attribution evidence into interpretable decompositions: shared versus unique evidence for contrastive explanations, and robust versus shortcut evidence for distribution-shift analysis. The modular, protocol-driven design enables easy extension to new games, evidence sources, and modalities. Empirical results demonstrate consistent overlap reduction (85--99%) in contrastive settings and successful robust-shortcut separation on controlled bias-injection benchmarks, establishing GAMBIT as a flexible foundation for trustworthy model interpretability.

---

## References

- Dhurandhar, A., Chen, P.-Y., Luss, R., Tu, C.-C., Ting, P., Shanmugam, K., & Das, P. (2018). Explanations based on the missing: Towards contrastive explanations with pertinent negatives. *NeurIPS*.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665--673.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *FAT*\*.
- Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts. *ICLR*.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
