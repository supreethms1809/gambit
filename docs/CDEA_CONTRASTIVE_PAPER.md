# CDEA-Contrastive: Contrastive Decomposition via Optimized Evidence Allocation

## Abstract

We introduce **CDEA-Contrastive**, a method for answering *why did the model predict class \(k\) rather than another top-scoring class?* given only a trained classifier and an input. Standard attribution maps (e.g., Grad-CAM, Integrated Gradients) assign nonnegative importance to each spatial region for every competing class, but those maps typically overlap heavily, so they do not isolate class-specific evidence. CDEA-Contrastive selects a small set of competing class hypotheses, builds nonnegative **base evidence** over a fixed grid of evidence units, and then **optimizes** continuous masks—one *unique* mask per hypothesis and an optional *shared* mask—using intervention-based objectives. The training signal encourages (i) predictive sufficiency under a *keep* intervention, (ii) a **contrastive margin** between the target logit and the strongest foil among the selected hypotheses, (iii) low overlap between unique masks, and (iv) parsimony and fidelity to the evidence mass budget. Optimization is differentiable: each step runs the frozen model on blended inputs \(x^{\mathrm{keep}}(m) = m \odot x + (1-m)\odot \bar{x}\). The method also reports probability splits (shared-only versus shared+unique) and full pairwise “\(k\) versus \(\ell\)” margin matrices for auditability. On vision benchmarks with ResNet-18 and both Grad-CAM and Integrated-Gradients evidence, optimized allocations reduce pairwise overlap between class-specific masks by large factors relative to raw evidence, while tracking sufficiency. CDEA-Contrastive is a self-contained contrastive explanation framework; it does not rely on a separate “meta-framework” at exposition time.

---

## 1. Introduction

Deep image classifiers can be probed with attribution methods that highlight where the model looks for each class. When several classes receive high probability, however, users need **contrastive** structure: which regions support class \(k\) *specifically*, as opposed to other plausible labels? Raw per-class heatmaps often highlight similar regions because discriminative cues are entangled in pixel space and because attribution is computed independently per class.

**CDEA-Contrastive** turns this into a **structured allocation problem** over a fixed set of **evidence units** (e.g., coarse spatial cells). For each input:

1. **Hypotheses** are the top-\(m\) classes by predicted probability.
2. A **base evidence provider** produces nonnegative scores \(E_k(u)\) for each hypothesis \(k\) and unit \(u\), pooled from a standard attribution method.
3. An **allocator** searches continuous masks in \([0,1]\) that reweight the input via keep interventions, optimizing a **contrastive objective** that rewards sufficiency and class separation and penalizes redundant overlap across hypotheses.

The output is interpretable spatially: **unique** masks localize hypothesis-specific evidence; an optional **shared** mask captures components that jointly support several top classes. Auxiliary metrics break down how much of the model’s preference for \(k\) is explained by shared versus unique regions, and pairwise margins summarize “\(k\) not \(\ell\)” under each intervention.

**Contributions.** (1) A complete, implementation-grounded formulation of contrastive explanation as **evidence allocation with intervention-based losses**. (2) A joint optimization procedure over sigmoid-parameterized masks with optional attention-conditioned initialization. (3) **Allocation presets** (cooperative, mixed, competitive) that vary the emphasis on shared structure, margins, and separation. (4) Open **example** pipelines and **ablation** scripts that log metrics to `examples/out/` and `scripts/out/journal/`, including large overlap reductions vs. raw evidence in controlled ablations.

---

## 2. Related Work

**Attribution and saliency.** Grad-CAM (Selvaraju et al., 2017) and Integrated Gradients (Sundararajan et al., 2017) produce class-conditional importance maps. They are used here as **evidence generators**, not as the final explanation.

**Contrastive explanations.** Pertinent positives/negatives and counterfactual approaches (Dhurandhar et al., 2018; Mothilal et al., 2020) explain predictions by contrasting to alternative cases or inputs. CDEA-Contrastive stays **local** to the given image: it contrasts **hypotheses** under interventions on the same \(x\).

**Optimization and masks for interpretability.** Differentiable masks and perturbation-based objectives have a long history in visualization and attribution refinement. CDEA-Contrastive differs in **explicit multi-hypothesis** structure: it optimizes **a bank of unique masks** plus an optional shared component, with **pairwise** diagnostics.

**Game-theoretic metaphor.** The competing hypotheses and tunable cooperative versus competitive balance can be read as a small **multi-player allocation** problem. The implementation solves it via **joint gradient descent** on a weighted loss (a pragmatic equilibrium notion), not via combinatorial equilibrium solvers.

---

## 3. Method

### 3.1 Problem setup

Given an input \(x\) (e.g., an image tensor), a trained classifier \(f\) producing logits \(f(x)\), and a finite set of **evidence units** \(\mathcal{U}\) with \(|\mathcal{U}| = R\), we choose a hypothesis set \(H(x)\) consisting of the **top-\(m\)** class indices by softmax probability. For each \(k \in H(x)\), base evidence \(E_k \in \mathbb{R}_{\ge 0}^{R}\) is provided by an attribution backend, yielding a tensor \(E \in \mathbb{R}_{\ge 0}^{B \times K \times R}\) in batched form. Evidence is optionally **normalized** per \((B,k)\) across \(R\) so each hypothesis’s evidence sums to one.

### 3.2 Evidence units and interventions

Following the vision-grid design, \(\mathcal{U}\) is a coarse spatial grid (e.g., \(7\times 7\) regions for CNN backbones). The **unit space** implements **keep** (and optionally **remove**) by replacing masked-out mass with a baseline \(\bar{x}\) (channel mean or blur). The keep intervention is

\[
x^{\mathrm{keep}}(m) = m \odot x + (1 - m) \odot \bar{x},
\]

with \(m\) broadcast appropriately over channels. All sufficiency and margin terms are computed on \(f(x^{\mathrm{keep}}(m))\).

### 3.3 Optional hypothesis interaction

When unit embeddings \(\phi(u)\) are available, CDEA-Contrastive can form tokens \(t_k = \sum_u E_k(u)\, \phi(u)\) and run a lightweight interaction module (**none**, **attention-only**, or a **single Transformer layer**) over the top-\(m\) hypotheses. This yields attention weights that can **reweight** how strongly each hypothesis contributes to the averaged objective and can **mix** evidence for allocator initialization. Interaction is optional; the contrastive loss is still defined on interventions.

### 3.4 Mask parameterization

For each batch element, CDEA-Contrastive optimizes:

- **Unique masks** \(m_k^{\mathrm{unique}} \in [0,1]^R\) for each of the \(K\) active hypotheses.
- Optionally a **shared mask** \(m^{\mathrm{shared}} \in [0,1]^R\).

The **effective** mask for hypothesis \(k\) is

\[
m_k^{\mathrm{tot}} = m_k^{\mathrm{unique}} + m^{\mathrm{shared}}
\]

(elementwise), when the shared mask is enabled; otherwise \(m_k^{\mathrm{tot}} = m_k^{\mathrm{unique}}\). Masks are parameterized as sigmoids of unconstrained logits \(\ell\), with logits clamped after each step for numerical stability (e.g., to \([-12,12]\)).

### 3.5 Contrastive objective

For each valid hypothesis \(k\), run the model on \(x^{\mathrm{keep}}(m_k^{\mathrm{tot}})\). Let \(z_k\) be the logit for class \(k\) on that forward pass. Define:

- **Sufficiency (per hypothesis):** \(z_k\) (target logit under the hypothesis-specific keep mask).
- **Contrastive margin:** \(z_k - \max_{\ell \neq k,\, \ell \in H(x)} z_\ell\), using only logits for hypotheses in \(H(x)\).

Batch-level regularizers on the **unique** masks include:

- **Overlap:** sum of dot products \(\sum_{k<\ell} \langle m_k^{\mathrm{unique}}, m_\ell^{\mathrm{unique}}\rangle\) (symmetric off-diagonal mass).
- **Sparsity:** mean \(\ell_1\) mass of unique masks.
- **Mass deviation:** alignment of \(\sum_u m_k^{\mathrm{unique}}(u)\) with the (normalized) evidence mass \(\sum_u E_k(u)\).

Hypothesis contributions to the averaged sufficiency and margin can be **reweighted** uniformly or by an optional blend with interaction attention.

The **scalar loss** minimized during allocation is

\[
\mathcal{L}_{\mathrm{ctr}}
=
-\Big(\lambda_{\mathrm{suff}}\, \overline{\mathrm{suff}} + \lambda_{\mathrm{margin}}\, \overline{\mathrm{margin}}\Big)
+ \lambda_{\mathrm{overlap}}\, \overline{\mathrm{overlap}}
+ \lambda_{\mathrm{sparse}}\, \overline{\mathrm{sparse}}
+ \lambda_{\mathrm{mass}}\, \overline{\mathrm{mass\text{-}dev}}.
\]

**Allocator-level penalties** (added to \(\mathcal{L}_{\mathrm{ctr}}\)) further structure the solution:

- **Disjointness** among unique masks (same overlap structure as in the objective, applied as an extra weighted term on unique masks).
- **Partition** penalty: \(\mathrm{ReLU}\big(\sum_k m_k^{\mathrm{unique}} + m^{\mathrm{shared}} - 1\big)\) per unit, discouraging total assigned mass above one where shared is used.

Defaults in code use Adam on mask logits (typically on the order of tens of steps, learning rate \(\approx 0.5\)), with **warm start** from inverse-sigmoid of clamped normalized evidence, optionally mixed with attention-transformed evidence.

### 3.6 Reporting: splits and pairwise margins

Beyond scalar summaries, the implementation records:

- **Shared-only** versus **shared+unique** logits/probabilities for each hypothesis (isolating the incremental effect of \(m_k^{\mathrm{unique}}\)).
- **Pairwise** margin tensors: for all ordered pairs \((k,\ell)\), contrasts under shared-only interventions and under \(k\)’s full mask, plus the **delta** attributable to unique evidence.

These tensors support qualitative dashboards and failure analysis (e.g., when margin remains negative for a specific foil class).

### 3.7 Allocation presets

**Presets** adjust whether a shared mask is used and the relative weights on margin, overlap, disjointness, and partition terms. Three built-in presets (with a **manual** override) sketch a cooperative-to-competitive spectrum:

| Preset        | Shared mask | Margin / separation emphasis | Typical use |
|---------------|:-----------:|------------------------------|-------------|
| Cooperative   | On          | Margin off; overlap off; partition on | Emphasize common evidence across hypotheses |
| Mixed         | On          | Balanced margin and overlap | Default balanced decomposition |
| Competitive   | Off         | Stronger margin, overlap, disjointness | Push hypotheses into disjoint unique regions |

Exact numerical weights are configuration-time choices; the implementation freezes them per run for reproducibility.

---

## 4. Algorithm (sketch)

**Input:** batch \(x\), frozen model \(f\), unit space, top-\(m\) hypotheses, normalized evidence \(E\), optional interaction module.  
**Output:** masks \(\{m_k^{\mathrm{unique}}\}\), optional \(m^{\mathrm{shared}}\), metric dict.

1. Forward \(f(x)\); select \(H(x)\); compute \(E\) via Grad-CAM or Integrated Gradients (pooled to \(R\)); normalize \(E\) if enabled.
2. Optionally build tokens, run interaction, obtain attention.
3. Initialize mask logits (inverse-sigmoid of \(E\), optional attention mix).
4. Repeat for \(T\) steps: sigmoid \(\rightarrow\) masks \(\rightarrow\) compute \(\mathcal{L}_{\mathrm{ctr}}\) plus allocator penalties \(\rightarrow\) backprop through keep interventions \(\rightarrow\) Adam step \(\rightarrow\) clamp logits.
5. Final forward to compute reported metrics (splits, pairwise margins).

**Complexity.** Each optimization step requires \(K\) full forward passes through \(f\) for the per-hypothesis keep evaluations, times \(T\) steps—substantially more than a single attribution call, but still practical for moderate \(m\) and \(T\).

---

## 5. Experimental setup

**Datasets.** MNIST, CIFAR-10, Oxford-IIIT Pets (binary cat vs dog), and Stanford Dogs (fine-grained breeds), as supported by the example drivers under `data/` (see `examples/README.md`).

**Example drivers.** From the repository root:

- Grad-CAM evidence: `PYTHONPATH=. python examples/contrastive_explanation.py [--dataset …] [--train] [--checkpoint …] [other flags]`
- Integrated Gradients evidence: `PYTHONPATH=. python examples/contrastive_explanation_ig.py [--dataset …] [other flags]`

Each run writes scalar metrics and auxiliary tables under `examples/out/`, for example `contrastive_<dataset>_metrics.json` / `.csv` and `contrastive_ig_<dataset>_metrics.json` / `.csv`, plus figures (`contrastive_explanation_*.png`, HTML plots in notebooks’ export paths).

**Model.** ResNet-18 with dataset-appropriate `num_classes`. **Table 1** below reflects whatever is stored in each JSON’s `config` block (in the current workspace snapshots, `pretrained` is `false` unless you re-run with `--pretrained` or load a trained checkpoint). For interpretable masks, the examples README recommends training on the target dataset (`--train`) or passing `--checkpoint`.

**Evidence backends.** Grad-CAM pooled to the grid (`base_evidence/gradcam_regions.py`) and Integrated Gradients pooled to the grid (`base_evidence/integrated_gradients_regions.py`), with IG baseline and step count recorded per JSON (`ig_baseline`, `ig_steps`).

**Ablation and journal drivers (`scripts/`).** Systematic **base evidence vs. naive contrastive vs. optimized (CDEA)** aggregates are produced by `scripts/ablation_contrastive.py`, multi-seed/full-dataset drivers, and the journal rollup in `scripts/run_experiments.py`. Outputs on disk include:

- **Per-seed ablation JSON:** `scripts/out/ablation_<dataset>_<evidence>_seed<N>_metrics.json` (e.g. `ablation_cifar10_gradcam_seed0_metrics.json`), each with an `aggregates` block for the three methods.
- **Journal rollup:** `scripts/out/journal/summary_contrastive.json` and `summary_contrastive.csv` (means/std over seeds when multiple seeds are merged).
- **Human-readable log:** `scripts/out/journal/JOURNAL_REPORT.md` (headline overlap-reduction percentages for the MNIST/CIFAR-10 journal slice).
- **POC matrix (subset of datasets):** `scripts/out/poc_all_datasets_matrix.json`, `poc_all_datasets_matrix.csv`, and `scripts/out/POC_REPORT_ALL_DATASETS.md` (MNIST, CIFAR-10, Pets with `pretrained: false` in those POC ablations).
- **Alternate compact ablations:** `scripts/out/ablation_contrastive_<dataset>_<evidence>_metrics.json` (smaller image counts, `lambda_disjoint: 1`, typically `pretrained: false`).

**Metrics.** In both `examples/out/` and `scripts/out/`, **sufficiency** (`suff`) is the mean target-class logit under the per-hypothesis keep mask; **margin** is the mean \(z_k - \max_{\text{foil}} z\) over valid top-\(m\) hypotheses; **overlap** is the pairwise dot-product sum among unique masks (unnormalized, scale depends on mask mass); **sparsity** is mean \(\ell_1\) mass of unique masks. These match `ContrastiveObjective.compute` unless a script notes a modified sufficiency (some reports subtract a zero-input baseline for exposition—check the driver for that run).

---

## 6. Results

Table 1 is taken **directly** from the `scalar_metrics` and `config` sections of the artifacts in `examples/out/` (Grad-CAM: `contrastive_{dataset}_metrics.json`; IG: `contrastive_ig_{dataset}_metrics.json`). Numbers are **means over the batch** used in that run’s final metric computation, not multi-seed aggregates.

**Table 1.** CDEA-Contrastive metrics from the examples pipeline (post-optimization). \(B\) = batch size, \(T\) = `num_alloc_steps`, \(m\) = top hypotheses (`top_k` in JSON). Overlap is pairwise among unique masks (lower is usually better).

| Dataset       | Evidence   | Suff. | Margin | Overlap | Sparse | \(B\) | \(T\) | \(m\) | Notes (from `config`) |
|---------------|------------|------:|-------:|--------:|-------:|------:|------:|------:|-------------------------|
| MNIST         | Grad-CAM   | 0.119 | −0.228 | 0.145 | 0.830 | 1 | 1 | 5 | `game_mode`: cooperative; very few alloc. steps |
| CIFAR-10      | Grad-CAM   | 0.204 | −0.249 | 0.166 | 0.989 | 4 | 25 | 5 | `interaction`: none; \(\lambda_{\mathrm{mass}}\)=2, partition 0.1 |
| Pets          | Grad-CAM   | −0.157 | 0.0003 | **0.0035** | 0.992 | 4 | 25 | 2 | Binary top-\(m\); overlap very low |
| Stanford Dogs | Grad-CAM   | 8.356 | −2.561 | 0.063 | 0.978 | 64 | 25 | 5 | Large batch; logits scale reflects run |
| MNIST         | IG         | 0.145 | −0.134 | 0.469 | 0.971 | 4 | 25 | 5 | `ig_steps`: 8, baseline zero |
| CIFAR-10      | IG         | 0.148 | −0.088 | 0.215 | 0.994 | 4 | 25 | 5 | Same pattern as Grad-CAM row |
| Pets          | IG         | −0.166 | \(\approx\)0 | 0.024 | 1.009 | 4 | 25 | 2 | Margin \(\approx 2.4\times 10^{-6}\) in file |
| Stanford Dogs | IG         | 1.749 | −0.524 | 0.348 | 0.975 | 4 | **5** | 5 | `ig_steps`: 4; fewer alloc. steps than other IG rows |

**Interpretation.** These rows are **not** a controlled sweep: they mix batch sizes, allocation steps, and (for MNIST Grad-CAM) a cooperative preset with a single optimization step. They are still useful as **reproducible snapshots** of what the examples emit on disk. Pets (both evidence types) achieves **very low overlap** with \(m=2\). Stanford Dogs Grad-CAM and IG rows used **different** batch sizes and IG used **shorter** allocation; direct comparison between those two cells is misleading. Negative sufficiency on Pets indicates that, for that checkpoint and intervention baseline, keep-masks still leave the target logit below the reported aggregate—consistent with the limitations discussion.

**Qualitative artifacts.** The same runs produce `contrastive_*_split.csv` and `contrastive_*_pairwise.csv` (and IG counterparts) for probability splits and “why \(k\) vs. \(\ell\)” margins; figures are under `examples/out/` (e.g. `contrastive_explanation_cifar10.png`, tutorial HTML exports). Probability-split tensors in code: `split_shared_only_*` vs `split_shared_plus_unique_*`.

### 6.1 Script ablations: MNIST and CIFAR-10 (journal summary)

**Table 2** reproduces the **three-way** comparison aggregated in `scripts/out/journal/summary_contrastive.json` (current workspace: **seed 0 only**, `n_seeds: 1`). The companion markdown `scripts/out/journal/JOURNAL_REPORT.md` reports a **mean overlap reduction vs. base of 92.8%** averaged over the four (dataset \(\times\) evidence) cells below, with near-zero mean absolute sufficiency drift between base and optimized in that slice.

**Table 2.** Base evidence, naive contrastive, and optimized CDEA (`scripts/out/journal/summary_contrastive.json`).

| Dataset | Evidence | Method | Suff. \(\uparrow\) | Margin \(\uparrow\) | Overlap \(\downarrow\) | Sparse \(\downarrow\) |
|---------|----------|--------|-------------------:|--------------------:|-----------------------:|----------------------:|
| MNIST | Grad-CAM | Base | 0.1539 | −0.0709 | 0.4966 | 1.0000 |
| MNIST | Grad-CAM | Naive | 0.1539 | −0.0709 | 0.2338 | 1.0000 |
| MNIST | Grad-CAM | **CDEA** | 0.1537 | −0.0709 | **0.0066** | 0.9959 |
| MNIST | IG | Base | 0.1829 | −0.0957 | 1.8007 | 1.0000 |
| MNIST | IG | Naive | 0.1830 | −0.0956 | 1.0012 | 1.0000 |
| MNIST | IG | **CDEA** | 0.1818 | −0.0950 | **0.0719** | 0.7834 |
| CIFAR-10 | Grad-CAM | Base | 0.1612 | −0.4044 | 0.1112 | 0.6188 |
| CIFAR-10 | Grad-CAM | Naive | 0.1612 | −0.4044 | 0.0261 | 0.6188 |
| CIFAR-10 | Grad-CAM | **CDEA** | 0.1612 | −0.4044 | **0.0019** | 0.6189 |
| CIFAR-10 | IG | Base | 0.1953 | −0.1141 | 0.5032 | 1.0000 |
| CIFAR-10 | IG | Naive | 0.1953 | −0.1141 | 0.3177 | 1.0000 |
| CIFAR-10 | IG | **CDEA** | 0.1952 | −0.1141 | **0.1105** | 0.7103 |

**Overlap reduction (optimized vs. base)** in this table: MNIST Grad-CAM **98.7%**; MNIST IG **96.0%**; CIFAR-10 Grad-CAM **98.3%**; CIFAR-10 IG **78.0%** (percent drop in overlap score, not log-space).

### 6.2 Script ablations: Pets and Stanford Dogs (seed 0, pretrained backbone)

**Table 3** uses `scripts/out/ablation_pets_*_seed0_metrics.json` and `scripts/out/ablation_stanford_dogs_*_seed0_metrics.json`. These runs use **ImageNet-pretrained** ResNet-18 (`pretrained: true` in the JSON), batch size 16, **`ig_steps`: 24** for IG, and aggregate over essentially the **full training split** (`num_images` \(\approx 2.5\times 10^4\) for Pets, \(\approx 2.06\times 10^4\) for Stanford Dogs).

**Table 3.** Full-dataset ablation, seed 0 (`aggregates` in each `ablation_*_seed0_metrics.json`).

| Dataset | Evidence | Method | Suff. \(\uparrow\) | Margin \(\uparrow\) | Overlap \(\downarrow\) | Sparse \(\downarrow\) |
|---------|----------|--------|-------------------:|--------------------:|-----------------------:|----------------------:|
| Pets | Grad-CAM | Base | 0.4709 | 0.0137 | 0.00185 | 0.9966 |
| Pets | Grad-CAM | Naive | 0.4706 | 0.0140 | 0.00000 | 0.9966 |
| Pets | Grad-CAM | **CDEA** | 0.6829 | 0.4443 | **0.00012** | 1.0187 |
| Pets | IG | Base | 0.4712 | 0.00036 | 0.0587 | 1.0000 |
| Pets | IG | Naive | 0.4640 | 0.00549 | 0.00000 | 1.0000 |
| Pets | IG | **CDEA** | 0.7266 | 0.5373 | **0.00502** | 1.0198 |
| Stanford Dogs | Grad-CAM | Base | −4.7305 | −1.8143 | 0.5017 | 1.0000 |
| Stanford Dogs | Grad-CAM | Naive | −4.6939 | −1.7872 | 0.2954 | 1.0000 |
| Stanford Dogs | Grad-CAM | **CDEA** | −4.3445 | −1.3176 | **0.0164** | 1.0182 |
| Stanford Dogs | IG | Base | −4.7377 | −1.8178 | 0.5637 | 1.0000 |
| Stanford Dogs | IG | Naive | −4.7127 | −1.8033 | 0.3284 | 1.0000 |
| Stanford Dogs | IG | **CDEA** | −4.3489 | −1.3482 | **0.0215** | 1.0128 |

**Takeaway.** On Pets, optimized allocation **increases** sufficiency and margin substantially while driving overlap toward **near zero** (Grad-CAM base overlap was already small for \(m=2\)). On Stanford Dogs, logits remain **negative** under keep interventions for this setup (hard fine-grained task), but **overlap still collapses** (roughly **97%** reduction vs. base for Grad-CAM, **96%** for IG) and **margins improve** (less negative) relative to base and naive.

### 6.3 POC matrix (optional third script slice)

`scripts/out/POC_REPORT_ALL_DATASETS.md` summarizes six ablations (MNIST, CIFAR-10, Pets \(\times\) Grad-CAM/IG) with **`pretrained: false`** and **`lambda_disjoint: 1`**, reporting **mean overlap reduction vs. base 80.84%** across those six runs. See `scripts/out/poc_all_datasets_matrix.csv` for the full numeric matrix (overlap reduction percentages, sparsity change, sufficiency delta).

---

## 7. Limitations

- **Cost:** \(O(T \cdot K)\) forward passes per explanation versus one attribution pass.
- **Sensitivity:** Results depend on backbone quality, baseline \(\bar{x}\), grid resolution \(R\), and \(\lambda\) weights; weak models can yield negative margins under aggressive masking.
- **Locality:** The method explains relative to **selected** hypotheses; it does not search the full label space exhaustively.
- **Non-convexity:** The joint loss is not guaranteed globally optimal; presets and warm starts matter.

---

## 8. Conclusion

CDEA-Contrastive is a **standalone** pipeline for structured contrastive explanations: it converts standard attributions into **optimized unique and shared masks** over a fixed evidence grid, with **intervention-based** sufficiency and margin training signals, **explicit overlap control**, and **rich pairwise diagnostics**. It is documented and evaluated in the open-source implementation accompanying this draft.

---

## References

- Dhurandhar, A., Chen, P.-Y., Luss, R., Tu, C.-C., Ting, P., Shanmugam, K., & Das, P. (2018). Explanations based on the missing: Towards contrastive explanations with pertinent negatives. *NeurIPS*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *FAT*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.

---

## Implementation note

The method described here corresponds to `CDEAExplainer` with `ContrastiveObjective` and `OptimizationAllocator` in the repository (`core/runner.py`, `instantiations/contrastive/objective.py`, `instantiations/contrastive/allocator.py`), plus pluggable evidence providers under `base_evidence/`. **Table 1** comes from `examples/contrastive_explanation.py` / `contrastive_explanation_ig.py` \(\rightarrow\) `examples/out/`. **Tables 2–3** and the POC summary come from ablation outputs under `scripts/out/` (`journal/`, `ablation_*_seed*_metrics.json`, and `poc_all_datasets_matrix.*`).
