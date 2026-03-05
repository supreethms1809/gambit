# CDEA Prelims: Implementation Overview

This document is a thesis-proposal starter focused on what is already implemented in this repository.

## 1) System Goal

The code implements a modular **Class-Distribution Evidence Allocation Explanation (CDEA)** framework that:

1. Builds per-class evidence over a fixed evidence-unit space.
2. Optionally models interaction among top hypotheses.
3. Allocates evidence into interpretable masks using optimization.
4. Evaluates explanations through intervention-based metrics.

The implementation supports two instantiations:

1. **Instantiation I (Contrastive shared-unique)** for "why class \(k\) rather than class \(l\)?"
2. **Instantiation II (Shift-aware robust-shortcut)** for separating robust evidence from shortcut evidence across environments.

## 2) Core Kernel Modules

Repository paths:

- `/Users/ssuresh/gambit/core/types.py`
- `/Users/ssuresh/gambit/core/runner.py`
- `/Users/ssuresh/gambit/core/unit_space.py`
- `/Users/ssuresh/gambit/core/base_evidence.py`
- `/Users/ssuresh/gambit/core/hypotheses.py`
- `/Users/ssuresh/gambit/core/interaction.py`
- `/Users/ssuresh/gambit/core/allocator.py`

Main data structures:

1. `HypothesisSet(ids, mask)`
2. `Explanation(hypotheses, masks, metrics, extras)`
3. `EnvBatch(xs, env_ids)`

Kernel orchestration is implemented in `CDEAExplainer.explain(...)`:

1. Run model forward pass to obtain logits/probabilities.
2. Select top hypotheses using `TopMSelector`.
3. Compute base evidence maps `(B, K, R)`.
4. Optionally normalize evidence across regions.
5. Build token embeddings from evidence and unit embeddings.
6. Optionally run interaction model (`none`, `attention`, `transformer`).
7. Allocate masks using an instantiation-specific optimizer.
8. Compute objective metrics under interventions.
9. Return standardized `Explanation`.

## 3) Evidence Unit Space

Concrete unit space:

- `/Users/ssuresh/gambit/modality/grid_regions.py`

`VisionGridUnitSpace` implements:

1. `num_units()` with `R = grid_h * grid_w`.
2. `keep(x, m)` intervention.
3. `remove(x, m)` intervention.
4. Optional `embed_units(x)` for token construction.

Current behavior:

1. Baseline can be `"mean"` or `"blur"` for interventions.
2. If `embed_dim=None`, `embed_units(...)` returns `None`.
3. If `embed_dim>0`, deterministic 2D sinusoidal embeddings are generated and returned as `(B, R, D)`.

## 4) Base Evidence Providers

Implemented providers:

1. Grad-CAM to regions: `/Users/ssuresh/gambit/base_evidence/gradcam_regions.py`
2. Integrated Gradients to regions: `/Users/ssuresh/gambit/base_evidence/integrated_gradients_regions.py`

Both return nonnegative evidence tensor of shape `(B, K, R)`.

## 5) Instantiation I: Contrastive Shared-Unique

Files:

1. `/Users/ssuresh/gambit/instantiations/contrastive/objective.py`
2. `/Users/ssuresh/gambit/instantiations/contrastive/allocator.py`
3. `/Users/ssuresh/gambit/examples/contrastive_explanation.py`
4. `/Users/ssuresh/gambit/examples/contrastive_explanation_ig.py`

Key outputs:

1. `masks["unique"]` with shape `(B, K, R)`
2. Optional `masks["shared"]` with shape `(B, R)`
3. Metrics including sufficiency, margin, overlap, sparsity, mass deviation, split probabilities, and pairwise margins.
4. Explicit game modes (`cooperative`, `competitive`, `mixed`, `manual`) configured through `/Users/ssuresh/gambit/core/game_modes.py`.

## 6) Instantiation II: Shift-aware Robust-Shortcut

Files:

1. `/Users/ssuresh/gambit/instantiations/shift/objective.py`
2. `/Users/ssuresh/gambit/instantiations/shift/allocator.py`
3. `/Users/ssuresh/gambit/instantiations/shift/env.py`
4. `/Users/ssuresh/gambit/instantiations/shift/biased_data.py`
5. `/Users/ssuresh/gambit/scripts/eval_robust_shortcut.py`

Key outputs:

1. `masks["robust"]` and `masks["shortcut"]`, each `(B, R)`
2. Metrics: `rob_mean`, `rob_var`, `sho_gap`, `sho_mean`, `disjoint`, `sparse`, and ID-OOD gap.
3. Explicit game modes (`cooperative`, `competitive`, `mixed`, `manual`) configured through `/Users/ssuresh/gambit/core/game_modes.py`.

## 7) Reporting and Artifacts

Helper utilities:

- `/Users/ssuresh/gambit/core/reporting.py`

Outputs are saved as:

1. JSON summaries
2. CSV scalar tables
3. CSV per-sample or per-batch analysis tables
4. Visualization PNGs

Existing output folders:

1. `/Users/ssuresh/gambit/examples/out`
2. `/Users/ssuresh/gambit/scripts/out`

## 8) One-paragraph Thesis Text (Starter)

The implemented CDEA system is organized as a reusable kernel with pluggable hypothesis selection, base evidence extraction, interaction modeling, and allocation objectives. The kernel is instantiated in two settings: (i) contrastive shared-unique evidence allocation for top-m hypotheses, and (ii) shift-aware robust-shortcut decomposition across environment batches. A concrete vision-grid evidence-unit space supports differentiable keep/remove interventions and optional deterministic unit embeddings for interaction-conditioned reasoning. The current implementation includes both Grad-CAM- and Integrated-Gradients-based evidence providers, optimization-based allocators, standardized explanation outputs, and structured reporting artifacts for quantitative and qualitative analysis.
