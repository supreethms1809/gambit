# CDEA Prelims: Limitations, Risks, and Milestones

This draft is intended for the "limitations and planned work" section of a PhD prelim proposal.

## 1) Current Limitations (Implementation-grounded)

### 1.1 Optimization stability

The contrastive optimizer remains sensitive to initialization, regularization balance, and backbone quality. Current pass-condition tests succeed, but empirical runs can still produce diffuse or weakly contrastive masks on harder datasets when training is insufficient.

### 1.2 Metric quality not yet consistently strong

Across current artifacts, some contrastive runs still show weak or negative margins, and visual quality varies by dataset/model quality. This is expected for a proof-of-concept phase but currently limits strong empirical claims.

### 1.3 Shift-aware setup still preliminary

The robust-shortcut script currently uses a lightweight setup (`/Users/ssuresh/gambit/scripts/eval_robust_shortcut.py`) intended as a demonstration. A stronger trained-backbone protocol and richer OOD shifts are needed for publishable conclusions.

### 1.4 Statistical validation gap

Current runs are mostly single-seed/small-scale. Confidence intervals, repeated trials, and significance testing are not yet part of the default pipeline.

## 2) Proposal Risks and Mitigation

### Risk A: Allocator collapse or diffuse masks

Mitigation:

1. Add multi-seed sweeps over `lambda_overlap`, `lambda_sparse`, `lambda_mass`, learning rate, and steps.
2. Add early-stop heuristics based on mask entropy or gradient norm.
3. Add explicit non-triviality constraints in tests and runtime warnings.

### Risk B: Weak margins despite lower overlap

Mitigation:

1. Emphasize margin-aware tuning and pairwise diagnostics.
2. Compare base/naive/optimized under matched sparsity budgets.
3. Track per-class and per-confusion-pair behavior, especially on Stanford Dogs.

### Risk C: Shift-aware metrics near zero

Mitigation:

1. Train a stronger base model for shift experiments.
2. Use controlled shortcut-inducing datasets and true domain-shift variants.
3. Expand EnvBatch generation beyond color-only cues.

## 3) Proposed Milestones (for prelim timeline)

## Milestone 1: Stabilize optimization and reproducibility

Deliverables:

1. All core tests passing in `marl`.
2. Deterministic and documented run scripts.
3. Multi-seed reporting harness for key metrics.

Success criteria:

1. No frequent mask-collapse failures.
2. Narrower variance for contrastive metrics across seeds.

## Milestone 2: Strengthen Instantiation I evidence

Deliverables:

1. Results on MNIST, CIFAR10, Pets, Stanford Dogs with both Grad-CAM and IG.
2. Comparative tables for base vs naive vs optimized CDEA.
3. Curated visual case studies on confusable classes.

Success criteria:

1. Consistent overlap reduction with minimal sufficiency drop.
2. Improved or stable contrastive margin under controlled sparsity.

## Milestone 3: Upgrade Instantiation II to research-grade

Deliverables:

1. Trained-backbone robust-shortcut experiments.
2. Expanded OOD shifts and ablations.
3. Robust/shortcut qualitative examples plus quantitative ID-OOD gaps.

Success criteria:

1. Positive and interpretable shortcut ID-OOD gap.
2. Lower robust variance with meaningful robust mean.

## Milestone 4: Pre-paper package

Deliverables:

1. Finalized evaluation protocol.
2. Statistical summary (mean/std/CI) across seeds.
3. Reproducibility bundle (scripts, configs, artifact schema).

Success criteria:

1. Ready-to-submit empirical section for workshop/demo or strong internal review.

## 4) Proposal-safe Claim Template

The current implementation establishes technical feasibility of CDEA as a modular evidence-allocation kernel and provides initial directional evidence that optimized allocation improves explanation structure (notably overlap reduction) while preserving core predictive signal in several settings. The next research phase focuses on improving allocator stability, strengthening margin-faithfulness tradeoffs, and validating robust-shortcut decomposition under stronger backbones and controlled domain shifts.
