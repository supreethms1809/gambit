# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GAMBIT** — Game theoretic Allocation for Model Based Interpretability and Trust.

A PyTorch-based research framework implementing the CDEA (Contrastive Decomposition via Evidence Allocation) pipeline: a unified, game-theoretic approach to producing interpretable, contrastive model explanations.

## Environment Setup

Uses the `marl` conda environment:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
```

All scripts must be run from the repository root with `PYTHONPATH=.` so that `core`, `modality`, `base_evidence`, and `instantiations` are importable.

## Running Examples

```bash
# Contrastive explanation (Grad-CAM evidence)
PYTHONPATH=. python examples/contrastive_explanation.py --dataset cifar10 --train --epochs 10

# Contrastive explanation (Integrated Gradients evidence)
PYTHONPATH=. python examples/contrastive_explanation_ig.py --dataset cifar10

# Shift-aware robust vs shortcut game
PYTHONPATH=. python scripts/eval_robust_shortcut.py --game_mode mixed
```

Datasets live in `data/` (git-ignored). Available: `mnist`, `cifar10`, `pets`, `stanford_dogs`. If the dataset is not found, a random batch is used as fallback.

## Running Tests

```bash
PYTHONPATH=. python -m pytest tests/
# Run a single test file
PYTHONPATH=. python -m pytest tests/test_game_modes.py
```

## Architecture

### Core Pipeline (`core/`)

The `CDEAExplainer` (`core/runner.py`) orchestrates the full pipeline:

1. **HypothesisSelector** (`core/hypotheses.py`) — selects competing hypotheses (e.g., top-K classes)
2. **BaseEvidenceProvider** (`core/base_evidence.py`) — computes raw attribution evidence per unit
3. **Interaction** (`core/interaction.py`) — optional hypothesis interaction (attention or transformer layers)
4. **Allocator** (`core/allocator.py`) — optimizes evidence allocation masks across hypotheses
5. **Objective** (`core/objective.py`) — defines the loss landscape driving allocation

All components are defined as Python Protocols, enabling pluggable implementations.

Key types: `HypothesisSet`, `Explanation`, `EnvBatch` (`core/types.py`).
Game mode presets (cooperative / competitive / mixed): `core/game_modes.py`.
Unit space abstraction (e.g., 7×7 spatial grid for vision): `core/unit_space.py`.

### Two Main Game Instantiations

**Contrastive Game** (`instantiations/contrastive/`) — "Why class K rather than L?"
- `OptimizationAllocator` performs gradient-based mask optimization
- `ContrastiveObjective` applies margin, overlap, sparsity, and partition penalties
- Outputs: shared evidence mask + unique evidence masks per class

**Shift-Aware Robust/Shortcut Game** (`instantiations/shift/`) — separates robust from shortcut evidence under distribution shift
- `RobustShortcutObjective` measures robustness across environments and shortcut gap
- Outputs: robust mask, shortcut mask, stability/gap diagnostics

### Modality Layer (`modality/`)

`VisionGridUnitSpace` (`modality/grid_regions.py`) divides images into spatial grid regions. Stubs exist for text tokens and graph nodes.

### Evidence Providers (`base_evidence/`)

- `gradcam_regions.py` — Grad-CAM pooled to grid
- `integrated_gradients_regions.py` — Integrated Gradients pooled to grid

### Adding a New Game Instantiation

See `docs/NEW_INSTANTIATION_GAME_GUIDE.md` for the step-by-step guide on implementing a new allocator + objective pair.

## Output Locations

- Example figures/metrics: `examples/out/`
- Script figures/metrics: `scripts/out/`
- Model checkpoints: `examples/out/checkpoints/<dataset>_<model>.pt`
