# AGENTS.md

Guidance for Cursor agents working in this repository. `CLAUDE.md` stays as it is for Claude Code. This file repeats the environment and architecture notes so the two stay consistent, and adds the paper-run protocol.

## Project

**GAMBIT** — Game theoretic Allocation for Model Based Interpretability and Trust.

A PyTorch research framework for the CDEA (Contrastive Decomposition via Evidence Allocation) pipeline.

## Environment

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
```

Run everything from the repository root with `PYTHONPATH=.`. Tests:

```bash
PYTHONPATH=. python -m pytest tests/
```

Datasets live in `data/` (git-ignored, except `data/splits/`). Available: `mnist`, `cifar10`, `pets`, `stanford_dogs`, `ham10000`, `brain_tumor`. Medical results and their known measurement problems are in `docs/MEDICAL_RESULTS.md`. Those numbers are pre-audit and are not paper results.

## Architecture

`CDEAExplainer` (`core/runner.py`) runs:

1. **HypothesisSelector** (`core/hypotheses.py`) — competing hypotheses, usually top-K classes
2. **BaseEvidenceProvider** (`core/base_evidence.py`) — raw attribution per unit
3. **Interaction** (`core/interaction.py`) — optional attention or transformer over hypotheses
4. **Allocator** (`core/allocator.py`) — evidence allocation masks
5. **Objective** (`core/objective.py`) — the loss that drives allocation

Key types live in `core/types.py`. Game presets live in `core/game_modes.py`.

**Contrastive game** (`instantiations/contrastive/`): why class K rather than L. Shared mask plus a unique mask per class.

**Shift-aware game** (`instantiations/shift/`): robust evidence versus shortcut evidence under a change of environment.

`VisionGridUnitSpace` (`modality/grid_regions.py`) is the spatial unit space. Evidence providers are Grad-CAM, Integrated Gradients, and occlusion (`base_evidence/`).

## Paper workflow

Read `docs/paper/PROGRESS.md` first. The stage specs are in `docs/paper/PLAN.md`. The same protocol is in `.cursor/rules/paper-workflow.mdc`.

Session start: `git pull`, read `PROGRESS.md` and the stage card, clean `git status`, run tests, check listed background jobs.

Session end: tests pass, small commits pushed, `PROGRESS.md` updated and pushed. `git add <specific paths>` only. Never `git add -A` or `git add .`.

Do not run `--final` or read the test split before the `eval-plan-frozen` tag. Do not edit `EVAL_PLAN.md` after it is frozen. Do not type a result number by hand. Do not weaken a test to make it pass. Do not change a baseline algorithm. Do not merge your own baseline PR. Ask before downloading a dataset or a package.

The kickoff prompt for a fresh chat is `docs/paper/KICKOFF_PROMPT.md`.
