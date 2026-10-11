# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

**GAMBIT** — Game theoretic Allocation for Model Based Interpretability and Trust.

CDEA allocates evidence across competing hypotheses. The method lives in `cdea/` and is specified by `docs/paper/FORMULATION.md`. Write that package from the formulation. Do not copy the earlier method. It lives only on tag `framing-v1`.

## Environment

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate gambit
```

Create that env from `requirements.txt`. Run everything from the repository root with `PYTHONPATH=.`.

```bash
PYTHONPATH=. python -m pytest tests/
PYTHONPATH=. python scripts/paper_run.py --game contrastive --dataset cifar10 --backbone resnet50 --seed 0 --split val --n 1 --methods core --fast
```

Records go to `results/paper/cells/`. A record counts only when its `method_code_hash` and `knobs_hash` match the current tree. Fast knobs never satisfy a gate cell.

## Layout

`cdea/` is the method (`payoffs`, `sinkhorn`, `allocation`, `first_order`, `shift`). It imports `torch`, `core`, and `base_evidence` only. Shift payoffs are specified by `docs/paper/SHIFT_FORMULATION.md`.

`core/` holds types, hypotheses, eval mode, device, reporting, and `grid.py` (the hard unit indicator and the one deletion baseline).

`evaluation/` scores explanations. `baselines/` are the comparators. `models/build.py` builds ResNet-50 and ViT-B/16. `analysis/` reads records.

Archive work that needs the earlier method checks out tag `framing-v1` in its own worktree. The GH200 seed-1 training chain stays pinned at `cf07ac2`.
