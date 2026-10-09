# AGENTS.md

Guidance for Cursor agents in this repository.

## Project

**GAMBIT** — Game theoretic Allocation for Model Based Interpretability and Trust.

CDEA is specified by `docs/paper/FORMULATION.md` and implemented in `cdea/`. A session that writes `cdea/` does not open the earlier method. That method is on tag `framing-v1` only.

## Environment

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate gambit
```

Run from the repository root with `PYTHONPATH=.`.

```bash
PYTHONPATH=. python -m pytest tests/
```

## Layout

`cdea/` imports `torch`, `core`, and `base_evidence` only. `core/grid.py` is the one deletion baseline. `evaluation/` scores. `baselines/` are the comparators. `models/build.py` builds ResNet-50 and ViT-B/16.

New records go to `results/paper/cells/`. Each record carries `method_code_hash` and `knobs_hash`. Fast knobs never satisfy a gate cell.

## Paper workflow

Read `docs/paper/PROGRESS.md` first. The stage card is `docs/paper/PLAN.md`. The same protocol is in `.cursor/rules/paper-workflow.mdc`.

Session start: `git pull`, read `PROGRESS.md` and the stage card, clean `git status`, run tests, check listed background jobs.

Session end: tests pass, small commits, `PROGRESS.md` updated. `git add <specific paths>` only.

Do not run `--final` or read the test split before the `eval-plan-frozen` tag. Do not edit `EVAL_PLAN.md` after it is frozen. Do not type a result number by hand. Do not weaken a test to make it pass. Do not change a baseline algorithm. Do not merge your own baseline PR. Ask before downloading a dataset or a package.

Work that needs the earlier tree uses a worktree pinned at tag `framing-v1`. The GH200 seed-1 chain stays pinned at `cf07ac2`. Its checkpoints are used. Its CDEA evaluation rows are not.
