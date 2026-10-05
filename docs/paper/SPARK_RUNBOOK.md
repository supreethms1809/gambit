# Running the paper evaluation on Spark

Every number in the paper comes from one entry point, `scripts/paper_run.py`, driven
over the grid by `scripts/launch_paper_eval.py`. `scripts/smoke_e2e.py` runs the same
code on a few val images to check the pipeline end to end. Its report is
`results/paper/smoke/REPORT.md`.

## 0. Before any evaluation run

1. `git pull` on `main`, with PR #23 and the runner branch merged. Spark's checkpoints
   are ImageNet-convention, and they only load correctly through `load_checkpoint_into`.
2. `conda activate gambit` and run `PYTHONPATH=. python -m pytest tests/`. Every test must pass.
3. Run the ImageNet split once: `PYTHONPATH=. python scripts/write_paper_splits.py --only imagenet`.
   It needs `data/imagenet/val/<wnid>/*.JPEG` (ImageFolder layout). It writes
   `data/splits/imagenet.json`: 10 images per class as val, 40 as test. Commit that file.
4. Check that the paper checkpoints are where `evaluation/run_models.paper_checkpoint_path`
   looks (`results/paper/checkpoints/<name>_imagenet_seed<s>.pt`). Run `--dry-run` (below):
   a missing checkpoint raises at run time and never falls back to a random head.

## 1. Smoke run on Spark (about the time of one small cell per unit)

```bash
PYTHONPATH=. python scripts/smoke_e2e.py --out results/paper/smoke_spark
```

Read `results/paper/smoke_spark/REPORT.md`. "Method errors" must say "None".
The "Per-image cost at the paper's knobs" table is what sets n per seed (step 3).

## 2. Val runs (before the freeze)

These feed val selection (EVAL_PLAN 6.2), the shift-area pilot (6.3), and G1 (section 11).
Test stays locked.

```bash
# the dev units, for selection and G1
PYTHONPATH=. python scripts/launch_paper_eval.py --split val --datasets cifar10,ham10000 --seeds 0
# the shift-area pilot: every shift unit, seed 0
PYTHONPATH=. python scripts/launch_paper_eval.py --split val --game shift --seeds 0
```

The candidate grids in EVAL_PLAN 6.2 still need a selection driver that loops
`CdeaConfig` / baseline knobs over these val cells. That is the next build step.

## 3. Freeze

Follow EVAL_PLAN section 12. Write n per seed from the timing table, then add
`frozen: true` and tag `eval-plan-frozen`.

## 4. The paper run (after the freeze only)

One machine owns each dataset (EVAL_PLAN 10). Example for Spark:

```bash
PYTHONPATH=. caffeinate -i python scripts/launch_paper_eval.py --split test --final \
    --config-hash <frozen hash> --datasets imagenet,cifar100,cub200,stanford_dogs \
    --n-contrastive <n> --n-shift <n> --ablations --extended \
    > results/paper/logs/eval_spark.log 2>&1 &
```

On Linux, drop `caffeinate` and use `nohup` or `tmux`. The run is resumable: rerunning
the same command skips cells with a done marker. A failed cell leaves
`results/paper/runs/_markers/test/<cell>.failed` with its error.

Records go to `results/paper/runs/<game>/<dataset>/<backbone>/seed<s>/records.csv.gz`,
with a `summary.json` (per-method status, seconds, pass counts, provenance).
`scripts/completion_check.py` and `analysis/build_results.py` read them.
