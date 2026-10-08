# Final runs

Final runs are not complete. The tag `final-runs-v1` was not created. No grid was launched, and no failed cell was rerun, because there are no run files.

## Why the check does not pass

`scripts/completion_check.py` counts a cell done only when its done marker exists. A failed marker stays on the rerun list. The contrastive log, the shift log, and the ablation log are absent, so every cell is pending.

The contrastive manifest names eight datasets, including ImageNet-S. ImageNet-S is still blocked on ImageNet-1k. The shift manifest is the six datasets from S21. The ablation manifest is A1–A8 plus the paired and unpaired mass-target rows. None of those cells has a done marker.

The G1 decision is not logged. `docs/paper/EVAL_PLAN.md` is a draft. The test split was not read. `--final` stays refused.

## What a finished record has to satisfy

A later run record has to pass `assert_run_record` before it can be called complete:

- masks within [0, 1];
- mask area within the planned fraction, plus or minus a tolerance;
- probabilities within [0, 1];
- every value finite;
- `n` equal to the planned image count.

No paper record has been checked. The provisional image counts in `docs/paper/PROGRESS.md` are not that planned count.
