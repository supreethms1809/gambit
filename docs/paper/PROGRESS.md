# Progress

History before this branch is on tag `framing-v1`. Do not score records written under `results/paper/runs/`. New records go to `results/paper/cells/`.

The seed-1 training chain on the GH200 stays pinned at `cf07ac2` (the archive commit). Its checkpoints are used. Its CDEA evaluation rows are not.

## Running

- **Seed-0 cells.** CIFAR-10 and HAM10000 val cells (ResNet-50, n = 64) are in `results/paper/cells/`. Their CDEA rows were merged in after the Sinkhorn catch-up (`d271283`). Seed 1 is not trained.
- **CDEA rerun with stage payoffs** (`ecbbfa2`, D2 inputs) goes to `results/paper/cdea_stages/`. The baseline rows are not rerun: no baseline calls `cdea/`.
- **Declared departure from the G1 stop rule.** One formulation change is under test: the boundary-robust payoff, `FORMULATION.md` section 4.1. Its read-out and decision rule are in `G1.md` and were fixed before it ran.

## Next session

Shift lives beside contrastive. `cdea/shift.py` is the transplant payoff and its own Sinkhorn loop. The contrastive loop was not refactored. `scripts/gate_shift.py` prints the read-out and the stop rule and does not launch the n = 64 pilot.

Smoke records, not paper numbers: shift Waterbirds is under `results/paper/smoke_shift_trained`; planted-patch is trained and not scored. Contrastive CIFAR-10 and HAM10000 are under `results/paper/smoke_contrastive`. Do not train seeds 1–4. The sixth dataset is still unchosen.
