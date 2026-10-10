# Progress

History before this branch is on tag `framing-v1`. Do not score records written under `results/paper/runs/`. New records go to `results/paper/cells/`.

The seed-1 training chain on the GH200 stays pinned at `cf07ac2` (the archive commit). Its checkpoints are used. Its CDEA evaluation rows are not.

## Next session

Shift lives beside contrastive. `cdea/shift.py` is the transplant payoff and its own Sinkhorn loop. The contrastive loop was not refactored. `scripts/gate_shift.py` prints the read-out and the stop rule and does not launch the n = 64 pilot.

Run `scripts/smoke_e2e.py --game shift` on Waterbirds and planted-patch val. Do not read a shift number. Do not train seeds 1–4. The sixth dataset is still unchosen.
