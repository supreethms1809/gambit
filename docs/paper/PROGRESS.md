# Progress

History before this branch is on tag `framing-v1`. Do not score records written under `results/paper/runs/`. New records go to `results/paper/cells/`.

The seed-1 training chain on the GH200 stays pinned at `cf07ac2` (the archive commit). Its checkpoints are used. Its CDEA evaluation rows are not.

## Next session

Run `scripts/smoke_e2e.py` on CIFAR-10 val, then `scripts/gate_pilot.py` only after that smoke is green. Do not launch the full n = 64 pilot until the smoke cell finishes.
