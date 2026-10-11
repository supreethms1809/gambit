# Progress

History before this branch is on tag `framing-v1`. Do not score records written under `results/paper/runs/`. New records go to `results/paper/cells/`.

The seed-1 training chain on the GH200 stays pinned at `cf07ac2` (the archive commit). Its checkpoints are used. Its CDEA evaluation rows are not.

## Running

- **Seed-0 cells.** CIFAR-10 and HAM10000 val cells (ResNet-50, n = 64) are in `results/paper/cells/`. Their CDEA rows were merged in after the Sinkhorn catch-up (`d271283`). Seed 1 is not trained.
- **CDEA was not reproducible before `d010f02`.**
  - Two runs at the same method hash (`540acbe0`: `results/paper/cdea_rerun/` and `results/paper/cdea_stages/`) shared no identical row.
  - At 5% on CIFAR-10 val, the median per-image difference in CD was 0.71 nats, and the mean moved from 2.39 to 1.99.
  - The cause was the backward of the cell upsampling, which accumulates in a varying order on GPUs; Adam amplified it.
  - Since `d010f02`, a repeated allocation on MPS is identical. CDEA rows from before it carry run-to-run noise of that size.
- **D2 fails** in `results/paper/cdea_stages/` (`scripts/check_degenerate.py --records`). At 5% the soft-minus-hard gap is above 0.5 nats for ranks 0 and 1 on both dev datasets.
- **Declared departure from the G1 stop rule.** One formulation change is under test: the boundary-robust payoff, `FORMULATION.md` section 4.1. Its read-out and decision rule are in `G1.md` and were fixed before it ran.
  - The fixed-boundary and boundary-robust CDEA cells run together at `f011c8e3` into `results/paper/cdea_boundary/`.
  - The baseline rows are not rerun: no baseline calls the changed code.
  - Read with `scripts/g1_readout.py --records results/paper/cells results/paper/cdea_boundary --cdea cdea cdea:A10_boundary_shift --seeds 0`.

## G1 declared read-out, 2026-10-11 (seed 0 only)

`scripts/g1_readout.py --records results/paper/cells results/paper/cdea_boundary --cdea cdea cdea:A10_boundary_shift --seeds 0`

- **Boundary-robust payoff** (`cdea:A10_boundary_shift`): "ahead of both comparators on both datasets". The declared rule is met.
- **Fixed-boundary payoff** (`cdea`): not ahead of margin Grad-CAM on either dataset.

**Open before section 4.1 becomes the method.**
- The 95% intervals against margin Grad-CAM include 0 on both datasets.
- D2 still fails at 5% on CIFAR-10, though the gap is smaller than with the fixed boundary.
- The G1 read is to be repeated on seed 1, which is not trained.

The default is unchanged until the user confirms adoption.

## Offset-seed check, declared 2026-10-11 before it ran

- **What runs.** The boundary-robust payoff with offset seeds 0, 1 and 2: `A10_boundary_shift`, `_o1`, `_o2`. Seed-0 val, both dev datasets, n = 64, CD@5% under ROAD, same baseline rows. Seed 0 is rerun so all three share one code hash, and so its rows can be compared with `results/paper/cdea_boundary/` as an end-to-end reproducibility check.
- **How it is read.** Every seed is reported; none is selected.
- **Verdict.** The G1 result counts as robust to the offset draw only if all three seeds are ahead (mean paired difference above 0) of margin Grad-CAM and of deletion Extremal Perturbations on both datasets. Otherwise the margin cannot be told apart from the method's sensitivity to its offsets.

**Result, 2026-10-11.**

`scripts/g1_readout.py --records results/paper/cells results/paper/cdea_offset_seeds --cdea cdea:A10_boundary_shift cdea:A10_boundary_shift_o1 cdea:A10_boundary_shift_o2 --seeds 0`

- **Verdict.** All three offset seeds print "ahead of both comparators on both datasets". By the declared rule, the G1 result is robust to the offset draw.
- **Margins.** Against margin Grad-CAM, every interval still includes 0, and the HAM10000 margins are the smallest.
- **Reproducibility.** The seed-0 rows reproduce `results/paper/cdea_boundary/` exactly.
- **D2.** Still fails at 5% on CIFAR-10 for all three seeds. It passes on HAM10000.

## Next session

Shift lives beside contrastive. `cdea/shift.py` is the transplant payoff and its own Sinkhorn loop. The contrastive loop was not refactored. `scripts/gate_shift.py` prints the read-out and the stop rule and does not launch the n = 64 pilot.

Smoke records, not paper numbers: shift Waterbirds is under `results/paper/smoke_shift_trained`; planted-patch is trained and not scored. Contrastive CIFAR-10 and HAM10000 are under `results/paper/smoke_contrastive`. Do not train seeds 1–4. The sixth dataset is still unchosen.
