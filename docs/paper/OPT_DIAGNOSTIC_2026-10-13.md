# CDEA optimisation diagnostic and question audit, 2026-10-13

Scratch diagnostic, not a paper result. Seed 0, val only. Nothing under the method code roots was changed, and no record under `results/` was written.

## 0. Summary

1. **The optimiser does not get stuck in a local minimum.** Its objective rises and plateaus, and different starts and step sizes reach the same plateau. It is not a code bug either: the copied loop matches `allocate()` to 4e-6.
2. **The binary-draw optimiser stops well short of its own objective.** The fractional plan (A12) reaches about 36% more of the *same* objective, the expected payoff over binary draws.
3. **More optimisation of CDEA's objective lowers CD.** With 400 steps instead of 100, CDEA's objective on its hard cells goes up, while CD goes down on both datasets. So the objective and the metric pull apart.
4. **CDEA's contrastive game answers a different question from CD.** CD asks "why k rather than l". The game asks how to split the image into evidence unique to each of the top-5 candidates, plus evidence shared by the shortlist. The same machinery aimed at the pair question (K = 2, no shared player, fractional masks) matches or beats margin Grad-CAM.
5. **CD can go far higher than margin Grad-CAM reaches.** Deleting one cell at a time and ranking cells by the change in z_k − z_l beats margin Grad-CAM by +2.49 on CIFAR-10 and +1.00 on ImageNet. Both intervals are clear of 0.
6. **The shift game has the same kind of mismatch.** Its shortcut player is scored by pasting cells in (substitution). ΔD scores the mask by deleting it (necessity).

## 1. Setup

- **Data:** ResNet-50, seed 0 val, the first 32 images of each cell's sample (n = 32), CIFAR-10 and ImageNet.
- **Area and scoring:** area 5%. Masks are scored with the cell's own `budget_or_floor` and `remove`, under ROAD and blur.
- **The loop:** `optcheck.py` copies `allocate()`'s loop so it can record the trajectory. Its largest difference from `allocate()` is 3.9e-6 on CIFAR-10 and 1.8e-6 on ImageNet.
- **Run time:** about 44 minutes on CIFAR-10 and 38 minutes on ImageNet. Both runs finished at 16:44 UTC.

**Columns**
- **CD / CD1, ROAD / blur:** the scored top-a pixel masks, as in the paper cells.
- **J_exp:** CDEA's objective as optimised, the mean over 16 binary draws and boundary offsets with common random numbers. It is scored the same way for every run, including the fractional one. The K = 2 runs use the K = 2 objective, so their J_exp is not comparable with the K = 5 runs.
- **J_hard:** the objective on hard top cells (each player keeps its top cells), under blur, with no offsets.
- **Jpair:** the pair part of J_hard, u_k + u_l.
- **CDcell:** the pair-margin CD under blur on the hard cells.
- **Δ vs margin GC:** CD under ROAD, minus margin Grad-CAM on the same images, with a 95% bootstrap interval over images.

**Runs**

| Name | What it is |
|---|---|
| default | CDEA as it runs in the cells: K = 5 plus a shared player, binary draws, Grad-CAM init, 100 steps, lr 0.1 |
| default_400steps | the same, with 400 steps |
| default_lr0.5 | the same, with lr 0.5 |
| uniform_init_400 | uniform start, 400 steps |
| margin_init | starts from margin Grad-CAM |
| A12_K5_fractional | the same game with fractional masks |
| pairCD_binary | K = 2 (k and l only), no shared player, binary draws |
| pairCD_fractional | K = 2, no shared player, fractional masks |
| pairCD_fractional_margin_init | pairCD_fractional started from margin Grad-CAM |
| greedy_oracle_blur | greedy search, 3 cells, under blur. For M_k it picks the cells whose deletion lowers z_k − z_l most; for M_l, those whose deletion raises it most |
| single_cell_occlusion | rank cells by the change in z_k − z_l when each is deleted alone, then score the top-a mask |

## 2. Results

### CIFAR-10, n = 32

| Method | CD ROAD | CD1 ROAD | CD blur | CD1 blur | J_exp | J_hard | Jpair | CDcell | Δ vs margin GC (CD ROAD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| margin Grad-CAM | 2.25 | 2.55 | 3.53 | 3.19 | | | | 2.32 | — |
| base evidence | 2.80 | 2.24 | 2.52 | 2.54 | | | | | +0.54 [−0.27, +1.44] |
| greedy oracle (blur) | 4.04 | 3.42 | 5.59 | 4.31 | | | | 7.67 | **+1.79 [+0.71, +2.92]** |
| single-cell occlusion | **4.74** | 3.72 | 5.92 | 3.93 | | | | | **+2.49 [+1.44, +3.58]** |
| default CDEA | 1.62 | 1.86 | 3.11 | 2.92 | 5.36 | 6.93 | 2.98 | 2.75 | −0.63 [−1.84, +0.72] |
| 400 steps | 1.14 | 1.78 | 2.31 | 2.47 | 4.98 | 7.22 | 2.97 | 2.76 | −1.11 [−2.31, +0.15] |
| lr 0.5 | 1.21 | 1.87 | 2.43 | 2.58 | 5.29 | 6.71 | 2.34 | 2.21 | −1.04 [−2.24, +0.16] |
| uniform init, 400 steps | 0.96 | 2.06 | 1.43 | 2.40 | 5.03 | 7.00 | 2.70 | 2.25 | −1.29 [−2.36, −0.20] |
| margin init | 2.15 | 2.17 | 3.60 | 3.09 | 5.30 | 7.13 | 3.43 | 3.06 | −0.11 [−1.32, +1.19] |
| A12, K = 5, fractional | 2.23 | 2.19 | 3.84 | 2.92 | **7.27** | **10.72** | 4.27 | 4.07 | −0.02 [−1.37, +1.42] |
| pair CD, binary | 2.57 | 2.39 | 3.42 | 3.16 | 2.38* | 2.89 | 2.89 | 2.89 | +0.32 [−1.06, +1.78] |
| pair CD, fractional | 2.92 | 3.04 | 4.27 | 3.49 | 2.87* | 4.88 | 4.88 | 4.88 | +0.67 [−0.64, +2.02] |
| pair CD, fractional, margin init | 2.86 | 2.69 | 3.80 | 3.12 | 2.88* | 4.57 | 4.57 | 4.57 | +0.61 [−0.49, +1.72] |

\* K = 2 objective; not comparable with the K = 5 J values.

**Support checks.**
- 98% of CDEA's final k cells lie inside Grad-CAM's nonzero support, but only 21% of them are among Grad-CAM's top cells.
- Grad-CAM is zero on 34% of cells.
- 19% of the greedy oracle's k cells lie outside Grad-CAM's support.

### ImageNet, n = 32

| Method | CD ROAD | CD1 ROAD | CD blur | CD1 blur | J_exp | J_hard | Jpair | CDcell | Δ vs margin GC (CD ROAD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| margin Grad-CAM | 1.95 | 1.31 | 1.95 | 1.36 | | | | 1.94 | — |
| base evidence | 1.19 | 0.96 | 1.11 | 0.94 | | | | | −0.76 [−1.38, −0.14] |
| greedy oracle (blur) | **3.26** | 1.91 | 4.09 | 2.38 | | | | 5.07 | **+1.31 [+0.62, +2.04]** |
| single-cell occlusion | 2.95 | 1.93 | 3.77 | 2.29 | | | | | **+1.00 [+0.19, +1.69]** |
| default CDEA | 1.12 | 0.72 | 1.50 | 0.96 | 3.24 | 4.67 | 1.90 | 1.83 | −0.84 [−1.53, −0.18] |
| 400 steps | 0.80 | 0.61 | 1.41 | 1.01 | 3.65 | 5.62 | 1.95 | 1.84 | −1.16 [−1.84, −0.46] |
| lr 0.5 | 1.16 | 0.72 | 1.34 | 0.82 | 3.53 | 4.59 | 1.50 | 1.41 | −0.80 [−1.34, −0.29] |
| uniform init, 400 steps | 1.09 | 0.61 | 1.12 | 0.89 | 3.35 | 4.97 | 1.85 | 1.81 | −0.86 [−1.41, −0.33] |
| margin init | 1.14 | 0.55 | 1.35 | 0.91 | 3.21 | 3.93 | 1.84 | 1.74 | −0.82 [−1.41, −0.27] |
| A12, K = 5, fractional | 1.76 | 1.21 | 1.94 | 1.26 | **4.43** | **6.59** | 2.25 | 2.13 | −0.19 [−1.02, +0.61] |
| pair CD, binary | 1.23 | 0.81 | 1.57 | 1.14 | 1.35* | 1.91 | 1.91 | 1.91 | −0.72 [−1.45, −0.03] |
| pair CD, fractional | 1.89 | 1.22 | 2.21 | 1.40 | 1.88* | 2.58 | 2.58 | 2.58 | −0.06 [−0.75, +0.59] |
| pair CD, fractional, margin init | 2.24 | 1.49 | 2.42 | 1.64 | 2.02* | 2.78 | 2.78 | 2.78 | +0.28 [−0.27, +0.86] |

**Support checks.**
- 100% of CDEA's final k cells lie inside Grad-CAM's support; 18% are among Grad-CAM's top cells.
- Grad-CAM is zero on only 2% of cells.
- 1% of the oracle's k cells lie outside Grad-CAM's support.

### Objective trajectories (J_exp, mean over images)

| Run | CIFAR-10 | ImageNet |
|---|---|---|
| default | t0 2.82, t20 3.96, t40 4.62, t60 5.04, t80 5.40, t100 5.36 | t0 1.26, t20 2.14, t40 2.93, t60 3.02, t80 3.25, t100 3.24 |
| 400 steps | t100 5.36, t200 5.15, t300 5.03, t400 4.98 | t100 3.24, t200 3.67, t300 3.74, t400 3.65 |
| lr 0.5 | t20 5.22, t40 5.40, t100 5.29 | t20 3.53, t40 3.54, t100 3.53 |
| uniform init, 400 steps | t0 −0.73, t100 3.98, t200 4.65, t400 5.03 | t0 0.48, t100 2.94, t200 3.17, t400 3.35 |
| margin init | t0 2.64, t100 5.30 | t0 1.66, t100 3.21 |
| A12 fractional | t0 2.82, t40 5.98, t80 6.88, t100 7.27 (still rising) | t0 1.26, t40 3.38, t80 4.26, t100 4.43 (still rising) |

### Transplant test

Margin Grad-CAM's top cells replace CDEA's k and l cells inside CDEA's own allocation. Scored on hard cells under blur, with no offsets.

| | CIFAR-10 | ImageNet |
|---|---|---|
| CDEA objective J_hard: CDEA → transplant | 6.93 → 5.93, diff −1.00 [−2.11, +0.02] | 4.67 → 4.67, diff −0.00 [−0.63, +0.61] |
| pair part u_k + u_l: CDEA → transplant | 2.98 → 2.44, diff −0.55 [−1.52, +0.39] | 1.90 → 1.94, diff +0.05 [−0.35, +0.45] |

By CDEA's own objective, its cells are as good as margin Grad-CAM's or better. Margin Grad-CAM still scores higher on CD.

## 3. Reading the results

### Is the optimiser stuck?

No, in the sense of a bad local minimum:
- The objective rises, then plateaus.
- Different starts (Grad-CAM, uniform, margin Grad-CAM) and step sizes (0.1, 0.5) end at a similar value: about 5.0–5.4 on CIFAR-10 and 3.2–3.7 on ImageNet.
- The copied loop matches `allocate()`.

The *binary-draw* optimiser is still weak, though:
- On the same expected-binary objective, the fractional plan reaches 7.27 against 5.36 on CIFAR-10, and 4.43 against 3.24 on ImageNet. That is about 36% more, and A12 was still rising at step 100.
- The straight-through gradient through the binary draws is a poor optimiser of the objective it was introduced to optimise.

Gradient search through the Sinkhorn plan also stops short on the plain pair question:
- pair CD, fractional, reaches CDcell 4.88 (CIFAR-10) and 2.58 (ImageNet).
- Greedy 3-cell search reaches 7.67 and 5.07.

**Trapped support.** θ starts at log(E + 1e-6), so cells where Grad-CAM is zero start with almost no mass. This matters a little on CIFAR-10, where Grad-CAM is zero on 34% of cells and 19% of the oracle's cells are there. It does not matter on ImageNet (2% and 1%).

### Where CD is lost, largest first

1. **The optimiser.** A12 and the greedy oracle show how much is left unreached.
2. **Cell plan to scored mask.**
   - On CIFAR-10, CDEA beats margin Grad-CAM on hard cells under blur (CDcell 2.75 vs 2.32) but loses on the scored pixel mask (CD ROAD 1.62 vs 2.25).
   - Margin Grad-CAM is a smooth pixel map, so its top-5% mask sits on the peak even when the peak crosses cell borders. A 7×7 cell plan at 5% is about 2.45 cells, and it loses that.
   - On ImageNet, CDEA does not win even on its hard cells (1.83 vs 1.94).
3. **The question mismatch** (section 4). K = 5 with a shared player costs 0.1–0.7 against aiming the same fractional machinery at the pair: A12 vs pair CD, fractional. It is real, but it is the smallest of the three.

### More optimisation, lower CD

Default vs 400 steps:

| | CIFAR-10 | ImageNet |
|---|---|---|
| J_hard | 6.93 → 7.22 | 4.67 → 5.62 |
| CD ROAD | 1.62 → 1.14 | 1.12 → 0.80 |

CDEA's objective is not a proxy for CD. Optimising it harder moves the masks away from what CD rewards.

### Margin Grad-CAM is far below what CD allows

Single-cell occlusion of z_k − z_l is cheap, has no game, and is not in the baseline grid. It beats margin Grad-CAM with intervals clear of 0 on both datasets. The evaluation-paper finding as currently framed ("margin attribution matches or beats mask optimisation") does not hold against a direct search on the margin.

## 4. What the two games answer, from first principles

### Contrastive

**The question CD scores.** "Why k rather than l?" is a question about one number, the model's preference m = z_k − z_l.
- CD = (drop in m from deleting M_k) + (rise in m from deleting M_l).
- These are two independent subset-selection problems on the same function: the area-a region that pushes m down most, and the one that pushes it up most.
- Nothing in it needs other classes, a shared player, exclusivity or a partition.
- Overlap between M_k and M_l cannot change CD, because each mask is deleted on its own.

**The question CDEA's game answers.** "Split the image among the top-5 candidates and the shortlist as a whole, so that each gets the area-a region most exclusively its own, maximising the total." That is a decomposition of the evidence for the whole shortlist. CD scores two of its six players. It differs from the pair question in four ways:
1. **Rivals.** k's payoff is against the log-sum-exp of {l, rank 3, rank 4, rank 5}, not against l. Cells that support both k and l, but not ranks 3–5, look unique to k. This fits the leak: deleting CDEA's M_k lowered z_l by 1.45 on CIFAR-10, against 0.68 for margin Grad-CAM (earlier records).
2. **Budget competition.** The shared player and ranks 3–5 each take a·R cells that CD never scores. A contested cell goes to whichever player values it most, and the shared player outbids k for object cells that support the whole shortlist.
3. **Operator.** The game is optimised under blur; the headline is scored under ROAD. Under ROAD the l side has a negative floor (section 5).
4. **Estimator.** Binary draws, boundary jitter, and whole 7×7 cells, against a top-a pixel mask at scoring.

**What would score CDEA's own question.** A measure of exclusivity and shared validity: removing M_k lowers z_k but not z_l, and removing the shared region lowers the whole shortlist. That is close to C4 (the K×K specificity check), not CD. The reason for reformulating was to separate shared from unique evidence, and CD cannot reward that separation. On the leak numbers above, CDEA does worse than margin Grad-CAM on that separation too.

### Shift

**The question ΔD scores.** "If region S is hidden in every environment, do the environments stop disagreeing?" ΔD is the drop in disagreement after ROAD-deleting the shortcut mask in every environment, minus random. That is **necessity, by deletion**.

**The question CDEA-shift's game answers.** "If region S is pasted from environment e into the in-distribution image, how much of the log-odds gap closes?" That is **substitution**, closer to sufficiency. The robust player (the worst-environment deletion drop) also takes budget under exclusivity, and ΔD never scores it.

**How this fits the numbers** (ViT, seed 0):
- **Planted patch,** where the shortcut is small and local, so both interventions agree. CDEA-shift − attribution difference = **+1.28 [+0.09, +2.52]** at 5%, then −0.39 at 10% and −0.34 at 25%.
- **Waterbirds,** where the shortcut is the background: large and diffuse. Hiding 5% of it leaves the rest still differing, so the two questions diverge. CDEA-shift − gap attribution = −0.87, −0.62, −0.49 at 5 / 10 / 25%.
- On both datasets, CDEA-shift beats random, EP-shift and SpRAy at every area.

The existing ablation `deletion_shortcut_payoff` matches ΔD's intervention. The optimiser checks above were **not** run on the shift game.

## 5. CD and CD1 across areas (seed 0 paper cells)

Contrastive areas are 2.5 / 5 / 10%. There is no contrastive CD at 25%: with 6 players the area is capped at 1/6. 25% belongs only to the shift game (areas 5 / 10 / 25%).

**Headline, CD@5% under ROAD**

| Cell | CDEA | margin GC | margin IG | deletion EP | base evidence | CDEA − better margin |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 R50 | 2.42 | 2.72 | 0.61 | 0.59 | 2.46 | −0.30 [−0.80, +0.19] |
| ImageNet R50 | 1.00 | 2.02 | 2.67 | 0.14 | — | −1.67 |
| ImageNet ViT (n = 64) | 1.15 | 0.71 | 2.83 | — | — | −1.68 |

**All finished contrastive cells, ROAD, n = 200 (ViT n = 64).** Read at 18:10 UTC, 2026-10-13.

| Cell | Area | CDEA | margin GC | margin IG | deletion EP | base evidence | CDEA − better margin |
|---|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 R50 | 2.5% | 1.38 | 1.94 | 0.45 | 0.09 | 1.75 | −0.57 [−0.94, −0.19] |
| | 5% | 2.42 | 2.72 | 0.61 | 0.59 | 2.46 | −0.30 [−0.77, +0.19] |
| | 10% | 3.43 | 3.79 | 0.66 | 0.93 | 3.17 | −0.36 [−0.92, +0.20] |
| CIFAR-100 R50 | 2.5% | 1.26 | 2.11 | 0.39 | 0.27 | 1.27 | −0.85 [−1.37, −0.32] |
| | 5% | 2.33 | 3.15 | 0.49 | 0.64 | 1.71 | −0.82 [−1.39, −0.24] |
| | 10% | 2.30 | 4.35 | 0.53 | 0.45 | 1.90 | −2.05 [−2.79, −1.33] |
| HAM10000 R50 | 2.5% | 0.58 | 0.69 | 0.00 | −0.35 | 0.97 | −0.11 [−0.45, +0.21] |
| | 5% | 0.65 | 0.81 | −0.02 | 0.10 | 0.79 | −0.16 [−0.51, +0.19] |
| | 10% | 0.83 | 0.66 | −0.12 | 0.22 | 0.54 | **+0.17** [−0.21, +0.55] |
| ImageNet R50 | 2.5% | 0.73 | 1.39 | 2.20 | −0.15 | 0.92 | −1.47 [−1.84, −1.11] |
| | 5% | 1.00 | 2.02 | 2.67 | 0.14 | 1.18 | −1.67 [−2.06, −1.29] |
| | 10% | 1.45 | 2.95 | 2.81 | 0.20 | 1.46 | −1.51 [−1.91, −1.12] |
| ImageNet ViT | 2.5% | 0.63 | 0.53 | 2.17 | 0.07 | 0.12 | −1.54 [−2.20, −0.99] |
| | 5% | 1.15 | 0.71 | 2.83 | 0.05 | 0.09 | −1.68 [−2.37, −1.06] |
| | 10% | 1.98 | 1.15 | 3.45 | 0.22 | −0.01 | −1.48 [−2.28, −0.73] |

- CDEA is behind the better margin variant in 14 of 15 area–cell pairs. The one exception is HAM10000 at 10% (+0.17, interval includes 0).
- CDEA is ahead of deletion EP everywhere.
- On CIFAR-100, CDEA's CD does not grow from 5% to 10% (2.33 → 2.30), while margin Grad-CAM's does (3.15 → 4.35). This fits budget competition in the game (section 4) and is worth checking.

**CD1 (the k side alone), ROAD, at 2.5 / 5 / 10%**

| | CDEA | margin GC |
|---|---|---|
| CIFAR-10 | 1.77 / 2.96 / 4.37 | 1.92 / 2.96 / 4.43 |
| ImageNet R50 | 0.75 / 0.99 / 1.70 | 0.92 / 1.44 / 2.18 (margin IG 1.47 at 5%) |

- **CIFAR-10:** CD1 ties, so CDEA's deficit there is on the l side.
- **ImageNet:** CDEA trails on both sides. The l side (CD − CD1) at 5% is 0.01 for CDEA and 0.58 for margin GC.

**The l side has a negative floor under ROAD.** For a random mask on CIFAR-10, CD − CD1 is −0.62 / −1.34 / −2.47 at 2.5 / 5 / 10%. Deleting almost anything lowers the confident top class. Read l-side numbers against that floor.

## 6. Decisions for you

1. **Which question is the paper about?**
   - **"Why k rather than l".** The multi-player game is not needed, and the bar is a direct search on the margin. Single-cell occlusion already clears margin Grad-CAM by a wide margin. A CDEA aimed at this question reduces to a deletion search, like Extremal Perturbations.
   - **"What evidence is unique to each candidate and what is shared".** That is what CDEA's game actually answers, and CD is the wrong headline metric for it. It needs a metric built for that question (exclusivity / shared validity, C4-like). On the existing leak numbers CDEA does not yet win there either.
   - **Shift.** Either realign the shortcut payoff to deletion (`deletion_shortcut_payoff`), or score the game by a substitution metric that matches what it optimises.
2. **The pre-declared fractional test on seed 1** (`G1.md`, "Reopened, 2026-10-13"). Recommendation: hold it. These numbers put A12 at parity with margin Grad-CAM on CD (−0.02 and −0.19), so it would likely spend the held-out seed-1 data on a predictable fail. If a test is run, it could be redeclared for the arm the chosen question implies, for example pair CD, fractional.
3. **Single-cell occlusion of the margin** should probably join the baseline grid on either route.

Any change chosen from these numbers is a post-hoc choice on seed-0 val, and has to be tested on data CDEA has not seen.

## 7. Caveats

- n = 32 per dataset, one seed, val, ResNet-50 only. The CIFAR-10 intervals are wide.
- The comparisons are paired on the same images, and the direction is the same on both datasets.
- The occlusion and greedy gains have intervals clear of 0 on both datasets.
- The greedy oracle optimises under blur and is scored under both operators.
- The shift-game account (section 4) is reasoning plus the existing ΔD numbers. No optimiser check was run on it.

## 8. Files and status

- **Scripts (scratchpad):** `/home/sureshm/tmp/claude-1000/-home-sureshm-gambit/eeb52147-4fa0-42e8-a76b-21253c74e00b/scratchpad/`
  - `optcheck.py` runs the diagnostic.
  - `optread.py` prints the tables: `python optread.py opt_cifar10.json opt_imagenet.json`.
  - Outputs: `opt_cifar10.json`, `opt_imagenet.json`, `opt_*.log`.
- **Repo:** this file is untracked. The fractional-test files (`scripts/fractional_readout.py`, `tests/test_fractional_readout.py`, and the `G1.md` and `PROGRESS.md` sections) are written, not committed, and not launched. They are waiting on the D2 sign-off.
- **Seed-0 chain at 17:58 UTC, 2026-10-13 (hash `9d84e7f2`, PID 873953):**
  - 9 of 24 cells done: imagenet ViT, imagenet R50, cifar10, cifar100, ham10000, planted_patch ViT and R50, waterbirds ViT and R50.
  - Running: cub200 R50 and oxford_pets R50 (small), imagenet9 ViT and imagenet9 R50 (big).
  - CIFAR-100 and HAM10000 are read in section 5. The planted_patch R50 and waterbirds R50 shift cells have finished but have not been read yet.
