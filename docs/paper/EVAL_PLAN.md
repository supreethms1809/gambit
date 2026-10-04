# Evaluation plan (draft)

This draft is not frozen. The tag `eval-plan-frozen` was not created. The test split stays locked. Hyperparameters are not selected. S19 freezes this file after val selection. Until that tag exists, no run reads the test split and no run passes `--final`.

## Budget

Every method's map becomes a binary pixel mask covering a fixed fraction of the image. Ties use the seeded jitter in `evaluation.masks.top_fraction_mask`. The primary fraction is 5%. Robustness fractions are 2.5% and 10%. Soft-mask scores are the D2 check only.

The provisional image counts in `docs/paper/PROGRESS.md` are not part of this draft's freeze. They stay provisional until a timed CDEA cell revises them.

## Removal

Optimisation uses blur keep. The primary score uses ROAD noisy-linear imputation, `evaluation.removal.noisy_linear_impute`, with 24 iterations and noise 0.01, the settings in `scripts/eval_faithfulness.py`. The blur operator is reported beside ROAD.

## Primary metrics

- Contrastive: CD@5% under ROAD. `evaluation.scores.contrastive_deletion` is m(x without M_l) − m(x without M_k), where m = z_k − z_l, k is hypothesis rank 0, and l is rank 1.
- Shift: ΔD. `evaluation.scores.disagreement_reduction` is the drop in |p_y(x_id) − p_y(x_ood)| after removing the shortcut mask in every environment, minus the same drop for a random mask of equal area.

## Secondary metrics named in this draft

- K×K deletion matrix. Entry (i, j) is the drop in class i's logit when mask j is removed. The summary is mean diagonal minus mean off-diagonal, and |diagonal| / |off-diagonal|.
- Two-patch recovery: share of M_k on patch A and of M_l on patch B.

Deletion AUC, model randomisation, compactness, stability, and lesion overlap stay in the plan and are not implemented in this draft.

## Confirmatory families

The unit is the dataset. Five seeds are averaged inside a dataset before methods are compared.

Family C, CD@5% under ROAD, eight contrastive datasets, two-sided Wilcoxon signed-rank with Holm:

- CDEA versus margin attribution
- CDEA versus contrastive Extremal Perturbations
- CDEA versus CVE

CDEA versus base evidence is in the main table and is not in this family.

Family S, ΔD, six shift datasets: CDEA-shift versus the strongest baseline on val. One comparison.

Each report has wins out of N, the mean difference, one bootstrap 95% interval over datasets, and the Holm-adjusted p. `analysis.stats.family_summary` computes them. Anything else is exploratory.

A confirmatory claim is called robust only after leave-one-dataset-out, dropping the two dev datasets, the second backbone, the blur operator, the 2.5% and 10% budgets, and a comparison with the measured MPS run-to-run variation. Those checks are not run in this draft.

## What is not decided here

- The selected λ, step count, learning rate, and evidence backend.
- Which baseline wins family S on val.
- The test-sample seeds.
- Config hashes.

The selection rule is `analysis.selection.select_config`. It maximises the mean CD@5% on the dev val sets among candidates that close D1–D5. It has not been applied to val. No config hash is written here.

Gate G0 is still closed. Gate G1 did not run. This draft does not start a pilot comparison.
