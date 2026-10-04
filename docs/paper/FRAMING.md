# Framing (draft)

Claims are tied to a table and a test. A method that beats CDEA on any metric or dataset stays in the table, and the text says where it is better. No number in the paper is typed by hand. Every number comes from `evaluation/` and `analysis/` on the shared inputs.

## Make only if the confirmatory test passes

- CDEA's unique evidence is more contrastive than margin attribution and Extremal Perturbations at matched area. Family C, CD@5% under ROAD.
- Allocation makes explanations more model-dependent. The model-randomisation check.
- The shortcut mask captures cross-environment disagreement. Family S, ΔD.
- Lesion overlap cannot validate contrastive explanations. The centre-prior comparison, reported as a negative finding.

## Do not make

- Good localisation.
- The same budget on every dataset.
- Equilibrium or convergence properties. The method is joint optimisation of one loss for a fixed number of steps.
- Scalability beyond the measured forward and backward counts.
- Usefulness to people. There is no user study.

## Exploratory, and labelled as such

- Deletion and insertion AUC.
- Compactness and stability.
- Worst-group accuracy after masking the shortcut. DFR and GroupDRO are reference points for that use case, not like-for-like explanation comparisons.
- Per-image paired tests. The confirmatory unit is the dataset.

This draft does not contain results. Gate G0 is still closed, and `docs/paper/EVAL_PLAN.md` is not frozen.
