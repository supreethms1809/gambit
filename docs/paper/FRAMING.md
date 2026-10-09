# Framing (draft)

Claims are tied to a table and a test. A method that beats CDEA on any metric or dataset stays in the table, and the text says where it is better. No number in the paper is typed by hand. Every number comes from `evaluation/` and `analysis/` on the shared inputs.

## Contrastive claims

- **C1.** CDEA's unique evidence is more necessary for the preference of k over l than margin attribution and deletion Extremal Perturbations at matched area. Family C, CD@5% under ROAD.
- **C2.** That evidence aligns with annotated discriminative parts.
- **C3.** That evidence aligns with planted cues.
- **C4.** Shared and unique evidence are the parts the payoffs define.
- **C5.** The allocation is model-dependent. The model-randomisation check.
- **N1.** Lesion overlap cannot validate contrastive explanations. The centre-prior comparison, reported as a negative finding.

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

This draft does not contain results. The G1 pilot is not scored, and `docs/paper/EVAL_PLAN.md` is not frozen. G1 picks between the method framing above and the evaluation-paper framing in `docs/paper/G1.md`.
