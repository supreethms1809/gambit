# Results audit

The audit does not pass. RESULTS.md is absent. Masks were not checked. No discrepancy was found in a paper file, because there is no paper file. The path that was not written is `results/paper/RESULTS.md`.

## What the checker does

`analysis/audit_results.py` compares a markdown file with a fresh call to `analysis.build_results.render_results`. A matching file passes that comparison. A hand-edited number does not. The checker does not rewrite the file. A discrepancy is fixed in code, then the file is generated again.

## What was not done

The plan asks for a fresh session that did not write the results text, and for an eye check of a sample of masks on every dataset. This session wrote the builder. The fresh session has not been run. The required datasets include ImageNet-S and the six shift datasets. No mask from those datasets was inspected.

The test split was not read. The final-runs tag was not created.
