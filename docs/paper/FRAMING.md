# Framing (draft)

Claims are tied to a table and a test. A method that beats CDEA on any metric or dataset stays in the table, and the text says where it is better. No number in the paper is typed by hand. Every number comes from `evaluation/` and `analysis/` on the shared inputs.

## Claims

The method is the decomposition game (`FORMULATION_DECOMPOSITION.md`). It splits one pair's evidence into unique to k, unique to l, and shared. The claims and their tests are `EVAL_PLAN.md` section 13.

- **D1.** On planted pairs, CDEA's three maps recover the planted roles better than per-class attribution split by the min-split rule. U_k should fall on A_k, U_l on A_l, and S on C. Family D-T, RR@2.5%.
- **D2.** On natural images, CDEA's unique maps are purer and its shared map is more valid than the min-split rows. Family D-C, PUR_k + PUR_l and SV at 5% under ROAD.
- **Joint search.** CDEA against occlusion_assign decides the framing (`EVAL_PLAN.md` 13.4):
  - If CDEA is ahead, the joint search is presented as the method.
  - Otherwise, the contribution is the definition, the scores and the ground truth, and CDEA is reported as the joint-search variant.
- **If neither D1 nor D2 holds**, the decomposition is a negative result. It is reported beside the evaluation-paper route: the contrastive evaluation protocol, under which margin attribution matches or beats mask optimisation, together with the lesion-overlap negative result.

## Do not make

- Good localisation.
- The same budget on every dataset.
- Equilibrium or convergence properties. The method optimises one objective for a fixed number of steps.
- Scalability beyond the measured forward and backward counts.
- Usefulness to people. There is no user study.

## Reported, not tested

- SPEC_k + SPEC_l (CD), BAL and SEP.
- The contrast game, ablation DA2 (`FORMULATION.md`). Its claims C1–C5 are reported as descriptive rows (`EVAL_PLAN.md` section 1).
- The robustness checks: the blur operator, the other areas, ViT-B/16, and every image beside the role-confirmed images.
- Deletion and insertion AUC, compactness and stability. They are exploratory and labelled as such.

This draft does not contain results. `docs/paper/EVAL_PLAN.md` is not frozen. Its section 13.4 was signed off on 2026-10-14.
