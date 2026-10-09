# Equation audit

Each quantity in `FORMULATION.md` is one function. The earlier formulation (tag `framing-v1`) is not in this tree.

| Quantity | Code |
|---|---|
| Unique log-odds c_k | `cdea.payoffs.unique_log_odds` |
| Shared log-odds s_H | `cdea.payoffs.shared_log_odds` |
| Deletion x ⊖ M and baseline b(x) | `core.grid.delete`, `core.grid.deletion_baseline` |
| Payoff u = score(x) − score(x ⊖ A) | `cdea.payoffs.deletion_payoffs` |
| Sinkhorn plan, I = 20, marginal error ≤ 1e-3 | `cdea.sinkhorn.sinkhorn` |
| Hard top-mass projection (A6) | `cdea.sinkhorn.hard_top_mass` |
| Allocation, Adam, per-image summed loss | `cdea.allocation.allocate` |
| First-order masks (A9) | `cdea.first_order.first_order_masks` |
| Contrastive deletion CD | `evaluation.scores.contrastive_deletion` |
| Sufficiency contrast SC | `evaluation.scores.sufficiency_contrast` |
| Shared validity SV | `evaluation.scores.shared_validity` |

`cdea/` imports `torch`, `core`, and `base_evidence` only.
