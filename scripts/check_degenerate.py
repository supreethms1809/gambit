"""D1–D5 checks on an allocation.

D1 and D5 are properties of the plan. D2, D3 and D4 are properties of the
scored cell and are reported from its records. D3 is not a gate.
"""

from __future__ import annotations

import argparse

import torch

TOL = 1e-3


def marginal_error(plan: torch.Tensor, row_target: torch.Tensor) -> tuple[float, float]:
    """Max absolute row error and column error. Columns sum to 1."""
    rows = plan.sum(dim=-1)
    cols = plan.sum(dim=-2)
    target = row_target.to(rows.dtype)
    while target.ndim < rows.ndim:
        target = target.unsqueeze(0)
    row_err = float((rows - target).abs().max())
    col_err = float((cols - 1).abs().max())
    return row_err, col_err


def d1_shared_on_budget(shared: torch.Tensor, budget: float, tol: float = TOL) -> bool:
    """The shared row sums to ``budget`` within ``tol`` on every image."""
    return bool((shared.sum(dim=-1) - budget).abs().max() <= tol)


def d5_marginals(plan: torch.Tensor, row_target: torch.Tensor, tol: float = TOL) -> bool:
    row_err, col_err = marginal_error(plan, row_target)
    return row_err <= tol and col_err <= tol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=None, help="a cell directory under results/paper/cells")
    args = parser.parse_args()
    if not args.records:
        raise SystemExit(
            "D1 and D5 are checked on the allocation inside a cell. "
            "Pass --records results/paper/cells/<split>/contrastive/..."
        )
    print(args.records)


if __name__ == "__main__":
    main()
