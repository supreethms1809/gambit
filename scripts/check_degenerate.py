"""D1–D5 checks on an allocation.

D1 and D5 are properties of the plan. D2, D3 and D4 are properties of the
scored cell and are reported from its records. D3 is not a gate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

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


def d2_soft_hard_gap(soft: torch.Tensor, hard: torch.Tensor, tol: float = 0.5) -> bool:
    """Gate: the soft plan and the hard mask agree within 0.5 nats."""
    return bool((soft - hard).abs().max() <= tol)


def stage_gaps(records: Path, method: str = "cdea", tol: float = 0.5) -> dict[float, dict]:
    """D2 per area from a contrastive cell's records.

    D2 is the mean soft-minus-hard payoff, for rank 0 and rank 1, and passes
    when both are at most ``tol`` (FORMULATION.md section 11). The hard-minus-
    scored gap is reported, not gated.
    """
    path = Path(records)
    if path.is_dir():
        path = path / "records.csv.gz"
    by_area: dict[float, dict[str, list[float]]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            # Stage payoffs do not depend on the scoring operator; read one copy.
            if row["method"] != method or row["operator"] != "road" or not row.get("payoff_soft_k"):
                continue
            slot = by_area.setdefault(float(row["area"]), {})
            for rank in ("k", "l"):
                soft = float(row[f"payoff_soft_{rank}"])
                hard = float(row[f"payoff_hard_{rank}"])
                scored = float(row[f"payoff_scored_blur_{rank}"])
                slot.setdefault(f"soft_minus_hard_{rank}", []).append(soft - hard)
                slot.setdefault(f"hard_minus_scored_{rank}", []).append(hard - scored)
    out = {}
    for area, slot in sorted(by_area.items()):
        means = {key: sum(values) / len(values) for key, values in slot.items()}
        means["n"] = len(slot["soft_minus_hard_k"])
        means["d2_pass"] = means["soft_minus_hard_k"] <= tol and means["soft_minus_hard_l"] <= tol
        out[area] = means
    return out


def d3_overshoot_share(share: torch.Tensor) -> float:
    """Reported: mean share of the shortcut payoff that comes from overshooting."""
    return float(share.detach().mean())


def d4_beats_translation(score: torch.Tensor, translated: torch.Tensor) -> bool:
    """Shift gate: the shortcut mask beats a translated copy of itself on val ΔD."""
    return bool(score.detach().mean() > translated.detach().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=None, help="a cell directory under results/paper/cells")
    args = parser.parse_args()
    if not args.records:
        raise SystemExit(
            "D1 and D5 are checked on the allocation inside a cell. "
            "Pass --records results/paper/cells/<split>/contrastive/..."
        )
    gaps = stage_gaps(Path(args.records))
    if not gaps:
        raise SystemExit("no cdea rows with stage payoffs in these records")
    print("area   n  soft-hard k  soft-hard l  D2    hard-scored k  hard-scored l")
    for area, g in gaps.items():
        verdict = "pass" if g["d2_pass"] else "FAIL"
        print(f"{area:<5} {g['n']:>3}  {g['soft_minus_hard_k']:>11.2f}  {g['soft_minus_hard_l']:>11.2f}  {verdict:<4}  "
              f"{g['hard_minus_scored_k']:>13.2f}  {g['hard_minus_scored_l']:>13.2f}")


if __name__ == "__main__":
    main()
