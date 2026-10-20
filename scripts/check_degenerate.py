"""D1–D5 checks on an allocation, and D2, D8, D9 for the decomposition game.

D1 and D5 are properties of the plan. D2, D3 and D4 are properties of the
scored cell and are reported from its records. D3 is not a gate.

Decomposition cells (``.../decomposition/...``, FORMULATION_DECOMPOSITION.md
section 8): D2 is gated for all three players. D8 (S found no shared evidence:
SV <= 0 under blur) and D9 (the final masks are the min-split start), and the
share of final cells outside the prior's support, are reported.

Shift cells (``.../shift/...``, SHIFT_FORMULATION.md): D2 gates soft-minus-hard
payoff <= 0.5 for R and S. D4 gates S beating its translated copy on val ΔD.
D3 (overshoot share) and D9 (same_as_init) are reported.
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


def _paper_rows(handle):
    """Record rows that can count toward a gate. Fast-knob rows never do."""
    return (row for row in csv.DictReader(handle) if row.get("fast", "0") != "1")


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
        for row in _paper_rows(handle):
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


DECOMP_PLAYERS = ("uk", "ul", "s")


def decomposition_gaps(records: Path, method: str = "cdea_decomp", tol: float = 0.5) -> dict[float, dict]:
    """Per area: mean soft-minus-hard and hard-minus-scored payoff per player, D2, D8, D9 and trapped support."""
    path = Path(records)
    if path.is_dir():
        path = path / "records.csv.gz"
    by_area: dict[float, dict[str, list[float]]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in _paper_rows(handle):
            if row["method"] != method or not row.get("payoff_soft_uk"):
                continue
            slot = by_area.setdefault(float(row["area"]), {})
            if row["operator"] == "blur":
                slot.setdefault("d8", []).append(float(float(row["sv"]) <= 0))
            if row["operator"] != "road":
                continue
            for player in DECOMP_PLAYERS:
                soft = float(row[f"payoff_soft_{player}"])
                hard = float(row[f"payoff_hard_{player}"])
                scored = float(row[f"payoff_scored_blur_{player}"])
                slot.setdefault(f"soft_minus_hard_{player}", []).append(soft - hard)
                slot.setdefault(f"hard_minus_scored_{player}", []).append(hard - scored)
            if row.get("same_as_init", "") != "":
                slot.setdefault("d9", []).append(float(row["same_as_init"]))
            if row.get("outside_prior", "") != "":
                slot.setdefault("outside_prior", []).append(float(row["outside_prior"]))
    out = {}
    for area, slot in sorted(by_area.items()):
        means = {key: sum(values) / len(values) for key, values in slot.items()}
        means["n"] = len(slot["soft_minus_hard_uk"])
        means["d2_pass"] = all(means[f"soft_minus_hard_{p}"] <= tol for p in DECOMP_PLAYERS)
        out[area] = means
    return out


def shift_gaps(records: Path, method: str = "cdea_shift", tol: float = 0.5) -> dict[float, dict]:
    """Per area: D2 (soft-minus-hard for R and S), D4 (S vs translated on ΔD), D3 (overshoot) and D9."""
    path = Path(records)
    if path.is_dir():
        path = path / "shift.csv.gz"
    by_area: dict[float, dict[str, list[float]]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in _paper_rows(handle):
            if row["method"] != method or not row.get("payoff_soft_r"):
                continue
            slot = by_area.setdefault(float(row["area"]), {})
            for player in ("r", "s"):
                soft = float(row[f"payoff_soft_{player}"])
                hard = float(row[f"payoff_hard_{player}"])
                slot.setdefault(f"soft_minus_hard_{player}", []).append(soft - hard)
            if row.get("delta_d", "") != "" and row.get("delta_d_translated", "") != "":
                slot.setdefault("delta_d", []).append(float(row["delta_d"]))
                slot.setdefault("delta_d_translated", []).append(float(row["delta_d_translated"]))
                slot.setdefault("d4_diff", []).append(float(row["delta_d"]) - float(row["delta_d_translated"]))
            if row.get("overshoot_share", "") != "":
                slot.setdefault("d3", []).append(float(row["overshoot_share"]))
            if row.get("same_as_init", "") != "":
                slot.setdefault("d9", []).append(float(row["same_as_init"]))
    out = {}
    for area, slot in sorted(by_area.items()):
        means = {key: sum(values) / len(values) for key, values in slot.items()}
        means["n"] = len(slot["soft_minus_hard_r"])
        means["d2_pass"] = means["soft_minus_hard_r"] <= tol and means["soft_minus_hard_s"] <= tol
        means["d4_pass"] = means.get("d4_diff", 0.0) > 0.0
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
    parser.add_argument("--method", default=None, help="the CDEA label (default: cdea, cdea_decomp, or cdea_shift)")
    args = parser.parse_args()
    if not args.records:
        raise SystemExit(
            "D1 and D5 are checked on the allocation inside a cell. "
            "Pass --records results/paper/cells/<split>/contrastive/..."
        )
    rec = Path(args.records)
    if "decomposition" in rec.parts:
        report_decomposition(rec, args.method or "cdea_decomp")
        return
    if "shift" in rec.parts or (rec.is_dir() and (rec / "shift.csv.gz").is_file()) or rec.name == "shift.csv.gz":
        report_shift(rec, args.method or "cdea_shift")
        return
    gaps = stage_gaps(rec, args.method or "cdea")
    if not gaps:
        raise SystemExit("no cdea rows with stage payoffs in these records")
    print("area   n  soft-hard k  soft-hard l  D2    hard-scored k  hard-scored l")
    for area, g in gaps.items():
        verdict = "pass" if g["d2_pass"] else "FAIL"
        print(f"{area:<5} {g['n']:>3}  {g['soft_minus_hard_k']:>11.2f}  {g['soft_minus_hard_l']:>11.2f}  {verdict:<4}  "
              f"{g['hard_minus_scored_k']:>13.2f}  {g['hard_minus_scored_l']:>13.2f}")


def report_decomposition(records: Path, method: str) -> None:
    gaps = decomposition_gaps(records, method)
    if not gaps:
        raise SystemExit(f"no {method} rows with stage payoffs in these records")
    print(f"{method}: D2 gates soft-hard <= 0.5 for every player. D8, D9 and outside-prior are reported.")
    print("area   n  soft-hard uk   ul    s   D2    hard-scored uk   ul    s    D8    D9   outside")
    for area, g in gaps.items():
        verdict = "pass" if g["d2_pass"] else "FAIL"
        gap = "  ".join(f"{g[f'soft_minus_hard_{p}']:5.2f}" for p in DECOMP_PLAYERS)
        scored = "  ".join(f"{g[f'hard_minus_scored_{p}']:5.2f}" for p in DECOMP_PLAYERS)
        print(f"{area:<5} {g['n']:>3}  {gap}  {verdict:<4}  {scored}  {g.get('d8', float('nan')):5.2f} "
              f"{g.get('d9', float('nan')):5.2f}  {g.get('outside_prior', float('nan')):5.2f}")


def report_shift(records: Path, method: str) -> None:
    gaps = shift_gaps(records, method)
    if not gaps:
        raise SystemExit(f"no {method} rows with stage payoffs in these records")
    print(f"{method}: D2 gates soft-hard <= 0.5 for R and S; D4 gates ΔD > translated. D3 and D9 are reported.")
    print("area   n  soft-hard r   s   D2    ΔD-trans  D4    D3     D9")
    for area, g in gaps.items():
        d2 = "pass" if g["d2_pass"] else "FAIL"
        d4 = "pass" if g["d4_pass"] else "FAIL"
        print(f"{area:<5} {g['n']:>3}  {g['soft_minus_hard_r']:>11.2f} {g['soft_minus_hard_s']:>5.2f}  {d2:<4}  "
              f"{g.get('d4_diff', float('nan')):>8.2f}  {d4:<4}  {g.get('d3', float('nan')):>5.2f}  "
              f"{g.get('d9', float('nan')):>5.2f}")


if __name__ == "__main__":
    main()
