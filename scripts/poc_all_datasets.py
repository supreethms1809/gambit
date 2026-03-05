"""
Run CDEA POC matrix:
- datasets: mnist, cifar10, pets
- evidence: gradcam, ig

For each combo, runs ablation (base/naive/optimized) and writes a consolidated report.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Optional

from scripts.ablation_contrastive import run_ablation
from core.reporting import save_json, save_rows_csv

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "scripts" / "out"


def _read_metrics_csv(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for row in rows:
            method = row["method"]
            out[method] = {
                "suff": float(row["suff"]),
                "margin": float(row["margin"]),
                "overlap": float(row["overlap"]),
                "sparse": float(row["sparse"]),
            }
    return out


def _pct_delta(old: float, new: float) -> Optional[float]:
    if abs(old) <= 1e-8:
        return None
    denom = abs(old)
    return (new - old) / denom * 100.0


def run_matrix(
    num_images: int = 64,
    batch_size: int = 16,
    model_name: str = "resnet18",
    pretrained: bool = False,
    ig_steps: int = 8,
    lambda_disjoint: float = 1.0,
    lambda_mass: float = 2.0,
) -> List[Dict[str, float]]:
    combos = [
        ("mnist", "gradcam"),
        ("mnist", "ig"),
        ("cifar10", "gradcam"),
        ("cifar10", "ig"),
        ("pets", "gradcam"),
        ("pets", "ig"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []

    for dataset, evidence in combos:
        prefix = f"ablation_contrastive_{dataset}_{evidence}"
        print("\n=== Running", dataset, evidence, "===")
        run_ablation(
            num_images=num_images,
            batch_size=batch_size,
            dataset=dataset,
            evidence=evidence,
            model_name=model_name,
            pretrained=pretrained,
            ig_steps=ig_steps,
            lambda_disjoint=lambda_disjoint,
            lambda_mass=lambda_mass,
            export_prefix=prefix,
        )
        metrics_csv = OUT / f"{prefix}_metrics.csv"
        metrics = _read_metrics_csv(metrics_csv)
        base = metrics["base_evidence"]
        naive = metrics["naive_contrastive"]
        opt = metrics["optimized"]

        overlap_red_vs_base = _pct_delta(base["overlap"], opt["overlap"])
        overlap_red_vs_naive = _pct_delta(naive["overlap"], opt["overlap"])
        sparse_red_vs_base = _pct_delta(base["sparse"], opt["sparse"])
        row = {
            "dataset": dataset,
            "evidence": evidence,
            "num_images": float(num_images),
            "batch_size": float(batch_size),
            "model": model_name,
            "pretrained": bool(pretrained),
            "ig_steps": float(ig_steps if evidence == "ig" else 0),
            "lambda_disjoint": float(lambda_disjoint),
            "lambda_mass": float(lambda_mass),
            "base_suff": base["suff"],
            "base_margin": base["margin"],
            "base_overlap": base["overlap"],
            "base_sparse": base["sparse"],
            "naive_suff": naive["suff"],
            "naive_margin": naive["margin"],
            "naive_overlap": naive["overlap"],
            "naive_sparse": naive["sparse"],
            "opt_suff": opt["suff"],
            "opt_margin": opt["margin"],
            "opt_overlap": opt["overlap"],
            "opt_sparse": opt["sparse"],
            "overlap_red_vs_base_pct": (-overlap_red_vs_base if overlap_red_vs_base is not None else None),
            "overlap_red_vs_naive_pct": (-overlap_red_vs_naive if overlap_red_vs_naive is not None else None),
            "sparse_red_vs_base_pct": (-sparse_red_vs_base if sparse_red_vs_base is not None else None),
            "suff_change_vs_base_abs": opt["suff"] - base["suff"],
            "margin_change_vs_base_abs": opt["margin"] - base["margin"],
        }
        rows.append(row)

    matrix_csv = OUT / "poc_all_datasets_matrix.csv"
    matrix_json = OUT / "poc_all_datasets_matrix.json"
    save_rows_csv(matrix_csv, rows)
    save_json(matrix_json, {"rows": rows})
    print("\nSaved matrix to", matrix_csv)
    print("Saved matrix to", matrix_json)
    return rows


def write_report(rows: List[Dict[str, float]]) -> Path:
    report = OUT / "POC_REPORT_ALL_DATASETS.md"

    lines: List[str] = []
    lines.append("# CDEA POC Report (All Datasets, Grad-CAM + IG)")
    lines.append("")
    lines.append("This report summarizes 6 concrete runs (3 datasets x 2 evidence providers).")
    lines.append("")
    lines.append("## Run Setup")
    lines.append("")
    lines.append("- Datasets: `mnist`, `cifar10`, `pets`")
    lines.append("- Evidence: `gradcam`, `ig`")
    models = sorted({str(r.get("model")) for r in rows if r.get("model") is not None})
    if models:
        lines.append("- Backbone(s): `{}`".format("`, `".join(models)))
    pre_flag = any(bool(r.get("pretrained", False)) for r in rows)
    lines.append("- ImageNet pretrained weights: `{}`".format("enabled" if pre_flag else "disabled"))
    lam_disjoint = sorted({float(r.get("lambda_disjoint", 0.0)) for r in rows})
    lam_mass = sorted({float(r.get("lambda_mass", 0.0)) for r in rows})
    lines.append("- `lambda_disjoint`: `{}`".format(", ".join(f"{v:g}" for v in lam_disjoint)))
    lines.append("- `lambda_mass`: `{}`".format(", ".join(f"{v:g}" for v in lam_mass)))
    lines.append("- Compared methods per run: `base_evidence`, `naive_contrastive`, `optimized (CDEA)`")
    lines.append("")
    lines.append("## Advantage Summary")
    lines.append("")
    lines.append("| dataset | evidence | overlap_red_vs_base_pct | overlap_red_vs_naive_pct | sparse_red_vs_base_pct | suff_change_vs_base_abs |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {dataset} | {evidence} | {ovb} | {ovn} | {spb} | {suff:+.6f} |".format(
                dataset=r["dataset"],
                evidence=r["evidence"],
                ovb=("NA" if r["overlap_red_vs_base_pct"] is None else f"{r['overlap_red_vs_base_pct']:.2f}%"),
                ovn=("NA" if r["overlap_red_vs_naive_pct"] is None else f"{r['overlap_red_vs_naive_pct']:.2f}%"),
                spb=("NA" if r["sparse_red_vs_base_pct"] is None else f"{r['sparse_red_vs_base_pct']:.2f}%"),
                suff=r["suff_change_vs_base_abs"],
            )
        )

    overlap_vals = [float(r["overlap_red_vs_base_pct"]) for r in rows if r["overlap_red_vs_base_pct"] is not None]
    sparse_vals = [float(r["sparse_red_vs_base_pct"]) for r in rows if r["sparse_red_vs_base_pct"] is not None]
    mean_overlap_red = sum(overlap_vals) / max(len(overlap_vals), 1)
    mean_sparse_red = sum(sparse_vals) / max(len(sparse_vals), 1)
    mean_suff_change = sum(abs(r["suff_change_vs_base_abs"]) for r in rows) / max(len(rows), 1)

    lines.append("")
    lines.append("## Aggregate Takeaway")
    lines.append("")
    lines.append(
        "- Mean overlap reduction vs base: **{:.2f}%**".format(mean_overlap_red)
    )
    lines.append(
        "- Mean sparsity reduction vs base: **{:.2f}%**".format(mean_sparse_red)
    )
    lines.append(
        "- Mean absolute sufficiency change vs base: **{:.6f}**".format(mean_suff_change)
    )
    lines.append("")
    overlap_trend = "reduces" if mean_overlap_red >= 0 else "increases"
    sparse_trend = "reduces" if mean_sparse_red >= 0 else "increases"
    lines.append(
        "Interpretation: with this run setup, optimized CDEA {} overlap and {} mask mass, while sufficiency stays close to baseline.".format(
            overlap_trend, sparse_trend
        )
    )
    lines.append("")
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("- `suff`: Mean target-class logit under keep-mask intervention.")
    lines.append("- `margin`: Mean contrastive margin (`z_k - max(z_foil)`) under keep-mask intervention.")
    lines.append("- `overlap`: Pairwise overlap among unique masks (lower is better).")
    lines.append("- `sparse`: Mean L1 mask mass (lower is better).")
    lines.append("- `overlap_red_vs_base_pct`: Percent overlap reduction of optimized vs base evidence.")
    lines.append("- `overlap_red_vs_naive_pct`: Percent overlap reduction of optimized vs naive contrastive.")
    lines.append("- `sparse_red_vs_base_pct`: Percent sparsity reduction of optimized vs base evidence.")
    lines.append("- `suff_change_vs_base_abs`: Absolute sufficiency delta (`optimized - base`).")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Consolidated matrix: `/Users/ssuresh/gambit/scripts/out/poc_all_datasets_matrix.csv`")
    lines.append("- Consolidated JSON: `/Users/ssuresh/gambit/scripts/out/poc_all_datasets_matrix.json`")
    lines.append("- Per-run ablations: `/Users/ssuresh/gambit/scripts/out/ablation_contrastive_<dataset>_<evidence>_metrics.csv`")

    report.write_text("\n".join(lines), encoding="utf-8")
    print("Saved report to", report)
    return report


if __name__ == "__main__":
    matrix_rows = run_matrix(
        num_images=64,
        batch_size=16,
        model_name="resnet18",
        pretrained=False,
        ig_steps=8,
        lambda_disjoint=1.0,
        lambda_mass=2.0,
    )
    write_report(matrix_rows)
