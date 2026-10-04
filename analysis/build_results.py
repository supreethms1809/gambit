"""Build RESULTS.md and LaTeX tables from run records.

Every number in the output is formatted from ``family_summary`` or from the
mean of the records. The paper path stays unwritten while
``runs_complete`` is false. This module does not read the test split and
does not launch a grid.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.stats import family_summary
from scripts.completion_check import assert_run_record, runs_complete

FAMILY_C = (
    ("margin", "CDEA versus margin attribution"),
    ("extremal", "CDEA versus contrastive Extremal Perturbations"),
    ("cve", "CDEA versus CVE"),
)
PRIMARY_METHODS = ("cdea", "margin", "extremal", "cve")


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _primary(records: Sequence[Mapping]) -> list[Mapping]:
    return [record for record in records if "ablation" not in record]


def _validate(records: Sequence[Mapping], *, area: float, tolerance: float, n: int) -> None:
    if not records:
        raise ValueError("results need at least one record")
    for record in records:
        value = float(record["value"])
        if value != value or value in (float("inf"), float("-inf")):
            raise AssertionError("metric is not finite")
        assert_run_record(record, area=area, tolerance=tolerance, n=n)


def dataset_means(records: Sequence[Mapping], *, model: str, game: str) -> dict[str, dict[str, float]]:
    """Mean metric per dataset and method. Seeds are averaged first."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in _primary(records):
        if record.get("model") != model or record.get("game") != game:
            continue
        buckets[(str(record["dataset"]), str(record["method"]))].append(float(record["value"]))
    means: dict[str, dict[str, float]] = defaultdict(dict)
    for (dataset, method), values in buckets.items():
        means[dataset][method] = sum(values) / len(values)
    return dict(means)


def _differences(means: Mapping[str, Mapping[str, float]], method: str, baseline: str) -> list[float]:
    diffs = []
    for dataset in sorted(means):
        row = means[dataset]
        if method not in row or baseline not in row:
            raise ValueError(f"{dataset} is missing {method} or {baseline}")
        diffs.append(row[method] - row[baseline])
    return diffs


def _family_c(means: Mapping[str, Mapping[str, float]], *, seed: int, n_boot: int) -> list[dict]:
    comparisons = {
        label: _differences(means, "cdea", method)
        for method, label in FAMILY_C
    }
    return family_summary(comparisons, seed=seed, n_boot=n_boot)


def _leave_one_out(means: Mapping[str, Mapping[str, float]], *, seed: int, n_boot: int) -> list[dict]:
    rows = []
    datasets = sorted(means)
    for dropped in datasets:
        kept = {name: row for name, row in means.items() if name != dropped}
        summary = _family_c(kept, seed=seed, n_boot=n_boot)
        rows.append({"dropped": dropped, "summary": summary})
    return rows


def _method_table(means: Mapping[str, Mapping[str, float]]) -> list[tuple[str, str, float]]:
    rows = []
    for dataset in sorted(means):
        for method in PRIMARY_METHODS:
            if method not in means[dataset]:
                raise ValueError(f"{dataset} is missing {method}")
            rows.append((dataset, method, means[dataset][method]))
    return rows


def _family_markdown(title: str, rows: Sequence[Mapping]) -> str:
    lines = [
        f"## {title}",
        "",
        "| Comparison | Wins | Mean | CI low | CI high | p | p Holm |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {wins:.0f} | {mean} | {low} | {high} | {p} | {holm} |".format(
                name=row["name"],
                wins=float(row["wins"]),
                mean=fmt(float(row["mean"])),
                low=fmt(float(row["ci_low"])),
                high=fmt(float(row["ci_high"])),
                p=fmt(float(row["p"])),
                holm=fmt(float(row["p_holm"])),
            )
        )
    return "\n".join(lines)


def _family_latex(title: str, rows: Sequence[Mapping]) -> str:
    body = [
        f"% {title}",
        "\\begin{tabular}{lrrrrrr}",
        "Comparison & Wins & Mean & CI low & CI high & p & p Holm \\\\",
    ]
    for row in rows:
        body.append(
            "{name} & {wins:.0f} & {mean} & {low} & {high} & {p} & {holm} \\\\".format(
                name=_latex(str(row["name"])),
                wins=float(row["wins"]),
                mean=fmt(float(row["mean"])),
                low=fmt(float(row["ci_low"])),
                high=fmt(float(row["ci_high"])),
                p=fmt(float(row["p"])),
                holm=fmt(float(row["p_holm"])),
            )
        )
    body.append("\\end{tabular}")
    return "\n".join(body)


def render_results(
    records: Sequence[Mapping],
    *,
    area: float,
    tolerance: float,
    n: int,
    model: str = "resnet50",
    selected_shift_baseline: str | None = None,
    seed: int = 0,
    n_boot: int = 1000,
) -> tuple[str, str, dict]:
    """Return markdown, LaTeX, and the family rows. Records are checked first."""
    _validate(records, area=area, tolerance=tolerance, n=n)
    means = dataset_means(records, model=model, game="contrastive")
    if not means:
        raise ValueError("contrastive records are missing")
    family = _family_c(means, seed=seed, n_boot=n_boot)
    left_out = _leave_one_out(means, seed=seed, n_boot=n_boot)
    per_dataset = _method_table(means)

    lines = [
        "# Results",
        "",
        "Generated by `analysis.build_results`. Each number is formatted from the records.",
        "",
        "## Main table",
        "",
        "| Dataset | Method | Mean |",
        "|---|---|---|",
    ]
    for dataset, method, value in per_dataset:
        lines.append(f"| {dataset} | {method} | {fmt(value)} |")
    lines.extend(["", "## Per-dataset table", "", "| Dataset | Method | Mean |", "|---|---|---|"])
    for dataset, method, value in per_dataset:
        lines.append(f"| {dataset} | {method} | {fmt(value)} |")
    lines.extend(["", _family_markdown("Family C", family), ""])

    if selected_shift_baseline is None:
        lines.extend(["## Family S", "", "Family S has no selected baseline.", ""])
    else:
        shift_means = dataset_means(records, model=model, game="shift")
        label = f"CDEA-shift versus {selected_shift_baseline}"
        shift_rows = family_summary(
            {label: _differences(shift_means, "cdea_shift", selected_shift_baseline)},
            seed=seed,
            n_boot=n_boot,
        )
        lines.extend([_family_markdown("Family S", shift_rows), ""])

    lines.extend(["## Robustness", "", "### Leave one dataset out", ""])
    for item in left_out:
        first = item["summary"][0]
        lines.append(
            f"- drop {item['dropped']}: {first['name']} mean {fmt(float(first['mean']))}"
        )
    datasets = set(means)
    lines.extend(["", "### Dev datasets", ""])
    if {"cifar10", "ham10000"} <= datasets:
        kept = {name: row for name, row in means.items() if name not in {"cifar10", "ham10000"}}
        if not kept:
            lines.append("Dropping the dev datasets leaves no dataset.")
        else:
            dropped = _family_c(kept, seed=seed, n_boot=n_boot)
            lines.append(
                f"After dropping the dev datasets, {dropped[0]['name']} mean {fmt(float(dropped[0]['mean']))}."
            )
    else:
        lines.append("The dev datasets are not both in this input.")
    lines.extend(["", "### Second backbone", ""])
    if any(record.get("model") == "vit_b_16" and "ablation" not in record for record in records):
        vit = dataset_means(records, model="vit_b_16", game="contrastive")
        vit_family = _family_c(vit, seed=seed, n_boot=n_boot)
        lines.append(f"{vit_family[0]['name']} mean {fmt(float(vit_family[0]['mean']))}.")
    else:
        lines.append("The second backbone was not measured.")
    lines.extend(["", "### Blur operator", ""])
    if any(record.get("operator") == "blur" for record in records):
        lines.append("Blur records are present.")
    else:
        lines.append("The blur operator was not measured.")
    lines.append("")
    budgets = {float(record["budget"]) for record in records if "budget" in record}
    lines.extend(["", "### Budgets", ""])
    if 0.025 in budgets and 0.1 in budgets:
        lines.append("The 2.5% and 10% budgets are present in the records.")
    else:
        lines.append("The 2.5% and 10% budgets were not measured.")
    lines.extend(["", "### Repeat variation", ""])
    if any("repeat" in record for record in records):
        lines.append("Repeat records are present.")
    else:
        lines.append("The repeat run was not measured.")

    lines.extend(["", "## Ablations", ""])
    ablations = [record for record in records if "ablation" in record]
    if not ablations:
        lines.append("Ablation records are not in the input.")
    else:
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in ablations:
            grouped[str(record["ablation"])].append(float(record["value"]))
        lines.extend(["| Ablation | Mean |", "|---|---|"])
        for name in sorted(grouped):
            values = grouped[name]
            lines.append(f"| {name} | {fmt(sum(values) / len(values))} |")

    lines.extend(["", "## Cost", ""])
    costs = [float(record["forward_passes"]) for record in records if "forward_passes" in record]
    if not costs:
        lines.append("Cost records are not in the input.")
    else:
        lines.append(f"Mean forward passes {fmt(sum(costs) / len(costs))}.")

    lines.extend(["", "## Model table", ""])
    accuracies = [
        (str(record["dataset"]), str(record["model"]), int(record["seed"]), float(record["accuracy"]))
        for record in records
        if "accuracy" in record
    ]
    if not accuracies:
        lines.append("The model table is not in the input.")
    else:
        lines.extend(["| Dataset | Model | Seed | Accuracy |", "|---|---|---|---|"])
        for dataset, model_name, seed_id, accuracy in accuracies:
            lines.append(f"| {dataset} | {model_name} | {seed_id} | {fmt(accuracy)} |")
    lines.append("")

    latex_parts = [_family_latex("Family C", family), ""]
    markdown = "\n".join(lines)
    return markdown, "\n".join(latex_parts), {"family_c": family, "leave_one_out": left_out}


def write_results(records: Sequence[Mapping], out_dir: Path, **kwargs) -> dict:
    """Write ``RESULTS.md`` and ``tables.tex`` under ``out_dir`` after the checks pass."""
    markdown, latex, summary = render_results(records, **kwargs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")
    (out_dir / "tables.tex").write_text(latex, encoding="utf-8")
    return summary


def main() -> None:
    """Do not write the paper results while the grids are unfinished."""
    report = runs_complete()
    if not report["complete"]:
        raise SystemExit(
            "refusing to write RESULTS.md: final runs are not complete; "
            "the tag final-runs-v1 was not created"
        )
    raise SystemExit("refusing to write RESULTS.md: no run directory was passed")


if __name__ == "__main__":
    main()
