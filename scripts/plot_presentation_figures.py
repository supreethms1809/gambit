"""
scripts/plot_presentation_figures.py

Slide-ready figures for the medical-results talk. Each figure reads result JSON
and renders nothing it cannot source from a file, so every number on a slide
traces back to a run.

Figures:
  F2  separation      base -> naive -> optimized overlap, per configuration
  F3  fixed budget    sufficiency alongside the mask budget that bought it
  F4  decomposition   probability spread + the K x K deletion matrix
  F5  center prior    why localization is the wrong yardstick, both datasets
  F6  foil ranks      localization by hypothesis rank, base vs CDEA

Run from repo root::

    PYTHONPATH=. python scripts/plot_presentation_figures.py \\
        --results_dir results/medical_presentation \\
        --out_dir results/medical_presentation/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Validated categorical slots (see the dataviz palette reference). Slots are
# assigned in fixed order and never cycled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Ordinal blue ramp for the three ablation methods: worse -> better reads as
# lighter -> darker, so the ordering is legible without relying on hue.
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"

DATASET_LABEL = {"ham10000": "HAM10000\n(skin lesions)", "brain_tumor": "Brain tumor MRI"}
EVIDENCE_LABEL = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"  missing (skipped): {path}")
        return None
    with path.open() as fh:
        return json.load(fh)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / (name + '.png')}")


def seed_aggregate(res: Path, ds: str, ev: str) -> Optional[dict]:
    """Mean and std across seeds for one (dataset, evidence) cell.

    Returns {method: {metric: {"mean":…, "std":…}}} so a figure can show error bars
    rather than a single run's point estimate.
    """
    import statistics as st
    per_seed: Dict[str, List[dict]] = {}
    for p in sorted((res / "seeds").glob(f"ablation_{ds}_{ev}_seed*_metrics.json")):
        d = load(p)
        if not d or "aggregates" not in d:
            continue
        for method, agg in d["aggregates"].items():
            per_seed.setdefault(method, []).append(agg)
    if not per_seed:
        return None
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, runs in per_seed.items():
        out[method] = {}
        for metric in ("overlap", "suff", "sparse", "margin"):
            vals = [r[metric] for r in runs if metric in r]
            if vals:
                out[method][metric] = {
                    "mean": st.mean(vals),
                    "std": st.stdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
    return out


def _val(cell: dict, method: str, metric: str) -> float:
    """Read a metric that may be a bare float or a {mean,std} dict."""
    v = (cell or {}).get(method, {}).get(metric)
    if isinstance(v, dict):
        return v.get("mean", np.nan)
    return v if v is not None else np.nan


def _err(cell: dict, method: str, metric: str) -> float:
    v = (cell or {}).get(method, {}).get(metric)
    return v.get("std", 0.0) if isinstance(v, dict) else 0.0


def _bar_labels(ax, bars, fmt="{:.3f}", dy=0.01, size=12) -> None:
    """Direct value labels — required relief for sub-3:1 fills, and readable when projected."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=size, color=INK2)


# --------------------------------------------------------------------------- F2
def fig_separation(configs: List[dict], out_dir: Path) -> None:
    """Overlap: base evidence -> naive subtraction -> optimized allocation."""
    configs = [c for c in configs if c.get("data")]
    if not configs:
        print("F2: no ablation results found")
        return
    methods = ["base_evidence", "naive_contrastive", "optimized"]
    labels = ["raw evidence", "naive subtraction", "CDEA allocation"]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    width = 0.26
    xs = np.arange(len(configs))
    for i, (m, lab) in enumerate(zip(methods, labels)):
        vals = [_val(c["data"], m, "overlap") for c in configs]
        errs = [_err(c["data"], m, "overlap") for c in configs]
        bars = ax.bar(xs + (i - 1) * width, vals, width * 0.92,
                      color=RAMP[i], label=lab, zorder=3,
                      yerr=errs if any(errs) else None, capsize=4,
                      error_kw={"ecolor": INK2, "lw": 1.2})
        _bar_labels(ax, bars, dy=0.006, size=11)

    ax.set_xticks(xs)
    ax.set_xticklabels([c["label"] for c in configs])
    ax.set_ylabel("mask overlap between classes")
    ax.set_title("Competing classes stop sharing evidence", pad=34, color=INK, loc="left")
    ax.text(0, 1.02, "lower is better", transform=ax.transAxes,
            fontsize=13, color=MUTED)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    save(fig, out_dir, "F2_separation")


# --------------------------------------------------------------------------- F3
def fig_budget(configs: List[dict], out_dir: Path) -> None:
    """Sufficiency rose — and the mask budget that paid for it stayed flat."""
    configs = [c for c in configs if c.get("data")]
    if not configs:
        print("F3: no ablation results found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    xs = np.arange(len(configs))
    width = 0.34

    for ax, metric, title, sub in [
        (axes[0], "suff", "The kept evidence still identifies the class", "higher is better"),
        (axes[1], "sparse", "…and it did not spend more highlight", "should stay near 1.0"),
    ]:
        base = [_val(c["data"], "base_evidence", metric) for c in configs]
        opt = [_val(c["data"], "optimized", metric) for c in configs]
        be = [_err(c["data"], "base_evidence", metric) for c in configs]
        oe = [_err(c["data"], "optimized", metric) for c in configs]
        b1 = ax.bar(xs - width / 2, base, width * 0.92, color=RAMP[0],
                    label="raw evidence", zorder=3,
                    yerr=be if any(be) else None, capsize=4,
                    error_kw={"ecolor": INK2, "lw": 1.2})
        b2 = ax.bar(xs + width / 2, opt, width * 0.92, color=RAMP[2],
                    label="CDEA allocation", zorder=3,
                    yerr=oe if any(oe) else None, capsize=4,
                    error_kw={"ecolor": INK2, "lw": 1.2})
        _bar_labels(ax, b1, dy=0.02, size=11)
        _bar_labels(ax, b2, dy=0.02, size=11)
        ax.set_xticks(xs)
        ax.set_xticklabels([c["label"] for c in configs], fontsize=11)
        ax.set_title(title, pad=32, color=INK, loc="left", fontsize=15)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=12, color=MUTED)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0, color=BASELINE, lw=1)
        if metric == "sparse":
            ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=2)
        # Headroom so the value labels and the legend never sit on a bar.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.18)
    axes[0].set_ylabel("sufficiency (logit gain)")
    axes[1].set_ylabel("total mask mass")
    # Tall bars are on the left in both panels, so the legend goes right.
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "F3_budget")


# --------------------------------------------------------------------------- F4
def fig_decomposition(runs: List[dict], out_dir: Path) -> None:
    """The headline validation: spread collapse/recovery + the deletion matrix."""
    runs = [r for r in runs if r.get("data")]
    if not runs:
        print("F4: no decomposition results found")
        return

    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

    # --- left: probability spread under the three conditions
    ax = fig.add_subplot(gs[0, 0])
    conds = ["full", "shared_only", "shared_plus_unique"]
    cond_lab = ["full image", "shared only", "shared + unique"]
    xs = np.arange(len(runs))
    width = 0.26
    for i, (c, lab) in enumerate(zip(conds, cond_lab)):
        vals = [r["data"]["spread"][c]["mean"] for r in runs]
        bars = ax.bar(xs + (i - 1) * width, vals, width * 0.92,
                      color=[BLUE, ORANGE, AQUA][i], label=lab, zorder=3)
        _bar_labels(ax, bars, dy=0.012, size=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in runs], fontsize=10)
    ax.set_ylabel("top-1 minus top-K probability")
    ax.set_title("Shared evidence alone cannot pick a diagnosis",
                 pad=34, color=INK, loc="left", fontsize=15)
    ax.text(0, 1.02, "the gap collapses without unique evidence, and returns with it",
            transform=ax.transAxes, fontsize=12, color=MUTED)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=3, fontsize=11)
    ax.set_ylim(0, 1.18)

    # --- right: deletion matrix for the headline run
    head = runs[0]
    mat = np.array(head["data"]["deletion_matrix"])
    K = mat.shape[0]
    lim = float(np.abs(mat).max())
    # Diverging blue<->red with a neutral gray midpoint at zero.
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", CRITICAL])
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(K):
        for j in range(K):
            ax2.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                     fontsize=11,
                     color=INK if abs(mat[i, j]) < lim * 0.55 else "white")
    ax2.set_xticks(range(K))
    ax2.set_yticks(range(K))
    ax2.set_xticklabels([f"remove\nunique {j}" for j in range(K)], fontsize=10)
    ax2.set_yticklabels([f"class {i}" for i in range(K)], fontsize=11)
    ax2.set_title("Removing a class's evidence hurts that class",
                  pad=34, color=INK, loc="left", fontsize=15)
    ax2.text(0, 1.02, f"change in logit · {head['label']}",
             transform=ax2.transAxes, fontsize=12, color=MUTED)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cb.set_label("Δ logit", fontsize=12, color=INK2)
    cb.outline.set_visible(False)
    save(fig, out_dir, "F4_decomposition")


# --------------------------------------------------------------------------- F5
def fig_resolution(res_runs: Dict[str, Dict[int, dict]], out_dir: Path) -> None:
    """The grid was the bottleneck — and the right grid depends on target size.

    Same model, same evidence, same budget, same images: only the grid changes. The
    centre-rectangle comparison flips sign on brain tumour and never does on
    HAM10000, which is a fact about each dataset's acquisition geometry rather than
    about the method.
    """
    if not any(res_runs.values()):
        print("F5: no resolution-sweep results found")
        return
    grids = [7, 14, 28]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    panels = [("brain_tumor", "Brain tumour: the grid was the bottleneck", axes[0]),
              ("ham10000", "HAM10000: no grid rescues the metric", axes[1])]

    for ds, title, ax in panels:
        runs = res_runs.get(ds, {})
        if not runs:
            ax.set_visible(False)
            continue
        gs = [g for g in grids if g in runs]
        rows = {g: {r["method"]: r["mean"] for r in runs[g]["rows"]} for g in gs}
        series = [
            ("cdea_unique", "CDEA unique", BLUE, "-", "o"),
            ("center_cell", "fixed centre cell (no model)", CRITICAL, "-", "s"),
            ("cdea_unique_translated", "same mask, position scrambled", AQUA, "--", "^"),
        ]
        for key, lab, col, ls, mk in series:
            ax.plot(gs, [rows[g][key] for g in gs], ls, marker=mk, color=col,
                    lw=2.6, ms=9, label=lab, zorder=4)
        ax.axhline(rows[gs[0]]["uniform"], color=MUTED, lw=1.6, ls=":",
                   zorder=2, label="chance (flat mask)")

        # Mark where CDEA overtakes the degenerate baseline, if it does.
        crossed = [g for g in gs if rows[g]["cdea_unique"] > rows[g]["center_cell"]]
        if crossed:
            g0 = crossed[0]
            ax.annotate("CDEA overtakes\nthe rectangle",
                        xy=(g0, rows[g0]["cdea_unique"]),
                        xytext=(g0 * 1.05, rows[g0]["cdea_unique"] * 0.55),
                        fontsize=12, color=BLUE,
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))

        ax.set_xscale("log", base=2)
        ax.set_xticks(gs)
        ax.set_xticklabels([f"{g}×{g}" for g in gs])
        ax.set_xlabel("grid resolution")
        ax.set_title(title, pad=34, color=INK, loc="left", fontsize=15)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        # Name the winning grid and how many cells the target spans there — that
        # ratio, not the raw resolution, is what predicts the optimum.
        area = rows[gs[0]]["uniform"]
        best_g = max(gs, key=lambda g: rows[g]["cdea_unique"])
        cells = area / (1.0 / (best_g * best_g))
        ax.text(0, 1.02,
                f"target is {area * 100:.1f}% of frame · best at {best_g}×{best_g}, "
                f"where it spans {cells:.0f} cells",
                transform=ax.transAxes, fontsize=12, color=MUTED)
    axes[0].set_ylabel("share of mask mass on the pathology")
    axes[0].legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, "F5_resolution")


def fig_center_prior(cp: dict, loc_runs: List[dict], out_dir: Path) -> None:
    """Why lesion overlap cannot validate the method."""
    if not cp:
        print("F5: no center-prior results found")
        return
    detail = cp["detail"]
    grids = [7, 14, 28, 56]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=False)
    for ax, ds, title in [
        (axes[0], "ham10000", "HAM10000: the centre prior swamps the metric"),
        (axes[1], "brain_tumor", "Brain tumour: the centre prior is weak"),
    ]:
        d = detail[ds]
        c1 = [d["by_grid"][str(g)]["center_1cell"] for g in grids]
        c3 = [d["by_grid"][str(g)]["center_3x3"] for g in grids]
        ax.plot(grids, c1, "-o", color=CRITICAL, lw=2.5, ms=9,
                label="fixed centre cell (no model)", zorder=4)
        ax.plot(grids, c3, "-s", color=ORANGE, lw=2.5, ms=8,
                label="fixed centre 3×3 (no model)", zorder=4)
        ax.axhline(d["chance_area_fraction"], color=MUTED, lw=1.6, ls=":",
                   zorder=2, label="chance (flat mask)")

        best, trans = None, None
        for r in loc_runs:
            if r.get("data") and r.get("dataset") == ds:
                rows = {row["method"]: row["mean"] for row in r["data"]["rows"]}
                v = rows.get("cdea_unique")
                if v is not None and (best is None or v > best):
                    best, trans = v, rows.get("cdea_unique_translated")
        if best is not None:
            ax.axhline(best, color=BLUE, lw=2.5, zorder=3, label="CDEA measured")
            ax.text(grids[-1], best, f"  {best:.3f}", va="center", ha="left",
                    fontsize=12, color=BLUE)
        # The honest null: the same mask, same budget, same shape, random position.
        # It is what separates "this mask localizes" from "this mask is small and central".
        if trans is not None:
            ax.axhline(trans, color=AQUA, lw=2.5, ls="--", zorder=3,
                       label="same mask, position scrambled")
            ax.text(grids[-1], trans, f"  {trans:.3f}", va="center", ha="left",
                    fontsize=12, color=AQUA)

        ax.set_xscale("log", base=2)
        ax.set_xticks(grids)
        ax.set_xticklabels([f"{g}×{g}" for g in grids])
        ax.set_xlabel("grid resolution")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, pad=34, color=INK, loc="left", fontsize=15)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        sd = d["centroid_std_yx"]
        ax.text(0, 1.02,
                f"target centroid spread: ±{sd[0]:.02f}, ±{sd[1]:.02f} of frame",
                transform=ax.transAxes, fontsize=12, color=MUTED)
    axes[0].set_ylabel("share of mask mass on the pathology")
    axes[0].legend(loc="center right", fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, "F5_center_prior")


# --------------------------------------------------------------------------- F6
def fig_foil_ranks(runs: List[dict], out_dir: Path) -> None:
    """Localization by hypothesis rank — does the split track the model's ranking?"""
    runs = [r for r in runs if r.get("data")]
    if not runs:
        print("F6: no foil results found")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(6.6 * len(runs), 5.0), squeeze=False)
    for ax, r in zip(axes[0], runs):
        rows = {row["method"]: row for row in r["data"]["rows"]}
        ranks = sorted(int(k.split("rank")[1]) for k in rows if k.startswith("cdea_unique_rank"))
        base = [rows[f"base_evidence_rank{k}"]["mean"] for k in ranks]
        cdea = [rows[f"cdea_unique_rank{k}"]["mean"] for k in ranks]
        chance = r["data"].get("mean_target_area_fraction",
                               r["data"].get("mean_lesion_area_fraction"))
        ax.plot(ranks, base, "-o", color=ORANGE, lw=2.5, ms=9, label="raw evidence")
        ax.plot(ranks, cdea, "-o", color=BLUE, lw=2.5, ms=9, label="CDEA unique")
        if chance:
            ax.axhline(chance, color=MUTED, lw=1.6, ls=":", label="chance")
        ax.set_xticks(ranks)
        ax.set_xticklabels(["predicted"] + [f"foil {k}" for k in ranks[1:]], fontsize=12)
        ax.set_title(r["label"], pad=20, color=INK, loc="left", fontsize=15)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 0.8)
    axes[0][0].set_ylabel("share of mask mass on the lesion")
    axes[0][0].legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    save(fig, out_dir, "F6_foil_ranks")


def main() -> None:
    p = argparse.ArgumentParser(description="Render presentation figures from result JSON")
    p.add_argument("--results_dir", type=str, default=str(REPO / "results" / "medical_presentation"))
    p.add_argument("--legacy_dir", type=str, default=str(REPO / "scripts" / "out"),
                   help="Fallback for results produced before the new results tree")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    style()
    res = Path(args.results_dir)
    legacy = Path(args.legacy_dir)
    out_dir = Path(args.out_dir) if args.out_dir else res / "figures"

    # --- ablations (F2, F3)
    abl_specs = [
        ("ham10000", "gradcam"), ("ham10000", "ig"),
        ("brain_tumor", "gradcam"), ("brain_tumor", "ig"),
    ]
    # All four cells must share a backbone or they are not comparable — which is the
    # very defect the unified-config runs exist to remove. Brain tumor only has a
    # ResNet-18 checkpoint, so ResNet-18 is the common denominator; the "_resnet"
    # variant is preferred for HAM10000 wherever an EfficientNetV2-S run also exists.
    ablations = []
    for ds, ev in abl_specs:
        seeded = seed_aggregate(res, ds, ev)          # 3 seeds -> error bars
        d = (load(res / "ablation" / f"ablation_unified_{ds}_{ev}_resnet_metrics.json")
             or load(res / "ablation" / f"ablation_unified_{ds}_{ev}_metrics.json")
             or load(legacy / f"ablation_contrastive_{ds}_{ev}_metrics.json"))
        ablations.append({
            "label": f"{DATASET_LABEL[ds]}\n{EVIDENCE_LABEL[ev]}",
            "data": seeded or (d or {}).get("aggregates"),
            "seeded": bool(seeded),
            "model": (d or {}).get("model"),
        })

    # --- decomposition (F4)
    decomp_specs = [
        ("decomp_ham10000_effnet_gradcam", "HAM10000 · EffNetV2-S · Grad-CAM"),
        ("decomp_ham10000_resnet_gradcam", "HAM10000 · ResNet-18 · Grad-CAM"),
        ("decomp_ham10000_resnet_ig", "HAM10000 · ResNet-18 · IG"),
        ("decomp_brain_resnet_gradcam", "Brain tumor · ResNet-18 · Grad-CAM"),
        ("decomp_brain_resnet_ig", "Brain tumor · ResNet-18 · IG"),
    ]
    # Prefer the lambda_shared_sparse=0.25 re-run: at 0.0 the shared mask is unpenalized
    # and blankets ~46% of the frame at 0.99x chance on base-evidence capture, so keeping
    # "only shared" degrades the image globally and inflates the collapse this figure
    # plots. See docs/MEDICAL_RESULTS.md section 9a.
    def _decomp(name: str):
        fixed = res / "decomposition_sharedfix" / f"{name}.json"
        return load(fixed if fixed.exists() else res / "decomposition" / f"{name}.json")

    decomps = [{"label": lab, "data": _decomp(name)} for name, lab in decomp_specs]

    # --- localization (F5, F6)
    loc_specs = [
        (res / "localization" / "localization_null_effnet_gradcam.json", "ham10000"),
        (legacy / "localization_foil_efficientnet_v2_s.json", "ham10000"),
        (legacy / "localization_ham10000_gradcam.json", "ham10000"),
        (res / "fine_grid" / "localization_brain_vit_gradcam.json", "brain_tumor"),
    ]
    loc_runs = [{"data": load(p_), "dataset": ds} for p_, ds in loc_specs]

    foil_specs = [
        (legacy / "localization_foil_resnet18.json", "HAM10000 · ResNet-18"),
        (legacy / "localization_foil_efficientnet_v2_s.json", "HAM10000 · EfficientNetV2-S"),
    ]
    foils = [{"label": lab, "data": load(p_)} for p_, lab in foil_specs]

    cp = load(out_dir / "center_prior.json") or load(res / "figures" / "center_prior.json")

    res_runs: Dict[str, Dict[int, dict]] = {"brain_tumor": {}, "ham10000": {}}
    for ds, tag in [("brain_tumor", "brain"), ("ham10000", "ham")]:
        for g in (7, 14, 28):
            d = load(res / "resolution" / f"res_loc_{tag}_g{g}.json")
            if d:
                res_runs[ds][g] = d

    print("\nRendering figures:")
    fig_separation(ablations, out_dir)
    fig_budget(ablations, out_dir)
    fig_decomposition(decomps, out_dir)
    fig_resolution(res_runs, out_dir)
    fig_center_prior(cp, loc_runs, out_dir)
    fig_foil_ranks(foils, out_dir)


if __name__ == "__main__":
    main()
