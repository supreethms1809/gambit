"""
scripts/plot_evidence_comparison.py

Plot Grad-CAM vs Integrated Gradients as base evidence for the contrastive game.

Reads the JSON/CSV that ``ablation_contrastive.py`` and ``eval_localization.py``
already write, so it never recomputes anything — run those first, then this.

Panels:
  A  Lesion localization (HAM10000): share of mask mass inside the expert lesion
     segmentation, per evidence provider. The uniform bar is the chance level (the
     lesion's own area fraction); anything at that height is not localizing.
  B  Mask overlap, base evidence vs after allocation, per dataset and provider.
     Lower is better — it is the quantity the contrastive objective is minimizing.
  C  Sufficiency over the same runs. Allocation should not buy lower overlap by
     giving up sufficiency, so this panel is the control.

Run from repo root::

    PYTHONPATH=. python scripts/plot_evidence_comparison.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "scripts" / "out"

EVIDENCE = ["gradcam", "ig"]
EVIDENCE_LABEL = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients"}
EVIDENCE_COLOR = {"gradcam": "#4C72B0", "ig": "#DD8452"}
DATASETS = ["ham10000", "brain_tumor"]
DATASET_LABEL = {"ham10000": "HAM10000\n(7 skin lesion classes)",
                 "brain_tumor": "Brain tumor MRI\n(3 classes)"}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"  missing (skipped): {path.name}")
        return None
    with path.open() as fh:
        return json.load(fh)


def _ablation(dataset: str, evidence: str) -> Optional[Dict[str, Dict[str, float]]]:
    d = _load_json(OUT / f"ablation_contrastive_{dataset}_{evidence}_metrics.json")
    if d is None:
        return None
    # ablation_contrastive.py writes {"aggregates": {method: {metric: value}}}
    agg = d.get("aggregates")
    if isinstance(agg, dict):
        return agg
    rows = d.get("rows") or []
    return {r["method"]: r for r in rows if "method" in r}


def _localization(evidence: str, prefix: str) -> Optional[Dict[str, dict]]:
    # Runs may carry a size suffix (e.g. localization_ham10000_ig200.json), so glob
    # rather than requiring an exact name.
    matches = sorted(OUT.glob(f"{prefix}_{evidence}*.json"))
    if not matches:
        print(f"  missing (skipped): {prefix}_{evidence}*.json")
        return None
    if len(matches) > 1:
        print(f"  note: {len(matches)} matches for {evidence}, using {matches[0].name}")
    d = _load_json(matches[0])
    return None if d is None else {r["method"]: r for r in d["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Grad-CAM vs IG base-evidence comparison")
    parser.add_argument("--loc_prefix", type=str, default="localization_ham10000",
                        help="Prefix of the paired localization runs, without the _<evidence> suffix")
    parser.add_argument("--out", type=str, default=str(OUT / "evidence_comparison.png"))
    parser.add_argument("--in_dir", type=str, default=None,
                        help="Directory to read result JSON from (default: scripts/out)")
    args = parser.parse_args()

    if args.in_dir:
        global OUT
        OUT = Path(args.in_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    print("Loading results from", OUT)
    loc = {e: _localization(e, args.loc_prefix) for e in EVIDENCE}
    abl = {(d, e): _ablation(d, e) for d in DATASETS for e in EVIDENCE}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # ---- Panel A: localization -------------------------------------------------
    ax = axes[0]
    methods = ["base_evidence", "cdea_shared", "cdea_unique"]
    method_label = {"base_evidence": "base\nevidence", "cdea_shared": "CDEA\nshared",
                    "cdea_unique": "CDEA\nunique"}
    xs = np.arange(len(methods))
    width = 0.36
    chance = None
    for i, e in enumerate(EVIDENCE):
        if loc[e] is None:
            continue
        means = [loc[e][m]["mean"] if m in loc[e] else np.nan for m in methods]
        errs = [loc[e][m]["std"] / max(loc[e][m]["n"], 1) ** 0.5 if m in loc[e] else np.nan
                for m in methods]
        ax.bar(xs + (i - 0.5) * width, means, width, yerr=errs, capsize=3,
               label=EVIDENCE_LABEL[e], color=EVIDENCE_COLOR[e])
        if chance is None and "uniform" in loc[e]:
            chance = loc[e]["uniform"]["mean"]
    if chance is not None:
        ax.axhline(chance, ls="--", c="0.35", lw=1.2)
        ax.text(len(methods) - 0.5, chance, f"  chance ({chance:.2f})",
                va="bottom", ha="right", fontsize=9, color="0.35")
    ax.set_xticks(xs)
    ax.set_xticklabels([method_label[m] for m in methods])
    ax.set_ylabel("share of mask mass inside lesion")
    ns = {e: loc[e]["cdea_unique"]["n"] for e in EVIDENCE
          if loc[e] and "cdea_unique" in loc[e]}
    n_txt = (f"n={next(iter(ns.values()))}" if len(set(ns.values())) == 1
             else ", ".join(f"{EVIDENCE_LABEL[e]} n={n}" for e, n in ns.items()))
    ax.set_title(f"A. Lesion localization (HAM10000, {n_txt})\nhigher is better", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- Panels B/C: ablation --------------------------------------------------
    for panel, (metric, better, title) in enumerate(
        [("overlap", "lower is better", "B. Mask overlap"),
         ("suff", "higher is better", "C. Sufficiency")], start=1
    ):
        ax = axes[panel]
        labels, base_vals, opt_vals, colors = [], [], [], []
        for d in DATASETS:
            for e in EVIDENCE:
                r = abl[(d, e)]
                if not r:
                    continue
                labels.append(f"{d.replace('_', ' ')}\n{EVIDENCE_LABEL[e]}")
                base_vals.append(r["base_evidence"][metric])
                opt_vals.append(r["optimized"][metric])
                colors.append(EVIDENCE_COLOR[e])
        xs = np.arange(len(labels))
        ax.bar(xs - 0.2, base_vals, 0.4, label="base evidence",
               color=colors, alpha=0.45, edgecolor="none")
        ax.bar(xs + 0.2, opt_vals, 0.4, label="after allocation",
               color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{title}\n{better}", fontsize=11)
        ax.axhline(0, c="0.7", lw=0.8)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CDEA contrastive game: Grad-CAM vs Integrated Gradients as base evidence",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print("Saved", out_path)


if __name__ == "__main__":
    main()
