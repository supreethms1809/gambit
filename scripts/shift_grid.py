"""Shift-grid and ablation manifest. It does not launch either job.

S20's contrastive grid is counted from its log directory. A missing directory,
or a directory with no done or failed marker, means that grid was not launched.
``--final`` stays refused while ``EVAL_PLAN.md`` says it is not frozen. This
module does not read the test split and does not start a background job.

The shift list is the six environments in the plan, five seeds, ResNet-50 and
ViT-B/16, and the five core shift methods. Ablation cells are one variant per
removed component. Nothing in either list is executed by ``main``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SHIFT_DATASETS = (
    "waterbirds",
    "imagenet9",
    "stanford_dogs",
    "planted_patch",
    "colored_mnist",
    # Placeholder until the sixth independent shift dataset is chosen
    # (EVAL_PLAN.md section 2.2). It keeps the completion check failing until
    # then. Waterbirds natural groups are ablation AS1, not a unit.
    "sixth_shift_tbd",
)
SHIFT_METHODS = (
    "cdea_shift",
    "attribution_difference",
    "spray",
    "extremal_per_env",
    "random_floor",
)
MODELS = ("resnet50", "vit_b_16")
SEEDS = (0, 1, 2, 3, 4)
CONTRASTIVE_LOG = REPO / "results" / "paper" / "logs" / "contrastive_grid"


def marker_progress(log_dir: Path) -> dict[str, int | bool]:
    """Count done and failed markers. ``launched`` is false when both are zero."""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return {"done": 0, "failed": 0, "launched": False}
    done = len(list(log_dir.glob("*.done")))
    failed = len(list(log_dir.glob("*.failed")))
    return {"done": done, "failed": failed, "launched": (done + failed) > 0}


def shift_cells() -> list[dict]:
    """Job list for the shift grid. Every unit uses the paired objective.

    The unpaired objective is ablation AS1 on Waterbirds, not a dataset unit.
    """
    cells = []
    for dataset in SHIFT_DATASETS:
        objective = "paired"
        for model in MODELS:
            for method in SHIFT_METHODS:
                for seed in SEEDS:
                    cells.append({
                        "id": f"shift_{dataset}_{model}_{method}_seed{seed}",
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": seed,
                        "game": "shift",
                        "objective": objective,
                    })
    return cells


def ablation_cells() -> list[dict]:
    """One cell per removed component. Variants of A5–A8 and the shift pair are expanded."""
    rows: list[dict] = [
        {"id": "abl_A1_independent", "ablation": "A1", "removes": "joint_allocation"},
        {"id": "abl_A2_no_margin", "ablation": "A2", "removes": "margin"},
        {"id": "abl_A3_no_overlap", "ablation": "A3", "removes": "overlap"},
        {"id": "abl_A4_no_shared", "ablation": "A4", "removes": "shared_mask"},
    ]
    for init in ("zero", "evidence"):
        rows.append({
            "id": f"abl_A5_init_{init}",
            "ablation": "A5",
            "removes": "init",
            "init": init,
        })
    for interaction in ("none", "attention", "transformer"):
        rows.append({
            "id": f"abl_A6_interaction_{interaction}",
            "ablation": "A6",
            "removes": "interaction",
            "interaction": interaction,
        })
    for steps in (10, 25, 50, 100):
        rows.append({
            "id": f"abl_A7_steps_{steps}",
            "ablation": "A7",
            "removes": "steps",
            "steps": steps,
        })
    for preset in ("mixed", "cooperative", "competitive"):
        rows.append({
            "id": f"abl_A8_preset_{preset}",
            "ablation": "A8",
            "removes": "game_preset",
            "preset": preset,
        })
    for objective in ("paired", "unpaired"):
        for mass_target in (True, False):
            flag = "mass" if mass_target else "nomass"
            rows.append({
                "id": f"abl_S_{objective}_{flag}",
                "ablation": "S",
                "game": "shift",
                "objective": objective,
                "mass_target": mass_target,
            })
    return rows


def main() -> None:
    """Do not start the shift grid or the ablations."""
    progress = marker_progress(CONTRASTIVE_LOG)
    launched = "true" if progress["launched"] else "false"
    raise SystemExit(
        "refusing to launch the shift grid and the ablations: "
        "EVAL_PLAN.md is not frozen; "
        f"contrastive grid launched={launched}"
    )


if __name__ == "__main__":
    main()
