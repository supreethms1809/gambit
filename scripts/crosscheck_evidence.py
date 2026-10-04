"""Grad-CAM and IG against pytorch-grad-cam and Captum on CIFAR-10 val.

The comparison is the one in ``baselines.crosscheck``: target-layer Grad-CAM
before the library's display resize, and right-Riemann integrated gradients.
The run stays on CPU. It does not read the test split.

    PYTHONPATH=. python scripts/crosscheck_evidence.py

Exit status is 0 when both mean Spearman correlations are at least 0.95.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.crosscheck import SPEARMAN_MIN, gradcam_spearman, ig_spearman
from baselines.versions import CAPTUM_VERSION, GRAD_CAM_VERSION
from core.reporting import save_json
from scripts.model_table import _load
from scripts.train_backbone import get_val_loader


def _take(loader: torch.utils.data.DataLoader, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    seen = 0
    for batch in loader:
        images.append(batch[0])
        labels.append(batch[1])
        seen += int(batch[0].shape[0])
        if seen >= n:
            break
    if seen < n:
        raise RuntimeError(f"val loader yielded {seen} images, needed {n}")
    return torch.cat(images)[:n], torch.cat(labels)[:n]


def _mean(rows: list[torch.Tensor]) -> tuple[float, float]:
    values = torch.cat(rows).float()
    return float(values.mean()), float(values.min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/paper_rerun/checkpoints/cifar10_resnet18_pt_lp_ep15_lr0.001_seed0.pt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/paper/crosscheck/val100.json"))
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("n must be >= 1")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    device = torch.device("cpu")
    model, blob = _load(args.checkpoint, device)
    if blob.get("dataset") != "cifar10":
        raise SystemExit(f"expected a cifar10 checkpoint, found {blob.get('dataset')}")
    loader = get_val_loader("cifar10", args.batch_size, args.data_root)
    if loader is None:
        raise SystemExit("cifar10 val loader is missing")
    images, labels = _take(loader, args.n)

    gradcam_rows, ig_rows = [], []
    for start in range(0, args.n, args.batch_size):
        xb = images[start : start + args.batch_size]
        yb = labels[start : start + args.batch_size]
        gradcam_rows.append(gradcam_spearman(model, xb, yb, args.grid, args.grid))
        ig_rows.append(ig_spearman(model, xb, yb, args.grid, args.grid, steps=args.steps))

    gradcam_mean, gradcam_min = _mean(gradcam_rows)
    ig_mean, ig_min = _mean(ig_rows)
    passed = gradcam_mean >= SPEARMAN_MIN and ig_mean >= SPEARMAN_MIN
    save_json(
        args.out,
        {
            "dataset": "cifar10",
            "split": "val",
            "n": args.n,
            "grid": args.grid,
            "ig_steps": args.steps,
            "ig_method": "riemann_right",
            "checkpoint": str(args.checkpoint),
            "grad_cam_version": GRAD_CAM_VERSION,
            "captum_version": CAPTUM_VERSION,
            "spearman_min": SPEARMAN_MIN,
            "gradcam_spearman_mean": gradcam_mean,
            "gradcam_spearman_min": gradcam_min,
            "ig_spearman_mean": ig_mean,
            "ig_spearman_min": ig_min,
            "passed": passed,
        },
    )
    print(f"wrote {args.out} passed={passed}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
