"""Descriptive measurements for decomposition version 2, before any method (PROGRESS.md plan R2).

FORMULATION_DECOMPOSITION.md version 2 sections 6.2, 6.6 and 10 (open decision 10).
Per image of a dataset's val sample: the overlap O and |Omega| at three rho,
O inside Omega, the gap z_l - z_m (K-sufficiency), the co-location share at
ResNet-50's last layer, and the pairwise interaction of the top Omega cells.
Forward passes and the reference evidence only.

    PYTHONPATH=. python scripts/decomp_descriptive.py --dataset cifar10 --n 64 \\
        --checkpoint results/paper/smoke_checkpoints/cifar10_resnet50_pt_lp_ep15_lr0.001_imagenet_seed0.pt \\
        --out results/paper/smoke/partition_local/descriptive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RHOS = (0.3, 0.5, 0.7)
TOP_CELLS = 6
GAPS = (0.5, 1.0, 2.0)


def measure(dataset: str, backbone: str, seed: int, split: str, n: int, model_source: str, device_name: str,
            checkpoint=None, image_batch: int = 8):
    import torch

    from cdea.region import cell_overlap, evidence_region
    from core.device import get_device, make_deterministic
    from core.grid import grid_of
    from core.types import HypothesisSet
    from evaluation.descriptive import colocation, head_contributions, pairwise_interaction
    from evaluation.run_data import decomposition_sample
    from evaluation.run_methods import Knobs, hypotheses_for, partition_reference
    from evaluation.run_models import load_cell_model

    device = get_device(None if device_name == "auto" else torch.device(device_name))
    make_deterministic(device)
    loaded = load_cell_model(dataset, backbone, seed, source=model_source, checkpoint=checkpoint, device=device)
    model = loaded.model
    sample = decomposition_sample(dataset, split, n, seed, model=model, device=device)
    grid = grid_of(backbone)
    rows = []
    for start in range(0, sample.x.shape[0], image_batch):
        x = sample.x[start:start + image_batch].to(device)
        top = hypotheses_for(model, x, 3)
        ids = top.ids.to(device)
        h = HypothesisSet(ids=ids[:, :2], mask=top.mask[:, :2].to(device))
        evidence = partition_reference(model, x, h, backbone, Knobs())
        overlap = cell_overlap(evidence)
        omegas = {rho: evidence_region(evidence, rho) for rho in RHOS}
        with torch.no_grad():
            logits = model(x).gather(1, ids)
        coloc = coloc_omega = None
        if backbone == "resnet50":
            contrib = head_contributions(model, x, ids[:, :2])
            coloc = colocation(contrib)
            coloc_omega = colocation(contrib, omegas[0.5])
        inter = pairwise_interaction(model, x, ids[:, :2], evidence.sum(dim=1), omegas[0.5], *grid, top=TOP_CELLS)
        for i in range(x.shape[0]):
            row = {
                "dataset": dataset, "backbone": backbone, "seed": seed, "split": split,
                "image_index": sample.index[start + i], "label": int(sample.labels[start + i]),
                "class_k": int(ids[i, 0]), "class_l": int(ids[i, 1]), "class_m": int(ids[i, 2]),
                "z_k": float(logits[i, 0]), "z_l": float(logits[i, 1]), "z_m": float(logits[i, 2]),
                "gap_lm": float(logits[i, 1] - logits[i, 2]), "overlap": float(overlap[i].sum()),
                "model_source": model_source,
            }
            for rho, omega in omegas.items():
                row[f"omega_size_{rho}"] = int(omega[i].sum())
                row[f"overlap_omega_{rho}"] = float((overlap[i] * omega[i]).sum())
            if coloc is not None:
                for name, out in (("all", coloc), ("omega", coloc_omega)):
                    for key in ("p", "p_pos", "q", "share", "share_pos"):
                        row[f"coloc_{key}_{name}"] = float(out[key][i])
            for j, cls in enumerate(("k", "l")):
                excess = inter["i"][i, :, j]
                single = inter["single"][i, :, j]
                valid = ~torch.isnan(excess)
                row[f"pairs_{cls}"] = int(valid.sum())
                if bool(valid.any()):
                    row[f"inter_mean_{cls}"] = float(excess[valid].mean())
                    row[f"inter_abs_{cls}"] = float(excess[valid].abs().mean())
                    row[f"single_abs_{cls}"] = float(single[valid].mean())
                    row[f"inter_rel_{cls}"] = float(excess[valid].abs().sum() / single[valid].sum().clamp_min(1e-6))
            rows.append(row)
    return rows, loaded


def summarise(frame) -> str:
    import numpy as np

    lines = [f"n = {len(frame)} images; top-1 accuracy {float((frame['class_k'] == frame['label']).mean()):.3f}"]

    def q(column):
        values = frame[column].dropna().to_numpy()
        if not len(values):
            return "n/a"
        a, b, c = np.percentile(values, [25, 50, 75])
        return f"median {b:.3f} (IQR {a:.3f} to {c:.3f})"

    lines.append(f"O (overlap of the clamped, normalised maps): {q('overlap')}")
    tertiles = np.percentile(frame["overlap"], [100 / 3, 200 / 3])
    lines.append(f"O tertile cuts on this sample: {tertiles[0]:.3f}, {tertiles[1]:.3f}")
    for rho in RHOS:
        lines.append(f"|Omega| at rho {rho}: {q(f'omega_size_{rho}')}; O inside Omega: {q(f'overlap_omega_{rho}')}")
    for gap in GAPS:
        lines.append(f"share with z_l - z_m < {gap}: {float((frame['gap_lm'] < gap).mean()):.3f}")
    if "coloc_share_all" in frame:
        lines.append(f"co-location share as written, all cells: {q('coloc_share_all')}; P {q('coloc_p_all')}; "
                     f"Q {q('coloc_q_all')}")
        lines.append(f"co-location share, positive parts, all cells: {q('coloc_share_pos_all')}; P+ {q('coloc_p_pos_all')}")
        lines.append(f"co-location share, positive parts, Omega cells: {q('coloc_share_pos_omega')}")
    for cls in ("k", "l"):
        if f"inter_rel_{cls}" in frame:
            lines.append(f"pairwise interaction ({cls}), sum |I| / sum (|d(a)| + |d(b)|): {q(f'inter_rel_{cls}')}; "
                         f"mean I {q(f'inter_mean_{cls}')}")
    return "\n".join(lines)


def main(argv=None) -> None:
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_b_16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val", choices=["val"])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--model-source", default="auto", choices=["auto", "paper", "checkpoint", "smoke"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    rows, loaded = measure(args.dataset, args.backbone, args.seed, args.split, args.n, args.model_source, args.device,
                           checkpoint=args.checkpoint)
    frame = pd.DataFrame(rows)
    out = Path(args.out) / args.split / args.dataset / args.backbone / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "descriptive.csv.gz", index=False)
    text = summarise(frame)
    (out / "summary.txt").write_text(f"model: {loaded.source} {loaded.path}\n{text}\n")
    print(text)
    print(f"records: {out}")


if __name__ == "__main__":
    main()
