"""Val top-1 and balanced accuracy for paper checkpoints.

Shift rows also record shortcut reliance: worst-group accuracy for Waterbirds,
and the accuracy drop when the shortcut is removed or randomized for the other
shift datasets. The test split is not read.

    PYTHONPATH=. python scripts/model_table.py --checkpoint path.pt --out results/paper/model_table/table.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.reporting import save_json
from evaluation.accuracy import worst_group_accuracy
from scripts.train_backbone import _build_model, get_val_loader, score_loader

SHIFT_DATASETS = ("planted_patch",)


def _load(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise ValueError(f"{path} is not a metadata checkpoint")
    from models.wrapper import load_checkpoint_into

    model = _build_model(blob["model_name"], int(blob["num_classes"]), pretrained=False)
    model = load_checkpoint_into(model, blob)
    model.to(device).eval()
    return model, blob


def _shift_fields(dataset: str, model, device, data_root: Path, batch_size: int) -> dict:
    """Shortcut reliance on val. Empty for contrastive datasets."""
    if dataset not in SHIFT_DATASETS:
        return {}
    if dataset == "planted_patch":
        from evaluation.planted_cues import PlantedPatchCIFAR
        ds = PlantedPatchCIFAR(split="val", root=data_root, seed=0)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        hit = {"present": 0, "removed": 0}
        total = 0
        with torch.no_grad():
            for batch in loader:
                y = batch["label"].to(device)
                total += int(y.shape[0])
                for name in hit:
                    pred = model(batch[name].to(device)).argmax(1)
                    hit[name] += int((pred == y).sum().item())
        present = hit["present"] / max(total, 1)
        removed = hit["removed"] / max(total, 1)
        return {"present_accuracy": present, "removed_accuracy": removed,
                "patch_gap": present - removed}
    return {}


def score_checkpoint(
    path: Path,
    data_root: Path,
    device: torch.device,
    batch_size: int = 32,
    shift_metrics: bool = True,
) -> dict:
    model, blob = _load(path, device)
    dataset = blob["dataset"]
    loader = get_val_loader(dataset, batch_size, data_root)
    if loader is None:
        raise RuntimeError(f"{dataset} has no val loader")
    top1, balanced = score_loader(model, loader, device, int(blob["num_classes"]))
    row = {
        "dataset": dataset,
        "model_name": blob["model_name"],
        "seed": int(blob.get("seed", -1)),
        "num_classes": int(blob["num_classes"]),
        "val_top1": top1,
        "val_balanced_accuracy": balanced,
        "checkpoint": str(path),
    }
    if shift_metrics:
        row.update(_shift_fields(dataset, model, device, data_root, batch_size))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Val model table from checkpoints")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(REPO / "data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-shift-metrics", action="store_true")
    args = parser.parse_args()
    paths = [Path(p) for p in args.checkpoint]
    if args.ckpt_dir:
        paths.extend(sorted(Path(args.ckpt_dir).glob("*.pt")))
    if not paths:
        raise SystemExit("pass --checkpoint or --ckpt-dir")
    from core.device import get_device
    device = get_device()
    rows = []
    for path in paths:
        print(f"scoring {path.name}", flush=True)
        rows.append(score_checkpoint(
            path, Path(args.data_root), device, args.batch_size,
            shift_metrics=not args.no_shift_metrics,
        ))
        print(
            f"  {rows[-1]['dataset']} seed={rows[-1]['seed']} "
            f"top1={rows[-1]['val_top1']:.4f} balanced={rows[-1]['val_balanced_accuracy']:.4f}",
            flush=True,
        )
    save_json(args.out, {"rows": rows})
    print(args.out)


if __name__ == "__main__":
    main()
