"""Train the paper backbones. Resumable: an existing checkpoint is skipped.

Contrastive datasets are ImageNet linear probes (frozen backbone, Adam, cosine,
cross-entropy, lr 1e-3). Shift datasets are full fine-tunes at lr 1e-4, because
lr 1e-3 on a pretrained backbone is the failure the trainer already warns about.
Checkpoint selection is val balanced accuracy. The test split is not read.

    PYTHONPATH=. python scripts/launch_paper_training.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.train_backbone import get_or_train

CONTRASTIVE = (
    "cifar10", "cifar100", "oxford_pets", "stanford_dogs", "cub200",
    "ham10000", "brain_tumor",
)
SHIFT = ("colored_mnist", "planted_patch", "imagenet9", "waterbirds")
MODELS = ("resnet50", "vit_b_16")
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 15
CKPT_DIR = REPO / "results" / "paper" / "checkpoints"
LOG_DIR = REPO / "results" / "paper" / "logs" / "train"


def paper_cells() -> list[dict]:
    cells = []
    for dataset in CONTRASTIVE:
        for model in MODELS:
            for seed in SEEDS:
                cells.append({
                    "dataset": dataset,
                    "model_name": model,
                    "seed": seed,
                    "freeze_backbone": True,
                    "lr": 1e-3,
                    "batch_size": 32 if model.startswith("resnet") else 16,
                    "balanced": None,
                })
    for dataset in SHIFT:
        for model in MODELS:
            for seed in SEEDS:
                cells.append({
                    "dataset": dataset,
                    "model_name": model,
                    "seed": seed,
                    "freeze_backbone": False,
                    "lr": 1e-4,
                    "batch_size": 16,
                    "balanced": dataset == "waterbirds",
                })
    return cells


def cell_id(cell: dict) -> str:
    mode = "lp" if cell["freeze_backbone"] else "ft"
    return (
        f"{cell['dataset']}_{cell['model_name']}_{mode}"
        f"_lr{cell['lr']:g}_seed{cell['seed']}"
    )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cells = paper_cells()
    print(f"{len(cells)} cells", flush=True)
    for index, cell in enumerate(cells, start=1):
        name = cell_id(cell)
        marker = LOG_DIR / f"{name}.done"
        if marker.is_file():
            print(f"[{index}/{len(cells)}] skip {name}", flush=True)
            continue
        print(f"[{index}/{len(cells)}] start {name}", flush=True)
        try:
            path = get_or_train(
                dataset=cell["dataset"],
                model_name=cell["model_name"],
                pretrained=True,
                ckpt_dir=CKPT_DIR,
                num_epochs=EPOCHS,
                lr=cell["lr"],
                freeze_backbone=cell["freeze_backbone"],
                batch_size=cell["batch_size"],
                seed=cell["seed"],
                balanced=cell["balanced"],
            )
            if not Path(path).is_file():
                raise RuntimeError(f"no checkpoint written for {name}")
            marker.write_text(str(path) + "\n", encoding="utf-8")
            print(f"[{index}/{len(cells)}] done {path}", flush=True)
        except Exception:
            failed = LOG_DIR / f"{name}.failed"
            failed.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{index}/{len(cells)}] failed {name}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
