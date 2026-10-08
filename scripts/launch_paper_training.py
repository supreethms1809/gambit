"""Train the paper backbones. Resumable: an existing checkpoint is skipped.

Contrastive datasets are ImageNet linear probes (frozen backbone, Adam, cosine,
cross-entropy, lr 1e-3). Shift datasets are full fine-tunes at lr 1e-4, because
lr 1e-3 on a pretrained backbone is the failure the trainer already warns about.
Checkpoint selection is val balanced accuracy. The test split is not read.

    PYTHONPATH=. python scripts/launch_paper_training.py

One seed at a time, eight cells of it in parallel on one GPU:

    PYTHONPATH=. python scripts/launch_paper_training.py --seeds 0 --jobs 8

``--jobs`` runs each cell in its own process with its own log in the log
directory, and starts the longest cells first. ``--epochs``, ``--ckpt-dir`` and
``--log-dir`` exist for smoke checkpoints, which must not land in the paper
checkpoint directory.
"""

from __future__ import annotations

import argparse
import os
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

# Longest training first, from the measured Spark wall times, so the parallel
# queue does not end on one long cell. Only the order matters here.
LONGEST_FIRST = (
    "planted_patch", "colored_mnist", "cifar10", "cifar100", "imagenet9", "waterbirds",
    "stanford_dogs", "ham10000", "cub200", "oxford_pets", "brain_tumor",
)


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


def select_cells(cells: list[dict], seeds=None, datasets=None, models=None) -> list[dict]:
    """Filter, keeping seeds in the requested order; within a seed, longest cells first."""
    seed_order = list(seeds) if seeds is not None else sorted({c["seed"] for c in cells})
    kept = [c for c in cells
            if c["seed"] in seed_order
            and (datasets is None or c["dataset"] in datasets)
            and (models is None or c["model_name"] in models)]
    return sorted(kept, key=lambda c: (seed_order.index(c["seed"]),
                                       LONGEST_FIRST.index(c["dataset"]),
                                       c["model_name"] != "vit_b_16"))


def train_cell(cell: dict, epochs: int, ckpt_dir: Path, log_dir: Path) -> bool:
    """Train one cell and write its done or failed marker. Returns success."""
    name = cell_id(cell)
    try:
        path = get_or_train(
            dataset=cell["dataset"],
            model_name=cell["model_name"],
            pretrained=True,
            ckpt_dir=ckpt_dir,
            num_epochs=epochs,
            lr=cell["lr"],
            freeze_backbone=cell["freeze_backbone"],
            batch_size=cell["batch_size"],
            seed=cell["seed"],
            balanced=cell["balanced"],
        )
        if not Path(path).is_file():
            raise RuntimeError(f"no checkpoint written for {name}")
        (log_dir / f"{name}.done").write_text(str(path) + "\n", encoding="utf-8")
        print(f"done {path}", flush=True)
        return True
    except Exception:
        (log_dir / f"{name}.failed").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"failed {name}", flush=True)
        traceback.print_exc()
        return False


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default=None, help="comma list, run in this order (default 0-4)")
    p.add_argument("--datasets", default=None, help="comma list (default every paper dataset)")
    p.add_argument("--models", default=None, help="comma list (default resnet50,vit_b_16)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    p.add_argument("--log-dir", default=str(LOG_DIR))
    p.add_argument("--jobs", type=int, default=1, help="cells trained at the same time "
                   "(launched in high/low memory balance order: full-finetune first, interleaved)")
    p.add_argument("--one", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    log_dir, ckpt_dir = Path(args.log_dir), Path(args.ckpt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in _csv(args.seeds)] if args.seeds else None
    cells = select_cells(paper_cells(), seeds, _csv(args.datasets), _csv(args.models))

    if args.one is not None:
        match = [c for c in cells if cell_id(c) == args.one]
        if not match:
            raise SystemExit(f"unknown cell {args.one!r}")
        raise SystemExit(0 if train_cell(match[0], args.epochs, ckpt_dir, log_dir) else 1)

    pending = [c for c in cells if not (log_dir / f"{cell_id(c)}.done").is_file()]
    print(f"{len(cells)} cells, {len(cells) - len(pending)} already done", flush=True)
    if args.jobs <= 1:
        for index, cell in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] start {cell_id(cell)}", flush=True)
            train_cell(cell, args.epochs, ckpt_dir, log_dir)
        return

    from scripts.parallel_cells import Job, order_balanced, python_command, run_parallel, thread_env

    env = thread_env(args.jobs)
    if "GAMBIT_LOADER_WORKERS" not in os.environ:
        env["GAMBIT_LOADER_WORKERS"] = str(max(1, (os.cpu_count() or 8) // args.jobs - 1))
    passthrough = ["--epochs", str(args.epochs), "--ckpt-dir", str(ckpt_dir), "--log-dir", str(log_dir)]
    jobs = []
    by_id = {}
    for cell in pending:
        name = cell_id(cell)
        by_id[name] = cell
        jobs.append(Job(name=name, log=log_dir / f"{name}.log", env=env,
                        command=python_command(Path(__file__), "--seeds", str(cell["seed"]),
                                               "--one", name, *passthrough)))
    jobs = order_balanced(jobs, lambda job: "high" if not by_id[job.name]["freeze_backbone"] else "low")
    codes = run_parallel(jobs, args.jobs)
    failed = sorted(name for name, code in codes.items() if code != 0)
    print(f"{len(codes)} cells run, {len(failed)} failed", flush=True)
    for name in failed:
        print(f"  failed: {name} (see {log_dir / (name + '.log')})", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
