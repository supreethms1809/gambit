"""Train one checkpoint, then run its games, before the next dataset.

The formulation in ``cdea/`` does not change classifier training, so a
checkpoint already on disk is reused. ImageNet has no training: it uses the
torchvision weights. Stanford Dogs trains one linear probe and then runs both
its contrastive game and its shift game.

Seed 0 finishes before seed 1 starts. A failed cell stops the seed from
starting anything further; cells already running finish, and the next seed is
not started. Within a seed, ``--jobs 4`` keeps two big cells and two small
cells in flight for the whole seed, so the queue never becomes four big cells.
Big means high GPU memory: every shift cell, and every contrastive ViT. Small
means a contrastive ResNet.

    PYTHONPATH=. python scripts/launch_paper_chain.py --seeds 0 --jobs 4

``--force-train`` retrains the cells this command names. It does not retrain
a seed on its own.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.environments import SHIFT_DATASETS  # noqa: E402
from scripts.launch_paper_eval import spec_for  # noqa: E402
from scripts.launch_paper_training import (  # noqa: E402
    CKPT_DIR,
    paper_cells,
    train_cell,
)
from scripts.parallel_cells import mem_class  # noqa: E402
from scripts.smoke_e2e import BACKBONES, CONTRASTIVE_UNITS  # noqa: E402

# Longest cells first within a memory class, so the queue does not finish on one hog.
# ImageNet is eval-only and the longest of those evals.
LONGEST = (
    "imagenet",
    "planted_patch", "waterbirds", "imagenet9", "colored_mnist",
    "cifar10", "cifar100",
    "stanford_dogs", "ham10000", "cub200", "oxford_pets", "brain_tumor",
)


def games_for(dataset: str) -> tuple[str, ...]:
    """Games that share this dataset's checkpoint, contrastive first."""
    games = []
    if dataset in CONTRASTIVE_UNITS:
        games.append("contrastive")
    if dataset in SHIFT_DATASETS:
        games.append("shift")
    if not games:
        raise ValueError(f"no game for dataset {dataset!r}")
    return tuple(games)


def needs_training(dataset: str) -> bool:
    """ImageNet is the torchvision classifier. Every other unit is trained."""
    return dataset != "imagenet"


def unit_id(unit: dict) -> str:
    return f"{unit['dataset']}_{unit['backbone']}_seed{unit['seed']}"


def mem_of(unit: dict) -> str:
    """``high`` when any of the unit's games is a memory hog, else ``low``."""
    if any(mem_class(game, unit["dataset"], unit["backbone"]) == "high" for game in unit["games"]):
        return "high"
    return "low"


def size_label(memory: str) -> str:
    return "big" if memory == "high" else "small"


def chain_units(seeds, datasets=None, backbones=None) -> list[dict]:
    """One unit per dataset × backbone × seed. Seeds stay in the requested order."""
    dataset_filter = set(datasets) if datasets is not None else None
    backbone_filter = list(backbones) if backbones is not None else list(BACKBONES)
    names = []
    for dataset in list(CONTRASTIVE_UNITS) + [d for d in SHIFT_DATASETS if d not in CONTRASTIVE_UNITS]:
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        names.append(dataset)
    units = []
    for seed in seeds:
        for dataset in names:
            for backbone in backbone_filter:
                units.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "seed": int(seed),
                    "games": games_for(dataset),
                })
    return units


def order_seed(units: list[dict]) -> list[dict]:
    """Longest first within each memory class, then two big interleaved with two small."""
    from scripts.parallel_cells import order_balanced

    def rank(unit: dict) -> tuple:
        dataset = unit["dataset"]
        place = LONGEST.index(dataset) if dataset in LONGEST else len(LONGEST)
        vit_first = unit["backbone"] != "vit_b_16"
        return (place, vit_first)

    ordered = sorted(units, key=rank)
    return order_balanced(ordered, mem_of)


def training_cell(unit: dict) -> dict:
    match = [c for c in paper_cells()
             if c["dataset"] == unit["dataset"]
             and c["model_name"] == unit["backbone"]
             and c["seed"] == unit["seed"]]
    if len(match) != 1:
        raise KeyError(f"no training cell for {unit_id(unit)}")
    return match[0]


def eval_cell(unit: dict, game: str, split: str) -> dict:
    return {
        "id": f"{split}_{game}_{unit['dataset']}_{unit['backbone']}_seed{unit['seed']}",
        "game": game,
        "dataset": unit["dataset"],
        "backbone": unit["backbone"],
        "seed": unit["seed"],
    }


def _release_device() -> None:
    """Drop the training model before the games allocate."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_unit(unit: dict, args) -> bool:
    """Train this checkpoint if it is missing, then run each of its games."""
    from scripts.final_grid import run_grid
    from scripts.paper_run import run

    name = unit_id(unit)
    if needs_training(unit["dataset"]):
        ok = train_cell(
            training_cell(unit), args.epochs, Path(args.ckpt_dir), Path(args.train_log_dir),
            force=args.force_train,
        )
        if not ok:
            print(f"failed training {name}", flush=True)
            return False
        _release_device()

    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_methods import FAST, Knobs

    stamp = f"{method_code_hash()} {knobs_hash(FAST if args.fast else Knobs())}"
    log_dir = Path(args.out) / "_markers" / args.split

    def execute(cell):
        spec = spec_for(cell, args)
        out = run(spec, args.device)
        print(f"done {cell['id']} -> {out}", flush=True)

    for game in unit["games"]:
        rows = run_grid([eval_cell(unit, game, args.split)], log_dir, execute,
                        final=args.final, stamp=stamp)
        if any(row["status"] == "failed" for row in rows):
            print(f"failed {name} game {game}", flush=True)
            return False
    return True


def unit_done(unit: dict, args, stamp: str) -> bool:
    from scripts.final_grid import is_done

    log_dir = Path(args.out) / "_markers" / args.split
    return all(is_done(log_dir / f"{eval_cell(unit, game, args.split)['id']}.done", stamp)
               for game in unit["games"])


def child_argv(argv: list[str], one: str) -> list[str]:
    """Parent arguments for one unit. Parallel flags drop off; ``--one`` is added."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in ("--jobs", "--mem-gate-gb"):
            skip = True
            continue
        if arg.startswith(("--jobs=", "--mem-gate-gb=")):
            continue
        out.append(arg)
    return out + ["--one", one]


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--seeds", default="0", help="comma list. Each seed finishes before the next starts.")
    p.add_argument("--datasets", default=None, help="comma list (default every paper dataset)")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    p.add_argument("--train-log-dir", default=str(REPO / "results" / "paper" / "logs" / "train"))
    p.add_argument("--n-contrastive", default=None, help="per backbone, e.g. resnet50:200,vit_b_16:64")
    p.add_argument("--areas", default="0.025,0.05,0.10")
    p.add_argument("--areas-shift", default="0.05,0.10,0.25")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--candidates", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--config-hash", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--image-batch", type=int, default=4)
    p.add_argument("--out", default=str(REPO / "results" / "paper" / "cells"))
    p.add_argument("--force-train", action="store_true",
                   help="retrain the named cells even when their checkpoint exists")
    p.add_argument("--dry-run", action="store_true", help="print the seed-major order and exit")
    p.add_argument("--jobs", type=int, default=4,
                   help="cells at once. 4 keeps two big and two small in flight.")
    p.add_argument("--mem-gate-gb", type=float, default=12.0,
                   help="start a cell only with this much GPU free (0 disables the gate).")
    p.add_argument("--one", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _stamp(args) -> str:
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_methods import FAST, Knobs

    return f"{method_code_hash()} {knobs_hash(FAST if args.fast else Knobs())}"


def _pending(units: list[dict], args, stamp: str) -> list[dict]:
    return [unit for unit in units if not unit_done(unit, args, stamp)]


def _run_seed(units: list[dict], args, parent_argv: list[str]) -> list[str]:
    """Run one seed's pending units. Returns the ids that failed."""
    from scripts.parallel_cells import Job, python_command, run_parallel, thread_env

    stamp = _stamp(args)
    pending = order_seed(_pending(units, args, stamp))
    if not pending:
        return []
    if args.jobs <= 1 or args.one is not None:
        failed = []
        for unit in pending:
            print(f"start {unit_id(unit)} ({size_label(mem_of(unit))})", flush=True)
            if not run_unit(unit, args):
                failed.append(unit_id(unit))
                break
        return failed

    env = thread_env(args.jobs)
    jobs = [Job(name=unit_id(unit), log=Path(args.train_log_dir) / f"chain_{unit_id(unit)}.log", env=env,
                command=python_command(Path(__file__), *child_argv(parent_argv, unit_id(unit))))
            for unit in pending]
    kind = {unit_id(unit): mem_of(unit) for unit in pending}
    per_class = max(1, args.jobs // 2)

    def admit(running_names, job):
        same = sum(1 for name in running_names if kind[name] == kind[job.name])
        return same < per_class

    print("launch order: " + ", ".join(f"{unit_id(u)}({size_label(mem_of(u))})" for u in pending),
          flush=True)
    print(f"at most {per_class} big and {per_class} small at once", flush=True)
    codes = run_parallel(jobs, args.jobs, min_free_gb=args.mem_gate_gb,
                         stop_on_failure=True, admit=admit)
    failed = sorted(name for name, code in codes.items() if code != 0)
    held = [job.name for job in jobs if job.name not in codes]
    if held:
        print(f"held {len(held)} cell(s) after a failure; they were not started", flush=True)
    return failed


def main(argv=None) -> None:
    args = parse(argv)
    parent_argv = list(sys.argv[1:] if argv is None else argv)
    if args.candidates and args.split == "test":
        raise SystemExit("selection candidates run on val only (EVAL_PLAN 6)")
    seeds = [int(s) for s in _csv(args.seeds)]
    units = chain_units(seeds, _csv(args.datasets), _csv(args.backbones))
    if args.one is not None:
        match = [u for u in units if unit_id(u) == args.one]
        if len(match) != 1:
            raise SystemExit(f"unknown cell {args.one!r}")
        raise SystemExit(0 if run_unit(match[0], args) else 1)

    if args.dry_run:
        stamp = _stamp(args)
        for seed in seeds:
            group = order_seed([u for u in units if u["seed"] == seed])
            print(f"seed {seed}: {len(group)} cells")
            for unit in group:
                games = ",".join(unit["games"])
                train = "train" if needs_training(unit["dataset"]) else "imagenet-weights"
                state = "done" if unit_done(unit, args, stamp) else "pending"
                print(f"  {unit_id(unit)}  {size_label(mem_of(unit))}  {train}  games={games}  {state}")
        return

    for seed in seeds:
        group = [u for u in units if u["seed"] == seed]
        print(f"seed {seed}: {len(group)} cells", flush=True)
        failed = _run_seed(group, args, parent_argv)
        if failed:
            print(f"seed {seed} stopped with {len(failed)} failed cell(s); the next seed was not started",
                  flush=True)
            for name in failed:
                print(f"  failed: {name}", flush=True)
            raise SystemExit(1)
        print(f"seed {seed} complete", flush=True)


if __name__ == "__main__":
    main()
