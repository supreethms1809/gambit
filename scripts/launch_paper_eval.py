"""Run the paper evaluation grid, resumably, through ``scripts/paper_run.py``.

One cell is one game × dataset × backbone × seed, running every method of
that game (and, on the ablation units, every ablation) and writing records
under ``--out``. A cell with a done marker is skipped, so the command can be
restarted after an interruption. ``--datasets`` assigns units to this
machine (EVAL_PLAN.md section 10: one dataset runs on one machine).

Val now (selection, the shift-area pilot, G1):

    PYTHONPATH=. python scripts/launch_paper_eval.py --split val --datasets cifar10,ham10000

Test, only after the freeze (refused before it, and on a dirty tree):

    PYTHONPATH=. python scripts/launch_paper_eval.py --split test --final --config-hash <hash>

Image counts per seed default to the plan's minimums (section 2.4); set the
frozen values per backbone, e.g. ``--n-contrastive resnet50:200,vit_b_16:64``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.run_methods import (  # noqa: E402
    ABLATIONS,
    CONTRASTIVE_CANDIDATES,
    CONTRASTIVE_CORE,
    CONTRASTIVE_EXTENDED,
    FAST,
    SHIFT_ABLATIONS,
    SHIFT_CANDIDATES,
    SHIFT_CORE,
    Knobs,
    candidate_applies,
)
from scripts.smoke_e2e import BACKBONES, CONTRASTIVE_UNITS, SHIFT_UNITS  # noqa: E402

# EVAL_PLAN.md section 8: ablations on the non-dev units, seeds 0-2.
ABLATION_UNITS = ("cifar100", "cub200", "brain_tumor", "imagenet")
ABLATION_SEEDS = (0, 1, 2)
SHIFT_ABLATION_UNIT = "waterbirds"
N_MIN = {("contrastive", "resnet50"): 200, ("contrastive", "vit_b_16"): 64,
         ("shift", "resnet50"): 128, ("shift", "vit_b_16"): 32}


def parse_n(value) -> dict:
    """``"resnet50:200,vit_b_16:64"`` -> per-backbone n. A bare number is refused:
    one n for both backbones would give ViT the ResNet budget (EVAL_PLAN 2.4)."""
    if not value:
        return {}
    out = {}
    for part in str(value).split(","):
        if ":" not in part:
            raise SystemExit(f"n must be per backbone, e.g. resnet50:200,vit_b_16:64 (got {value!r})")
        name, count = part.split(":", 1)
        if name not in BACKBONES:
            raise SystemExit(f"unknown backbone {name!r} in n")
        out[name] = int(count)
    return out


def grid(args) -> list[dict]:
    datasets = set(args.datasets.split(",")) if args.datasets else None
    backbones = args.backbones.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = []
    for game, units in (("contrastive", CONTRASTIVE_UNITS), ("shift", SHIFT_UNITS)):
        if args.game not in {"both", game}:
            continue
        for dataset in units:
            if datasets and dataset not in datasets:
                continue
            for backbone in backbones:
                for seed in seeds:
                    cells.append({
                        "id": f"{args.split}_{game}_{dataset}_{backbone}_seed{seed}",
                        "game": game, "dataset": dataset, "backbone": backbone, "seed": seed,
                    })
    return cells


def spec_for(cell: dict, args):
    from evaluation.run_cell import CellSpec

    game, dataset, backbone, seed = cell["game"], cell["dataset"], cell["backbone"], cell["seed"]
    n = parse_n(args.n_contrastive if game == "contrastive" else args.n_shift).get(backbone)
    n = n or N_MIN[(game, backbone)]
    if game == "contrastive":
        methods = [m for m in CONTRASTIVE_CORE + (CONTRASTIVE_EXTENDED if args.extended else ())
                   if not (m == "cve" and backbone != "resnet50")]
        ablations = (list(ABLATIONS) if args.ablations and dataset in ABLATION_UNITS
                     and seed in ABLATION_SEEDS and backbone == "resnet50" else [])
        areas = tuple(float(a) for a in args.areas.split(","))
        operators = ("road", "blur")
        candidates = list(CONTRASTIVE_CANDIDATES) if args.candidates else []
    else:
        methods = list(SHIFT_CORE)
        ablations = (list(SHIFT_ABLATIONS) if args.ablations and dataset == SHIFT_ABLATION_UNIT
                     and seed in ABLATION_SEEDS and backbone == "resnet50" else [])
        areas = tuple(float(a) for a in args.shift_areas.split(","))
        operators = ("road",)
        candidates = list(SHIFT_CANDIDATES) if args.candidates else []
    candidates = [c for c in candidates if candidate_applies(c, backbone)]
    return CellSpec(game=game, dataset=dataset, backbone=backbone, seed=seed, split=args.split, n=n,
                    methods=methods, ablations=ablations, candidates=candidates, areas=areas,
                    operators=operators,
                    model_source="auto", final=args.final, config_hash=args.config_hash,
                    knobs=FAST if args.fast else Knobs(), out_dir=args.out)


def child_argv(argv: list[str], cell_id: str) -> list[str]:
    """This launcher's arguments for one cell: ``--jobs`` dropped, ``--one`` added."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == "--jobs":
            skip = True
            continue
        if arg.startswith("--jobs="):
            continue
        out.append(arg)
    return out + ["--one", cell_id]


def _run_cells_in_parallel(cells: list[dict], log_dir: Path, jobs: int) -> None:
    from scripts.parallel_cells import Job, python_command, run_parallel, thread_env

    env = thread_env(jobs)
    run_parallel([Job(name=c["id"], log=log_dir / f"{c['id']}.log", env=env,
                      command=python_command(Path(__file__), *child_argv(sys.argv[1:], c["id"])))
                  for c in cells], jobs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--game", default="both", choices=["both", "contrastive", "shift"])
    p.add_argument("--datasets", default=None, help="units for this machine, comma list")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-contrastive", default=None, help="per backbone, e.g. resnet50:200,vit_b_16:64")
    p.add_argument("--n-shift", default=None, help="per backbone, e.g. resnet50:128,vit_b_16:32")
    p.add_argument("--areas", default="0.025,0.05,0.10")
    p.add_argument("--shift-areas", default="0.05,0.10,0.25")
    p.add_argument("--extended", action="store_true", help="also run the Extended baselines")
    p.add_argument("--ablations", action="store_true", help="run ablations on the ablation units")
    p.add_argument("--candidates", action="store_true",
                   help="run the val selection grid (EVAL_PLAN 6.2); refused with --split test")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--config-hash", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(REPO / "results" / "paper" / "runs"))
    p.add_argument("--dry-run", action="store_true", help="list the cells and exit")
    p.add_argument("--jobs", type=int, default=1,
                   help="cells run at the same time, one process each, logs beside the markers")
    p.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    from scripts.final_grid import pending_cells, run_grid
    from scripts.paper_run import run

    if args.candidates and args.split == "test":
        raise SystemExit("selection candidates run on val only (EVAL_PLAN 6)")
    cells = grid(args)
    if args.one is not None:
        cells = [c for c in cells if c["id"] == args.one]
        if not cells:
            raise SystemExit(f"unknown cell {args.one!r}")
    log_dir = Path(args.out) / "_markers" / args.split
    if args.dry_run:
        for cell in cells:
            spec = spec_for(cell, args)
            print(f"{cell['id']}  n={spec.n}  methods={len(spec.methods)}  ablations={len(spec.ablations)}"
                  f"  candidates={len(spec.candidates)}")
        print(f"{len(cells)} cells")
        return

    if args.jobs > 1 and args.one is None:
        run_grid([], log_dir, lambda cell: None, final=args.final)   # the freeze and clean-tree checks
        _run_cells_in_parallel(pending_cells(cells, log_dir), log_dir, args.jobs)
        rows = [{"id": c["id"], "status": "done" if (log_dir / f"{c['id']}.done").is_file() else "failed"}
                for c in cells]
        failed = [r["id"] for r in rows if r["status"] == "failed"]
        print(f"{len(rows)} cells: {len(rows) - len(failed)} done, {len(failed)} failed")
        for name in failed:
            print(f"  failed: {name} (see {log_dir / (name + '.log')})")
        if failed:
            raise SystemExit(1)
        return

    def execute(cell):
        spec = spec_for(cell, args)
        out = run(spec, args.device)
        print(f"done {cell['id']} -> {out}", flush=True)

    rows = run_grid(cells, log_dir, execute, final=args.final)
    failed = [r["id"] for r in rows if r["status"] == "failed"]
    print(f"{len(rows)} cells: {sum(r['status'] == 'done' for r in rows)} done, "
          f"{sum(r['status'] == 'skipped' for r in rows)} skipped, {len(failed)} failed")
    for name in failed:
        print(f"  failed: {name} (see {log_dir / (name + '.failed')})")


if __name__ == "__main__":
    main()
