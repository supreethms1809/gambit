"""Launch the paper runs on one GPU: the decomposition game (EVAL_PLAN.md
section 13) and the shift game (SHIFT_EVAL_PLAN.md).

A phase is a set of cells. A cell is one game on one dataset × backbone ×
seed. Every requested phase goes into one queue. Two large and two small
cells run at once:

- large: every shift cell and every ViT-B/16 decomposition cell
  (``parallel_cells.MEM_SEED_HIGH``). A summary that recorded a peak of
  20 GB or more makes that cell large next time.
- small: ResNet-50 decomposition cells.

Large cells never exceed two at once. When no large cell is waiting, small
cells take the free large slots. Each cell runs in its own process with its
own log in ``results/paper/logs/runs/``.

Before a cell runs, its checkpoint is reused when it exists and trained when
it is missing, with the paper recipe. ImageNet uses the torchvision weights.
A lock per checkpoint keeps two cells from training the same one. A cell is
done when its marker carries the current code and knobs hashes and no method
in it errored. Rerunning the same command skips done cells.

Phases (val unless noted):

  dev          Decomposition. CIFAR-10 and HAM10000, ResNet-50, seed 0, n = 64,
               every method, DA1-DA9 and the 4 CDEA candidates (13.3).
  planted-val  Decomposition. Planted pairs, both backbones, seeds 0-4, n = 200,
               every method and the 4 candidates; DA1-DA9 on ResNet-50.
  natural-val  Decomposition. The other 6 natural datasets on ResNet-50 and all 8
               on ViT-B/16, seed 0, n = 64, every method and the 4 candidates
               (only ``--cdea`` once it is chosen). Picks the best min-split rows.
  test         Decomposition, after the freeze only (--final, --config-hash,
               --cdea). Planted pairs and the 8 natural datasets, both backbones,
               seeds 0-4; DA1-DA9 on section 8's units and seeds, ResNet-50.
  shift-gate   Shift. Waterbirds and planted-patch, ResNet-50, seeds 0 and 1,
               n = 64, 5%, the gate's four methods. Written to
               results/paper/gate_shift, so shift-val never overwrites it.
  shift-val    Shift. The 5 shift datasets, both backbones, seed 0, n = 200 / 64,
               areas 5/10/25%, every core method, the candidates, and S1-S8
               without S6 on Waterbirds and planted-patch ResNet-50.

Decomposition areas are 2.5, 5 and 10%, under ROAD and blur.

    PYTHONPATH=. python scripts/launch_runs.py --phases dev,planted-val,natural-val,shift-gate --dry-run
    nohup env PYTHONPATH=. python scripts/launch_runs.py --phases dev,planted-val,natural-val,shift-gate \\
        > results/paper/logs/launch_runs.log 2>&1 &
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.environments import SHIFT_DATASETS  # noqa: E402
from evaluation.run_methods import (  # noqa: E402
    DECOMP_ABLATIONS,
    DECOMP_CANDIDATES,
    DECOMP_CORE,
    DECOMP_EXTENDED,
    SHIFT_ABLATIONS,
    SHIFT_CANDIDATES,
    SHIFT_CORE,
    candidate_applies,
)
from scripts.launch_paper_eval import ABLATION_SEEDS, ABLATION_UNITS, N_MIN, PEAK_HIGH_MB  # noqa: E402
from scripts.smoke_e2e import CONTRASTIVE_UNITS  # noqa: E402

PHASES = ("dev", "planted-val", "natural-val", "test", "shift-gate", "shift-val")
BACKBONES = ("resnet50", "vit_b_16")
SEEDS = (0, 1, 2, 3, 4)
NATURAL = CONTRASTIVE_UNITS
DEV = ("cifar10", "ham10000")
PLANTED = "planted_pair"
DECOMP_AREAS = (0.025, 0.05, 0.10)
SHIFT_AREAS = (0.05, 0.10, 0.25)
GATE = ("waterbirds", "planted_patch")
GATE_SEEDS = (0, 1)
GATE_METHODS = ("cdea_shift", "shift_assign", "gap_attribution", "attribution_difference")
SHIFT_ABLATION_UNITS = ("waterbirds", "planted_patch")
VAL_N = 64
PLANTED_N = 200
# These need maps exported from the archive tag (PROGRESS.md, Parked). Any other
# method error fails the cell.
ALLOWED_ERRORS = ("cdea:A8_precomputed", "cdea_shift:S6_precomputed")
EPOCHS = 15

RESULTS = REPO / "results" / "paper"
CKPT_DIR = RESULTS / "checkpoints"
TRAIN_LOG_DIR = RESULTS / "logs" / "train"

# Longest first within a size class, so the queue does not end on one long cell.
LONGEST = (
    "imagenet", "planted_pair", "planted_patch", "waterbirds", "imagenet9", "colored_mnist",
    "cifar10", "cifar100", "stanford_dogs", "ham10000", "cub200", "oxford_pets", "brain_tumor",
)


@dataclass(frozen=True)
class Cell:
    phase: str
    game: str
    dataset: str
    backbone: str
    seed: int
    split: str
    n: int
    methods: tuple
    out: str
    ablations: tuple = ()
    candidates: tuple = ()
    areas: tuple = DECOMP_AREAS

    @property
    def id(self) -> str:
        return f"{self.phase}__{self.split}_{self.game}_{self.dataset}_{self.backbone}_seed{self.seed}"

    @property
    def target(self) -> tuple:
        """The records directory this cell writes. Two cells must never share one."""
        return (self.out, self.split, self.game, self.dataset, self.backbone, self.seed)

    @property
    def cell_dir(self) -> Path:
        return Path(self.out) / self.split / self.game / self.dataset / self.backbone / f"seed{self.seed}"


def _decomp_methods(args) -> tuple:
    return DECOMP_CORE + (() if args.core_only else DECOMP_EXTENDED)


def _phase_cells(phase: str, args) -> list[Cell]:
    cells_root = str(Path(args.out_root) / "cells")
    gate_root = str(Path(args.out_root) / "gate_shift")
    methods = _decomp_methods(args)
    candidates = tuple(DECOMP_CANDIDATES)
    ablations = tuple(DECOMP_ABLATIONS)
    out: list[Cell] = []
    if phase == "dev":
        for dataset in DEV:
            out.append(Cell(phase, "decomposition", dataset, "resnet50", 0, "val", VAL_N, methods, cells_root,
                            ablations, candidates))
    elif phase == "planted-val":
        for seed in SEEDS:
            for backbone in BACKBONES:
                out.append(Cell(phase, "decomposition", PLANTED, backbone, seed, "val", PLANTED_N, methods,
                                cells_root, ablations if backbone == "resnet50" else (), candidates))
    elif phase == "natural-val":
        chosen = (args.cdea,) if args.cdea else candidates
        for backbone in BACKBONES:
            for dataset in NATURAL:
                if backbone == "resnet50" and dataset in DEV:
                    continue   # the dev phase already runs these, with more
                out.append(Cell(phase, "decomposition", dataset, backbone, 0, "val", VAL_N, methods, cells_root,
                                (), chosen))
    elif phase == "test":
        # The paper's CDEA row is the chosen candidate (13.3), so the default label is not run.
        test_methods = tuple(m for m in methods if m != "cdea_decomp")
        n_natural = {**{b: N_MIN[("contrastive", b)] for b in BACKBONES}, **args.n_test}
        for seed in SEEDS:
            for backbone in BACKBONES:
                out.append(Cell(phase, "decomposition", PLANTED, backbone, seed, "test", PLANTED_N, test_methods,
                                cells_root, (), (args.cdea,)))
                for dataset in NATURAL:
                    abl = (ablations if dataset in ABLATION_UNITS and seed in ABLATION_SEEDS
                           and backbone == "resnet50" else ())
                    out.append(Cell(phase, "decomposition", dataset, backbone, seed, "test", n_natural[backbone],
                                    test_methods, cells_root, abl, (args.cdea,)))
    elif phase == "shift-gate":
        for dataset in GATE:
            for seed in GATE_SEEDS:
                out.append(Cell(phase, "shift", dataset, "resnet50", seed, "val", VAL_N, GATE_METHODS, gate_root,
                                areas=(0.05,)))
    elif phase == "shift-val":
        shift_methods = SHIFT_CORE + (("r2r",) if args.r2r else ())
        shift_ablations = tuple(a for a in SHIFT_ABLATIONS if a != "S6_precomputed")
        for backbone in BACKBONES:
            for dataset in SHIFT_DATASETS:
                abl = shift_ablations if dataset in SHIFT_ABLATION_UNITS and backbone == "resnet50" else ()
                cands = tuple(c for c in SHIFT_CANDIDATES if candidate_applies(c, backbone))
                out.append(Cell(phase, "shift", dataset, backbone, 0, "val", N_MIN[("contrastive", backbone)],
                                shift_methods, cells_root, abl, cands, SHIFT_AREAS))
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return out


def build_cells(args) -> list[Cell]:
    """Every requested phase's cells, after the filters. Refuses two cells that write one directory."""
    phases = _csv(args.phases)
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; choose from {PHASES}")
    if args.cdea is not None and args.cdea not in DECOMP_CANDIDATES:
        raise SystemExit(f"--cdea must be one of {list(DECOMP_CANDIDATES)}")
    if "test" in phases:
        missing = [flag for flag, value in (("--cdea", args.cdea), ("--final", args.final),
                                            ("--config-hash", args.config_hash)) if not value]
        if missing:
            raise SystemExit(f"the test phase needs {', '.join(missing)} (after the freeze only)")
    cells = [cell for phase in phases for cell in _phase_cells(phase, args)]
    datasets, backbones, seeds = _csv(args.datasets), _csv(args.backbones), _csv(args.seeds)
    cells = [c for c in cells
             if (datasets is None or c.dataset in datasets)
             and (backbones is None or c.backbone in backbones)
             and (seeds is None or str(c.seed) in seeds)]
    if args.n is not None:
        cells = [replace(c, n=args.n) for c in cells]
    seen: dict[tuple, str] = {}
    for cell in cells:
        if cell.target in seen:
            raise SystemExit(f"{cell.id} and {seen[cell.target]} would write the same records")
        seen[cell.target] = cell.id
    return cells


def size_of(cell: Cell) -> str:
    """``large`` or ``small``. A recorded peak in this cell's summary wins over the table."""
    from scripts.parallel_cells import mem_class

    try:
        peak = float(json.loads((cell.cell_dir / "summary.json").read_text()).get("device_peak_mb", "nan"))
        if peak == peak and peak >= 0:
            return "large" if peak >= PEAK_HIGH_MB else "small"
    except (OSError, ValueError, TypeError):
        pass
    return "large" if mem_class(cell.game, cell.dataset, cell.backbone) == "high" else "small"


def order(cells: list[Cell]) -> list[Cell]:
    """Seed first, then the longest datasets, ViT before ResNet within a dataset."""
    def rank(cell: Cell) -> tuple:
        place = LONGEST.index(cell.dataset) if cell.dataset in LONGEST else len(LONGEST)
        return (cell.seed, place, cell.backbone != "vit_b_16", cell.phase)

    return sorted(cells, key=rank)


def make_admit(kind: dict[str, str], large: int, small: int, fill: bool = True):
    """At most ``large`` large cells. Small cells get ``small`` slots, plus idle large slots once no large waits."""
    started: set[str] = set()

    def admit(running, job) -> bool:
        run_large = sum(1 for name in running if kind[name] == "large")
        run_small = len(running) - run_large
        if kind[job.name] == "large":
            ok = run_large < large
        else:
            large_waiting = any(size == "large" and name not in started for name, size in kind.items())
            ok = run_small < small or (fill and not large_waiting and run_small < small + large - run_large)
        if ok:
            started.add(job.name)
        return ok

    return admit


def stamp(args) -> str:
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_methods import FAST, Knobs

    return f"{method_code_hash()} {knobs_hash(FAST if args.fast else Knobs())}"


def marker_dir(cell: Cell) -> Path:
    return Path(cell.out) / "_markers" / cell.split


def is_done(cell: Cell, current: str) -> bool:
    from scripts.final_grid import is_done as marker_done

    return marker_done(marker_dir(cell) / f"{cell.id}.done", current)


def checkpoint_state(cell: Cell, args) -> str:
    if args.model_source != "auto":
        return f"model-source {args.model_source}"
    if cell.dataset == "imagenet":
        return "torchvision"
    from evaluation.run_models import paper_checkpoint_path

    return "present" if paper_checkpoint_path(cell.dataset, cell.backbone, cell.seed).is_file() else "will train"


def training_cell(cell: Cell) -> dict:
    from scripts.launch_paper_training import paper_cells

    match = [c for c in paper_cells() if c["dataset"] == cell.dataset
             and c["model_name"] == cell.backbone and c["seed"] == cell.seed]
    if len(match) != 1:
        raise KeyError(f"no training cell for {cell.dataset} {cell.backbone} seed {cell.seed}")
    return match[0]


@contextlib.contextmanager
def _lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_checkpoint(cell: Cell, args) -> None:
    """Reuse the paper checkpoint when it exists; otherwise train it, one trainer per checkpoint."""
    if checkpoint_state(cell, args) in ("present", "torchvision") or args.model_source != "auto":
        return
    from evaluation.run_models import paper_checkpoint_path
    from scripts.launch_paper_training import train_cell

    path = paper_checkpoint_path(cell.dataset, cell.backbone, cell.seed)
    with _lock(CKPT_DIR / ".locks" / f"{path.name}.lock"):
        if path.is_file():
            return   # another cell trained it while this one waited
        print(f"training {path.name}", flush=True)
        if not train_cell(training_cell(cell), EPOCHS, CKPT_DIR, TRAIN_LOG_DIR) or not path.is_file():
            raise RuntimeError(f"training did not produce {path}")
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def method_errors(summary_path: Path) -> list[str]:
    """Methods that errored in this cell, other than the known map-export gaps."""
    methods = json.loads(Path(summary_path).read_text(encoding="utf-8")).get("methods", {})
    return sorted(name for name, status in methods.items()
                  if status.get("status") == "error" and name not in ALLOWED_ERRORS)


def spec_of(cell: Cell, args):
    from evaluation.run_cell import CellSpec
    from evaluation.run_methods import FAST, Knobs

    return CellSpec(game=cell.game, dataset=cell.dataset, backbone=cell.backbone, seed=cell.seed,
                    split=cell.split, n=cell.n, methods=list(cell.methods), ablations=list(cell.ablations),
                    candidates=list(cell.candidates), areas=cell.areas, operators=("road", "blur"),
                    model_source=args.model_source, final=cell.split == "test" and args.final,
                    config_hash=args.config_hash, knobs=FAST if args.fast else Knobs(), out_dir=cell.out,
                    image_batch=args.image_batch)


def run_one(cell: Cell, args) -> bool:
    """Train if needed, run the cell, and fail it if any method errored. Writes its marker."""
    from scripts.final_grid import run_grid
    from scripts.paper_run import run

    def execute(_):
        try:
            ensure_checkpoint(cell, args)
            out = run(spec_of(cell, args), args.device)
            errors = method_errors(Path(out) / "summary.json")
            if errors:
                raise RuntimeError(f"method errors in {cell.id}: {errors}")
            print(f"done {cell.id} -> {out}", flush=True)
        except Exception:
            traceback.print_exc()
            raise

    rows = run_grid([{"id": cell.id}], marker_dir(cell), execute,
                    final=cell.split == "test" and args.final, stamp=stamp(args))
    return all(row["status"] != "failed" for row in rows)


_PARALLEL_FLAGS = ("--large", "--small", "--mem-gate-gb", "--log-dir")


def child_argv(argv: list[str], cell_id: str) -> list[str]:
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in _PARALLEL_FLAGS:
            skip = True
            continue
        if arg.startswith(tuple(f"{flag}=" for flag in _PARALLEL_FLAGS)) or arg in ("--no-fill", "--keep-going"):
            continue
        out.append(arg)
    return out + ["--one", cell_id]


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _n_map(value) -> dict:
    out = {}
    for part in _csv(value) or []:
        backbone, _, n = part.partition(":")
        if backbone not in BACKBONES or not n:
            raise SystemExit("--n-test is per backbone, e.g. resnet50:200,vit_b_16:64")
        out[backbone] = int(n)
    return out


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phases", required=True, help=f"comma list of {', '.join(PHASES)}")
    p.add_argument("--dry-run", action="store_true", help="print the queue and each cell's state, then exit")
    p.add_argument("--large", type=int, default=2, help="large cells at once")
    p.add_argument("--small", type=int, default=2, help="small cells at once")
    p.add_argument("--no-fill", action="store_true", help="never let small cells use idle large slots")
    p.add_argument("--keep-going", action="store_true", help="keep starting cells after a failure")
    p.add_argument("--mem-gate-gb", type=float, default=12.0,
                   help="start a cell only with this much GPU memory free (0 disables)")
    p.add_argument("--cdea", default=None, help="the chosen CDEA candidate, e.g. cdea_decomp@lr0.2_t100")
    p.add_argument("--core-only", action="store_true", help="leave out the extended decomposition rows")
    p.add_argument("--r2r", action="store_true", help="add R2R to shift-val (needs scripts/fetch_r2r.sh)")
    p.add_argument("--n-test", type=_n_map, default={}, help="test n per backbone, set at the freeze")
    p.add_argument("--final", action="store_true")
    p.add_argument("--config-hash", default=None)
    p.add_argument("--datasets", default=None, help="comma list filter")
    p.add_argument("--backbones", default=None, help="comma list filter")
    p.add_argument("--seeds", default=None, help="comma list filter")
    p.add_argument("--n", type=int, default=None, help="override n in every cell (smoke only)")
    p.add_argument("--fast", action="store_true", help="smoke knobs; needs --out-root outside results/paper")
    p.add_argument("--model-source", default="auto", choices=["auto", "smoke"],
                   help="smoke builds untrained models and never trains (smoke only)")
    p.add_argument("--out-root", default=str(RESULTS))
    p.add_argument("--log-dir", default=str(RESULTS / "logs" / "runs"))
    p.add_argument("--device", default="auto")
    p.add_argument("--image-batch", type=int, default=4)
    p.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    paper_root = Path(args.out_root).resolve() == RESULTS.resolve()
    if paper_root and (args.fast or args.model_source != "auto" or args.n is not None):
        raise SystemExit("--fast, --model-source smoke and --n write smoke records; give --out-root elsewhere")
    return args


def main(argv=None) -> None:
    args = parse(argv)
    parent_argv = list(sys.argv[1:] if argv is None else argv)
    cells = build_cells(args)
    if args.one is not None:
        match = [c for c in cells if c.id == args.one]
        if len(match) != 1:
            raise SystemExit(f"unknown cell {args.one!r}")
        raise SystemExit(0 if run_one(match[0], args) else 1)

    current = stamp(args)
    pending = order([c for c in cells if not is_done(c, current)])
    kind = {c.id: size_of(c) for c in pending}
    print(f"{len(cells)} cells, {len(cells) - len(pending)} done, {len(pending)} pending "
          f"({sum(1 for v in kind.values() if v == 'large')} large, "
          f"{sum(1 for v in kind.values() if v == 'small')} small); stamp {current}", flush=True)
    if args.dry_run:
        for cell in order(cells):
            state = "done" if cell.id not in kind else "pending"
            size = kind.get(cell.id, size_of(cell))
            print(f"  {cell.id:70s} {size:5s}  n={cell.n:<4d} {checkpoint_state(cell, args):12s} {state}")
        return
    if not pending:
        return

    from scripts.parallel_cells import Job, python_command, run_parallel, thread_env

    total = args.large + args.small
    env = thread_env(total)
    jobs = [Job(name=c.id, log=Path(args.log_dir) / f"{c.id}.log", env=env,
                command=python_command(Path(__file__), *child_argv(parent_argv, c.id)))
            for c in pending]
    print(f"at most {args.large} large and {args.small} small at once"
          + ("" if args.no_fill else "; small cells use idle large slots once no large cell waits"), flush=True)
    codes = run_parallel(jobs, total, min_free_gb=args.mem_gate_gb, stop_on_failure=not args.keep_going,
                         admit=make_admit(kind, args.large, args.small, fill=not args.no_fill))
    failed = sorted(name for name, code in codes.items() if code != 0)
    held = [job.name for job in jobs if job.name not in codes]
    if held:
        print(f"held {len(held)} cell(s) after a failure; they were not started", flush=True)
    if failed:
        for name in failed:
            print(f"  failed: {name} (log {Path(args.log_dir) / f'{name}.log'})", flush=True)
        raise SystemExit(1)
    print("all requested cells done", flush=True)


if __name__ == "__main__":
    main()
