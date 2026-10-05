"""Run one paper cell: every requested method (and CDEA ablation) on one
dataset × backbone × seed, scored as EVAL_PLAN.md defines, written as records.

    # contrastive, val, all core + extended methods, every ablation
    PYTHONPATH=. python scripts/paper_run.py --game contrastive --dataset cifar10 \\
        --backbone resnet50 --seed 0 --split val --n 200 --methods all --ablations all

    # shift, with the area pilot
    PYTHONPATH=. python scripts/paper_run.py --game shift --dataset waterbirds \\
        --backbone resnet50 --seed 0 --split val --n 128 --methods all --areas 0.05,0.10,0.25

``--split test`` is refused until EVAL_PLAN.md is frozen and ``--final`` and
``--config-hash`` are given; ``--final`` also refuses a dirty tree.
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
    CONTRASTIVE_CORE,
    CONTRASTIVE_EXTENDED,
    FAST,
    SHIFT_ABLATIONS,
    SHIFT_CORE,
    Knobs,
)


def _list(value: str, every: tuple, core: tuple = ()) -> list[str]:
    if value in {"", "none"}:
        return []
    if value == "all":
        return list(every)
    if value == "core":
        return list(core)
    names = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [n for n in names if n not in every]
    if unknown:
        raise SystemExit(f"unknown names: {unknown}. Choices: {list(every)}")
    return names


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", choices=["contrastive", "shift"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_b_16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--methods", default="core", help="core | all | none | comma list")
    p.add_argument("--ablations", default="none", help="all | none | comma list")
    p.add_argument("--areas", default="0.05")
    p.add_argument("--operators", default="road,blur")
    p.add_argument("--model-source", default="auto", choices=["auto", "paper", "checkpoint", "imagenet", "smoke"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--fast", action="store_true", help="smoke knobs; never a paper number")
    p.add_argument("--out", default=str(REPO / "results" / "paper" / "runs"))
    p.add_argument("--device", default="auto")
    p.add_argument("--final", action="store_true")
    p.add_argument("--config-hash", default=None)
    return p.parse_args(argv)


def spec_from_args(args):
    from evaluation.run_cell import CellSpec

    if args.game == "contrastive":
        methods = _list(args.methods, CONTRASTIVE_CORE + CONTRASTIVE_EXTENDED, CONTRASTIVE_CORE)
        ablations = _list(args.ablations, tuple(ABLATIONS))
        operators = tuple(o.strip() for o in args.operators.split(","))
    else:
        methods = _list(args.methods, SHIFT_CORE, SHIFT_CORE)
        ablations = _list(args.ablations, tuple(SHIFT_ABLATIONS))
        operators = ("road",)
    if args.game == "contrastive" and args.backbone != "resnet50" and "cve" in methods:
        methods.remove("cve")   # CVE is ResNet-50 only (EVAL_PLAN 4.2)
    return CellSpec(
        game=args.game, dataset=args.dataset, backbone=args.backbone, seed=args.seed,
        split=args.split, n=args.n, methods=methods, ablations=ablations,
        areas=tuple(float(a) for a in args.areas.split(",")), operators=operators,
        model_source=args.model_source, checkpoint=args.checkpoint, final=args.final,
        config_hash=args.config_hash, knobs=FAST if args.fast else Knobs(), out_dir=args.out,
    )


def run(spec, device_name: str = "auto"):
    from core.device import get_device
    from core.reporting import refuse_final_if_dirty
    from evaluation.run_cell import run_contrastive, run_shift

    refuse_final_if_dirty(spec.final)
    import torch

    device = get_device(None if device_name == "auto" else torch.device(device_name))
    return (run_contrastive if spec.game == "contrastive" else run_shift)(spec, device)


def main(argv=None) -> None:
    args = parse(argv)
    out = run(spec_from_args(args), args.device)
    print(f"records: {out}")


if __name__ == "__main__":
    main()
