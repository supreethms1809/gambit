"""End-to-end smoke run of the whole evaluation on a few val images.

Runs every contrastive and shift unit, on both backbones, with every method,
plus every ablation, through the same ``scripts/paper_run.py`` code the paper
run uses. It reads val only. It exists so the full run on Spark starts from a
pipeline that is known to run end to end.

* Contrastive cells use 1 image. Shift cells use 4, because SpRAy clusters
  relevance maps across images and needs at least 3.
* ``FAST`` knobs shorten the expensive loops. One extra cell per backbone
  runs the core methods at the paper's knobs, to time a real explanation.
* A unit with no paper checkpoint on this machine uses the smoke model (an
  ImageNet backbone with a random head). Its numbers are meaningless; the
  report marks it. A unit whose data is missing is reported as skipped.
* ``--checkpoint-dir`` reads short-trained smoke checkpoints from their own
  directory instead (``launch_paper_training.py --epochs 1 --ckpt-dir ...``),
  so smoke models never sit beside the paper checkpoints.
* ``--jobs N`` runs N datasets at a time in separate processes, then runs the
  paper-knob timing cells alone so their seconds are not inflated by sharing.

    PYTHONPATH=. python scripts/smoke_e2e.py --out results/paper/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
    Knobs,
    candidate_applies,
)

BACKBONES = ("resnet50", "vit_b_16")
CONTRASTIVE_UNITS = ("cifar10", "cifar100", "oxford_pets", "stanford_dogs", "cub200",
                     "ham10000", "brain_tumor", "imagenet")
ABLATION_UNIT = "cifar10"


def _source(dataset: str, backbone: str, seed: int, checkpoint_dir=None, checkpoint_epochs: int = 15):
    """``(model_source, checkpoint path or None)`` for one smoke cell."""
    from evaluation.run_models import paper_checkpoint_path

    if dataset == "imagenet":
        return "imagenet", None
    if checkpoint_dir is not None:
        path = paper_checkpoint_path(dataset, backbone, seed, ckpt_dir=Path(checkpoint_dir),
                                     num_epochs=checkpoint_epochs)
        return ("checkpoint", str(path)) if path.is_file() else ("smoke", None)
    return ("paper" if paper_checkpoint_path(dataset, backbone, seed).is_file() else "smoke"), None


def _data_present(dataset: str) -> bool:
    if dataset == "imagenet":
        return (REPO / "data" / "imagenet" / "val").is_dir() and (REPO / "data" / "splits" / "imagenet.json").is_file()
    return True


SHIFT_SMOKE_UNITS = ("waterbirds", "planted_patch")


def shift_cells():
    from evaluation.run_cell import CellSpec
    from evaluation.run_methods import SHIFT_CORE

    return [
        CellSpec(
            game="shift", dataset=dataset, backbone=backbone, seed=0, n=4,
            methods=list(SHIFT_CORE), knobs=FAST,
        )
        for dataset in SHIFT_SMOKE_UNITS
        for backbone in BACKBONES
    ]


def cells(production_timing: bool = True):
    from evaluation.run_cell import CellSpec

    out = []
    for dataset in CONTRASTIVE_UNITS:
        for backbone in BACKBONES:
            methods = [m for m in CONTRASTIVE_CORE + CONTRASTIVE_EXTENDED
                       if not (m in ("cve", "sc_cve") and backbone != "resnet50")
                       and not (m == "chefer" and not backbone.startswith("vit"))]
            ablations = list(ABLATIONS) if dataset == ABLATION_UNIT else []
            candidates = ([c for c in CONTRASTIVE_CANDIDATES if candidate_applies(c, backbone)]
                          if dataset == ABLATION_UNIT else [])
            out.append(CellSpec(game="contrastive", dataset=dataset, backbone=backbone, seed=0, n=1,
                                methods=methods, ablations=ablations, candidates=candidates, knobs=FAST))
    if production_timing:
        for backbone in BACKBONES:
            methods = [m for m in CONTRASTIVE_CORE if not (m == "cve" and backbone != "resnet50")]
            out.append(CellSpec(game="contrastive", dataset="cifar10", backbone=backbone, seed=0, n=1,
                                methods=methods, knobs=Knobs()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="contrastive", choices=["contrastive", "shift"])
    p.add_argument("--out", default=str(REPO / "results" / "paper" / "smoke"))
    p.add_argument("--device", default="auto")
    p.add_argument("--only", default=None, help="comma list of datasets to keep")
    p.add_argument("--no-production-timing", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="reuse cells whose summary.json already exists; the report covers all cells")
    p.add_argument("--checkpoint-dir", default=None,
                   help="read smoke checkpoints from here instead of the paper checkpoint directory")
    p.add_argument("--checkpoint-epochs", type=int, default=15,
                   help="epoch count in the smoke checkpoint names")
    p.add_argument("--jobs", type=int, default=1, help="datasets run at the same time")
    args = p.parse_args()
    from scripts.paper_run import run

    out = Path(args.out)
    keep = set(args.only.split(",")) if args.only else None
    if args.jobs > 1:
        _run_datasets_in_parallel(args, keep)
        args.resume = True
    results = []
    planned = shift_cells() if args.game == "shift" else cells(not args.no_production_timing)
    for spec in planned:
        if keep and spec.dataset not in keep:
            continue
        timing = not spec.knobs.fast
        spec.out_dir = str(out / ("production_knobs" if timing else "fast_knobs"))
        tag = f"{spec.game}/{spec.dataset}/{spec.backbone}{' [paper knobs]' if timing else ''}"
        if not _data_present(spec.dataset):
            results.append({"cell": tag, "status": "skipped", "reason": "data not on this machine"})
            print(f"skip  {tag}: data not on this machine", flush=True)
            continue
        spec.model_source, spec.checkpoint = _source(spec.dataset, spec.backbone, spec.seed,
                                                     args.checkpoint_dir, args.checkpoint_epochs)
        done = Path(spec.out_dir) / spec.split / spec.game / spec.dataset / spec.backbone / f"seed{spec.seed}" / "summary.json"
        if args.resume and done.is_file():
            summary = json.loads(done.read_text())
            method_seconds = sum(m.get("seconds", 0.0) for m in summary["methods"].values())
            results.append({"cell": tag, "status": "ok", "seconds": method_seconds,
                            "model_source": spec.model_source, "summary": str(done),
                            "methods": summary["methods"], "rows": summary["rows"], "resumed": True})
            print(f"reuse {tag}", flush=True)
            continue
        start = time.perf_counter()
        try:
            path = run(spec, args.device)
            summary = json.loads((Path(path) / "summary.json").read_text())
            results.append({"cell": tag, "status": "ok", "seconds": time.perf_counter() - start,
                            "model_source": spec.model_source, "summary": str(Path(path) / "summary.json"),
                            "methods": summary["methods"], "rows": summary["rows"]})
            print(f"done  {tag}  {time.perf_counter() - start:.0f}s  errors={summary['errors']}", flush=True)
        except Exception as exc:
            results.append({"cell": tag, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "trace": traceback.format_exc(limit=6)})
            print(f"FAIL  {tag}: {type(exc).__name__}: {exc}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "REPORT.md").write_text(report(results))
    print(f"report: {out / 'REPORT.md'}")


def _run_datasets_in_parallel(args, keep) -> None:
    """One process per dataset at the fast knobs. The caller then resumes over every cell."""
    from scripts.parallel_cells import Job, python_command, run_parallel, thread_env

    datasets = []
    for spec in cells(production_timing=False):
        if spec.dataset not in datasets and (keep is None or spec.dataset in keep):
            datasets.append(spec.dataset)
    passthrough = ["--out", str(args.out), "--device", args.device, "--no-production-timing",
                   "--checkpoint-epochs", str(args.checkpoint_epochs)]
    if args.checkpoint_dir:
        passthrough += ["--checkpoint-dir", str(args.checkpoint_dir)]
    if args.resume:
        passthrough.append("--resume")
    log_dir = Path(args.out) / "logs"
    env = thread_env(args.jobs)
    jobs = [Job(name=d, log=log_dir / f"{d}.log", env=env,
                command=python_command(Path(__file__), "--only", d, *passthrough))
            for d in datasets]
    run_parallel(jobs, args.jobs)


def report(results) -> str:
    lines = ["# End-to-end smoke run", "",
             "Val images only. Generated by `scripts/smoke_e2e.py`. Numbers from a `smoke` model "
             "(random head) or `fast` knobs are not results; this checks that every path runs.", "",
             "## Cells", "", "| cell | status | model | seconds | rows | method errors |", "|---|---|---|---|---|---|"]
    for r in results:
        errs = [m for m, s in r.get("methods", {}).items() if s["status"] == "error"]
        lines.append(f"| {r['cell']} | {r['status']} | {r.get('model_source', '')} | "
                     f"{r.get('seconds', 0):.0f} | {r.get('rows', '')} | {', '.join(errs) or ''} |")
    lines += ["", "## Method errors", ""]
    any_err = False
    for r in results:
        if r["status"] == "error":
            any_err = True
            lines.append(f"- **{r['cell']}** (cell): `{r['error']}`")
        for m, s in r.get("methods", {}).items():
            if s["status"] == "error":
                any_err = True
                lines.append(f"- **{r['cell']}** / `{m}`: `{s['error']}`")
    if not any_err:
        lines.append("None.")
    lines += ["", "## Per-image cost at the paper's knobs", "",
              "| cell | method | seconds / image | forward images | backward images |", "|---|---|---|---|---|"]
    for r in results:
        if "[paper knobs]" not in r["cell"] or r["status"] != "ok":
            continue
        for m, s in r["methods"].items():
            if s["status"] == "ok":
                lines.append(f"| {r['cell']} | {m} | {s['seconds']:.1f} | {s['forward_images']} | {s['backward_images']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
