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
    SHIFT_ABLATIONS,
    SHIFT_CANDIDATES,
    SHIFT_CORE,
    Knobs,
    candidate_applies,
)

BACKBONES = ("resnet50", "vit_b_16")
CONTRASTIVE_UNITS = ("cifar10", "cifar100", "oxford_pets", "stanford_dogs", "cub200",
                     "ham10000", "brain_tumor", "imagenet")
SHIFT_UNITS = ("waterbirds", "imagenet9", "stanford_dogs", "planted_patch", "colored_mnist")
ABLATION_UNIT = "cifar10"
SHIFT_ABLATION_UNIT = "waterbirds"


def _source(dataset: str, backbone: str, seed: int) -> str:
    from evaluation.run_models import paper_checkpoint_path

    if dataset == "imagenet":
        return "imagenet"
    return "paper" if paper_checkpoint_path(dataset, backbone, seed).is_file() else "smoke"


def _data_present(dataset: str) -> bool:
    if dataset == "imagenet":
        return (REPO / "data" / "imagenet" / "val").is_dir() and (REPO / "data" / "splits" / "imagenet.json").is_file()
    return True


def cells(production_timing: bool = True):
    from evaluation.run_cell import CellSpec

    out = []
    for dataset in CONTRASTIVE_UNITS:
        for backbone in BACKBONES:
            methods = [m for m in CONTRASTIVE_CORE + CONTRASTIVE_EXTENDED
                       if not (m == "cve" and backbone != "resnet50")]
            ablations = list(ABLATIONS) if dataset == ABLATION_UNIT else []
            candidates = ([c for c in CONTRASTIVE_CANDIDATES if candidate_applies(c, backbone)]
                          if dataset == ABLATION_UNIT else [])
            out.append(CellSpec(game="contrastive", dataset=dataset, backbone=backbone, seed=0, n=1,
                                methods=methods, ablations=ablations, candidates=candidates, knobs=FAST))
    for dataset in SHIFT_UNITS:
        for backbone in BACKBONES:
            ablations = list(SHIFT_ABLATIONS) if dataset == SHIFT_ABLATION_UNIT else []
            candidates = ([c for c in SHIFT_CANDIDATES if candidate_applies(c, backbone)]
                          if dataset == SHIFT_ABLATION_UNIT else [])
            out.append(CellSpec(game="shift", dataset=dataset, backbone=backbone, seed=0, n=4,
                                methods=list(SHIFT_CORE), ablations=ablations, candidates=candidates,
                                areas=(0.05, 0.10, 0.25), operators=("road",), knobs=FAST))
    if production_timing:
        for backbone in BACKBONES:
            methods = [m for m in CONTRASTIVE_CORE if not (m == "cve" and backbone != "resnet50")]
            out.append(CellSpec(game="contrastive", dataset="cifar10", backbone=backbone, seed=0, n=1,
                                methods=methods, knobs=Knobs()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO / "results" / "paper" / "smoke"))
    p.add_argument("--device", default="auto")
    p.add_argument("--only", default=None, help="comma list of datasets to keep")
    p.add_argument("--no-production-timing", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="reuse cells whose summary.json already exists; the report covers all cells")
    args = p.parse_args()
    from scripts.paper_run import run

    out = Path(args.out)
    keep = set(args.only.split(",")) if args.only else None
    results = []
    for spec in cells(not args.no_production_timing):
        if keep and spec.dataset not in keep:
            continue
        timing = not spec.knobs.fast
        spec.out_dir = str(out / ("production_knobs" if timing else "fast_knobs"))
        tag = f"{spec.game}/{spec.dataset}/{spec.backbone}{' [paper knobs]' if timing else ''}"
        if not _data_present(spec.dataset):
            results.append({"cell": tag, "status": "skipped", "reason": "data not on this machine"})
            print(f"skip  {tag}: data not on this machine", flush=True)
            continue
        spec.model_source = _source(spec.dataset, spec.backbone, spec.seed)
        done = Path(spec.out_dir) / spec.game / spec.dataset / spec.backbone / f"seed{spec.seed}" / "summary.json"
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
