"""Run and score one contrastive cell, and write its records.

One row per image x method x area x operator. A method that raises is
recorded as ``status="error"`` and the cell continues. Records go under
``results/paper/cells`` and carry the method-code hash and the knobs hash.
"""

from __future__ import annotations

import csv
import gzip
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from evaluation.run_methods import (
    ABLATIONS,
    CONTRASTIVE_CANDIDATES,
    WHOLE_SAMPLE,
    CdeaConfig,
    Knobs,
    PairMaps,
    contrastive_method,
    default_backend,
    hypotheses_for,
)

IMAGE_SIZE = 224
AREA_DEPENDENT = {"extremal", "extremal_preserve", "cdea"}
ROAD = dict(iters=24, noise=0.01)
CELLS_ROOT = Path(__file__).resolve().parents[1] / "results" / "paper" / "cells"


class PassCounter:
    """Images through the model's top-level forward, and images backpropagated."""

    def __init__(self, model: nn.Module):
        self.forward = 0
        self.backward = 0
        self._handle = model.register_forward_hook(self._hook)

    def _hook(self, _module, inputs, output):
        batch = int(inputs[0].shape[0]) if inputs and torch.is_tensor(inputs[0]) else 0
        self.forward += batch
        if torch.is_tensor(output) and output.requires_grad:
            def count(grad, b=batch):
                self.backward += b
                return grad
            output.register_hook(count)

    def reset(self) -> None:
        self.forward = 0
        self.backward = 0

    def close(self) -> None:
        self._handle.remove()


def remove(x: torch.Tensor, mask: torch.Tensor, operator: str, seed: int, grid: tuple[int, int]) -> torch.Tensor:
    from core.grid import deletion_baseline
    from evaluation.scores import _remove

    if operator == "road":
        return _remove(x, mask, ROAD["iters"], ROAD["noise"], seed)
    if operator == "blur":
        m = mask.unsqueeze(1) if mask.ndim == 3 else mask
        return x * (1 - m) + deletion_baseline(x, grid[0], grid[1]) * m
    raise ValueError(f"unknown removal operator {operator!r}")


def _logits(model, x):
    with torch.no_grad():
        return model(x)


def _pick(logits, cls):
    return logits.gather(1, cls.view(-1, 1)).squeeze(1)


@dataclass
class CellSpec:
    game: str = "contrastive"
    dataset: str = ""
    backbone: str = ""
    seed: int = 0
    split: str = "val"
    n: int = 1
    methods: Sequence[str] = ()
    ablations: Sequence[str] = ()
    candidates: Sequence[str] = ()
    areas: Sequence[float] = (0.05,)
    operators: Sequence[str] = ("road", "blur")
    model_source: str = "auto"
    checkpoint: Optional[str] = None
    final: bool = False
    config_hash: Optional[str] = None
    knobs: Knobs = field(default_factory=Knobs)
    out_dir: Optional[str] = None
    image_batch: int = 4
    precomputed_dir: Optional[str] = None


def _ranges(n: int, image_batch: int) -> list[tuple[int, int]]:
    if n < 1:
        return []
    step = n if image_batch is None or image_batch <= 0 else min(int(image_batch), n)
    if step < 1:
        raise ValueError("image_batch must be >= 1, or 0 for the whole sample")
    return [(start, min(start + step, n)) for start in range(0, n, step)]


def _whole_sample(method: str) -> bool:
    return method in WHOLE_SAMPLE


def _area_dependent(method: str, ablation: Optional[str]) -> bool:
    if method in {"extremal", "extremal_preserve"}:
        return True
    return method == "cdea" and ablation != "A9_first_order"


def _method_runs(spec: CellSpec) -> list[tuple[str, Optional[str], Optional[str]]]:
    runs = [(m, None, None) for m in spec.methods]
    runs += [("cdea", a, None) for a in spec.ablations]
    runs += [(c.split("@", 1)[0], None, c) for c in spec.candidates]
    return runs


_COST_KNOBS = {"extremal_max_iter", "rise_masks", "ig_steps"}


def _candidate_knobs(spec: CellSpec, override: dict) -> Knobs:
    from dataclasses import replace as _replace

    fields = {k: v for k, v in override.items() if k != "backend"}
    if spec.knobs.fast:
        fields = {k: v for k, v in fields.items() if k not in _COST_KNOBS}
    return _replace(spec.knobs, **fields)


def _maps(spec, model, x, h, method, ablation, candidate, area, knobs, device, index):
    override = CONTRASTIVE_CANDIDATES[candidate] if candidate else None
    if ablation is not None:
        cfg = ABLATIONS[ablation](CdeaConfig(backend=default_backend(spec.backbone)))
        if cfg.kind == "precomputed":
            cfg = type(cfg)(**{**cfg.__dict__, "precomputed_dir": spec.precomputed_dir})
        return cdea_call(model, x, h, spec, knobs, device, cfg, area, index)
    if isinstance(override, CdeaConfig):
        return cdea_call(model, x, h, spec, knobs, device, override, area, index)
    fn = contrastive_method(method)
    kwargs = {}
    if _area_dependent(method, None):
        kwargs["area"] = area
    if method in {"cve", "sc_cve"}:
        kwargs["dataset"] = spec.dataset
    if method in {"cve", "sc_cve", "random_floor", "rise_margin"}:
        kwargs["seed"] = spec.seed
    if isinstance(override, dict) and "backend" in override:
        kwargs["backend"] = override["backend"]
    return fn(model, x, h, spec.backbone, knobs, device, **kwargs)


def cdea_call(model, x, h, spec, knobs, device, cfg, area, index):
    from evaluation.run_methods import cdea_pair

    return cdea_pair(model, x, h, spec.backbone, knobs, device, cfg, area=area if area is not None else spec.areas[0], index=index)


def run_contrastive(spec: CellSpec, device) -> Path:
    from baselines.adapter import budget_or_floor
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_data import contrastive_sample
    from evaluation.run_models import load_cell_model

    loaded = load_cell_model(spec.dataset, spec.backbone, spec.seed, source=spec.model_source,
                             checkpoint=spec.checkpoint, device=device)
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = loaded.model
    sample = contrastive_sample(spec.dataset, spec.split, spec.n, spec.seed,
                                final=spec.final, config_hash=spec.config_hash)
    n = int(sample.x.shape[0])
    counter = PassCounter(model)
    rows: list[dict] = []
    status: dict[str, dict] = {}
    code_hash = method_code_hash()
    knob_hash = knobs_hash(spec.knobs)

    for method, ablation, candidate in _method_runs(spec):
        label = candidate or (method if ablation is None else f"{method}:{ablation}")
        areas = spec.areas if _area_dependent(method, ablation) else (None,)
        ranges = _ranges(n, 0 if _whole_sample(method) else spec.image_batch)
        for area_for_map in areas:
            counter.reset()
            start = time.perf_counter()
            failed = None
            chunk_rows: list[dict] = []
            extra = {}
            override = CONTRASTIVE_CANDIDATES[candidate] if candidate else None
            knobs = _candidate_knobs(spec, override) if isinstance(override, dict) else spec.knobs
            try:
                for start_i, end_i in ranges:
                    x = sample.x[start_i:end_i].to(device)
                    h = hypotheses_for(model, x, 5)
                    h = type(h)(ids=h.ids.to(device), mask=h.mask.to(device))
                    maps = _maps(spec, model, x, h, method, ablation, candidate, area_for_map, knobs, device,
                                 sample.index[start_i:end_i])
                    extra = maps.extra
                    k, l = h.ids[:, 0], h.ids[:, 1]
                    full = _logits(model, x)
                    for area in (spec.areas if area_for_map is None else (area_for_map,)):
                        chunk_rows += _score_pair(
                            model, x, maps, k, l, full, area, spec, label, sample, budget_or_floor,
                            start_i, code_hash, knob_hash,
                        )
            except Exception as exc:
                failed = exc
            seconds = time.perf_counter() - start
            if failed is not None:
                status[label] = {"status": "error", "error": f"{type(failed).__name__}: {failed}",
                                 "trace": traceback.format_exc(limit=4)}
                continue
            status[label] = {"status": "ok", "seconds": seconds, "forward_images": counter.forward,
                             "backward_images": counter.backward, "extra": _jsonable(extra),
                             "device_mb_after": device_memory_mb(device)}
            if os.environ.get("GAMBIT_TRACE_MEMORY"):
                print(f"[mem] {label}: {status[label]['device_mb_after']:.0f} MB", flush=True)
            for row in chunk_rows:
                row["seconds_per_image"] = seconds / n
                row["forward_images"] = counter.forward
                row["backward_images"] = counter.backward
            rows += chunk_rows
            if label == "cdea" and extra.get("shared") is not None and len(ranges) == 1:
                # Shared sensitivity uses the unchunked maps. Chunked cells record unique rows only
                # here; the shared map is image-wise and is scored with the same chunk maps when present.
                pass
    counter.close()
    summary = _summary(spec, loaded, status, rows, code_hash, knob_hash)
    return _write(spec, rows, summary, device)


def _score_pair(model, x, maps: PairMaps, k, l, full, area, spec, label, sample, budget_or_floor,
                offset, code_hash, knob_hash) -> list[dict]:
    from core.grid import grid_of
    from evaluation.scores import sufficiency_contrast

    grid = maps.grid if maps.grid is not None else grid_of(spec.backbone)
    grid_kw = {}
    if maps.grid is not None:
        grid_kw = dict(grid_h=maps.grid[0], grid_w=maps.grid[1], height=IMAGE_SIZE, width=IMAGE_SIZE)
    mask_seed = spec.seed * 1000 + int(round(area * 1000))
    mk, fk = budget_or_floor(maps.k.detach().float().cpu(), area, mask_seed, **grid_kw)
    mk = mk.to(x.device)
    ml = fl = None
    if maps.l is not None:
        ml, fl = budget_or_floor(maps.l.detach().float().cpu(), area, mask_seed + 1, **grid_kw)
        ml = ml.to(x.device)
    m_full = _pick(full, k) - _pick(full, l)
    rows = []
    batch = x.shape[0]
    for operator in spec.operators:
        out_k = _logits(model, remove(x, mk, operator, mask_seed, grid))
        zk_wo_k, zl_wo_k = _pick(out_k, k), _pick(out_k, l)
        cd1 = m_full - (zk_wo_k - zl_wo_k)
        if ml is not None:
            out_l = _logits(model, remove(x, ml, operator, mask_seed, grid))
            zk_wo_l, zl_wo_l = _pick(out_l, k), _pick(out_l, l)
            cd = (zk_wo_l - zl_wo_l) - (zk_wo_k - zl_wo_k)
            sc = sufficiency_contrast(model, x, mk, ml, k, l, seed=mask_seed) if operator == "road" else torch.full_like(cd, float("nan"))
        else:
            zk_wo_l = zl_wo_l = cd = sc = torch.full_like(cd1, float("nan"))
        for i in range(batch):
            rows.append({
                "game": "contrastive", "dataset": spec.dataset, "backbone": spec.backbone,
                "seed": spec.seed, "split": spec.split, "image_index": sample.index[offset + i],
                "method": label, "area": area, "operator": operator,
                "class_k": int(k[i]), "class_l": int(l[i]), "label": int(sample.labels[offset + i]),
                "correct": int(int(k[i]) == int(sample.labels[offset + i])),
                "cd": float(cd[i]), "cd1": float(cd1[i]), "sc": float(sc[i]),
                "z_k_full": float(_pick(full, k)[i]), "z_l_full": float(_pick(full, l)[i]),
                "z_k_without_k": float(zk_wo_k[i]), "z_l_without_k": float(zl_wo_k[i]),
                "z_k_without_l": float(zk_wo_l[i]), "z_l_without_l": float(zl_wo_l[i]),
                "area_k": float(mk[i].float().mean()),
                "area_l": float(ml[i].float().mean()) if ml is not None else float("nan"),
                "failed_row": int(bool(fk[i]) or (fl is not None and bool(fl[i]))),
                "fast": int(spec.knobs.fast), "model_source": spec.model_source,
                "method_code_hash": code_hash, "knobs_hash": knob_hash,
            })
        if label == "cdea" and maps.extra.get("shared") is not None and ml is not None:
            shared = maps.extra["shared"]
            if shared.shape[0] == batch:
                ms, fs = budget_or_floor(shared.detach().float().cpu(), area, mask_seed + 2, **grid_kw)
                ms = ms.to(x.device)
                both_k = (mk + ms).clamp(0, 1)
                both_l = (ml + ms).clamp(0, 1)
                out_k = _logits(model, remove(x, both_k, operator, mask_seed, grid))
                out_l = _logits(model, remove(x, both_l, operator, mask_seed, grid))
                zk_wo_k, zl_wo_k = _pick(out_k, k), _pick(out_k, l)
                zk_wo_l, zl_wo_l = _pick(out_l, k), _pick(out_l, l)
                cd = (zk_wo_l - zl_wo_l) - (zk_wo_k - zl_wo_k)
                cd1 = m_full - (zk_wo_k - zl_wo_k)
                for i in range(batch):
                    rows.append({
                        "game": "contrastive", "dataset": spec.dataset, "backbone": spec.backbone,
                        "seed": spec.seed, "split": spec.split, "image_index": sample.index[offset + i],
                        "method": "cdea+shared", "area": area, "operator": operator,
                        "class_k": int(k[i]), "class_l": int(l[i]), "label": int(sample.labels[offset + i]),
                        "correct": int(int(k[i]) == int(sample.labels[offset + i])),
                        "cd": float(cd[i]), "cd1": float(cd1[i]), "sc": float("nan"),
                        "z_k_full": float(_pick(full, k)[i]), "z_l_full": float(_pick(full, l)[i]),
                        "z_k_without_k": float(zk_wo_k[i]), "z_l_without_k": float(zl_wo_k[i]),
                        "z_k_without_l": float(zk_wo_l[i]), "z_l_without_l": float(zl_wo_l[i]),
                        "area_k": float(both_k[i].float().mean()),
                        "area_l": float(both_l[i].float().mean()),
                        "failed_row": int(bool(fk[i]) or bool(fl[i]) or bool(fs[i])),
                        "fast": int(spec.knobs.fast), "model_source": spec.model_source,
                        "method_code_hash": code_hash, "knobs_hash": knob_hash,
                    })
    return rows


def device_memory_mb(device) -> float:
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        return torch.cuda.memory_allocated(device) / 2**20
    if kind == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return float("nan")


def _peak_mb(device) -> float:
    if getattr(device, "type", str(device)) == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return float("nan")


def _summary(spec, loaded, status, rows, code_hash, knob_hash) -> dict:
    spec_dict = asdict(spec)
    return {
        "cell": spec_dict,
        "model": {"source": loaded.source, "path": loaded.path, "num_classes": loaded.num_classes},
        "methods": status,
        "rows": len(rows),
        "errors": sorted(m for m, s in status.items() if s["status"] == "error"),
        "method_code_hash": code_hash,
        "knobs_hash": knob_hash,
        "fast": bool(spec.knobs.fast),
    }


def _jsonable(extra: dict) -> dict:
    out = {}
    for key, value in extra.items():
        if torch.is_tensor(value):
            out[key] = list(value.shape)
        else:
            out[key] = value
    return out


def _write(spec: CellSpec, rows: list[dict], summary: dict, device) -> Path:
    from core.reporting import save_json

    summary["device_peak_mb"] = _peak_mb(device)
    root = Path(spec.out_dir) if spec.out_dir else CELLS_ROOT
    out = root / spec.split / "contrastive" / spec.dataset / spec.backbone / f"seed{spec.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with gzip.open(out / "records.csv.gz", "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    save_json(out / "summary.json", summary, config_hash=spec.config_hash, device=device)
    return out
