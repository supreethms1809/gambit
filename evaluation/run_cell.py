"""Run and score one paper cell, and write its records (EVAL_PLAN.md section 10).

One row per image × method × area × operator. A method that raises is
recorded as ``status="error"`` with its message, and the cell continues; a
row whose map is non-finite is scored on the random floor and flagged
``failed_row``. Nothing is dropped.
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
import torch.nn.functional as F

from evaluation.run_methods import (
    ABLATIONS,
    CONTRASTIVE_CANDIDATES,
    SHIFT_ABLATIONS,
    SHIFT_CANDIDATES,
    CdeaConfig,
    Knobs,
    PairMaps,
    ShiftConfig,
    ShiftMaps,
    cdea_pair,
    cdea_shift_maps,
    contrastive_method,
    default_backend,
    hypotheses_for,
    shift_method,
)

IMAGE_SIZE = 224
AREA_DEPENDENT = {"extremal", "extremal_class", "extremal_per_env"}
ROAD = dict(iters=24, noise=0.01)


# ---------------------------------------------------------------------------
# Pass counting: the hardware-independent cost unit (EVAL_PLAN section 9).
# ---------------------------------------------------------------------------

class PassCounter:
    """Images through the model's top-level forward, and images backpropagated.

    Methods that call submodules directly (CVE's feature trunk) bypass the
    top-level hook; their counts are a lower bound and say so in the record.
    """

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


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def _blur(x: torch.Tensor, kernel: int = 15) -> torch.Tensor:
    """The same blur baseline as VisionGridUnitSpace (the optimisation operator)."""
    out = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    if out.shape[-2:] != x.shape[-2:]:
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return out


def remove(x: torch.Tensor, mask: torch.Tensor, operator: str, seed: int, offset: int = 0) -> torch.Tensor:
    from evaluation.scores import _remove

    if operator == "road":
        return _remove(x, mask, ROAD["iters"], ROAD["noise"], seed, offset=offset)
    if operator == "blur":
        m = mask.unsqueeze(1) if mask.ndim == 3 else mask
        return x * (1 - m) + _blur(x) * m
    raise ValueError(f"unknown removal operator {operator!r}")


def _logits(model, x):
    with torch.no_grad():
        return model(x)


def _pick(logits, cls):
    return logits.gather(1, cls.view(-1, 1)).squeeze(1)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class CellSpec:
    game: str                      # contrastive | shift
    dataset: str
    backbone: str
    seed: int
    split: str = "val"
    n: int = 1
    methods: Sequence[str] = ()
    ablations: Sequence[str] = ()
    candidates: Sequence[str] = ()     # "method@variant" names for val selection
    areas: Sequence[float] = (0.05,)
    operators: Sequence[str] = ("road", "blur")
    model_source: str = "auto"
    checkpoint: Optional[str] = None
    final: bool = False
    config_hash: Optional[str] = None
    knobs: Knobs = field(default_factory=Knobs)
    out_dir: Optional[str] = None
    image_batch: int = 4          # images on the device at once; 0 keeps the whole sample


def _reset_peak(device) -> None:
    """Peak allocator stats for this cell, so the summary carries its profile."""
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device) -> float:
    if getattr(device, "type", str(device)) == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return float("nan")


def _write(spec: CellSpec, rows: list[dict], summary: dict, device) -> Path:
    from core.reporting import save_json

    summary["device_peak_mb"] = _peak_mb(device)

    # The split is in the path: a test run must never overwrite the val records
    # that selection was made from.
    out = Path(spec.out_dir) / spec.split / spec.game / spec.dataset / spec.backbone / f"seed{spec.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with gzip.open(out / "records.csv.gz", "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    save_json(out / "summary.json", summary, config_hash=spec.config_hash, device=device)
    return out


def _ranges(n: int, image_batch: int) -> list[tuple[int, int]]:
    """``(start, end)`` slices. ``image_batch <= 0`` is the whole sample."""
    if n < 1:
        return []
    step = n if image_batch is None or image_batch <= 0 else min(int(image_batch), n)
    if step < 1:
        raise ValueError("image_batch must be >= 1, or 0 for the whole sample")
    return [(start, min(start + step, n)) for start in range(0, n, step)]


def _whole_sample(method: str, ablation: Optional[str]) -> bool:
    """True when splitting the sample would change the result.

    The random floor and CVE draw from a seed that walks the sample in order.
    SpRAy clusters every image together. The unpaired shift objective's reward
    is the variance of the group means, so a group has to stay intact.
    """
    if method in {"random_floor", "cve", "spray"}:
        return True
    return bool(ablation and "unpaired" in ablation)


def _release(device) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()


def _method_runs(spec: CellSpec) -> list[tuple[str, Optional[str], Optional[str]]]:
    """``(method, ablation, candidate)``: the methods, CDEA per ablation, then each candidate."""
    runs = [(m, None, None) for m in spec.methods]
    runs += [("cdea_shift" if spec.game == "shift" else "cdea", a, None) for a in spec.ablations]
    runs += [(c.split("@", 1)[0], None, c) for c in spec.candidates]
    return runs


_COST_KNOBS = {"extremal_max_iter", "rise_masks", "ig_steps"}


def _candidate_knobs(spec: CellSpec, override: dict) -> "Knobs":
    """Knob overrides for a candidate. Under ``fast`` the cost knobs keep the fast
    values: the smoke run checks that a candidate runs, not what it scores."""
    from dataclasses import replace as _replace

    fields = {k: v for k, v in override.items() if k != "backend"}
    if spec.knobs.fast:
        fields = {k: v for k, v in fields.items() if k not in _COST_KNOBS}
    return _replace(spec.knobs, **fields)


# ---------------------------------------------------------------------------
# Contrastive
# ---------------------------------------------------------------------------

def _contrastive_maps(spec, model, x, h, method, ablation, candidate, area_for_map, knobs, device, loss_scale):
    override = CONTRASTIVE_CANDIDATES[candidate] if candidate else None
    if ablation is not None:
        cfg = ABLATIONS[ablation](CdeaConfig(backend=default_backend(spec.backbone)))
        return cdea_pair(model, x, h, spec.backbone, spec.knobs, device, cfg, loss_scale=loss_scale)
    if isinstance(override, CdeaConfig):
        return cdea_pair(model, x, h, spec.backbone, spec.knobs, device, override, loss_scale=loss_scale)
    fn = contrastive_method(method)
    kwargs = {}
    if method in AREA_DEPENDENT:
        kwargs["area"] = area_for_map
    if method in {"cve"}:
        kwargs["dataset"] = spec.dataset
    if method in {"cve", "random_floor", "rise_margin"}:
        kwargs["seed"] = spec.seed
    return fn(model, x, h, spec.backbone, knobs, device, **kwargs)


def run_contrastive(spec: CellSpec, device) -> Path:
    from baselines.adapter import budget_or_floor
    from evaluation.run_data import ContrastiveSample, contrastive_sample
    from evaluation.run_models import load_cell_model

    loaded = load_cell_model(spec.dataset, spec.backbone, spec.seed, source=spec.model_source,
                             checkpoint=spec.checkpoint, device=device)
    _reset_peak(device)
    model = loaded.model
    sample = contrastive_sample(spec.dataset, spec.split, spec.n, spec.seed,
                                final=spec.final, config_hash=spec.config_hash)
    n = int(sample.x.shape[0])
    counter = PassCounter(model)
    rows: list[dict] = []
    status: dict[str, dict] = {}

    for method, ablation, candidate in _method_runs(spec):
        label = candidate or (method if ablation is None else f"{method}:{ablation}")
        areas = spec.areas if method in AREA_DEPENDENT else (None,)
        ranges = _ranges(n, 0 if _whole_sample(method, ablation) else spec.image_batch)
        for area_for_map in areas:
            counter.reset()
            pending: list[dict] = []
            extra: dict = {}
            failed = None
            trace = ""
            map_seconds = 0.0
            map_forward = 0
            map_backward = 0
            override = CONTRASTIVE_CANDIDATES[candidate] if candidate else None
            knobs = _candidate_knobs(spec, override) if isinstance(override, dict) else spec.knobs
            for start_i, end_i in ranges:
                if len(ranges) > 1:
                    print(f"{spec.dataset}/{spec.backbone} {label} images {start_i}:{end_i}/{n}", flush=True)
                x = sample.x[start_i:end_i].to(device)
                try:
                    h = hypotheses_for(model, x, 5)
                    h = type(h)(ids=h.ids.to(device), mask=h.mask.to(device))
                    k, l = h.ids[:, 0], h.ids[:, 1]
                    full = _logits(model, x)
                    scale = (end_i - start_i) / n
                    before_f, before_b = counter.forward, counter.backward
                    tick = time.perf_counter()
                    maps = _contrastive_maps(spec, model, x, h, method, ablation, candidate,
                                             area_for_map, knobs, device, scale)
                    map_seconds += time.perf_counter() - tick
                    map_forward += counter.forward - before_f
                    map_backward += counter.backward - before_b
                    extra = _jsonable(maps.extra)
                    chunk = ContrastiveSample(x=x, labels=sample.labels[start_i:end_i],
                                               index=list(sample.index[start_i:end_i]))
                    for area in (spec.areas if area_for_map is None else (area_for_map,)):
                        pending += _score_pair(model, x, maps, k, l, full, area, spec, label, chunk,
                                               budget_or_floor, map_seconds, counter, offset=start_i, n_images=n)
                        if label == "cdea" and "shared" in maps.extra:
                            with_shared = PairMaps(k=maps.k + maps.extra["shared"],
                                                   l=maps.l + maps.extra["shared"], grid=maps.grid)
                            pending += _score_pair(model, x, with_shared, k, l, full, area, spec, "cdea+shared",
                                                   chunk, budget_or_floor, map_seconds, counter, offset=start_i,
                                                   n_images=n)
                except Exception as exc:  # recorded, not raised: one method must not sink the cell
                    failed = exc
                    trace = traceback.format_exc(limit=4)
                    break
                finally:
                    del x
                    _release(device)
            if failed is not None:
                status[label] = {"status": "error", "error": f"{type(failed).__name__}: {failed}",
                                 "trace": trace}
                continue
            for row in pending:
                row["seconds_per_image"] = map_seconds / n
                row["forward_images"] = map_forward
                row["backward_images"] = map_backward
            rows += pending
            status[label] = {"status": "ok", "seconds": map_seconds, "forward_images": map_forward,
                             "backward_images": map_backward, "extra": extra,
                             "device_mb_after": device_memory_mb(device)}
            if os.environ.get("GAMBIT_TRACE_MEMORY"):
                print(f"[mem] {label}: {status[label]['device_mb_after']:.0f} MB", flush=True)
    counter.close()
    summary = _summary(spec, loaded, status, rows)
    return _write(spec, rows, summary, device)


def _score_pair(model, x, maps: PairMaps, k, l, full, area, spec, label, sample, budget_or_floor,
                seconds, counter, offset: int = 0, n_images: Optional[int] = None) -> list[dict]:
    grid_kw = {}
    if maps.grid is not None:
        grid_kw = dict(grid_h=maps.grid[0], grid_w=maps.grid[1], height=IMAGE_SIZE, width=IMAGE_SIZE)
    mask_seed = spec.seed * 1000 + int(round(area * 1000))
    budget_kw = dict(grid_kw, offset=offset)
    mk, fk = budget_or_floor(maps.k.detach().float(), area, mask_seed, **budget_kw)
    ml = fl = None
    if maps.l is not None:
        ml, fl = budget_or_floor(maps.l.detach().float(), area, mask_seed + 1, **budget_kw)
    m_full = _pick(full, k) - _pick(full, l)
    rows = []
    for operator in spec.operators:
        out_k = _logits(model, remove(x, mk, operator, mask_seed, offset=offset))
        zk_wo_k, zl_wo_k = _pick(out_k, k), _pick(out_k, l)
        cd1 = m_full - (zk_wo_k - zl_wo_k)
        if ml is not None:
            out_l = _logits(model, remove(x, ml, operator, mask_seed, offset=offset))
            zk_wo_l, zl_wo_l = _pick(out_l, k), _pick(out_l, l)
            cd = (zk_wo_l - zl_wo_l) - (zk_wo_k - zl_wo_k)
        else:
            zk_wo_l = zl_wo_l = cd = torch.full_like(cd1, float("nan"))
        for i in range(x.shape[0]):
            rows.append({
                "game": "contrastive", "dataset": spec.dataset, "backbone": spec.backbone,
                "seed": spec.seed, "split": spec.split, "image_index": sample.index[i],
                "method": label, "area": area, "operator": operator,
                "class_k": int(k[i]), "class_l": int(l[i]), "label": int(sample.labels[i]),
                "correct": int(int(k[i]) == int(sample.labels[i])),
                "cd": float(cd[i]), "cd1": float(cd1[i]),
                "z_k_full": float(_pick(full, k)[i]), "z_l_full": float(_pick(full, l)[i]),
                "z_k_without_k": float(zk_wo_k[i]), "z_l_without_k": float(zl_wo_k[i]),
                "z_k_without_l": float(zk_wo_l[i]), "z_l_without_l": float(zl_wo_l[i]),
                "area_k": float(mk[i].float().mean()),
                "area_l": float(ml[i].float().mean()) if ml is not None else float("nan"),
                "failed_row": int(bool(fk[i]) or (fl is not None and bool(fl[i]))),
                "seconds_per_image": seconds / (x.shape[0] if n_images is None else n_images),
                "forward_images": counter.forward, "backward_images": counter.backward,
                "fast": int(spec.knobs.fast), "model_source": spec.model_source,
            })
    return rows


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------

def _shift_chunk(sample, start: int, end: int, device):
    from evaluation.run_data import ShiftSample

    sl = slice(start, end)
    env = type(sample.env)(xs=[v[sl].to(device) for v in sample.env.xs], env_ids=list(sample.env.env_ids))
    fg = None if sample.foreground is None else sample.foreground[sl].to(device)
    return ShiftSample(x_id=sample.x_id[sl].to(device), env=env, labels=sample.labels[sl],
                       index=list(sample.index[sl]), foreground=fg, notes=sample.notes)


def _shift_maps(spec, model, sample, method, ablation, candidate, area_for_map, device, loss_scale, init_offset):
    override = SHIFT_CANDIDATES[candidate] if candidate else None
    if isinstance(override, ShiftConfig):
        return cdea_shift_maps(model, sample, spec.backbone, spec.knobs, device, override, seed=spec.seed,
                               loss_scale=loss_scale, init_offset=init_offset)
    if isinstance(override, dict):
        knobs = _candidate_knobs(spec, override)
        extra = {"backend": override["backend"]} if "backend" in override else {}
        return shift_method(method)(model, sample, spec.backbone, knobs, device, **extra)
    if ablation is not None:
        cfg = SHIFT_ABLATIONS[ablation](ShiftConfig(backend=default_backend(spec.backbone)))
        if cfg.objective == "unpaired" and spec.dataset != "waterbirds":
            raise ValueError("the unpaired objective (AS1) is defined on Waterbirds only")
        return cdea_shift_maps(model, sample, spec.backbone, spec.knobs, device, cfg, seed=spec.seed,
                               loss_scale=loss_scale, init_offset=init_offset)
    kwargs = {}
    if method in AREA_DEPENDENT:
        kwargs["area"] = area_for_map
    if method in {"spray", "random_floor", "cdea_shift"}:
        kwargs["seed"] = spec.seed
    if method == "cdea_shift":
        kwargs["loss_scale"] = loss_scale
        kwargs["init_offset"] = init_offset
    return shift_method(method)(model, sample, spec.backbone, spec.knobs, device, **kwargs)


def run_shift(spec: CellSpec, device) -> Path:
    from baselines.adapter import budget_or_floor, random_floor
    from evaluation.masks import mass_in
    from evaluation.run_data import shift_sample
    from evaluation.run_models import load_cell_model
    from evaluation.scores import disagreement_reduction, logit_disagreement_reduction

    loaded = load_cell_model(spec.dataset, spec.backbone, spec.seed, source=spec.model_source,
                             checkpoint=spec.checkpoint, device=device)
    _reset_peak(device)
    model = loaded.model
    sample = shift_sample(spec.dataset, spec.split, spec.n, spec.seed,
                          final=spec.final, config_hash=spec.config_hash)
    n = int(sample.x_id.shape[0])
    counter = PassCounter(model)
    rows: list[dict] = []
    status: dict[str, dict] = {}

    for method, ablation, candidate in _method_runs(spec):
        label = candidate or (method if ablation is None else f"{method}:{ablation}")
        areas = spec.areas if method in AREA_DEPENDENT else (None,)
        ranges = _ranges(n, 0 if _whole_sample(method, ablation) else spec.image_batch)
        for area_for_map in areas:
            counter.reset()
            pending: list[dict] = []
            extra: dict = {}
            failed = None
            trace = ""
            map_seconds = 0.0
            map_forward = 0
            map_backward = 0
            for start_i, end_i in ranges:
                if len(ranges) > 1:
                    print(f"{spec.dataset}/{spec.backbone} {label} images {start_i}:{end_i}/{n}", flush=True)
                chunk = None
                try:
                    chunk = _shift_chunk(sample, start_i, end_i, device)
                    # The predicted class on the full image, not the label (EVAL_PLAN 5.1).
                    y = _logits(model, chunk.x_id).argmax(-1)
                    before_f, before_b = counter.forward, counter.backward
                    tick = time.perf_counter()
                    scale = (end_i - start_i) / n
                    maps = _shift_maps(spec, model, chunk, method, ablation, candidate, area_for_map,
                                       device, scale, start_i)
                    map_seconds += time.perf_counter() - tick
                    map_forward += counter.forward - before_f
                    map_backward += counter.backward - before_b
                    extra = _jsonable(maps.extra)
                    grid_kw = {}
                    if maps.grid is not None:
                        grid_kw = dict(grid_h=maps.grid[0], grid_w=maps.grid[1], height=IMAGE_SIZE, width=IMAGE_SIZE)
                    budget_kw = dict(grid_kw, offset=start_i)
                    for area in (spec.areas if area_for_map is None else (area_for_map,)):
                        mask_seed = spec.seed * 1000 + int(round(area * 1000))
                        sho, failed_row = budget_or_floor(maps.shortcut.detach().float(), area, mask_seed, **budget_kw)
                        rob, _ = budget_or_floor(maps.robust.detach().float(), area, mask_seed + 2, **budget_kw)
                        rand = random_floor(maps.shortcut.detach().float(), area, mask_seed + 1, **budget_kw)
                        oods = list(chunk.env.xs[1:])
                        ld = logit_disagreement_reduction(model, chunk.x_id, oods, y, sho, rand,
                                                          seed=mask_seed, offset=start_i, **ROAD)
                        pd = disagreement_reduction(model, chunk.x_id, oods, y, sho, rand,
                                                    seed=mask_seed, offset=start_i, **ROAD)
                        fg = chunk.foreground
                        for i in range(chunk.x_id.shape[0]):
                            pending.append({
                                "game": "shift", "dataset": spec.dataset, "backbone": spec.backbone,
                                "seed": spec.seed, "split": spec.split, "image_index": chunk.index[i],
                                "method": label, "area": area, "operator": "road",
                                "pred": int(y[i]), "label": int(chunk.labels[i]),
                                "logit_delta_d": float(ld[i]), "prob_delta_d": float(pd[i]),
                                "shortcut_on_background": float(mass_in(sho[i:i + 1], 1 - fg[i:i + 1])) if fg is not None else float("nan"),
                                "robust_on_foreground": float(mass_in(rob[i:i + 1], fg[i:i + 1])) if fg is not None else float("nan"),
                                "area_shortcut": float(sho[i].float().mean()),
                                "failed_row": int(bool(failed_row[i])),
                                "seconds_per_image": 0.0,
                                "forward_images": 0, "backward_images": 0,
                                "fast": int(spec.knobs.fast), "model_source": spec.model_source,
                            })
                except Exception as exc:
                    failed = exc
                    trace = traceback.format_exc(limit=4)
                    break
                finally:
                    del chunk
                    _release(device)
            if failed is not None:
                status[label] = {"status": "error", "error": f"{type(failed).__name__}: {failed}",
                                 "trace": trace}
                continue
            for row in pending:
                row["seconds_per_image"] = map_seconds / n
                row["forward_images"] = map_forward
                row["backward_images"] = map_backward
            rows += pending
            status[label] = {"status": "ok", "seconds": map_seconds, "forward_images": map_forward,
                             "backward_images": map_backward, "extra": extra,
                             "device_mb_after": device_memory_mb(device)}
            if os.environ.get("GAMBIT_TRACE_MEMORY"):
                print(f"[mem] {label}: {status[label]['device_mb_after']:.0f} MB", flush=True)
    counter.close()
    summary = _summary(spec, loaded, status, rows)
    summary["sample_notes"] = sample.notes
    return _write(spec, rows, summary, device)


# ---------------------------------------------------------------------------

def device_memory_mb(device) -> float:
    """Live tensor memory on the device, for spotting leaks across methods."""
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        return torch.cuda.memory_allocated(device) / 2**20
    if kind == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return float("nan")


def _summary(spec: CellSpec, loaded, status: dict, rows: list[dict]) -> dict:
    spec_dict = asdict(spec)
    return {
        "cell": spec_dict,
        "model": {"source": loaded.source, "path": loaded.path, "num_classes": loaded.num_classes},
        "methods": status,
        "rows": len(rows),
        "errors": sorted(m for m, s in status.items() if s["status"] == "error"),
    }


def _jsonable(extra: dict) -> dict:
    out = {}
    for key, value in extra.items():
        if torch.is_tensor(value):
            out[key] = list(value.shape)
        else:
            out[key] = value
    return out
