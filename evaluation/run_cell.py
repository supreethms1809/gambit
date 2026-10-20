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

from evaluation.removal import ROAD
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
PAPER_ROOT = Path(__file__).resolve().parents[1] / "results" / "paper"
CELLS_ROOT = PAPER_ROOT / "cells"
FAST_ROOT = PAPER_ROOT / "smoke" / "fast_knobs"


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


def remove(x: torch.Tensor, mask: torch.Tensor, operator: str, seed: int, grid: tuple[int, int],
           offset: int = 0) -> torch.Tensor:
    from core.grid import deletion_baseline
    from evaluation.scores import _remove

    if operator == "road":
        return _remove(x, mask, ROAD["iters"], ROAD["noise"], seed, offset)
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
    # offset: tie-breaks and ROAD noise follow the image's place in the sample,
    # so a chunked method and a whole-sample method score an image identically.
    mk, fk = budget_or_floor(maps.k.detach().float().cpu(), area, mask_seed, offset=offset, **grid_kw)
    mk = mk.to(x.device)
    ml = fl = None
    if maps.l is not None:
        ml, fl = budget_or_floor(maps.l.detach().float().cpu(), area, mask_seed + 1, offset=offset, **grid_kw)
        ml = ml.to(x.device)
    m_full = _pick(full, k) - _pick(full, l)
    stages = {}
    if "payoff_ids" in maps.extra and maps.grid is not None and ml is not None:
        stages = _stage_payoffs(model, x, maps, mk, ml, area, grid)
    stages.update(_edit_columns(maps, area))
    rows = []
    batch = x.shape[0]
    for operator in spec.operators:
        out_k = _logits(model, remove(x, mk, operator, mask_seed, grid, offset))
        zk_wo_k, zl_wo_k = _pick(out_k, k), _pick(out_k, l)
        cd1 = m_full - (zk_wo_k - zl_wo_k)
        if ml is not None:
            out_l = _logits(model, remove(x, ml, operator, mask_seed, grid, offset))
            zk_wo_l, zl_wo_l = _pick(out_l, k), _pick(out_l, l)
            cd = (zk_wo_l - zl_wo_l) - (zk_wo_k - zl_wo_k)
            sc = sufficiency_contrast(model, x, mk, ml, k, l, seed=mask_seed, offset=offset) if operator == "road" else torch.full_like(cd, float("nan"))
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
                **{key: float(value[i]) for key, value in stages.items()},
            })
        if label == "cdea" and maps.extra.get("shared") is not None and ml is not None:
            shared = maps.extra["shared"]
            if shared.shape[0] == batch:
                ms, fs = budget_or_floor(shared.detach().float().cpu(), area, mask_seed + 2, offset=offset, **grid_kw)
                ms = ms.to(x.device)
                both_k = (mk + ms).clamp(0, 1)
                both_l = (ml + ms).clamp(0, 1)
                out_k = _logits(model, remove(x, both_k, operator, mask_seed, grid, offset))
                out_l = _logits(model, remove(x, both_l, operator, mask_seed, grid, offset))
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


def _edit_columns(maps: PairMaps, area: float) -> dict[str, torch.Tensor]:
    """CVE and SC-CVE: edited cells, their share of the area-a budget, and whether the class flipped.

    Their map ranks the edited cells by edit order and leaves the rest at 0. When
    the flip takes fewer than a·R edits, the shared top-a conversion fills the
    rest of area a with seeded untouched cells. The mask keeps area a, and
    ``edit_share`` is the part of it that is the search's own edits.
    """
    flipped = maps.extra.get("cve_flipped", maps.extra.get("sc_cve_flipped"))
    if flipped is None:
        return {}
    budget = float(area) * maps.k.shape[-1]
    edits = (maps.k.detach().float().cpu() > 0).sum(dim=-1).float()
    return {
        "edit_cells": edits,
        "edit_share": edits.clamp(max=budget) / budget,
        "flipped": torch.tensor([float(bool(f)) for f in flipped]),
    }


def _stage_payoffs(model, x, maps: PairMaps, mk, ml, area, grid) -> dict[str, torch.Tensor]:
    """Inputs to D2 for an allocation, ranks 0 and 1, all under blur deletion.

    ``soft`` is the payoff the allocation optimised (the plan, nearest cells).
    ``hard`` is its top ``a * R`` cells. ``scored_blur`` is the mask the scorer
    uses. Each is the drop in the player's unique log-odds over the hypotheses
    the allocation used.
    """
    from cdea.first_order import top_mass
    from cdea.payoffs import unique_log_odds
    from core.grid import delete

    ids = maps.extra["payoff_ids"].to(x.device).long()
    gh, gw = grid
    budget = float(area) * gh * gw
    reference = unique_log_odds(_logits(model, x), ids)
    out = {}
    for rank, name, plan, scored in ((0, "k", maps.k, mk), (1, "l", maps.l, ml)):
        plan = plan.detach().to(device=x.device, dtype=x.dtype)
        views = {
            "soft": delete(x, plan, gh, gw),
            "hard": delete(x, top_mass(plan, budget), gh, gw),
            "scored_blur": remove(x, scored, "blur", 0, grid),
        }
        for stage, view in views.items():
            dropped = unique_log_odds(_logits(model, view), ids)
            out[f"payoff_{stage}_{name}"] = (reference[:, rank] - dropped[:, rank]).detach().float().cpu()
    return out


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


def run_shift(spec: CellSpec, device) -> Path:
    """Score every requested shift method. Records go to ``shift.csv.gz``."""
    from baselines.adapter import budget_or_floor, random_floor
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_data import shift_sample
    from evaluation.run_methods import SHIFT_CANDIDATES, SHIFT_WHOLE_SAMPLE, shift_ablation, shift_method
    from evaluation.run_models import load_cell_model
    from evaluation.scores import (
        null_pixel_masks,
        null_shift_drops,
        shift_drop_matrix,
        shift_scores,
        shortcut_patch_contrast,
        transplant_closure,
    )
    from evaluation.masks import mass_in, top_fraction_mask

    loaded = load_cell_model(
        spec.dataset, spec.backbone, spec.seed, source=spec.model_source,
        checkpoint=spec.checkpoint, device=device,
    )
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = loaded.model
    sample = shift_sample(
        spec.dataset, spec.split, spec.n, spec.seed,
        final=spec.final, config_hash=spec.config_hash,
    )
    xs_cpu = sample.env.xs
    n = int(xs_cpu[0].shape[0])
    counter = PassCounter(model)
    code_hash = method_code_hash()
    knob_hash = knobs_hash(spec.knobs)
    rows: list[dict] = []
    status: dict[str, dict] = {}
    runs = [(method, method, None) for method in spec.methods]
    runs += [("cdea_shift", f"cdea_shift:{ablation}", shift_ablation(ablation)) for ablation in spec.ablations]
    runs += [("cdea_shift", candidate, SHIFT_CANDIDATES[candidate]) for candidate in spec.candidates]
    grid_h, grid_w = __import__("core.grid", fromlist=["grid_of"]).grid_of(spec.backbone)
    area_dependent = {"cdea_shift", "shift_assign", "extremal_shift"}

    for method, label, cfg in runs:
        areas = spec.areas if method in area_dependent else (None,)
        whole = method in SHIFT_WHOLE_SAMPLE
        ranges = [(0, n)] if whole else _ranges(n, spec.image_batch)
        for area_for_map in areas:
            counter.reset()
            start = time.perf_counter()
            failed = None
            chunk_rows: list[dict] = []
            try:
                for start_i, end_i in ranges:
                    views = [view[start_i:end_i].to(device) for view in xs_cpu]
                    if cfg == "precomputed":
                        raise FileNotFoundError("precomputed maps need a directory")
                    fn = shift_method(method)
                    kwargs = {}
                    if method in area_dependent:
                        kwargs["area"] = area_for_map if area_for_map is not None else spec.areas[0]
                    if method == "cdea_shift":
                        kwargs["cfg"] = cfg
                    if method in {"spray", "random_floor"}:
                        kwargs["seed"] = spec.seed
                    maps = fn(model, views, spec.backbone, spec.knobs, device, **kwargs)
                    with torch.no_grad():
                        y = model(views[0]).argmax(dim=-1)
                    for area in (spec.areas if area_for_map is None else (area_for_map,)):
                        chunk_rows += _score_shift(
                            model, views, maps, y, area, spec, label, sample, start_i,
                            code_hash, knob_hash, budget_or_floor, random_floor,
                            null_pixel_masks, null_shift_drops, shift_drop_matrix, shift_scores,
                            transplant_closure, shortcut_patch_contrast, mass_in, top_fraction_mask,
                            grid_h, grid_w,
                        )
            except Exception as exc:
                failed = exc
            seconds = time.perf_counter() - start
            if failed is not None:
                status[label] = {"status": "error", "error": f"{type(failed).__name__}: {failed}",
                                 "trace": traceback.format_exc(limit=4)}
                continue
            status[label] = {"status": "ok", "seconds": seconds, "forward_images": counter.forward,
                             "backward_images": counter.backward,
                             "device_mb_after": device_memory_mb(device)}
            for row in chunk_rows:
                row["seconds_per_image"] = seconds / n
                row["forward_images"] = counter.forward
                row["backward_images"] = counter.backward
            rows += chunk_rows
    counter.close()
    summary = {
        "game": "shift",
        "dataset": spec.dataset,
        "backbone": spec.backbone,
        "seed": spec.seed,
        "split": spec.split,
        "n": n,
        "fast": bool(spec.knobs.fast),
        "model_source": loaded.source,
        "method_code_hash": code_hash,
        "knobs_hash": knob_hash,
        "methods": status,
        "rows": len(rows),
        "errors": sorted(name for name, item in status.items() if item["status"] == "error"),
    }
    return _write(spec, rows, summary, device)


def _score_shift(
    model, views, maps, y, area, spec, label, sample, start_i,
    code_hash, knob_hash, budget_or_floor, random_floor,
    null_pixel_masks, null_shift_drops, shift_drop_matrix, shift_scores,
    transplant_closure, shortcut_patch_contrast, mass_in, top_fraction_mask,
    grid_h, grid_w,
) -> list[dict]:
    from baselines.adapter import row_constant

    device = views[0].device
    height, width = views[0].shape[-2:]
    grid = maps.grid
    kwargs = dict(
        grid_h=None if grid is None else grid[0],
        grid_w=None if grid is None else grid[1],
        height=height, width=width,
    )
    # start_i: tie-breaks and ROAD noise follow the image's place in the sample.
    shortcut, failed_s = budget_or_floor(maps.shortcut.detach().float().cpu(), area, spec.seed, offset=start_i, **kwargs)
    robust, failed_r = budget_or_floor(maps.robust.detach().float().cpu(), area, spec.seed + 1, offset=start_i, **kwargs)
    constant = row_constant(maps.shortcut.detach().float().cpu()) | row_constant(maps.robust.detach().float().cpu())
    shortcut = shortcut.to(device)
    robust = robust.to(device)
    random_mask = random_floor(maps.shortcut.detach().float().cpu(), area, spec.seed + 2, offset=start_i, **kwargs).to(device)
    null_masks = null_pixel_masks(grid_h, grid_w, area, height, width)
    null, null_prob = null_shift_drops(
        model, views, y, null_masks, ROAD["iters"], ROAD["noise"], spec.seed, offset=start_i,
    )
    stacked = torch.stack([robust, shortcut], dim=1)
    drops, prob_drops = shift_drop_matrix(
        model, views, y, stacked, ROAD["iters"], ROAD["noise"], spec.seed, offset=start_i,
    )
    scores = shift_scores(drops, null, prob_drops, null_prob)
    if grid is not None:
        units = top_fraction_mask(maps.shortcut.detach().float().cpu(), area, spec.seed, offset=start_i).to(device)
    else:
        units = _units_from_pixels(shortcut, grid_h, grid_w)
    closure = transplant_closure(model, views, y, units, grid_h, grid_w)
    rolled = torch.roll(shortcut, shifts=max(height // grid_h, 1), dims=-1)
    rolled_drops, _ = shift_drop_matrix(
        model, views, y, rolled.unsqueeze(1), ROAD["iters"], ROAD["noise"], spec.seed, offset=start_i,
    )
    odds_rolled = rolled_drops[:, 0, 1] - null[:, 1]
    extra = getattr(maps, "extra", None) or {}
    stages = _shift_stage_payoffs(model, views, maps, y, area, grid_h, grid_w) if "robust_payoff" in extra else {}
    foreground = None if sample.foreground is None else sample.foreground[start_i:start_i + shortcut.shape[0]].to(device)
    rows = []
    for i in range(shortcut.shape[0]):
        row = {
            "game": "shift",
            "dataset": spec.dataset,
            "backbone": spec.backbone,
            "seed": spec.seed,
            "split": spec.split,
            "method": label,
            "area": area,
            "image_index": sample.index[start_i + i],
            "label": int(sample.labels[start_i + i]),
            "pred": int(y[i]),
            "correct": int(int(y[i]) == int(sample.labels[start_i + i])),
            "d_r_rob": float(drops[i, 0, 0]),
            "d_r_gap": float(drops[i, 0, 1]),
            "d_s_rob": float(drops[i, 1, 0]),
            "d_s_gap": float(drops[i, 1, 1]),
            "null_rob": float(null[i, 0]),
            "null_gap": float(null[i, 1]),
            "delta_d": float(scores["delta_d"][i]),
            "delta_d_prob": float(scores["delta_d_prob"][i]),
            "rn": float(scores["rn"][i]),
            "pur_r": float(scores["pur_r"][i]),
            "pur_s": float(scores["pur_s"][i]),
            "role_sep": float(scores["role_sep"][i]),
            "tc": float(closure[i]),
            "delta_d_translated": float(odds_rolled[i]),
            "area_r": float(robust[i].float().mean()),
            "area_s": float(shortcut[i].float().mean()),
            "failed_row": int(bool(failed_s[i] or failed_r[i])),
            "constant_row": int(bool(constant[i])),
            "method_code_hash": code_hash,
            "knobs_hash": knob_hash,
            "fast": int(spec.knobs.fast),
            "model_source": spec.model_source,
        }
        row.update({name: float(value[i]) for name, value in stages.items()})
        if sample.mask_a is not None:
            row["spc"] = float(shortcut_patch_contrast(
                shortcut[i:i + 1],
                sample.mask_a[start_i + i:start_i + i + 1].to(device),
                sample.mask_b[start_i + i:start_i + i + 1].to(device),
            )[0])
        if foreground is not None:
            row["mass_shortcut_background"] = float(mass_in(shortcut[i:i + 1], 1.0 - foreground[i:i + 1])[0])
            row["mass_robust_foreground"] = float(mass_in(robust[i:i + 1], foreground[i:i + 1])[0])
            row["mass_random_background"] = float(mass_in(random_mask[i:i + 1], 1.0 - foreground[i:i + 1])[0])
        rows.append(row)
    return rows


def _shift_stage_payoffs(model, views, maps, y, area: float, grid_h: int, grid_w: int) -> dict[str, torch.Tensor]:
    """D2, D3 and D9 inputs for a shift allocation: soft vs hard payoffs, overshoot, and same_as_init."""
    from cdea.first_order import top_mass
    from cdea.shift import deletion_shortcut_payoff, log_odds, robust_payoff, transplant_payoff

    device = views[0].device
    dtype = views[0].dtype
    budget = float(area) * grid_h * grid_w
    extra = maps.extra
    rob_mode = extra.get("robust_mode", "min")
    short_mode = extra.get("shortcut_mode", "transplant")
    plan_r = maps.robust.detach().to(device=device, dtype=dtype)
    plan_s = maps.shortcut.detach().to(device=device, dtype=dtype)
    hard_r = top_mass(plan_r, budget)
    hard_s = top_mass(plan_s, budget)
    with torch.no_grad():
        reference = [log_odds(model(v), y) for v in views]
        soft_r = extra["robust_payoff"].detach().float().cpu()
        soft_s = extra["shortcut_payoff"].detach().float().cpu()
        hard_pay_r = robust_payoff(model, list(views), y, hard_r, grid_h, grid_w, rob_mode, reference).detach().float().cpu()
        if short_mode == "deletion":
            hard_pay_s = deletion_shortcut_payoff(model, list(views), y, hard_s, grid_h, grid_w, reference).detach().float().cpu()
        else:
            hard_pay_s = transplant_payoff(model, list(views), y, hard_s, grid_h, grid_w, reference).detach().float().cpu()
    out = {
        "payoff_soft_r": soft_r,
        "payoff_hard_r": hard_pay_r,
        "payoff_soft_s": soft_s,
        "payoff_hard_s": hard_pay_s,
    }
    if extra.get("overshoot") is not None:
        out["overshoot_share"] = extra["overshoot"].detach().float().cpu()
    initial = extra.get("initial")
    if initial is not None:
        top = max(1, int(round(budget)))
        plan = torch.stack([plan_r, plan_s], dim=1)
        final_top = plan.topk(top, dim=-1).indices.sort(dim=-1).values
        init_top = initial.to(device).topk(top, dim=-1).indices.sort(dim=-1).values
        out["same_as_init"] = (final_top == init_top).all(dim=-1).all(dim=-1).float().cpu()
    return out


def _units_from_pixels(mask: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    pooled = torch.nn.functional.adaptive_avg_pool2d(mask.unsqueeze(1), (grid_h, grid_w))
    return (pooled.flatten(1) > 0.5).to(dtype=mask.dtype)


DECOMP_PLAYERS = ("uk", "ul", "s")
DECOMP_PATCHES = ("ak", "al", "c")


def _decomp_runs(spec: CellSpec) -> list[tuple[str, Optional[str]]]:
    runs = [(m, None) for m in spec.methods] + [("cdea_decomp", a) for a in spec.ablations]
    return runs + [("cdea_decomp", c) for c in spec.candidates]


def _decomp_area_dependent(method: str, ablation: Optional[str]) -> bool:
    from evaluation.run_methods import DECOMP_AREA_DEPENDENT

    return method in DECOMP_AREA_DEPENDENT


def _decomp_maps(spec, model, x, h, method, ablation, area, knobs, device):
    from evaluation.run_methods import DECOMP_ABLATIONS, DECOMP_CANDIDATES, DecompConfig, decomposition_method

    fn = decomposition_method(method)
    kwargs = {}
    if ablation in DECOMP_CANDIDATES:
        kwargs["cfg"] = DECOMP_CANDIDATES[ablation]
    elif ablation is not None:
        kwargs["cfg"] = DECOMP_ABLATIONS[ablation](DecompConfig())
    if _decomp_area_dependent(method, ablation):
        kwargs["area"] = area
    if method in {"random_decomp", "rise_minsplit"}:
        kwargs["seed"] = spec.seed
    return fn(model, x, h, spec.backbone, knobs, device, **kwargs)


def run_decomposition(spec: CellSpec, device) -> Path:
    """Score every requested decomposition method. Records go to ``records.csv.gz`` under ``decomposition``."""
    from baselines.adapter import budget_or_floor
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_data import decomposition_sample
    from evaluation.run_methods import DECOMP_WHOLE_SAMPLE
    from evaluation.run_models import load_cell_model

    loaded = load_cell_model(spec.dataset, spec.backbone, spec.seed, source=spec.model_source,
                             checkpoint=spec.checkpoint, device=device)
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = loaded.model
    sample = decomposition_sample(spec.dataset, spec.split, spec.n, spec.seed,
                                  final=spec.final, config_hash=spec.config_hash, model=model, device=device)
    n = int(sample.x.shape[0])
    counter = PassCounter(model)
    rows: list[dict] = []
    status: dict[str, dict] = {}
    code_hash = method_code_hash()
    knob_hash = knobs_hash(spec.knobs)

    for method, ablation in _decomp_runs(spec):
        label = method if ablation is None else (ablation if "@" in ablation else f"{method}:{ablation}")
        areas = spec.areas if _decomp_area_dependent(method, ablation) else (None,)
        ranges = _ranges(n, 0 if method in DECOMP_WHOLE_SAMPLE else spec.image_batch)
        for area_for_map in areas:
            counter.reset()
            start = time.perf_counter()
            failed = None
            chunk_rows: list[dict] = []
            try:
                for start_i, end_i in ranges:
                    x = sample.x[start_i:end_i].to(device)
                    h = hypotheses_for(model, x, 2)
                    h = type(h)(ids=h.ids.to(device), mask=h.mask.to(device))
                    maps = _decomp_maps(spec, model, x, h, method, ablation, area_for_map, spec.knobs, device)
                    for area in (spec.areas if area_for_map is None else (area_for_map,)):
                        chunk_rows += _score_decomp(model, x, maps, h, area, spec, label, sample,
                                                    budget_or_floor, start_i, code_hash, knob_hash)
            except Exception as exc:
                failed = exc
            seconds = time.perf_counter() - start
            if failed is not None:
                status[label] = {"status": "error", "error": f"{type(failed).__name__}: {failed}",
                                 "trace": traceback.format_exc(limit=4)}
                continue
            status[label] = {"status": "ok", "seconds": seconds, "forward_images": counter.forward,
                             "backward_images": counter.backward,
                             "device_mb_after": device_memory_mb(device)}
            for row in chunk_rows:
                row["seconds_per_image"] = seconds / n
                row["forward_images"] = counter.forward
                row["backward_images"] = counter.backward
            rows += chunk_rows
    counter.close()
    summary = _summary(spec, loaded, status, rows, code_hash, knob_hash)
    summary["sample"] = dict(sample.notes)
    if sample.roles is not None:
        summary["sample"]["roles_confirmed"] = [int(v) for v in sample.roles.sum(dim=0)]
        summary["sample"]["all_roles_confirmed"] = int(sample.roles.all(dim=1).sum())
    return _write(spec, rows, summary, device)


def _score_decomp(model, x, maps, h, area, spec, label, sample, budget_or_floor, offset,
                  code_hash, knob_hash) -> list[dict]:
    from baselines.adapter import row_constant
    from core.grid import grid_of
    from evaluation.scores import decomposition_scores, drop_matrix, mask_patch_confusion, null_drops_scored, null_pixel_masks

    grid = grid_of(spec.backbone)
    grid_kw = {}
    if maps.grid is not None:
        grid_kw = dict(grid_h=maps.grid[0], grid_w=maps.grid[1], height=x.shape[-2], width=x.shape[-1])
    mask_seed = spec.seed * 1000 + int(round(area * 1000))
    sources = (maps.k, maps.l, maps.shared)
    masks, failed, constant = [], [], []
    for index, source in enumerate(sources):
        if source is None:
            masks.append(torch.full(x.shape[:1] + x.shape[-2:], float("nan"), device=x.device))
            failed.append(torch.zeros(x.shape[0], dtype=torch.bool))
            constant.append(torch.zeros(x.shape[0], dtype=torch.bool))
            continue
        m, f = budget_or_floor(source.detach().float().cpu(), area, mask_seed + index, offset=offset, **grid_kw)
        masks.append(m.to(x.device))
        failed.append(f)
        constant.append(row_constant(source.detach().float().cpu()))
    stacked = torch.stack(masks, dim=1)
    ids = h.ids[:, :2].long()
    k, l = ids[:, 0], ids[:, 1]
    full = _logits(model, x)
    null_masks = null_pixel_masks(grid[0], grid[1], area, x.shape[-2], x.shape[-1])
    stages = _decomp_stage_payoffs(model, x, maps, stacked, ids, area, grid) if "null_blur" in maps.extra else {}
    patches = getattr(sample, "patches", None)
    confusion = None
    if patches is not None:
        safe = torch.nan_to_num(stacked, nan=0.0)
        precision, recall = mask_patch_confusion(safe, patches[offset:offset + x.shape[0]].to(x.device))
        # A player the method does not have (margin_gradcam_ref's S) has no
        # precision or recall, not a zero one.
        absent = [p for p in range(3) if sources[p] is None]
        precision[:, absent] = float("nan")
        recall[:, absent] = float("nan")
        confusion = (precision, recall)
    rows = []
    for operator in spec.operators:
        def remove_op(images, mask, operator=operator):
            return remove(images, mask, operator, mask_seed, grid, offset)

        present = [p for p in range(3) if sources[p] is not None]
        drops = torch.full((x.shape[0], 3, 2), float("nan"), device=x.device)
        drops[:, present] = drop_matrix(model, x, stacked[:, present], ids, remove=remove_op)
        null = null_drops_scored(model, x, ids, null_masks, remove_op)
        scores = decomposition_scores(drops, null)
        for i in range(x.shape[0]):
            row = {
                "game": "decomposition", "dataset": spec.dataset, "backbone": spec.backbone,
                "seed": spec.seed, "split": spec.split, "image_index": sample.index[offset + i],
                "method": label, "area": area, "operator": operator,
                "class_k": int(k[i]), "class_l": int(l[i]), "label": int(sample.labels[offset + i]),
                "correct": int(int(k[i]) == int(sample.labels[offset + i])),
                "z_k_full": float(_pick(full, k)[i]), "z_l_full": float(_pick(full, l)[i]),
                "null_k": float(null[i, 0]), "null_l": float(null[i, 1]),
                "failed_row": int(any(bool(f[i]) for f in failed)),
                "constant_row": int(any(bool(c[i]) for c in constant)),
                "fast": int(spec.knobs.fast), "model_source": spec.model_source,
                "method_code_hash": code_hash, "knobs_hash": knob_hash,
            }
            for p, player in enumerate(DECOMP_PLAYERS):
                row[f"d_{player}_k"] = float(drops[i, p, 0])
                row[f"d_{player}_l"] = float(drops[i, p, 1])
                row[f"area_{player}"] = float(stacked[i, p].float().mean())
            row.update({name: float(value[i]) for name, value in scores.items()})
            row.update({name: float(value[i]) for name, value in stages.items()})
            if getattr(sample, "roles", None) is not None:
                for t, target in enumerate(DECOMP_PATCHES):
                    row[f"role_{target}"] = int(sample.roles[offset + i, t])
            if confusion is not None:
                precision, recall = confusion
                for p, player in enumerate(DECOMP_PLAYERS):
                    for t, target in enumerate(DECOMP_PATCHES + ("rest",)):
                        row[f"prec_{player}_{target}"] = float(precision[i, p, t])
                    for t, target in enumerate(DECOMP_PATCHES):
                        row[f"rec_{player}_{target}"] = float(recall[i, p, t])
            rows.append(row)
    return rows


def _decomp_stage_payoffs(model, x, maps, scored, ids, area, grid) -> dict[str, torch.Tensor]:
    """D2, D9 and trapped-support inputs for a decomposition plan, under blur, with CDEA's blur null.

    ``soft``: the payoff of the plan rows. ``hard``: of their top a·R cells.
    ``scored_blur``: of the scored pixel masks. ``same_as_init``: 1 when every
    player's top cells equal those of the step-0 plan (D9). ``outside_prior``:
    the share of final top cells where the min-split prior was 0.
    """
    from cdea.first_order import top_mass
    from cdea.payoffs import class_drops, decomposition_payoffs
    from core.grid import delete

    gh, gw = grid
    budget = float(area) * gh * gw
    null = maps.extra["null_blur"].to(device=x.device, dtype=x.dtype)
    plan = torch.stack([maps.k, maps.l, maps.shared], dim=1).detach().to(device=x.device, dtype=x.dtype)
    hard = torch.stack([top_mass(plan[:, p], budget) for p in range(3)], dim=1)
    full = _logits(model, x)
    out = {}
    views = {
        "soft": [delete(x, plan[:, p], gh, gw) for p in range(3)],
        "hard": [delete(x, hard[:, p], gh, gw) for p in range(3)],
        "scored_blur": [remove(x, scored[:, p], "blur", 0, grid) for p in range(3)],
    }
    for stage, edited in views.items():
        logits = torch.stack([_logits(model, view) for view in edited], dim=1)
        payoff = decomposition_payoffs(class_drops(full, logits, ids), null)
        for p, player in enumerate(DECOMP_PLAYERS):
            out[f"payoff_{stage}_{player}"] = payoff[:, p].detach().float().cpu()
    top = max(1, int(round(budget)))
    final_top = plan.topk(top, dim=-1).indices.sort(dim=-1).values
    initial = maps.extra.get("initial")
    if initial is not None:
        init_top = initial.to(x.device).topk(top, dim=-1).indices.sort(dim=-1).values
        out["same_as_init"] = (final_top == init_top).all(dim=-1).all(dim=-1).float().cpu()
    prior = maps.extra.get("prior_split")
    if prior is not None:
        outside = prior.to(x.device).gather(-1, final_top) <= 1e-12
        out["outside_prior"] = outside.float().mean(dim=(-1, -2)).cpu()
    return out


def records_root(spec: CellSpec) -> Path:
    """The root a cell writes under. Fast-knob records never share a path with paper records.

    Without ``out_dir``, a fast cell writes under ``results/paper/smoke/fast_knobs``.
    A fast cell asked to write anywhere else under ``results/paper`` raises,
    because it would overwrite the paper cell at the same path.
    """
    if not spec.out_dir:
        return FAST_ROOT if spec.knobs.fast else CELLS_ROOT
    root = Path(spec.out_dir)
    resolved = root.resolve()
    if spec.knobs.fast and resolved.is_relative_to(PAPER_ROOT) and not resolved.is_relative_to(PAPER_ROOT / "smoke"):
        raise ValueError(f"fast-knob records would overwrite paper records under {root}; "
                         "write them under results/paper/smoke or outside results/paper")
    return root


def _write(spec: CellSpec, rows: list[dict], summary: dict, device) -> Path:
    from core.reporting import save_json

    summary["device_peak_mb"] = _peak_mb(device)
    root = records_root(spec)
    out = root / spec.split / spec.game / spec.dataset / spec.backbone / f"seed{spec.seed}"
    out.mkdir(parents=True, exist_ok=True)
    name = "shift.csv.gz" if spec.game == "shift" else "records.csv.gz"
    if rows:
        keys = sorted({k for r in rows for k in r})
        with gzip.open(out / name, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    save_json(out / "summary.json", summary, config_hash=spec.config_hash, device=device)
    return out
