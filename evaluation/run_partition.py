"""Run and score one version-2 decomposition cell.

FORMULATION_DECOMPOSITION.md version 2 sections 5, 6 and 8; EVAL_PLAN.md 13.5-13.8.

Batch by batch: the reference evidence and Omega are computed once and every
method divides that Omega. The scorer's null, its 95th percentile and the
single-cell drops are measured once per image and shared by every method.
One row per image x method x operator, under ``partition``. A method
that raises is recorded as ``status="error"`` and the cell continues.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import torch

from core.types import HypothesisSet
from evaluation.run_cell import CellSpec, PassCounter, _logits, _pick, _ranges, _summary, _write, device_memory_mb

GAME = "partition"
ROLE_CHARS = "kls0"     # U_k, U_l, S, empty; "." is outside Omega
PATCHES = ("ak", "al", "c")


def _runs(spec: CellSpec) -> list[tuple[str, Optional[str]]]:
    runs = [(m, None) for m in spec.methods] + [("cdea", a) for a in spec.ablations]
    return runs + [("cdea", c) for c in spec.candidates]


def _label(method: str, variant: Optional[str]) -> str:
    if variant is None:
        return method
    return variant if "@" in variant else f"{method}:{variant}"


def _maps(spec, model, x, h, method, variant, device, ctx):
    from evaluation.run_methods import (
        PARTITION_ABLATIONS,
        PARTITION_CANDIDATES,
        PARTITION_SEEDED,
        PartitionRunConfig,
        partition_method,
    )

    kwargs = {}
    if variant in PARTITION_CANDIDATES:
        kwargs["cfg"] = PARTITION_CANDIDATES[variant]
    elif variant is not None:
        kwargs["cfg"] = PARTITION_ABLATIONS[variant](PartitionRunConfig())
    if method in PARTITION_SEEDED:
        kwargs["seed"] = spec.seed
    return partition_method(method)(model, x, h, spec.backbone, spec.knobs, device, ctx, **kwargs)


def labels_string(roles: torch.Tensor) -> str:
    """One character per cell. A cell in several roles (DA1) shows the first of S, U_k, U_l, empty."""
    chars = []
    for r in range(roles.shape[-1]):
        held = [p for p in (2, 0, 1, 3) if bool(roles[p, r])]
        chars.append(ROLE_CHARS[held[0]] if held else ".")
    return "".join(chars)


def run_partition(spec: CellSpec, device) -> Path:
    """Score every requested version-2 method on one dataset x backbone x seed."""
    from cdea.region import evidence_region
    from core.grid import grid_of
    from evaluation.provenance import knobs_hash, method_code_hash
    from evaluation.run_data import decomposition_sample
    from evaluation.run_methods import PartitionContext, hypotheses_for, partition_reference
    from evaluation.run_models import load_cell_model

    if spec.game != GAME:
        raise ValueError(f"run_partition runs the {GAME} game")
    loaded = load_cell_model(spec.dataset, spec.backbone, spec.seed, source=spec.model_source,
                             checkpoint=spec.checkpoint, device=device)
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = loaded.model
    sample = decomposition_sample(spec.dataset, spec.split, spec.n, spec.seed,
                                  final=spec.final, config_hash=spec.config_hash, model=model, device=device)
    n = int(sample.x.shape[0])
    grid = grid_of(spec.backbone)
    counter = PassCounter(model)
    code_hash = method_code_hash()
    knob_hash = knobs_hash(spec.knobs)
    runs = [(m, v, _label(m, v)) for m, v in _runs(spec)]
    status = {label: {"status": "ok", "seconds": 0.0, "forward_images": 0, "backward_images": 0}
              for _, _, label in runs}
    shared = {"reference_seconds": 0.0, "scoring_seconds": 0.0}
    rows: list[dict] = []

    for start, end in _ranges(n, spec.image_batch):
        x = sample.x[start:end].to(device)
        top = hypotheses_for(model, x, 3)
        ids = top.ids.to(device)
        h = HypothesisSet(ids=ids[:, :2], mask=top.mask[:, :2].to(device))
        began = time.perf_counter()
        evidence = partition_reference(model, x, h, spec.backbone, spec.knobs)
        ctx = PartitionContext(evidence=evidence, omega=evidence_region(evidence), rows=list(range(start, end)),
                               grid=grid)
        shared["reference_seconds"] += time.perf_counter() - began
        maps = {}
        for method, variant, label in runs:
            if status[label]["status"] == "error":
                continue
            counter.reset()
            began = time.perf_counter()
            try:
                maps[label] = _maps(spec, model, x, h, method, variant, device, ctx)
            except Exception as exc:
                status[label] = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc(limit=4)}
                continue
            entry = status[label]
            entry["seconds"] += time.perf_counter() - began
            entry["forward_images"] += counter.forward
            entry["backward_images"] += counter.backward
        began = time.perf_counter()
        rows += _score_batch(model, x, ids, ctx, maps, spec, sample, start, code_hash, knob_hash)
        shared["scoring_seconds"] += time.perf_counter() - began
    counter.close()
    for _, _, label in runs:
        if status[label]["status"] == "ok":
            status[label]["seconds_per_image"] = status[label]["seconds"] / max(n, 1)
            status[label]["device_mb_after"] = device_memory_mb(device)
    summary = _summary(spec, loaded, status, rows, code_hash, knob_hash)
    summary["shared"] = shared
    summary["sample"] = dict(sample.notes)
    if sample.roles is not None:
        summary["sample"]["roles_confirmed"] = [int(v) for v in sample.roles.sum(dim=0)]
        summary["sample"]["all_roles_confirmed"] = int(sample.roles.all(dim=1).sum())
    return _write(spec, rows, summary, device)


def _null_at_sizes(null: dict, sizes: torch.Tensor, which: int) -> torch.Tensor:
    """``(B, P, J)`` the scorer's null (``which=0``), 95th (``1``) or 5th (``2``) percentile at each size; NaN at size 0."""
    batch, players = sizes.shape
    some = next(iter(null.values()))[which] if null else None
    width = some.shape[-1] if some is not None else 3
    out = torch.full((batch, players, width), float("nan"))
    for b in range(batch):
        for p in range(players):
            s = int(sizes[b, p])
            if s > 0:
                out[b, p] = null[s][which][b].cpu()
    return out


def _score_batch(model, x, ids, ctx, maps, spec, sample, offset, code_hash, knob_hash) -> list[dict]:
    from cdea.region import cell_overlap
    from evaluation.scores import (
        VERDICTS,
        half_verdict,
        partition_scores,
        planted_role_scores,
        removal_drops,
        scored_size_null,
        split_halves,
    )

    batch = x.shape[0]
    grid = ctx.grid
    units = grid[0] * grid[1]
    omega = ctx.omega.cpu()
    overlap = cell_overlap(ctx.evidence).cpu()
    full = _logits(model, x)
    classes = ids.shape[1]
    roles_by = {label: out.roles.cpu() for label, out in maps.items()}
    halves_by = {}
    for label, roles in roles_by.items():
        halves = torch.zeros(batch, 2, units, dtype=torch.bool)
        for b in range(batch):
            if int(roles[b, 2].sum()) >= 2:
                halves[b] = split_halves(roles[b, 2])
        halves_by[label] = halves
    sizes_needed = [{1} for _ in range(batch)]
    cells_needed = torch.zeros(batch, units, dtype=torch.bool)
    for label, roles in roles_by.items():
        for b in range(batch):
            sizes_needed[b] |= {int(s) for s in roles[b].sum(dim=-1)}
            sizes_needed[b] |= {int(s) for s in halves_by[label][b].sum(dim=-1)}
        cells_needed |= roles.any(dim=1)
    cells = torch.nonzero(cells_needed.any(dim=0)).flatten().tolist()
    one_hot = torch.zeros(batch, len(cells), units)
    for i, r in enumerate(cells):
        one_hot[cells_needed[:, r], i, r] = 1.0
    patch_cells = None
    if getattr(sample, "patches", None) is not None:
        from evaluation.run_cell import _units_from_pixels

        patches = sample.patches[offset:offset + batch].float()
        patch_cells = torch.stack([_units_from_pixels(patches[:, t], *grid) > 0.5 for t in range(3)], dim=1)
    stages = {label: _stage_payoffs(model, x, ids[:, :2], out.extra["partition"], grid)
              for label, out in maps.items() if "partition" in out.extra}
    seed = spec.seed * 1000
    rows = []
    for operator in spec.operators:
        null = scored_size_null(model, x, ids, grid, sizes_needed, operator, seed, offset)
        cell_drop = removal_drops(model, x, one_hot, ids, operator, grid, seed, offset).cpu()
        cell_excess = torch.full((batch, classes, units), float("nan"))
        for i, r in enumerate(cells):
            value = cell_drop[:, i] - null[1][0].cpu()
            cell_excess[:, :, r] = torch.where(cells_needed[:, r].unsqueeze(-1), value, torch.full_like(value, float("nan")))
        labels = list(roles_by)
        stacked = torch.cat([torch.cat([roles_by[label], halves_by[label]], dim=1) for label in labels], dim=1)
        drops = removal_drops(model, x, stacked.float(), ids, operator, grid, seed, offset).cpu().view(
            batch, len(labels), 6, classes)
        for index, label in enumerate(labels):
            roles = roles_by[label]
            sizes = roles.sum(dim=-1)
            role_drops = drops[:, index, :4].clone()
            role_drops[sizes == 0] = float("nan")
            scores = partition_scores(role_drops, _null_at_sizes(null, sizes, 0), _null_at_sizes(null, sizes, 1),
                                      sizes, cell_excess, roles, overlap, omega, _null_at_sizes(null, sizes, 2))
            half_sizes = halves_by[label].sum(dim=-1)
            half_q = _null_at_sizes(null, half_sizes, 1)
            half_count = drops[:, index, 4:, :2] > half_q[..., :2]
            planted = planted_role_scores(roles, patch_cells, omega) if patch_cells is not None else {}
            for i in range(batch):
                row = {
                    "game": spec.game, "dataset": spec.dataset, "backbone": spec.backbone, "seed": spec.seed,
                    "split": spec.split, "image_index": sample.index[offset + i], "method": label,
                    "operator": operator, "label": int(sample.labels[offset + i]),
                    "class_k": int(ids[i, 0]), "class_l": int(ids[i, 1]),
                    "class_m": int(ids[i, 2]) if classes > 2 else -1,
                    "correct": int(int(ids[i, 0]) == int(sample.labels[offset + i])),
                    "z_k_full": float(_pick(full, ids[:, 0])[i]), "z_l_full": float(_pick(full, ids[:, 1])[i]),
                    "z_m_full": float(_pick(full, ids[:, 2])[i]) if classes > 2 else float("nan"),
                    "omega_size": int(omega[i].sum()), "overlap": float(overlap[i].sum()),
                    "overlap_omega": float((overlap[i] * omega[i]).sum()),
                    "labels": labels_string(roles[i]),
                    "split_verdict": VERDICTS[half_verdict(half_count[i])] if int(half_sizes[i, 1]) > 0 else "",
                    "fast": int(spec.knobs.fast), "model_source": spec.model_source,
                    "method_code_hash": code_hash, "knobs_hash": knob_hash,
                }
                row.update({name: float(value[i]) for name, value in scores.items()})
                row.update({name: float(value[i]) for name, value in planted.items()})
                if label in stages:
                    row.update({name: float(value[i]) for name, value in stages[label].items()})
                if getattr(sample, "roles", None) is not None:
                    for t, target in enumerate(PATCHES):
                        row[f"role_{target}"] = int(sample.roles[offset + i, t])
                rows.append(row)
    return rows


def _stage_payoffs(model, x, ids, part, grid) -> dict[str, torch.Tensor]:
    """D2, D9 and trapped-role inputs for a version 2 plan, under blur, with CDEA's own blur null and payoff.

    ``soft``: the payoff of the search's final plan on Omega. ``search``: of
    the hard plan the search ends at. ``hard``: of the final partition, after
    the polish. ``init``: of the step-0 partition. ``polish_changed`` is the
    share of Omega cells the polish moved, and ``polish_welfare`` the final
    partition's welfare under the polish's own deletion (blur, or ROAD for
    DA13). ``same_as_init`` is 1 when
    the final partition equals step 0 on every Omega cell (D9). ``trapped`` is
    the share of Omega cells whose final role, U_k, U_l or S, started from a
    zero part (initial logit log eps). ``empty_share`` is the share labelled
    empty, which starts at log eps everywhere.
    """
    from cdea._plan import _player_logits
    from cdea.null import null_at
    from cdea.payoffs import class_drops, cross_penalty, partition_payoffs

    omega = part.omega
    inside = omega.to(x.dtype).unsqueeze(1)
    views = {
        "soft": part.plan[:, :3] * inside,
        "search": part.search_roles[:, :3].to(x.dtype),
        "hard": part.roles[:, :3].to(x.dtype),
        "init": part.initial_roles[:, :3].to(x.dtype),
    }
    out = {}
    with torch.no_grad():
        full = model(x)
        for stage, masks in views.items():
            edited = _player_logits(model, x, masks, grid[0], grid[1], False, None)
            null = null_at(part.null.to(x.device), masks.sum(dim=-1))
            penalty = None if part.cell_excess is None else cross_penalty(masks, part.cell_excess, part.payoff)
            payoff = partition_payoffs(class_drops(full, edited, ids), null, part.payoff, penalty)
            for p, player in enumerate(("uk", "ul", "s")):
                out[f"payoff_{stage}_{player}"] = payoff[:, p].float().cpu()
    size = omega.sum(dim=-1).clamp_min(1).float()
    same = ((part.roles == part.initial_roles) | ~omega.unsqueeze(1)).all(dim=-1).all(dim=-1)
    out["same_as_init"] = same.float().cpu()
    out["polish_welfare"] = part.welfare.float().cpu()
    moved = ((part.roles != part.search_roles).any(dim=1) & omega).sum(dim=-1).float()
    out["polish_changed"] = (moved / size).cpu()
    out["empty_share"] = (part.roles[:, 3].sum(dim=-1).float() / size).cpu()
    if part.init_parts is None:
        out["trapped"] = torch.full((x.shape[0],), float("nan"))
    else:
        zero = part.init_parts <= 0
        out["trapped"] = ((part.roles[:, :3] & zero).any(dim=1).sum(dim=-1).float() / size).cpu()
    return out
