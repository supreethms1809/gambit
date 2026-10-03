from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch


def to_serializable(value: Any) -> Any:
    """Convert tensors/paths/numbers into JSON-serializable structures."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def extract_scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Keep only scalar numeric metrics for compact summaries."""
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            out[k] = float(v.detach().cpu().item())
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def extract_metric_shapes(metrics: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Report tensor metric shapes without writing large arrays into summary JSON."""
    out: Dict[str, List[int]] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor) and v.numel() > 1:
            out[k] = list(v.shape)
    return out


def save_json(path: Path, data: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # allow_nan=False turns NaN/Infinity into an error rather than the bare
        # `NaN` literal Python emits by default. That literal is not valid JSON —
        # JavaScript, jq and most other readers reject the whole file — so a single
        # undefined std silently poisons a result file. _json_safe maps them to null.
        json.dump(_json_safe(to_serializable(dict(data))), f, indent=2,
                  sort_keys=True, allow_nan=False)
    return path


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Infinity with None so the output is valid JSON everywhere."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_rows_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows_list = [dict(r) for r in rows]
    if fieldnames is None:
        keys = set()
        for row in rows_list:
            keys.update(row.keys())
        fieldnames = sorted(keys)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({k: to_serializable(row.get(k, "")) for k in fieldnames})
    return path

