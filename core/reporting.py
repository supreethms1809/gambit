from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
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


def git_state(repo: Optional[Path] = None) -> tuple[Optional[str], bool]:
    """Return ``(commit, dirty)``. A missing git directory is ``(None, True)``."""
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit or None, bool(dirty)


def config_hash(payload: Mapping[str, Any]) -> str:
    """Stable short hash of a run config. This is what the frozen eval plan must list."""
    blob = json.dumps(_json_safe(to_serializable(dict(payload))), sort_keys=True, allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def collect_provenance(
    *,
    config_hash_value: Optional[str] = None,
    device: Optional[Any] = None,
    data_split_hash: Optional[str] = None,
) -> Dict[str, Any]:
    commit, dirty = git_state()
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": config_hash_value,
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "platform": platform.platform(),
        "device": None if device is None else str(device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_split_hash": data_split_hash,
    }


def refuse_final_if_dirty(final: bool) -> None:
    """Final runs record a commit hash, so they refuse a dirty tree."""
    if not final:
        return
    commit, dirty = git_state()
    if dirty or not commit:
        raise RuntimeError("refusing --final on a dirty or non-git tree")


def save_json(
    path: Path,
    data: Mapping[str, Any],
    *,
    provenance: bool = True,
    config_hash: Optional[str] = None,
    device: Optional[Any] = None,
    data_split_hash: Optional[str] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    if provenance:
        auto = collect_provenance(
            config_hash_value=config_hash, device=device, data_split_hash=data_split_hash
        )
        supplied = payload.get("provenance") or {}
        # A caller-supplied field wins, so a rerun can record the hash it actually used.
        payload["provenance"] = {**auto, **{k: v for k, v in supplied.items() if v is not None}}
        if config_hash is not None:
            payload["provenance"]["config_hash"] = config_hash
        if device is not None:
            payload["provenance"]["device"] = str(device)
        if data_split_hash is not None:
            payload["provenance"]["data_split_hash"] = data_split_hash
    with path.open("w", encoding="utf-8") as f:
        # allow_nan=False turns NaN/Infinity into an error rather than the bare
        # `NaN` literal Python emits by default. That literal is not valid JSON —
        # JavaScript, jq and most other readers reject the whole file — so a single
        # undefined std silently poisons a result file. _json_safe maps them to null.
        json.dump(_json_safe(to_serializable(payload)), f, indent=2,
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

