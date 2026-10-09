"""Hashes that keep an old record from counting as a finished cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_ROOTS = ("cdea", "core", "base_evidence")


def method_code_hash() -> str:
    """sha256 of the method, the core, and the evidence providers."""
    digest = hashlib.sha256()
    for folder in CODE_ROOTS:
        for path in sorted((REPO / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def knobs_hash(knobs, extra: dict | None = None) -> str:
    payload = {key: getattr(knobs, key) for key in sorted(knobs.__dataclass_fields__)}
    if extra:
        payload.update(extra)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
