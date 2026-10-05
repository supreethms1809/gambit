"""Val selection rule. This does not freeze the evaluation plan.

The score is the mean CD@5% on the dev val sets, already averaged by the
caller. A candidate is eligible only when the gating routes (D1, D2, D4, D5)
are closed. D1 and D5 close by construction under the hard mass budget, so
they gate as an assertion. D3, the share of the margin won by foil
suppression, is recorded and reported but does not gate: for "k rather than
l", evidence that lowers l is legitimate (EVAL_PLAN.md section 6.1). The
eligible candidate with the highest score is selected. Ties keep the
earliest candidate. If none are eligible, nothing is selected: a candidate
that leaves a gating route open is not a fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

ROUTES = ("D1", "D2", "D3", "D4", "D5")
GATING_ROUTES = ("D1", "D2", "D4", "D5")


def config_hash(config: Mapping) -> str:
    """Stable hash of a JSON-compatible config. Key order does not matter."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _open_routes(routes: Mapping) -> list[str]:
    missing = [name for name in ROUTES if name not in routes]
    if missing:
        raise ValueError(f"routes missing {', '.join(missing)}")
    opened = []
    for name in ROUTES:
        flag = routes[name]
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be a bool")
        if flag and name in GATING_ROUTES:
            opened.append(name)
    return opened


def select_config(candidates: Sequence[Mapping]) -> dict:
    """Return the selected name, its score, its config hash, and the rejections."""
    if not candidates:
        raise ValueError("selection needs at least one candidate")
    eligible: list[tuple[int, Mapping, float]] = []
    rejected: list[dict] = []
    for index, candidate in enumerate(candidates):
        name = str(candidate["name"])
        score = float(candidate["score"])
        if not math.isfinite(score):
            raise ValueError(f"{name} has a non-finite score")
        if "config" not in candidate or not isinstance(candidate["config"], Mapping):
            raise TypeError(f"{name} needs a config mapping")
        opened = _open_routes(candidate["routes"])
        if opened:
            rejected.append({"name": name, "open": opened})
            continue
        eligible.append((index, candidate, score))
    if not eligible:
        return {"selected": None, "score": None, "config_hash": None, "rejected": rejected}
    _index, chosen, score = max(eligible, key=lambda item: (item[2], -item[0]))
    return {
        "selected": str(chosen["name"]),
        "score": score,
        "config_hash": config_hash(chosen["config"]),
        "rejected": rejected,
    }
