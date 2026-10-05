"""The val rule maximises the score among candidates that close the gating routes.

Nothing here freezes the evaluation plan or reads a split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from analysis.selection import config_hash, select_config

ROOT = Path(__file__).resolve().parents[1]
CLOSED = {"D1": False, "D2": False, "D3": False, "D4": False, "D5": False}


def _candidate(name: str, score: float, routes: dict, step: int = 50) -> dict:
    return {
        "name": name,
        "score": score,
        "routes": routes,
        "config": {"steps": step, "lr": 0.5, "name": name},
    }


def test_an_open_route_loses_to_a_lower_closed_score() -> None:
    open_d1 = dict(CLOSED)
    open_d1["D1"] = True
    chosen = select_config(
        [
            _candidate("wide", 0.9, open_d1),
            _candidate("tight", 0.4, CLOSED),
        ]
    )
    assert chosen["selected"] == "tight"
    assert chosen["score"] == 0.4
    assert chosen["rejected"] == [{"name": "wide", "open": ["D1"]}]
    assert chosen["config_hash"] == config_hash({"lr": 0.5, "name": "tight", "steps": 50})


def test_no_closed_candidate_selects_nothing() -> None:
    open_d5 = dict(CLOSED)
    open_d5["D5"] = True
    chosen = select_config([_candidate("only", 1.0, open_d5)])
    assert chosen["selected"] is None
    assert chosen["config_hash"] is None
    assert chosen["rejected"][0]["open"] == ["D5"]


def test_ties_keep_the_earliest_candidate() -> None:
    chosen = select_config(
        [
            _candidate("first", 0.5, CLOSED, step=10),
            _candidate("second", 0.5, CLOSED, step=20),
        ]
    )
    assert chosen["selected"] == "first"
    assert config_hash({"steps": 10, "lr": 0.5, "name": "first"}) == config_hash(
        {"name": "first", "lr": 0.5, "steps": 10}
    )


def test_a_missing_route_raises() -> None:
    routes = dict(CLOSED)
    del routes["D4"]
    try:
        select_config([_candidate("partial", 0.2, routes)])
        raise AssertionError("a missing route should raise")
    except ValueError:
        pass


def test_the_plan_stays_unfrozen() -> None:
    text = (ROOT / "docs" / "paper" / "EVAL_PLAN.md").read_text(encoding="utf-8")
    assert "not frozen" in text
    assert "has not been applied" in text
    tags = subprocess.check_output(
        ["git", "tag", "--list", "eval-plan-frozen"],
        cwd=ROOT,
        text=True,
    )
    assert tags.strip() == ""


def test_open_d3_is_reported_but_does_not_gate() -> None:
    open_d3 = dict(CLOSED)
    open_d3["D3"] = True
    chosen = select_config(
        [
            _candidate("suppresses_foil", 0.9, open_d3),
            _candidate("tight", 0.4, CLOSED),
        ]
    )
    assert chosen["selected"] == "suppresses_foil"
    assert chosen["rejected"] == []


def test_d3_must_still_be_recorded() -> None:
    missing_d3 = {k: v for k, v in CLOSED.items() if k != "D3"}
    try:
        select_config([_candidate("x", 0.5, missing_d3)])
    except ValueError:
        return
    raise AssertionError("a candidate without a D3 record should raise")
