"""The evaluation plan and the framing note are drafts.

Neither file freezes the test split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eval_plan_is_a_draft_and_the_freeze_tag_is_absent() -> None:
    text = (ROOT / "docs" / "paper" / "EVAL_PLAN.md").read_text(encoding="utf-8")
    assert "not frozen" in text
    assert "eval-plan-frozen" in text
    assert "was not created" in text
    assert "CD@5%" in text
    assert "test split stays locked" in text
    tags = subprocess.check_output(
        ["git", "tag", "--list", "eval-plan-frozen"],
        cwd=ROOT,
        text=True,
    )
    assert tags.strip() == ""


def test_framing_keeps_the_claims_that_are_out_of_scope() -> None:
    text = (ROOT / "docs" / "paper" / "FRAMING.md").read_text(encoding="utf-8")
    for phrase in (
        "Good localisation",
        "Equilibrium or convergence",
        "Usefulness to people",
        "stays in the table",
    ):
        assert phrase in text, phrase
