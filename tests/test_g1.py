"""S18. G1 is the framing decision. Its read-out is fixed before the pilot is scored."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "docs" / "paper" / "G1.md"


def test_g1_fixes_the_read_out_and_does_not_gate_writing() -> None:
    text = G1.read_text(encoding="utf-8")
    assert "g1-pilot" in text
    assert "does not gate writing" in text
    assert "## Read-out, fixed before the pilot is scored" in text
    for phrase in ("CIFAR-10", "HAM10000", "margin attribution", "CD@5%", "val only"):
        assert phrase in text, phrase


def test_the_pilot_tag_is_absent_until_a_decision_is_logged() -> None:
    text = G1.read_text(encoding="utf-8")
    tags = subprocess.check_output(
        ["git", "tag", "--list", "g1-pilot"],
        cwd=ROOT,
        text=True,
    )
    if "No stop-or-continue decision is logged" in text:
        assert tags.strip() == ""
