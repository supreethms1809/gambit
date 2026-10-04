"""S18. The pilot gate stays closed until G0 passes and the dev checkpoints exist."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_g1_does_not_run_and_the_pilot_tag_is_absent() -> None:
    text = (ROOT / "docs" / "paper" / "G1.md").read_text(encoding="utf-8")
    assert "G1 does not run" in text
    assert "g1-pilot" in text
    assert "was not created" in text
    assert "No stop-or-continue decision is logged" in text
    for phrase in ("CIFAR-10", "HAM10000", "margin attribution", "G0 does not pass"):
        assert phrase in text, phrase
    tags = subprocess.check_output(
        ["git", "tag", "--list", "g1-pilot"],
        cwd=ROOT,
        text=True,
    )
    assert tags.strip() == ""
