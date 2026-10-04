"""S15. The baseline gate stays closed until the published Core runs exist."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "docs" / "paper" / "G0.md"
DOSSIER = ROOT / "docs" / "paper" / "BASELINES.md"

CORE_SECTIONS = (
    "## Margin attribution",
    "## Base evidence (Grad-CAM, integrated gradients)",
    "## Extremal Perturbations",
    "## Counterfactual visual explanations",
    "## Random area-a mask",
    "## Attribution difference",
    "## Spectral Relevance Analysis",
    "## Per-environment Extremal Perturbations",
)

OPEN_REPRODUCTIONS = (
    "Grad-CAM pointing game on VOC 2007",
    "Extremal Perturbations pointing game on VOC 2007",
    "CVE edit counts on CUB",
    "SpRAy horse analysis on VOC 2007",
)


def test_core_methods_have_dossier_entries() -> None:
    text = DOSSIER.read_text(encoding="utf-8")
    for heading in CORE_SECTIONS:
        assert heading in text, heading


def test_g0_does_not_pass_while_core_reproductions_are_open() -> None:
    text = G0.read_text(encoding="utf-8")
    assert "G0 does not pass" in text
    assert "g0-baselines" in text
    assert "was not created" in text
    for item in OPEN_REPRODUCTIONS:
        assert item in text, item
    tags = subprocess.check_output(
        ["git", "tag", "--list", "g0-baselines"],
        cwd=ROOT,
        text=True,
    )
    assert tags.strip() == ""
