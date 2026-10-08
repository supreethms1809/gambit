"""S15. G0 is an entry check. Reproductions are reported, and only CVE's gates family C."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "docs" / "paper" / "G0.md"
DOSSIER = ROOT / "docs" / "paper" / "BASELINES.md"
EVAL_PLAN = ROOT / "docs" / "paper" / "EVAL_PLAN.md"

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


def test_g0_states_the_entry_rule_and_tracks_open_reproductions() -> None:
    text = G0.read_text(encoding="utf-8")
    assert "## Entry rule" in text
    assert "## Reproduction rule" in text
    assert "g0-baselines" in text
    for item in OPEN_REPRODUCTIONS:
        assert item in text, item


def test_cve_fallback_is_fixed_in_g0_and_the_eval_plan() -> None:
    g0 = G0.read_text(encoding="utf-8")
    plan = EVAL_PLAN.read_text(encoding="utf-8")
    assert "Holm over 2" in g0
    assert "Holm over 2" in plan
    assert "exploratory row" in g0
    assert "exploratory row" in plan
