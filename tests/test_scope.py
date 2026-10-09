"""The earlier framing does not re-enter this tree.

Banned names are identifiers, keyword arguments, attributes, and CLI flags.
Version tokens are banned in code and in ``docs/paper``. ``framing-v1`` is the
archive tag and is the only allowed ``v1``.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED = {
    "lambda_suff",
    "lambda_overlap",
    "lambda_mass",
    "lambda_sparse",
    "lambda_shared_sparse",
    "lambda_partition",
    "mass_ref_regions",
    "budgeted_mask",
    "hard_budget",
    "game_mode",
    "preset",
    "interaction",
    "naive_contrastive",
    "ContrastiveObjective",
    "OptimizationAllocator",
    "CDEAExplainer",
    "loss_scale",
    "foil_masks",
}

CDEA_WORDS = ("equilibrium", "best response", "provably")
VERSION = re.compile(r"\bv[12]\b")


_ROOTS = ("cdea", "core", "base_evidence", "models", "evaluation", "baselines", "analysis", "scripts", "tests")


def _py_files() -> list[Path]:
    out = []
    for name in _ROOTS:
        for path in (ROOT / name).rglob("*.py"):
            if path.name == "test_scope.py":
                continue
            out.append(path)
    return out


def _hits(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in BANNED:
            found.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg in BANNED:
            found.append(node.arg)
        elif isinstance(node, ast.arg) and node.arg in BANNED:
            found.append(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text in BANNED or text.lstrip("-") in BANNED:
                found.append(text)
    return found


def test_banned_names_are_absent():
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _hits(tree)
        if hits:
            offenders.append(f"{path.relative_to(ROOT)}: {sorted(set(hits))}")
    assert offenders == []


def test_version_tokens_are_absent():
    offenders = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        if VERSION.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    for path in (ROOT / "docs" / "paper").rglob("*.md"):
        text = path.read_text(encoding="utf-8").replace("framing-v1", "")
        if "v2" in text or "G1v2" in text or VERSION.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_method_package_avoids_the_retired_claims():
    for path in (ROOT / "cdea").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for word in CDEA_WORDS:
            assert word not in text, f"{path.name} contains {word}"


def test_retired_packages_are_not_importable():
    for name in ("instantiations", "modality", "core.runner"):
        assert importlib.util.find_spec(name) is None, name


def test_method_does_not_import_the_scorer():
    banned = {"evaluation", "baselines", "analysis"}
    for path in (ROOT / "cdea").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            assert not banned.intersection(names), path.name
