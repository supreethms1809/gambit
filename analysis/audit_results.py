"""Audit generated results against the records that produced them.

A markdown file passes when it matches a fresh ``render_results`` call. The
check is read-only: a mismatch is reported and the file is left as it is.
The paper file is absent, and no mask has been inspected, so the paper audit
does not pass. This module does not read the test split.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.build_results import render_results
from scripts.completion_check import CONTRASTIVE_DATASETS

PAPER_RESULTS = REPO / "results" / "paper" / "RESULTS.md"
_NUMBER = re.compile(r"-?\d+\.\d+")


def required_datasets() -> tuple[str, ...]:
    """Contrastive datasets whose masks still need an eye check."""
    return tuple(CONTRASTIVE_DATASETS)


def masks_checked(seen: Sequence[str]) -> bool:
    """True only when every required dataset was actually inspected."""
    return set(required_datasets()) <= set(seen)


def number_tokens(text: str) -> list[str]:
    return _NUMBER.findall(text)


def audit_text(text: str, records: Sequence[Mapping], **kwargs) -> dict:
    """Compare ``text`` with a fresh render. Does not write."""
    markdown, _latex, _summary = render_results(records, **kwargs)
    file_numbers = number_tokens(text)
    fresh_numbers = number_tokens(markdown)
    return {
        "passed": text == markdown and file_numbers == fresh_numbers,
        "numbers_match": file_numbers == fresh_numbers,
        "file_numbers": file_numbers,
        "fresh_numbers": fresh_numbers,
    }


def audit_file(path: Path, records: Sequence[Mapping], **kwargs) -> dict:
    """Read ``path`` and compare it with a fresh render. The file is not edited."""
    text = Path(path).read_text(encoding="utf-8")
    report = audit_text(text, records, **kwargs)
    report["path"] = str(path)
    return report


def paper_audit(path: Path = PAPER_RESULTS) -> dict:
    """The paper audit does not pass. The results file is absent and no mask was checked."""
    present = Path(path).is_file()
    return {
        "passed": False,
        "results_present": present,
        "masks_checked": False,
        "reason": "RESULTS.md is absent" if not present else "RESULTS.md is present and unaudited",
    }


def main() -> None:
    """Do not certify the paper. There is no results file and no mask inspection."""
    report = paper_audit()
    if report["passed"]:
        raise SystemExit("the audit does not pass")
    raise SystemExit(
        "the audit does not pass; RESULTS.md is absent; masks were not checked"
    )


if __name__ == "__main__":
    main()
