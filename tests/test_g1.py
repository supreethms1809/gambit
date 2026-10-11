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


def test_readout_pairs_images_and_later_roots_win(tmp_path):
    import csv
    import gzip

    from scripts.g1_readout import load, paired

    fields = ["split", "seed", "operator", "area", "dataset", "method", "image_index", "cd"]

    def write(root, rows):
        cell = root / "val" / "contrastive" / "cifar10" / "resnet50" / "seed0"
        cell.mkdir(parents=True)
        with gzip.open(cell / "records.csv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    write(tmp_path / "a", [
        ["val", 0, "road", 0.05, "cifar10", "cdea", 1, 1.0],
        ["val", 0, "road", 0.05, "cifar10", "cdea", 2, 2.0],
        ["val", 0, "road", 0.05, "cifar10", "margin_gradcam", 1, 0.5],
        ["val", 0, "road", 0.05, "cifar10", "margin_gradcam", 2, 3.0],
        ["val", 0, "blur", 0.05, "cifar10", "cdea", 1, 9.0],
    ])
    write(tmp_path / "b", [
        ["val", 0, "road", 0.05, "cifar10", "cdea", 1, 4.0],
        ["val", 0, "road", 0.05, "cifar10", "cdea", 2, 4.0],
    ])
    data = load([tmp_path / "a", tmp_path / "b"], [0], 0.05, "road")
    diff = paired(data[("cifar10", "cdea")], data[("cifar10", "margin_gradcam")])
    assert sorted(diff.tolist()) == [1.0, 3.5]
