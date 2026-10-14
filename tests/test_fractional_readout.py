"""The fractional-mask test's read-out applies the rule fixed in G1.md."""

from __future__ import annotations

import csv
import gzip

import numpy as np

from scripts.fractional_readout import (
    BINARY,
    DATASETS,
    FRACTIONAL,
    arm_passes,
    decide,
    main,
    pooled_ci,
)

FIELDS = ["split", "seed", "operator", "area", "dataset", "method", "image_index", "cd", "cd1",
          "z_k_full", "z_k_without_k", "z_k_without_l", "z_l_full", "z_l_without_k", "z_l_without_l",
          "method_code_hash", "knobs_hash"]


def write_cells(root, offsets: dict[str, dict[str, float]], n: int = 40, skip: tuple[str, ...] = ()) -> None:
    """One seed-1 cell per dataset. ``offsets[dataset][method]`` shifts that method's CD."""
    rng = np.random.default_rng(0)
    for dataset in DATASETS:
        if dataset in skip:
            continue
        cell = root / "val" / "contrastive" / dataset / "resnet50" / "seed1"
        cell.mkdir(parents=True)
        base = rng.normal(2.0, 1.0, size=n)
        with gzip.open(cell / "records.csv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            for method in (FRACTIONAL, BINARY, "margin_gradcam", "margin_ig", "extremal"):
                for i in range(n):
                    cd = base[i] + offsets.get(dataset, {}).get(method, 0.0) + rng.normal(0, 0.1)
                    writer.writerow(["val", 1, "road", 0.05, dataset, method, i, cd, cd,
                                     5, 4, 4, 3, 3, 2, "9d84e7f2abc", "4d561871"])


def test_pooled_interval_weights_datasets_equally():
    point, low, high = pooled_ci([np.zeros(500), np.full(10, 2.0)], n_boot=200)
    assert point == 1.0
    assert low == high == 1.0


def test_rule_needs_every_dataset_and_no_dataset_clearly_behind():
    ahead = (0.5, 0.1, 0.9)
    per = {d: {"margin": ahead, "extremal": ahead} for d in DATASETS}
    pooled = {"margin": ahead, "extremal": ahead}
    assert arm_passes(per, pooled)[0]
    behind = dict(per, cifar10={"margin": (-0.5, -0.9, -0.1), "extremal": ahead})
    passed, reasons = arm_passes(behind, pooled)
    assert not passed and any("rule 3" in r and "cifar10" in r for r in reasons)
    missing = dict(per, imagenet={"margin": None, "extremal": None})
    passed, reasons = arm_passes(missing, pooled)
    assert not passed and any("rule 4" in r for r in reasons)
    assert not arm_passes(per, {"margin": (0.5, -0.1, 1.1), "extremal": ahead})[0]


def test_binary_is_kept_when_it_passes():
    assert decide(True, True).startswith("method paper with CDEA as it is")
    assert decide(False, True).startswith("method paper with fractional masks")
    assert decide(False, False).startswith("evaluation-paper route stands")


def test_end_to_end_fractional_ahead(tmp_path, capsys):
    write_cells(tmp_path, {d: {FRACTIONAL: 1.0} for d in DATASETS})
    main(["--records", str(tmp_path), "--expect-hash", "9d84e7f2", "--n-boot", "500"])
    out = capsys.readouterr().out
    assert f"{FRACTIONAL}: PASS" in out
    assert f"{BINARY}: does not pass" in out
    assert "Decision: method paper with fractional masks" in out


def test_end_to_end_missing_cell_fails_the_arm(tmp_path, capsys):
    write_cells(tmp_path, {d: {FRACTIONAL: 1.0} for d in DATASETS}, skip=("brain_tumor",))
    main(["--records", str(tmp_path), "--n-boot", "200"])
    out = capsys.readouterr().out
    assert "Decision: evaluation-paper route stands" in out


def test_comparator_is_the_better_margin_variant(tmp_path, capsys):
    write_cells(tmp_path, {d: {FRACTIONAL: 1.0, "margin_ig": 1.5} for d in DATASETS})
    main(["--records", str(tmp_path), "--n-boot", "200"])
    out = capsys.readouterr().out
    assert "better margin variant: margin_ig" in out
    assert f"{FRACTIONAL}: does not pass" in out
