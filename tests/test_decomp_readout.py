"""The decomposition read-out applies EVAL_PLAN.md section 13's metrics and selection."""

from __future__ import annotations

import csv
import gzip

import numpy as np

from scripts.decomp_readout import _paired, _value, main

FIELDS = ["game", "dataset", "backbone", "seed", "split", "image_index", "method", "area", "operator",
          "pur_k", "pur_l", "sv", "cd", "bal", "sep", "fast", "method_code_hash", "knobs_hash",
          "role_ak", "role_al", "role_c", "rec_uk_ak", "rec_ul_al", "rec_s_c",
          "prec_s_c", "prec_uk_ak", "prec_ul_al", "prec_s_ak", "prec_s_al"]


def record(method: str, image: int, **values) -> dict:
    row = {"method": method, "image_index": image, "area": 0.05, "operator": "road", "fast": 0,
           "pur_k": 1.0, "pur_l": 1.0, "sv": 0.5, "cd": 1.0, "bal": 0.0, "sep": 0.0,
           "method_code_hash": "abc123", "knobs_hash": "k1"}
    row.update(values)
    return row


def write_cell(root, split: str, dataset: str, rows: list[dict], backbone: str = "resnet50", seed: int = 0) -> None:
    cell = root / split / "decomposition" / dataset / backbone / f"seed{seed}"
    cell.mkdir(parents=True)
    with gzip.open(cell / "records.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, game="decomposition", dataset=dataset, backbone=backbone, seed=seed,
                                 split=split))


def planted(method: str, image: int, recall: float, confirmed: bool = True, **values) -> dict:
    role = 1 if confirmed else 0
    return record(method, image, area=0.025, role_ak=role, role_al=role, role_c=role,
                  rec_uk_ak=recall, rec_ul_al=recall, rec_s_c=recall, **values)


def test_rr_is_the_mean_of_each_players_own_recall():
    row = {"rec_uk_ak": "1.0", "rec_ul_al": "0.5", "rec_s_c": "0.0"}
    assert _value(row, "rr") == 0.5
    assert np.isnan(_value({"pur_k": "1"}, "rr"))


def test_paired_rr_can_keep_only_role_confirmed_images():
    a = {0: {"rec_uk_ak": "1", "rec_ul_al": "1", "rec_s_c": "1", "role_ak": "1", "role_al": "1", "role_c": "1"},
         1: {"rec_uk_ak": "1", "rec_ul_al": "1", "rec_s_c": "1", "role_ak": "1", "role_al": "0", "role_c": "1"}}
    b = {i: dict(row, rec_uk_ak="0", rec_ul_al="0", rec_s_c="0") for i, row in a.items()}
    assert _paired(a, b, "rr").size == 2
    assert _paired(a, b, "rr", confirmed_only=True).tolist() == [1.0]


def test_planted_cell_prints_rr_and_s_spill(tmp_path, capsys):
    rows = []
    for i in range(6):
        rows.append(planted("cdea_decomp", i, 0.9, prec_s_ak=0.1, prec_s_al=0.2))
        rows.append(planted("gradcam_minsplit", i, 0.5, prec_s_ak=0.0, prec_s_al=0.0))
    write_cell(tmp_path, "val", "planted_pair", rows)
    main(["--records", str(tmp_path)])
    out = capsys.readouterr().out
    assert "roles confirmed" in out and "every image" in out
    assert "cdea_decomp - gradcam_minsplit" in out
    assert "RR 0.90 | 0.90" in out
    assert "S on A_k 0.10 | 0.10" in out and "S on A_l 0.20 | 0.20" in out
