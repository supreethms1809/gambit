"""The declared-test read-out (scripts/partition_tests.py) on synthetic test records with known answers.

Every planted CDEA row beats its comparators by a fixed amount on every image, and every natural
dataset has CDEA's S confirmed on more images, so each test's mean and verdict are known in advance.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")

from evaluation.provenance import knobs_hash, method_code_hash  # noqa: E402
from evaluation.run_methods import Knobs  # noqa: E402
from scripts import partition_tests as pt  # noqa: E402

N_IMAGES = 12
PATCH = [f"conf_{t}_{r}" for t in ("ak", "al", "c") for r in ("uk", "ul", "s", "none", "outside")]


def _write_cell(base, root, dataset, backbone, seed, rows, config=pt.FROZEN_CONFIG):
    cell = base / root / "test" / "partition" / dataset / backbone / f"seed{seed}"
    cell.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(cell / "records.csv.gz", index=False)
    summary = {"method_code_hash": method_code_hash(), "knobs_hash": knobs_hash(Knobs()),
               "cell": {"split": "test", "seed": seed},
               "provenance": {"config_hash": config, "git_dirty": False},
               "methods": {m: {"status": "ok"} for m in frame["method"].unique()}}
    (cell / "summary.json").write_text(json.dumps(summary))


def _rows(dataset, backbone, seed, scores):
    """``scores`` maps a method to (ra_all, ra_all_ak, ra_all_al, ra_all_c, confirmed_s)."""
    out = []
    for image in range(N_IMAGES):
        for op in ("road", "blur"):
            for method, (ra, ak, al, c, cs) in scores.items():
                row = {"dataset": dataset, "backbone": backbone, "seed": seed, "operator": op, "image_index": image,
                       "method": method, "method_code_hash": method_code_hash(), "knobs_hash": knobs_hash(Knobs()),
                       "fast": 0, "class_k": 1, "class_l": 2, "omega_size": 5 + image, "overlap_omega": 1.5,
                       "role_ak": 1, "role_al": 1, "role_c": 1, "ra_all": ra, "ra_all_ak": ak, "ra_all_al": al,
                       "ra_all_c": c, "labels": "k.l" if method.startswith("cdea") else "l.k",
                       "confirmed_s": cs, "confirmed_uk": 0.5, "confirmed_ul": 0.5,
                       "size_s": 2, "size_uk": 1, "size_ul": 1}
                row.update({col: 0.0 for col in PATCH})
                row["conf_ak_uk"] = ak
                out.append(row)
    return out


@pytest.fixture()
def base(tmp_path):
    planted = {"cdea@lr0.05_t50": (0.90, 1.0, 1.0, 0.70, 0), "cdea": (0.90, 1.0, 1.0, 0.70, 0),
               "cdea:DA12_polish_only": (0.90, 1.0, 1.0, 0.70, 0),
               "subtract": (0.80, 1.0, 1.0, 0.71, 0), "occlusion_partition": (0.60, 0.5, 0.5, 0.70, 0),
               "occlusion_onesided": (0.85, 0.9, 0.9, 0.70, 0)}
    for seed in range(5):
        for dataset in ("planted_pair", "planted_pair_adjacent"):
            _write_cell(tmp_path, "cells", dataset, "resnet50", seed, _rows(dataset, "resnet50", seed, planted))
            _write_cell(tmp_path, "cells", dataset, "vit_b_16", seed, _rows(dataset, "vit_b_16", seed, planted))
        for dataset in pt.NATURAL:
            for backbone in ("resnet50", "vit_b_16"):
                natural = {"cdea@lr0.05_t50": (np.nan,) * 4 + (1,), "occlusion_onesided": (np.nan,) * 4 + (0,)}
                _write_cell(tmp_path, "cells", dataset, backbone, seed, _rows(dataset, backbone, seed, natural))
    return tmp_path


def test_known_answers(base) -> None:
    out = pt.run(base, "test")
    t = out["tests"]
    # C's accuracy is 0.01 below subtract on every image: inside the -0.02 margin.
    assert t["D1''a"]["mean"] == pytest.approx(-0.01) and t["D1''a"]["holds"]
    assert t["D1''b"]["mean"] == pytest.approx(0.10) and t["D1''b"]["holds"] and t["D1''"]["holds"]
    assert t["framing"]["mean"] == pytest.approx(0.30) and t["framing"]["verdict"] == "ahead"
    assert t["S1"]["mean"] == pytest.approx(0.05) and t["S1"]["holds"]
    # Unique patches 1.0 against 0.9, both conditions stacked.
    assert t["S2"]["mean"] == pytest.approx(0.10) and t["S2"]["n"] == 2 * 5 * N_IMAGES and t["S2"]["holds"]
    for backbone in ("resnet50", "vit_b_16"):
        r = t["S3"]["per_backbone"][backbone]
        assert r["positive"] == 8 and r["median"] == pytest.approx(1.0)
        assert r["p"] == pytest.approx(2 / 256) and r["holds"]
    assert out["beside"]["S1"]["share of images whose patch cells are labelled differently"] == 1.0


def test_test_split_refuses_other_configs(base) -> None:
    summary = base / "cells" / "test" / "partition" / "planted_pair" / "resnet50" / "seed0" / "summary.json"
    s = json.loads(summary.read_text())
    s["provenance"]["config_hash"] = "not-the-frozen-one"
    summary.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="not a final test cell"):
        pt.run(base, "test")


def test_roots_must_agree_on_the_images(base) -> None:
    records = base / "cells" / "test" / "partition" / "planted_pair" / "resnet50" / "seed1" / "records.csv.gz"
    d = pd.read_csv(records)
    d.loc[(d["method"] == "subtract") & (d["image_index"] == 3), "omega_size"] = 99
    d.to_csv(records, index=False)
    with pytest.raises(SystemExit, match="disagree on the images"):
        pt.run(base, "test")
