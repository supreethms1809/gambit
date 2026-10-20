"""The decomposition read-out applies EVAL_PLAN.md section 13's metrics and selection."""

from __future__ import annotations

import csv
import gzip

import numpy as np

from scripts import decomp_readout
from scripts.decomp_readout import (_paired, _value, family_dt_differences, load, main, rank_candidates,
                                    select_minsplit)

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
    # A method with no S map averages the players it has.
    assert _value({"rec_uk_ak": "1.0", "rec_ul_al": "0.5", "rec_s_c": "nan"}, "rr") == 0.75


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


def natural(scores: dict[str, float], n: int = 6, **values) -> list[dict]:
    """``scores[method]`` sets that method's SV; PUR_k + PUR_l stays 2."""
    return [record(method, i, sv=sv, **values) for method, sv in scores.items() for i in range(n)]


def test_min_split_row_is_chosen_on_val_even_when_reading_test(tmp_path, capsys):
    write_cell(tmp_path, "val", "cifar10", natural({"cdea_decomp": 1.0, "gradcam_minsplit": 0.0, "ig_minsplit": 0.5}))
    write_cell(tmp_path, "test", "cifar10", natural({"cdea_decomp": 1.0, "gradcam_minsplit": 0.9, "ig_minsplit": 0.1}))
    main(["--records", str(tmp_path), "--split", "test"])
    out = capsys.readouterr().out
    assert "cifar10 resnet50 by PUR + SV@0.05, ROAD: ig_minsplit" in out
    assert "cdea_decomp - ig_minsplit (best min-split, chosen on val)" in out
    assert "cdea_decomp - gradcam_minsplit" not in out


def test_planted_min_split_row_is_chosen_by_rr_on_confirmed_images(tmp_path):
    rows = []
    for i in range(6):
        rows.append(planted("gradcam_minsplit", i, 0.2, pur_k=9.0))
        rows.append(planted("ig_minsplit", i, 0.6))
        rows.append(planted("occlusion_minsplit", i, 0.1))
    for i in range(6, 12):
        rows.append(planted("occlusion_minsplit", i, 1.0, confirmed=False))
        rows.append(planted("ig_minsplit", i, 0.0, confirmed=False))
    write_cell(tmp_path, "val", "planted_pair", rows)
    best, scores = select_minsplit(load(tmp_path, "val", None))[("planted_pair", "resnet50")]
    assert best == "ig_minsplit"
    assert np.isclose(scores["ig_minsplit"], 0.6)


def test_fast_rows_and_partial_seeds_never_choose(tmp_path):
    write_cell(tmp_path, "val", "cifar10", natural({"gradcam_minsplit": 0.0, "ig_minsplit": 5.0}, fast=1))
    write_cell(tmp_path, "val", "ham10000", natural({"gradcam_minsplit": 0.0, "ig_minsplit": 5.0}))
    write_cell(tmp_path, "val", "ham10000", natural({"gradcam_minsplit": 0.0}), seed=1)
    selection = select_minsplit(load(tmp_path, "val", None))
    assert ("cifar10", "resnet50") not in selection
    assert selection[("ham10000", "resnet50")][0] == "gradcam_minsplit"


def test_candidates_rank_by_pur_plus_sv_with_each_dev_dataset_counted_once(tmp_path, capsys):
    t50, t100, partial = "cdea_decomp@lr0.05_t50", "cdea_decomp@lr0.2_t100", "cdea_decomp@lr0.2_t50"
    # Pooled over images t50 would win (3.64 vs 3.0); averaged over datasets t100 wins (3.0 vs 2.0).
    write_cell(tmp_path, "val", "cifar10", natural({t50: -2.0, t100: 1.0, partial: 9.0}))
    write_cell(tmp_path, "val", "ham10000", natural({t100: 1.0}) + natural({t50: 2.0}, n=60))
    ranked = rank_candidates(load(tmp_path, "val", None))
    assert [label for label, _, _ in ranked[:2]] == [t100, t50]
    assert np.isclose(ranked[0][1], 3.0) and np.isclose(ranked[1][1], 2.0)
    assert {label for label, score, _ in ranked if score is None} == {partial, "cdea_decomp@lr0.05_t100"}
    main(["--records", str(tmp_path)])
    out = capsys.readouterr().out
    assert f"chosen: {t100}" in out
    assert "not ranked, missing ham10000" in out


def dt_seeds(root, split: str, gaps: list[tuple[int, float]], **recalls) -> None:
    """One planted ResNet-50 cell per seed: ``n`` confirmed images where CDEA's RR is 0.8 and
    ig_minsplit's is ``0.8 - gap``. ``recalls[method]`` sets another method's RR."""
    for seed, (n, gap) in enumerate(gaps):
        rows = []
        for i in range(n):
            rows += [planted("cdea_decomp", i, 0.8), planted("ig_minsplit", i, 0.8 - gap)]
            rows += [planted(method, i, recall) for method, recall in recalls.items()]
        write_cell(root, split, "planted_pair", rows, seed=seed)


def test_family_dt_pairs_confirmed_resnet_images_and_keeps_seeds_apart(tmp_path):
    for seed, n in enumerate((2, 3)):
        rows = []
        for i in range(n):
            rows += [planted("cdea_decomp", i, 0.8), planted("ig_minsplit", i, 0.8 - 0.1 * (seed + 1))]
        # None of these reach family D-T: an unconfirmed image, a fast-knob image, an image only
        # CDEA has, and the same images at 5% and under blur.
        rows += [planted("cdea_decomp", 10, 1.0, confirmed=False), planted("ig_minsplit", 10, 0.0, confirmed=False)]
        rows += [planted("cdea_decomp", 11, 1.0, fast=1), planted("ig_minsplit", 11, 0.0, fast=1)]
        rows += [planted("cdea_decomp", 12, 1.0)]
        rows += [dict(planted("cdea_decomp", 0, 1.0), area=0.05), dict(planted("ig_minsplit", 0, 0.0), area=0.05)]
        rows += [dict(planted("cdea_decomp", 0, 1.0), operator="blur"),
                 dict(planted("ig_minsplit", 0, 0.0), operator="blur")]
        write_cell(tmp_path, "val", "planted_pair", rows, seed=seed)
    vit = [planted("cdea_decomp", 0, 1.0), planted("ig_minsplit", 0, 0.0)]
    write_cell(tmp_path, "val", "planted_pair", vit, backbone="vit_b_16", seed=2)
    write_cell(tmp_path, "val", "planted_pair", [planted("cdea_decomp", 0, 1.0)], seed=3)
    per_seed, empty = family_dt_differences(load(tmp_path, "val", None), "cdea_decomp", "ig_minsplit")
    assert sorted(per_seed) == [0, 1] and empty == [3]
    assert np.allclose(per_seed[0], [0.1, 0.1]) and np.allclose(per_seed[1], [0.2, 0.2, 0.2])


def test_family_dt_prints_seed_pooled_stats_on_the_reported_split(tmp_path, capsys, monkeypatch):
    calls = []
    real = decomp_readout.seed_pooled_summary

    def spy(comparisons, **kwargs):
        calls.append({name: {seed: np.asarray(d).tolist() for seed, d in diffs.items()}
                      for name, diffs in comparisons.items()})
        return real(comparisons, **kwargs)

    monkeypatch.setattr(decomp_readout, "seed_pooled_summary", spy)
    # Val chooses ig_minsplit; on test gradcam_minsplit would win, but test chooses nothing.
    dt_seeds(tmp_path, "val", [(4, 0.2)], gradcam_minsplit=0.1)
    # Seed 0 gains 0.6 on 2 images, seed 1 gains 0 on 6: the seed-pooled mean is 0.3, pooled
    # over images it would be 0.15.
    dt_seeds(tmp_path, "test", [(2, 0.6), (6, 0.0)], gradcam_minsplit=0.9, occlusion_assign=0.9)
    main(["--records", str(tmp_path), "--split", "test"])
    out = capsys.readouterr().out.split("Family D-T")[1]
    assert "cdea_decomp - ig_minsplit (best min-split, chosen on val):" in out
    assert "cdea_decomp - gradcam_minsplit" not in out
    assert "rr   mean +0.300" in out and "rr   mean -0.100" in out
    assert "images seed0 2  seed1 6" in out
    assert "sign-flip p" in out and "95% two-level" in out
    assert "holds" not in out
    # One comparison per call, so no Holm across them; each seed's images stay a separate array.
    assert len(calls) == 2 and all(list(call) == ["rr"] for call in calls)
    assert np.allclose(calls[0]["rr"][0], [0.6, 0.6]) and np.allclose(calls[0]["rr"][1], [0.0] * 6)
    assert np.allclose(calls[1]["rr"][0], [-0.1, -0.1]) and np.allclose(calls[1]["rr"][1], [-0.1] * 6)


def test_family_dt_says_when_nothing_is_compared(tmp_path, capsys):
    write_cell(tmp_path, "val", "cifar10", natural({"cdea_decomp": 1.0, "gradcam_minsplit": 0.0}))
    main(["--records", str(tmp_path)])
    assert "no planted_pair resnet50 cells at area 0.025 under ROAD" in capsys.readouterr().out
    # Fast-knob rows choose no min-split row and give no differences; occlusion_assign is absent.
    fast = [planted(method, i, 0.5, fast=1) for method in ("cdea_decomp", "ig_minsplit") for i in range(3)]
    write_cell(tmp_path, "val", "planted_pair", fast)
    main(["--records", str(tmp_path)])
    out = capsys.readouterr().out.split("Family D-T")[1]
    assert "best min-split: none chosen on val, so not compared" in out
    assert "cdea_decomp - occlusion_assign (additive per-cell assignment):" in out
    assert "no paired role-confirmed non-fast images in seed 0" in out
    assert "rr   mean" not in out
