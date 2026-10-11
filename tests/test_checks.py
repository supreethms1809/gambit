"""D1 and D5 are properties of the plan."""

from __future__ import annotations

import torch

from scripts.check_degenerate import d1_shared_on_budget, d5_marginals


def test_shared_row_matches_the_budget_and_marginals_match():
    budget = 2.45
    shared = torch.zeros(2, 49)
    shared[:, :2] = 1
    shared[:, 2] = 0.45
    plan = torch.zeros(2, 3, 49)
    plan[:, 0, 0] = 1
    plan[:, 1, 1] = 1
    plan[:, 2] = 1
    plan[:, 2, 0] = 0
    plan[:, 2, 1] = 0
    row_target = torch.tensor([1.0, 1.0, 47.0])
    assert d1_shared_on_budget(shared, budget)
    assert d5_marginals(plan, row_target)
    assert not d1_shared_on_budget(shared, budget + 1)


def test_d2_reads_stage_payoffs_from_records(tmp_path):
    import csv
    import gzip

    from scripts.check_degenerate import stage_gaps

    fields = ["method", "operator", "area", "payoff_soft_k", "payoff_hard_k", "payoff_scored_blur_k",
              "payoff_soft_l", "payoff_hard_l", "payoff_scored_blur_l"]
    rows = [
        # Two images at 5%: soft-hard gaps 0.2 and 0.4 for k, 1.0 and 1.2 for l.
        ["cdea", "road", "0.05", "2.0", "1.8", "1.0", "3.0", "2.0", "1.5"],
        ["cdea", "road", "0.05", "1.0", "0.6", "0.6", "2.2", "1.0", "1.0"],
        # The blur copy of the same rows and baseline rows are ignored.
        ["cdea", "blur", "0.05", "9.0", "0.0", "0.0", "9.0", "0.0", "0.0"],
        ["margin_gradcam", "road", "0.05", "", "", "", "", "", ""],
        # One image at 10% that closes D2.
        ["cdea", "road", "0.1", "1.0", "0.9", "0.5", "1.0", "0.8", "0.4"],
    ]
    with gzip.open(tmp_path / "records.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    gaps = stage_gaps(tmp_path)
    assert set(gaps) == {0.05, 0.1}
    assert gaps[0.05]["n"] == 2
    assert abs(gaps[0.05]["soft_minus_hard_k"] - 0.3) < 1e-9
    assert abs(gaps[0.05]["soft_minus_hard_l"] - 1.1) < 1e-9
    assert abs(gaps[0.05]["hard_minus_scored_k"] - 0.4) < 1e-9
    assert not gaps[0.05]["d2_pass"]
    assert gaps[0.1]["d2_pass"]
