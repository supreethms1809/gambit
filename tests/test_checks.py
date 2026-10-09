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
