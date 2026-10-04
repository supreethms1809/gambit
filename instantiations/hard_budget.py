"""Project mask logits onto a fixed mass inside [0, 1].

The mass penalty is a cost the optimizer can pay. This projection is a
constraint: each row sums to ``max(1, R / mass_ref_regions)`` and no entry
exceeds 1. At 7×7 that budget is 1, matching the objective's mass target.
"""

from __future__ import annotations

import torch

from core.types import Tensor


def budget_mass(regions: int, mass_ref_regions: int = 49) -> float:
    if regions < 1:
        raise ValueError("regions must be >= 1")
    if mass_ref_regions <= 0:
        raise ValueError("mass_ref_regions must be > 0")
    return max(1.0, regions / float(mass_ref_regions))


def budgeted_mask(logits: Tensor, mass_ref_regions: int = 49) -> Tensor:
    """``logits`` is ``(..., R)``. The result is in ``[0, 1]`` and sums to the budget."""
    regions = logits.shape[-1]
    target = budget_mass(regions, mass_ref_regions)
    if target > regions:
        raise ValueError("budget cannot exceed the number of regions")
    target_t = torch.as_tensor(target, dtype=logits.dtype, device=logits.device)
    lo = logits.amin(dim=-1, keepdim=True) - 1.0
    hi = logits.amax(dim=-1, keepdim=True) + 1.0
    for _ in range(40):
        tau = (lo + hi) / 2
        mass = (logits - tau).clamp(0, 1).sum(dim=-1, keepdim=True)
        go_up = mass > target_t
        lo = torch.where(go_up, tau, lo)
        hi = torch.where(go_up, hi, tau)
    # Tau is the threshold that meets the sum. Detach it so the backward pass
    # is the projection gradient: 1 on free entries, 0 on entries pinned at 0 or 1.
    tau = ((lo + hi) / 2).detach()
    return (logits - tau).clamp(0, 1)
