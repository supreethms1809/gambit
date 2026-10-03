"""Thresholds for the degenerate-optimum checks D1–D6.

A route is open when the allocated mask takes the cheap way out of the loss.
The numbers below are the constraints the paper has to name. They are not
accuracy targets.
"""

from __future__ import annotations

import torch

# Shared mask: the measured blanket covered about half the grid.
D1_MAX_FRACTION_ABOVE_HALF = 0.15
D1_MAX_MASS_FRACTION = 0.15

# Soft mask logit minus same-area hard mask logit. Above this, the soft mask
# is doing something a same-budget hard mask does not.
D2_MAX_SOFT_MINUS_HARD = 0.5

# Share of a risen margin that comes from the foil logit falling.
D3_MAX_SUPPRESSION_SHARE = 0.5

# Unique-mask evidence capture must beat both chance and a translated copy.
D4_MIN_CAPTURE_RATIO = 1.2

# Achieved unique-mask mass over the objective's mass target.
D5_MAX_MASS_RATIO = 1.1

# Shift robust mask, and shortcut equal to the complement of robust.
D6_MAX_FRACTION_ABOVE_HALF = 0.25
D6_MAX_COMPLEMENT_DEVIATION = 0.25
D6_MIN_COMPLEMENT_MASS = 0.5


def _mean(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().item())


def evidence_capture_ratio(mask: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    """Capture of nonnegative evidence divided by the mask's area fraction. ``(B,)``."""
    area = mask.clamp_min(0).mean(dim=-1).clamp_min(1e-8)
    field = evidence.clamp_min(0)
    total = field.sum(dim=-1).clamp_min(1e-8)
    captured = (mask.clamp_min(0) * field).sum(dim=-1) / total
    return captured / area


def same_area_hard(mask: torch.Tensor) -> torch.Tensor:
    """Binary mask with the same per-example mass, on the highest soft entries."""
    hard = torch.zeros_like(mask)
    regions = mask.shape[-1]
    for i in range(mask.shape[0]):
        k = int(round(float(mask[i].detach().clamp_min(0).sum().item())))
        k = min(max(k, 0), regions)
        if k == 0:
            continue
        chosen = mask[i].detach().topk(k).indices
        hard[i, chosen] = 1.0
    return hard


def d1_shared_blanket(shared: torch.Tensor) -> dict:
    """``shared`` is ``(B, R)``."""
    regions = shared.shape[-1]
    fraction = (shared > 0.5).float().mean(dim=-1)
    mass_fraction = shared.clamp_min(0).sum(dim=-1) / regions
    fraction_mean = _mean(fraction)
    mass_mean = _mean(mass_fraction)
    return {
        "fraction_above_half": fraction_mean,
        "mass_fraction": mass_mean,
        "open": fraction_mean > D1_MAX_FRACTION_ABOVE_HALF or mass_mean > D1_MAX_MASS_FRACTION,
    }


def d2_soft_versus_hard(soft_logit: torch.Tensor, hard_logit: torch.Tensor) -> dict:
    gap = _mean(soft_logit - hard_logit)
    return {"soft_minus_hard": gap, "open": gap > D2_MAX_SOFT_MINUS_HARD}


def d3_foil_suppression(
    z_k_keep: torch.Tensor,
    z_l_keep: torch.Tensor,
    z_k_full: torch.Tensor,
    z_l_full: torch.Tensor,
) -> dict:
    delta_k = z_k_keep - z_k_full
    delta_l = z_l_keep - z_l_full
    support = delta_k.clamp_min(0)
    suppress = (-delta_l).clamp_min(0)
    share = suppress / (support + suppress).clamp_min(1e-8)
    margin_rose = (z_k_keep - z_l_keep) > (z_k_full - z_l_full)
    if bool(margin_rose.any()):
        suppression_share = _mean(share[margin_rose])
    else:
        suppression_share = 0.0
    return {
        "z_k_keep": _mean(z_k_keep),
        "z_l_keep": _mean(z_l_keep),
        "z_k_full": _mean(z_k_full),
        "z_l_full": _mean(z_l_full),
        "suppression_share": suppression_share,
        "open": suppression_share > D3_MAX_SUPPRESSION_SHARE,
    }


def d4_arbitrary_cells(real_ratio: torch.Tensor, null_ratio: torch.Tensor) -> dict:
    real = _mean(real_ratio)
    null = _mean(null_ratio)
    return {
        "capture_ratio": real,
        "null_capture_ratio": null,
        "open": real < D4_MIN_CAPTURE_RATIO or real <= null,
    }


def d5_budget(mask_mass: torch.Tensor, target_mass: torch.Tensor) -> dict:
    ratio = mask_mass / target_mass.clamp_min(1e-8)
    value = _mean(ratio)
    return {"mass_ratio": value, "open": value > D5_MAX_MASS_RATIO}


def d6_shift_masks(robust: torch.Tensor, shortcut: torch.Tensor) -> dict:
    """``robust`` and ``shortcut`` are ``(B, R)``.

    Disjointness alone is not the failure. The failure is a shortcut mask that
    copies ``1 - robust`` and therefore covers the rest of the frame.
    """
    fraction = _mean((robust > 0.5).float().mean(dim=-1))
    deviation = _mean((shortcut - (1.0 - robust)).abs())
    shortcut_mass = _mean(shortcut.clamp_min(0).mean(dim=-1))
    copies_complement = (
        deviation < D6_MAX_COMPLEMENT_DEVIATION and shortcut_mass > D6_MIN_COMPLEMENT_MASS
    )
    return {
        "robust_fraction_above_half": fraction,
        "complement_deviation": deviation,
        "shortcut_mass_fraction": shortcut_mass,
        "open": fraction > D6_MAX_FRACTION_ABOVE_HALF or copies_complement,
    }
