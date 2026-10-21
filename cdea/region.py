"""The evidence region and the subtraction rule. FORMULATION_DECOMPOSITION.md v2 sections 3.1, 3.2 and 5.

``E_k`` and ``E_l`` are per-class evidence on the R cells, clamped at 0 and
sum-normalised. The overlap of a cell is ``o(r) = min(E_k(r), E_l(r))``. The
evidence region ``Omega`` is the union of each class's smallest set of top
cells holding at least ``rho`` of its mass. Every method divides ``Omega`` and
only ``Omega``.

Roles are indexed ``U_K, U_L, SHARED, NEITHER``. A cell outside ``Omega`` is
``OUTSIDE``. The subtraction rule labels each cell of ``Omega`` with the
largest of ``E_k - m``, ``E_l - m`` and ``m``; the first role wins a tie, and
the rule never labels a cell ``NEITHER``.
"""

from __future__ import annotations

import torch

RHO = 0.5
ROLES = ("uk", "ul", "s", "none")
U_K, U_L, SHARED, NEITHER = 0, 1, 2, 3
OUTSIDE = -1
# Slack on the cumulative mass, so that float rounding does not add a cell.
MASS_TOL = 1e-6


def normalise(evidence: torch.Tensor) -> torch.Tensor:
    """Clamp at 0 and sum-normalise over the last dimension. An all-zero map stays zero."""
    positive = evidence.clamp_min(0)
    return positive / positive.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def top_mass_cells(evidence: torch.Tensor, rho: float = RHO) -> torch.Tensor:
    """``(..., R)`` bool: the smallest set of top cells holding at least ``rho`` of the mass.

    ``evidence`` is normalised. Ties go to the lower cell index. Cells with no
    mass are never included, so a map with no mass gives an empty set.
    """
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    values, order = torch.sort(evidence, dim=-1, descending=True, stable=True)
    before = values.cumsum(dim=-1) - values
    take = (before < float(rho) - MASS_TOL) & (values > 0)
    return torch.zeros_like(take).scatter(-1, order, take)


def evidence_region(evidence: torch.Tensor, rho: float = RHO) -> torch.Tensor:
    """``(B, R)`` bool ``Omega = C_k(rho) | C_l(rho)`` from the ``(B, 2, R)`` normalised evidence."""
    if evidence.ndim != 3 or evidence.shape[1] != 2:
        raise ValueError("evidence must be (batch, 2 classes, cells)")
    cells = top_mass_cells(evidence, rho)
    return cells[:, 0] | cells[:, 1]


def cell_overlap(evidence: torch.Tensor) -> torch.Tensor:
    """``(B, R)`` overlap ``o(r) = min(E_k(r), E_l(r))``. Its sum O is in [0, 1]."""
    return torch.minimum(evidence[:, 0], evidence[:, 1])


def rule_parts(evidence: torch.Tensor) -> torch.Tensor:
    """``(B, 3, R)`` subtraction parts ``(E_k - m, E_l - m, m)`` with ``m = o(r)``."""
    m = cell_overlap(evidence)
    return torch.stack([evidence[:, 0] - m, evidence[:, 1] - m, m], dim=1)


def rule_labels(parts: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """``(B, R)`` long: each cell of ``Omega`` takes its largest part; ``OUTSIDE`` elsewhere."""
    labels = parts.argmax(dim=1)
    return torch.where(omega, labels, torch.full_like(labels, OUTSIDE))


def roles_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """``(B, 4, R)`` bool role membership from ``(B, R)`` labels. ``OUTSIDE`` is in no role."""
    return torch.stack([labels == role for role in range(len(ROLES))], dim=1)
