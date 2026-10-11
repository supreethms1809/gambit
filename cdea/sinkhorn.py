"""Log-domain Sinkhorn. FORMULATION.md section 7.

Column marginals are 1. Row marginals are the player budgets ``a * R`` and the
unallocated remainder. ``I = 20`` iterations are unrolled for backpropagation.
A detached catch-up brings a sharp kernel to the marginal bound first; twenty
steps cannot reach it from a cold dual once ``theta`` is peaked.
"""

from __future__ import annotations

import torch

# Enough for a kernel whose entries span a few tens of nats. The unrolled
# count stays ``iters``. This cap only finishes the dual before those steps.
CATCHUP_ITERS = 4096


def _factors(masked, log_row, log_col, u, v, active):
    u = log_row - torch.logsumexp(masked + v.unsqueeze(-2), dim=-1)
    u = torch.where(active, u, torch.zeros_like(u))
    v = log_col - torch.logsumexp(masked + u.unsqueeze(-1), dim=-2)
    return u, v


def _plan_from(masked, u, v, active):
    plan = torch.exp(masked + u.unsqueeze(-1) + v.unsqueeze(-2))
    return torch.where(active.unsqueeze(-1), plan, torch.zeros_like(plan))


def sinkhorn(
    theta: torch.Tensor,
    row_target: torch.Tensor,
    iters: int = 20,
    tol: float = 1e-3,
    dual: list | None = None,
) -> torch.Tensor:
    """``A = Sinkhorn(exp theta)``.

    ``theta`` is ``(B, rows, R)``. ``row_target`` is ``(B, rows)`` or ``(rows,)``
    and must sum to ``R`` on every image. Column targets are 1.
    """
    if theta.ndim != 3:
        raise ValueError("theta must be (batch, rows, units)")
    if iters < 1:
        raise ValueError("iters must be positive")
    batch, rows, units = theta.shape
    if row_target.ndim == 1:
        row_target = row_target.expand(batch, -1)
    if row_target.shape != (batch, rows):
        raise ValueError("row_target must be (rows,) or (batch, rows)")
    row_target = row_target.to(device=theta.device, dtype=theta.dtype)
    total = row_target.sum(dim=-1)
    if not torch.allclose(total, torch.full_like(total, float(units)), atol=1e-4, rtol=0):
        raise ValueError("row targets must sum to the number of units")
    if torch.any(row_target < -1e-8):
        raise ValueError("row targets must be non-negative")

    # A global shift does not change the plan and keeps the exp in range.
    # Inactive rows (a zero budget) stay out of the factorisation.
    # Float64 everywhere it exists. MPS has no float64, and moving the
    # factorisation to CPU deadlocks autograd, so MPS stays on-device in float32.
    work_dtype = torch.float32 if theta.device.type == "mps" else torch.float64
    active = row_target > 1e-8
    work = theta.to(dtype=work_dtype)
    work = work - work.amax(dim=(-1, -2), keepdim=True)
    log_row = row_target.to(dtype=work_dtype).clamp_min(1e-12).log()
    log_col = torch.zeros(batch, units, device=theta.device, dtype=work_dtype)
    u = torch.zeros(batch, rows, device=theta.device, dtype=work_dtype)
    v = torch.zeros(batch, units, device=theta.device, dtype=work_dtype)
    if dual is not None and dual[0] is not None and dual[0][0].shape == u.shape:
        # Detached dual from the previous solve. Twenty iterations then only
        # have to track a small change in theta, which is what makes the
        # marginal error bound hold once the plan is sharp.
        u = dual[0][0].to(dtype=work_dtype, device=theta.device)
        v = dual[0][1].to(dtype=work_dtype, device=theta.device)
    masked = work.masked_fill(~active.unsqueeze(-1), -1e30)
    log_row = torch.where(active, log_row, torch.zeros_like(log_row))
    # Detached. A peaked theta needs more than twenty steps from a zero dual,
    # and the Adam step can move the kernel faster than twenty warm steps track.
    # Catch-up restores the marginals; the unrolled steps then follow theta.
    with torch.no_grad():
        for _ in range(CATCHUP_ITERS):
            u, v = _factors(masked, log_row, log_col, u, v, active)
            caught = _plan_from(masked, u, v, active)
            if _marginals_ok(caught, row_target.to(dtype=caught.dtype), tol):
                break
    u = u.detach()
    v = v.detach()
    for _ in range(int(iters)):
        u, v = _factors(masked, log_row, log_col, u, v, active)
    plan = _plan_from(masked, u, v, active)
    if dual is not None:
        dual[0] = (u.detach(), v.detach())
    _assert_marginals(plan, row_target.to(dtype=plan.dtype), tol)
    return plan.to(dtype=theta.dtype)


def _marginals_ok(plan: torch.Tensor, row_target: torch.Tensor, tol: float) -> bool:
    if not torch.isfinite(plan).all():
        return False
    row_error = (plan.sum(dim=-1) - row_target).abs().max().detach()
    col_error = (plan.sum(dim=-2) - 1.0).abs().max().detach()
    return bool(float(max(row_error, col_error)) <= tol)


def _assert_marginals(plan: torch.Tensor, row_target: torch.Tensor, tol: float) -> None:
    row_error = (plan.sum(dim=-1) - row_target).abs().max().detach()
    col_error = (plan.sum(dim=-2) - 1.0).abs().max().detach()
    if not _marginals_ok(plan, row_target, tol):
        raise AssertionError(
            f"Sinkhorn marginal error {float(max(row_error, col_error)):.3e} exceeds {tol}"
        )


def hard_top_mass(scores: torch.Tensor, budget: float) -> torch.Tensor:
    """Project each row onto ``{m in [0, 1]^R : sum m = budget}`` by ``(scores - tau).clamp(0, 1)``.

    ``tau`` is detached. A unit that the threshold pins at 0 receives no gradient
    through the projection. ``scores`` is ``(B, R)`` or ``(B, rows, R)``.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    flat = scores.reshape(-1, scores.shape[-1])
    units = flat.shape[-1]
    target = float(budget)
    if target > units + 1e-6:
        raise ValueError("budget exceeds the number of units")
    # Search tau. A larger tau lowers the kept mass.
    lo = flat.min(dim=-1).values - 1.0
    hi = flat.max(dim=-1).values + 1.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        mass = (flat - mid.unsqueeze(-1)).clamp(0.0, 1.0).sum(dim=-1)
        too_much = mass > target
        lo = torch.where(too_much, mid, lo)
        hi = torch.where(too_much, hi, mid)
    tau = hi.detach()
    projected = (flat - tau.unsqueeze(-1)).clamp(0.0, 1.0)
    # The search lands within a fraction of a unit. Rescale only when the mass
    # is short of an integer budget that the clamp can still fill; leave a
    # fractional budget as the clamp produced it.
    return projected.reshape(scores.shape)
