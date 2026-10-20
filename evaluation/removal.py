"""Removal operators. Optimisation uses blur. Scoring uses ROAD imputation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# ROAD's linear imputer (Rong et al. 2022), road_evaluation commit
# e6b9987ffbd79654125af2f581b5dae70e831b09, road/imputations.py
# ``neighbors_weights``: 1/6 for the four edge neighbours, 1/12 for the corners.
ROAD_WEIGHTS = torch.tensor([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0.0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]])
# Max-abs residual at which the solve stops. In float32 this leaves the fill
# within about 5e-4 of ROAD's direct solve, against ROAD's own noise of 0.01.
ROAD_TOL = 1e-6
# Cap on solver steps for the converged fill. A 224 px image with 95% removed
# converges in about 600; reaching the cap raises.
ROAD_MAX_ITER = 4000
# The ROAD removal every score uses: the converged fill and ROAD's noise.
ROAD = dict(iters=None, noise=0.01)


def _neighbour_sum(u: torch.Tensor) -> torch.Tensor:
    """Weighted sum of each pixel's in-image neighbours. Zero padding drops the outside ones, as ROAD does."""
    channels = u.shape[1]
    kernel = ROAD_WEIGHTS.to(device=u.device, dtype=u.dtype).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    return F.conv2d(u, kernel, padding=1, groups=channels)


def noisy_linear_impute(
    x: torch.Tensor,
    keep: torch.Tensor,
    iters: Optional[int],
    noise: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Fill ``keep == 0`` pixels as ROAD's ``NoisyLinearImputer`` does.

    Each removed pixel equals the weighted mean of its in-image neighbours
    (``ROAD_WEIGHTS``, renormalised at the border), and kept pixels are pinned.
    ROAD solves that sparse system directly with scipy. Here the same system,
    ``s * u - W u = W x_kept`` on the removed pixels, is solved by conjugate
    gradient with a diagonal preconditioner, batched over images and channels.
    The system is symmetric and positive definite whenever an image keeps a
    pixel. Each image and channel stops at its own residual, so a chunk gets
    the same fill as the whole sample.

    ``iters=None`` is the converged fill used for every paper score. It stops
    when the max-abs residual is at most ``ROAD_TOL`` and raises if
    ``ROAD_MAX_ITER`` steps are not enough. An integer runs at most that many
    steps without the check; ``0`` leaves removed pixels black, which tests use
    for exact numbers.

    ``keep`` is ``(B, 1, H, W)`` and is read as binary at 0.5. ``x`` is
    ``(B, C, H, W)`` in ``[0, 1]``. Gaussian noise of std ``noise`` is then
    added to removed pixels, as in ROAD, and the image is clamped to
    ``[0, 1]``. An image with no kept pixel stays black.
    """
    known = (keep > 0.5).to(dtype=x.dtype)
    hole = 1.0 - known
    weight = _neighbour_sum(torch.ones_like(known))
    rhs = hole * _neighbour_sum(x * known)

    def system(u):
        return hole * (weight * u - _neighbour_sum(u))

    dims = (-1, -2)
    fill = torch.zeros_like(x)
    residual = rhs
    direction = residual / weight
    rz = (residual * direction).sum(dims, keepdim=True)
    limit = ROAD_MAX_ITER if iters is None else int(iters)
    converged = False
    for _ in range(limit):
        active = residual.abs().amax(dims, keepdim=True) > ROAD_TOL
        if iters is None and not bool(active.any()):
            converged = True
            break
        step = system(direction)
        curvature = (direction * step).sum(dims, keepdim=True)
        alpha = torch.where(active & (curvature > 0), rz / curvature.clamp_min(1e-30), torch.zeros_like(rz))
        fill = fill + alpha * direction
        residual = residual - alpha * step
        preconditioned = residual / weight
        rz_next = (residual * preconditioned).sum(dims, keepdim=True)
        beta = torch.where(rz > 0, rz_next / rz.clamp_min(1e-30), torch.zeros_like(rz))
        direction = preconditioned + beta * direction
        rz = rz_next
    if iters is None and not converged:
        if bool((residual.abs().amax(dims) > ROAD_TOL).any()):
            raise RuntimeError(f"ROAD fill did not converge in {ROAD_MAX_ITER} steps")
    current = x * known + hole * fill
    if noise > 0:
        draw = torch.randn(current.shape, generator=gen, device="cpu").to(current.device)
        current = current + draw * noise * hole
    return current.clamp(0, 1)
