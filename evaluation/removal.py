"""Removal operators. Optimisation uses blur. Scoring uses ROAD imputation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def noisy_linear_impute(
    x: torch.Tensor,
    keep: torch.Tensor,
    iters: int,
    noise: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Fill ``keep == 0`` pixels from their neighbours (ROAD).

    Jacobi iteration on the neighbour-average system. Known pixels stay pinned.
    ``keep`` is ``(B, 1, H, W)`` and ``x`` is ``(B, C, H, W)`` in ``[0, 1]``.
    """
    kernel = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        device=x.device,
    ).view(1, 1, 3, 3) / 8.0
    channels = x.shape[1]
    kernel_c = kernel.expand(channels, 1, 3, 3)
    current = x * keep
    for _ in range(iters):
        neighbours = F.conv2d(
            F.pad(current, (1, 1, 1, 1), mode="replicate"), kernel_c, groups=channels
        )
        weight = F.conv2d(
            F.pad(keep, (1, 1, 1, 1), mode="replicate"), kernel
        ).clamp_min(1e-6)
        filled = neighbours / weight
        current = x * keep + filled * (1.0 - keep)
    if noise > 0:
        draw = torch.randn(current.shape, generator=gen, device="cpu").to(current.device)
        current = current + draw * noise * (1.0 - keep)
    return current.clamp(0, 1)
