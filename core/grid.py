"""Grid units and the one deletion baseline.

FORMULATION.md section 2. ``phi_r`` is the hard (nearest-cell) indicator.
``b(x)`` is a Gaussian blur with ``sigma`` equal to half a unit side,
kernel ``2 * ceil(3 * sigma) + 1``, and reflect padding. The scorer's blur
operator calls this same function.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def grid_of(backbone: str) -> tuple[int, int]:
    """ResNet-50 is 7 by 7. ViT-B/16 is 14 by 14. Anything else is 7 by 7."""
    if backbone == "vit_b_16":
        return 14, 14
    return 7, 7


def unit_side(height: int, grid_h: int) -> float:
    return float(height) / float(grid_h)


def blur_sigma(height: int, grid_h: int) -> float:
    """Half a unit side. 16 px at 7 by 7 on 224, 8 px at 14 by 14 on 224."""
    return 0.5 * unit_side(height, grid_h)


def blur_kernel_size(sigma: float) -> int:
    return int(2 * math.ceil(3.0 * float(sigma)) + 1)


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    size = blur_kernel_size(sigma)
    radius = size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. Reflect padding, so a constant image is unchanged."""
    if x.ndim != 4:
        raise ValueError("blur expects (batch, channels, height, width)")
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    kernel = _gaussian_kernel(sigma, x.device, x.dtype)
    channels = x.shape[1]
    pad = kernel.shape[0] // 2
    if pad >= x.shape[-1] or pad >= x.shape[-2]:
        raise ValueError("blur kernel is wider than the image; use a finer grid or a larger image")
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    blurred = F.conv2d(padded, horizontal, groups=channels)
    return F.conv2d(blurred, vertical, groups=channels)


def deletion_baseline(x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """``b(x)``. Sigma uses the height and ``grid_h``; width uses ``grid_w`` only as a check."""
    if grid_h < 1 or grid_w < 1:
        raise ValueError("grid size must be positive")
    if x.shape[-2] < 1:
        raise ValueError("image has no spatial size")
    return gaussian_blur(x, blur_sigma(x.shape[-2], grid_h))


def _cell_index(length: int, cells: int, device) -> torch.Tensor:
    coords = torch.arange(length, device=device)
    index = torch.div((coords.float() + 0.5) * cells, length, rounding_mode="floor")
    return index.long().clamp(max=cells - 1)


def _cell_onehot(length: int, cells: int, device, dtype) -> torch.Tensor:
    """``(length, cells)``: row ``i`` is the indicator of the cell that pixel ``i`` falls in."""
    return F.one_hot(_cell_index(length, cells, device), cells).to(dtype)


def upsample_units(
    unit_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Hard ``phi``: each pixel takes the value of its nearest cell. ``(B, 1, H, W)``.

    A product with one-hot cell indicators, not indexing. Each pixel has one
    nonzero term, so values are exact, and the backward pass is a fixed-order
    sum. The backward of indexing accumulates in a different order each call
    on GPUs, and Adam amplifies that into a different allocation.
    """
    if unit_mask.ndim != 2 or unit_mask.shape[1] != grid_h * grid_w:
        raise ValueError("unit mask must be (batch, grid_h * grid_w)")
    grid = unit_mask.reshape(unit_mask.shape[0], grid_h, grid_w)
    rows = _cell_onehot(height, grid_h, unit_mask.device, unit_mask.dtype)
    cols = _cell_onehot(width, grid_w, unit_mask.device, unit_mask.dtype)
    return (rows @ grid @ cols.t()).unsqueeze(1)


def pool_sum(pixel: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Sum pixels into their cells. ``pixel`` is ``(B, H, W)`` and the result is ``(B, R)``.

    A fixed-order product with one-hot cell indicators, so it is deterministic
    on GPUs, unlike ``scatter_add``.
    """
    if pixel.ndim != 3:
        raise ValueError("pool_sum expects (batch, height, width)")
    batch, height, width = pixel.shape
    rows = _cell_onehot(height, grid_h, pixel.device, pixel.dtype)
    cols = _cell_onehot(width, grid_w, pixel.device, pixel.dtype)
    return (rows.t() @ pixel @ cols).reshape(batch, grid_h * grid_w)


def boundary_offset(height: int, width: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    """``s``: the largest pixel offset of the boundary-robust payoff, half a unit side, floored."""
    return int(unit_side(height, grid_h) // 2), int(unit_side(width, grid_w) // 2)


def shift_pixels(pixel: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Move a ``(B, 1, H, W)`` mask by ``(dy, dx)`` pixels. Vacated pixels are 0; nothing wraps."""
    dy, dx = int(dy), int(dx)
    if dy == 0 and dx == 0:
        return pixel
    if abs(dy) >= pixel.shape[-2] or abs(dx) >= pixel.shape[-1]:
        raise ValueError("offset is larger than the image")
    # Negative padding crops, so this pads one side and crops the other.
    return F.pad(pixel, (dx, -dx, dy, -dy))


def delete(
    x: torch.Tensor,
    unit_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """``x ⊖ M = (1 - M̃) x + M̃ b(x)`` with hard ``M̃``, moved by ``offset`` pixels."""
    pixel = shift_pixels(upsample_units(unit_mask, grid_h, grid_w, x.shape[-2], x.shape[-1]), *offset)
    base = deletion_baseline(x, grid_h, grid_w)
    return (1.0 - pixel) * x + pixel * base


def transplant(
    x_id: torch.Tensor,
    x_env: torch.Tensor,
    unit_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Copy the masked cells from ``x_env`` into ``x_id``. Hard ``phi``."""
    if x_id.shape != x_env.shape:
        raise ValueError("transplant views must share a shape")
    pixel = upsample_units(unit_mask, grid_h, grid_w, x_id.shape[-2], x_id.shape[-1])
    return (1.0 - pixel) * x_id + pixel * x_env


def keep(
    x: torch.Tensor,
    unit_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Keep ``M``, moved by ``offset`` pixels, and replace the rest with ``b(x)``."""
    pixel = shift_pixels(upsample_units(unit_mask, grid_h, grid_w, x.shape[-2], x.shape[-1]), *offset)
    base = deletion_baseline(x, grid_h, grid_w)
    return pixel * x + (1.0 - pixel) * base
