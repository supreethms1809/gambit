"""Vision grid EvidenceUnitSpace: regions are grid cells, keep/remove via mask + baseline."""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from core.types import Tensor
from core.unit_space import EvidenceUnitSpace


class VisionGridUnitSpace(EvidenceUnitSpace):
    """Unit space over a grid_h x grid_w spatial grid (e.g. 7x7 -> 49 units)."""

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        baseline: str = "blur",
        blur_kernel_size: int = 15,
        embed_dim: Optional[int] = None,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self._num_units = grid_h * grid_w
        self.baseline = baseline  # "mean" or "blur"
        self.blur_kernel_size = blur_kernel_size
        if embed_dim is not None and embed_dim <= 0:
            raise ValueError("embed_dim must be > 0 when provided")
        self.embed_dim = embed_dim
        self._embed_cache: Dict[Tuple[str, str], Tensor] = {}

    def num_units(self) -> int:
        return self._num_units

    def _region_to_pixel_mask(self, m: Tensor, h: int, w: int) -> Tensor:
        """Convert region mask (B, R) to pixel mask (B, 1, H, W)."""
        B, R = m.shape
        assert R == self._num_units
        # (B, R) -> (B, 1, grid_h, grid_w)
        pm = m.view(B, 1, self.grid_h, self.grid_w)
        # Upsample to input spatial size
        pm = F.interpolate(pm, size=(h, w), mode="bilinear", align_corners=False)
        return pm  # (B, 1, H, W)

    def _baseline(self, x: Tensor) -> Tensor:
        """Baseline image: mean or blur."""
        if self.baseline == "mean":
            return x.mean(dim=(2, 3), keepdim=True).expand_as(x)
        # blur: average pooling then upsample back
        k = self.blur_kernel_size
        if min(x.shape[2], x.shape[3]) < k:
            k = min(x.shape[2], x.shape[3]) | 1
        x_blur = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        # handle size mismatch from stride
        if x_blur.shape[2:] != x.shape[2:]:
            x_blur = F.interpolate(x_blur, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x_blur

    def keep(self, x: Tensor, m: Tensor) -> Tensor:
        """Keep regions in m: x_keep = mask*x + (1-mask)*baseline(x)."""
        B, C, H, W = x.shape
        pixel_mask = self._region_to_pixel_mask(m, H, W)  # (B, 1, H, W)
        base = self._baseline(x)
        return pixel_mask * x + (1 - pixel_mask) * base

    def remove(self, x: Tensor, m: Tensor) -> Tensor:
        """Remove regions in m: replace those with baseline."""
        B, C, H, W = x.shape
        pixel_mask = self._region_to_pixel_mask(m, H, W)
        base = self._baseline(x)
        return (1 - pixel_mask) * x + pixel_mask * base

    def _sinusoid(self, pos: Tensor, dim: int) -> Tensor:
        """Standard sinusoidal encoding for a 1D coordinate list."""
        if dim <= 0:
            return pos.new_zeros((pos.shape[0], 0))
        out = pos.new_zeros((pos.shape[0], dim))
        steps = torch.arange(0, dim, 2, device=pos.device, dtype=pos.dtype)
        scale = -math.log(10000.0) / max(dim, 1)
        inv_freq = torch.exp(steps * scale)
        angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        out[:, 0::2] = torch.sin(angles)
        if dim > 1:
            out[:, 1::2] = torch.cos(angles[:, : out[:, 1::2].shape[1]])
        return out

    def _build_base_embedding(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.embed_dim is None:
            raise RuntimeError("embed_dim is None; no embeddings are available")
        D = self.embed_dim
        d_row = D // 2
        d_col = D - d_row

        row_pos = torch.linspace(0.0, 1.0, self.grid_h, device=device, dtype=dtype)
        col_pos = torch.linspace(0.0, 1.0, self.grid_w, device=device, dtype=dtype)
        row_emb = self._sinusoid(row_pos, d_row)  # (grid_h, d_row)
        col_emb = self._sinusoid(col_pos, d_col)  # (grid_w, d_col)

        parts = []
        if d_row > 0:
            parts.append(row_emb.unsqueeze(1).expand(self.grid_h, self.grid_w, d_row))
        if d_col > 0:
            parts.append(col_emb.unsqueeze(0).expand(self.grid_h, self.grid_w, d_col))
        return torch.cat(parts, dim=-1).reshape(self._num_units, D)

    def embed_units(self, units: Tensor) -> Optional[Tensor]:
        """Optional deterministic embeddings for grid regions: (B, R, D)."""
        if self.embed_dim is None:
            return None
        key = (str(units.device), str(units.dtype))
        base = self._embed_cache.get(key)
        if base is None:
            base = self._build_base_embedding(units.device, units.dtype)
            self._embed_cache[key] = base
        B = units.shape[0]
        return base.unsqueeze(0).expand(B, -1, -1)
