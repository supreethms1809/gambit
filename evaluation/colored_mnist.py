"""ColoredMNIST pairs.

The digit is tinted with a hue tied to the label. Recolouring the same digit
is the other environment. The colour is not a spatial region, so robust and
shortcut players compete for the same units.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset

from core.types import EnvBatch


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def _hue_rgb(hue: float) -> Tuple[float, float, float]:
    """Convert a hue value in [0, 1) to (R, G, B) using the cosine formula."""
    r = 0.5 * (1.0 + math.cos(2.0 * math.pi * hue))
    g = 0.5 * (1.0 + math.cos(2.0 * math.pi * hue + 2.0943951))  # 2π/3
    b = 0.5 * (1.0 + math.cos(2.0 * math.pi * hue + 4.1887902))  # 4π/3
    return r, g, b


def _class_hue(class_idx: int, num_classes: int) -> Tuple[float, float, float]:
    return _hue_rgb(class_idx / max(num_classes, 1))


def _stamp_patch(
    im: torch.Tensor,
    patch_h: int,
    patch_w: int,
    color: Tuple[float, float, float],
) -> torch.Tensor:
    """Stamp a flat-color rectangle into the top-left corner of ``im`` (C, H, W).

    Returns a modified copy; does not mutate the input.
    """
    im = im.clone()
    ph = min(patch_h, im.shape[1])
    pw = min(patch_w, im.shape[2])
    im[0, :ph, :pw] = color[0]
    im[1, :ph, :pw] = color[1]
    im[2, :ph, :pw] = color[2]
    return im


def _stripe_texture(
    class_idx: int,
    H: int,
    W: int,
    num_classes: int = 10,
    frequency: float = 6.0,
) -> torch.Tensor:
    """Sinusoidal stripe texture for *class_idx*, shape (3, H, W) in [0, 1].

    Orientation angle = class_idx * π / num_classes.
    Different classes produce stripes at different angles, giving a
    texture shortcut that is spatially distributed (not color-based).
    """
    theta = class_idx * math.pi / max(num_classes, 1)
    ys = torch.linspace(-1.0, 1.0, H)
    xs = torch.linspace(-1.0, 1.0, W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
    freq = max(frequency, 1.0)
    pattern = 0.5 + 0.5 * torch.sin(freq * (xx * math.cos(theta) + yy * math.sin(theta)))
    return pattern.unsqueeze(0).repeat(3, 1, 1)       # (3, H, W)


def _composite_texture(
    digit_im: torch.Tensor,
    texture: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Composite ``texture`` behind the digit in ``digit_im`` (3, H, W).

    Foreground = pixels where mean channel value > ``threshold``.
    Returns a new tensor; does not mutate inputs.
    """
    fg = (digit_im.mean(dim=0, keepdim=True) > threshold).float()   # (1, H, W)
    return digit_im * fg + texture * (1.0 - fg)


# ---------------------------------------------------------------------------
# ColoredMNIST  (original)
# ---------------------------------------------------------------------------

def _hue_channels(color_idx: torch.Tensor, num_colors: int) -> torch.Tensor:
    """(B, 3) RGB hues. Same constants as ``colorize_mnist`` so the inverse matches."""
    hues = color_idx.float() / max(num_colors, 1)
    r = (1 + torch.cos(2 * 3.14159 * hues)) * 0.5
    g = (1 + torch.cos(2 * 3.14159 * hues + 2.094)) * 0.5
    b = (1 + torch.cos(2 * 3.14159 * hues + 4.188)) * 0.5
    return torch.stack([r, g, b], dim=1)


def recover_mnist_gray(
    x: torch.Tensor,
    color_idx: int | torch.Tensor,
    num_colors: int = 10,
) -> torch.Tensor:
    """Invert ``colorize_mnist`` when the color index is known.

    ``x`` is (3, H, W) or (B, 3, H, W). The largest hue channel is always at
    least 0.75 for these ten colors, so the division is stable. Returns the
    grayscale digit, shape (1, H, W) or (B, 1, H, W).
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    batch = x.shape[0]
    if isinstance(color_idx, int):
        color_idx = torch.full((batch,), color_idx, device=x.device, dtype=torch.long)
    else:
        color_idx = color_idx.to(device=x.device, dtype=torch.long).view(batch)
    rgb = _hue_channels(color_idx, num_colors).to(device=x.device)
    channel = rgb.argmax(dim=1)
    scale = rgb.gather(1, channel.view(batch, 1)).clamp(min=1e-3)
    picked = x.gather(1, channel.view(batch, 1, 1, 1).expand(batch, 1, x.shape[2], x.shape[3]))
    gray = (picked / scale.view(batch, 1, 1, 1)).clamp(0, 1)
    return gray.squeeze(0) if single else gray


def colorize_mnist(im: torch.Tensor, color_idx: int, num_colors: int = 10) -> torch.Tensor:
    """
    im: (B, 1, H, W) or (1, H, W) MNIST digit in [0,1].
    color_idx: int or (B,) tensor; color index in [0, num_colors-1].
    Returns (B, 3, H, W) RGB with hue determined by color_idx.
    """
    if im.dim() == 3:
        im = im.unsqueeze(0)
    B, _, H, W = im.shape
    device = im.device
    if isinstance(color_idx, int):
        color_idx = torch.full((B,), color_idx, device=device, dtype=torch.long)
    rgb = _hue_channels(color_idx.to(device=device), num_colors).view(B, 3, 1, 1)
    return (im * rgb).clamp(0, 1)


class ColoredMNIST(Dataset):
    """MNIST with color correlated to label (shortcut)."""

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        correlation: float = 1.0,
        num_colors: int = 10,
        seed: Optional[int] = None,
    ):
        from torchvision.datasets import MNIST
        from torchvision import transforms
        self.mnist = MNIST(root=root, train=train, download=download,
                           transform=transforms.ToTensor())
        self.correlation = correlation
        self.num_colors = num_colors
        # A seed fixes the color of each index. Without it, every read draws a
        # new color, so a val score changes between epochs.
        self._colors: Optional[torch.Tensor] = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            labels = torch.as_tensor(self.mnist.targets).long()
            agree = torch.rand(len(self.mnist), generator=generator) < correlation
            random_color = torch.randint(0, num_colors, (len(self.mnist),), generator=generator)
            self._colors = torch.where(agree, labels, random_color)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        im, label = self.mnist[i]
        if self._colors is not None:
            color_idx = int(self._colors[i].item())
        elif torch.rand(1).item() < self.correlation:
            color_idx = label
        else:
            color_idx = int(torch.randint(0, self.num_colors, (1,)).item())
        im_c = colorize_mnist(im.unsqueeze(0), color_idx, self.num_colors).squeeze(0)
        return im_c, label


def env_batch_colored_mnist(
    x: torch.Tensor,
    y: torch.Tensor,
    num_colors: int = 10,
) -> EnvBatch:
    """xs = [x_id, x_ood1, x_ood2].

    OOD views are the same digit recolored with hue ``label+1`` and ``label+2``.
    The digit is recovered by inverting the label hue. Averaging the colored
    channels is not that inverse: it scales the digit by the mean of the hue.
    """
    y_cpu = y.detach().cpu()
    ood1 = []
    ood2 = []
    for b in range(x.shape[0]):
        label = int(y_cpu[b].item()) % num_colors
        gray = recover_mnist_gray(x[b], label, num_colors)
        ood1.append(colorize_mnist(gray, (label + 1) % num_colors, num_colors).squeeze(0))
        ood2.append(colorize_mnist(gray, (label + 2) % num_colors, num_colors).squeeze(0))
    return EnvBatch(
        xs=[x, torch.stack(ood1), torch.stack(ood2)],
        env_ids=["id", "ood1", "ood2"],
    )

