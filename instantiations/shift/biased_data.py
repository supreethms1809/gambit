"""
Biased datasets for robust vs shortcut (Instantiation II).

Three datasets, each with a clear spatial shortcut correlated with the label:

- **ColoredMNIST**     — global color tint (hue) on the digit
- **ColoredCIFAR10**   — class-correlated color patch in the top-left corner
- **TextureBiasedMNIST** — class-correlated sinusoidal stripe texture in the background

Each dataset ships with an ``env_batch_*`` function that produces three environments:
  xs[0] = ID  (original shortcut)
  xs[1] = OOD1 (shortcut changed to class+1)
  xs[2] = OOD2 (shortcut changed to class+2)

Use the factory functions ``build_biased_dataset`` / ``get_env_batch_fn`` to route
between datasets by name string in ``eval_robust_shortcut.py``.
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
    hues = color_idx.float() / max(num_colors, 1)
    r = (1 + torch.cos(2 * 3.14159 * hues)) * 0.5
    g = (1 + torch.cos(2 * 3.14159 * hues + 2.094)) * 0.5
    b = (1 + torch.cos(2 * 3.14159 * hues + 4.188)) * 0.5
    rgb = torch.stack([r, g, b], dim=1).view(B, 3, 1, 1)
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
    ):
        from torchvision.datasets import MNIST
        from torchvision import transforms
        self.mnist = MNIST(root=root, train=train, download=download,
                           transform=transforms.ToTensor())
        self.correlation = correlation
        self.num_colors = num_colors

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        im, label = self.mnist[i]
        if torch.rand(1).item() < self.correlation:
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
    """xs = [x_id, x_ood1, x_ood2] — same digit, different hue for OOD."""
    B = x.shape[0]
    device = x.device
    y_cpu = y.cpu()
    x_ood1 = x.clone()
    x_ood2 = x.clone()
    for b in range(B):
        other  = (int(y_cpu[b].item()) + 1) % num_colors
        other2 = (int(y_cpu[b].item()) + 2) % num_colors
        g = x[b:b+1].mean(dim=1, keepdim=True)
        x_ood1[b:b+1] = colorize_mnist(g, other, num_colors)
        x_ood2[b:b+1] = colorize_mnist(g, other2, num_colors)
    return EnvBatch(xs=[x, x_ood1, x_ood2], env_ids=["id", "ood1", "ood2"])


# ---------------------------------------------------------------------------
# ColoredCIFAR10
# ---------------------------------------------------------------------------

# Patch size as a fraction of image width (25%). At 32×32 → 8×8 patch.
# At 224×224 (after upscale) → 56×56 patch.
_CIFAR_PATCH_FRAC = 0.25


class ColoredCIFAR10(Dataset):
    """CIFAR-10 with a class-correlated solid-color patch in the top-left corner.

    Shortcut: the patch color (hue) is correlated with the class label.
    Robust:   the CIFAR-10 object content in the remaining image area.

    The patch occupies the top-left 25% × 25% of the image (8×8 px at 32×32).
    In the 7×7 grid unit space at 224×224, this maps to roughly the top-left
    2×2 cells — clearly separable from the object in the remaining cells.
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        correlation: float = 1.0,
        num_classes: int = 10,
    ):
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        self.cifar = CIFAR10(root=root, train=train, download=download,
                             transform=transforms.ToTensor())
        self.correlation = correlation
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        im, label = self.cifar[i]   # (3, 32, 32) float32 in [0,1]
        if torch.rand(1).item() < self.correlation:
            color_class = label
        else:
            color_class = int(torch.randint(0, self.num_classes, (1,)).item())
        color = _class_hue(color_class, self.num_classes)
        H, W = im.shape[1], im.shape[2]
        ph = max(1, int(H * _CIFAR_PATCH_FRAC))
        pw = max(1, int(W * _CIFAR_PATCH_FRAC))
        return _stamp_patch(im, ph, pw, color), label


def env_batch_colored_cifar10(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
) -> EnvBatch:
    """xs = [x_id, x_ood1, x_ood2] — same image content, different corner color."""
    B, C, H, W = x.shape
    ph = max(1, int(H * _CIFAR_PATCH_FRAC))
    pw = max(1, int(W * _CIFAR_PATCH_FRAC))
    x_ood1 = x.clone()
    x_ood2 = x.clone()
    y_cpu = y.cpu()
    for b in range(B):
        other  = (int(y_cpu[b].item()) + 1) % num_classes
        other2 = (int(y_cpu[b].item()) + 2) % num_classes
        x_ood1[b] = _stamp_patch(x[b], ph, pw, _class_hue(other,  num_classes))
        x_ood2[b] = _stamp_patch(x[b], ph, pw, _class_hue(other2, num_classes))
    return EnvBatch(xs=[x, x_ood1, x_ood2], env_ids=["id", "ood1", "ood2"])


# ---------------------------------------------------------------------------
# TextureBiasedMNIST
# ---------------------------------------------------------------------------

class TextureBiasedMNIST(Dataset):
    """MNIST digits composited onto a class-correlated sinusoidal stripe texture.

    Shortcut: the orientation angle of the background stripe pattern is correlated
              with the class label (each class gets a unique stripe angle).
    Robust:   the digit shape (foreground pixels, identified by luminance threshold).

    This is a purely texture-based shortcut — no color dependency — making it
    categorically diverse from ColoredMNIST (hue shortcut) and ColoredCIFAR10
    (localized spatial color shortcut).
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        correlation: float = 1.0,
        num_classes: int = 10,
        texture_frequency: float = 6.0,
        fg_threshold: float = 0.1,
    ):
        from torchvision.datasets import MNIST
        from torchvision import transforms
        self.mnist = MNIST(root=root, train=train, download=download,
                           transform=transforms.ToTensor())
        self.correlation = correlation
        self.num_classes = num_classes
        self.texture_frequency = texture_frequency
        self.fg_threshold = fg_threshold

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        im, label = self.mnist[i]    # (1, 28, 28) grayscale
        digit_rgb = im.repeat(3, 1, 1)
        if torch.rand(1).item() < self.correlation:
            tex_class = label
        else:
            tex_class = int(torch.randint(0, self.num_classes, (1,)).item())
        H, W = digit_rgb.shape[1], digit_rgb.shape[2]
        texture = _stripe_texture(tex_class, H, W, self.num_classes, self.texture_frequency)
        return _composite_texture(digit_rgb, texture, self.fg_threshold), label


def env_batch_texture_biased_mnist(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
    texture_frequency: float = 6.0,
    fg_threshold: float = 0.1,
) -> EnvBatch:
    """xs = [x_id, x_ood1, x_ood2] — same digit, different background texture."""
    B, C, H, W = x.shape
    x_ood1 = x.clone()
    x_ood2 = x.clone()
    y_cpu = y.cpu()
    for b in range(B):
        other  = (int(y_cpu[b].item()) + 1) % num_classes
        other2 = (int(y_cpu[b].item()) + 2) % num_classes
        tex1 = _stripe_texture(other,  H, W, num_classes, texture_frequency).to(x.device)
        tex2 = _stripe_texture(other2, H, W, num_classes, texture_frequency).to(x.device)
        x_ood1[b] = _composite_texture(x[b], tex1, fg_threshold)
        x_ood2[b] = _composite_texture(x[b], tex2, fg_threshold)
    return EnvBatch(xs=[x, x_ood1, x_ood2], env_ids=["id", "ood1", "ood2"])


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

BIASED_DATASETS = ["colored_mnist", "colored_cifar10", "texture_mnist"]


def build_biased_dataset(
    name: str,
    root: str = "data",
    train: bool = False,
    download: bool = True,
    correlation: float = 0.95,
) -> Dataset:
    """Return the biased dataset for *name*.

    Args:
        name:        One of ``colored_mnist``, ``colored_cifar10``, ``texture_mnist``.
        root:        Data root directory.
        train:       Train split (True) or test split (False).
        download:    Auto-download if not present.
        correlation: Fraction of samples where the shortcut matches the label.
    """
    if name == "colored_mnist":
        return ColoredMNIST(root=root, train=train, download=download, correlation=correlation)
    if name == "colored_cifar10":
        return ColoredCIFAR10(root=root, train=train, download=download, correlation=correlation)
    if name == "texture_mnist":
        return TextureBiasedMNIST(root=root, train=train, download=download, correlation=correlation)
    raise ValueError(
        f"Unknown biased dataset: '{name}'. "
        f"Choose from: {BIASED_DATASETS}"
    )


def get_env_batch_fn(name: str) -> Callable:
    """Return the env_batch_* function for *name*."""
    if name == "colored_mnist":
        return env_batch_colored_mnist
    if name == "colored_cifar10":
        return env_batch_colored_cifar10
    if name == "texture_mnist":
        return env_batch_texture_biased_mnist
    raise ValueError(
        f"Unknown biased dataset: '{name}'. "
        f"Choose from: {BIASED_DATASETS}"
    )


# ---------------------------------------------------------------------------
# Shared metric: ID-OOD gap
# ---------------------------------------------------------------------------

def compute_id_ood_gap(
    model: torch.nn.Module,
    env: EnvBatch,
    unit_space: Any,
    m_shortcut: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Shortcut gap: logit of predicted class under shortcut mask on ID vs OOD mean."""
    with torch.no_grad():
        logits_id = model(unit_space.keep(env.xs[0], m_shortcut))
        z_id = logits_id.gather(1, y.unsqueeze(1)).squeeze(1)
        z_ood = []
        for xe in env.xs[1:]:
            logits_o = model(unit_space.keep(xe, m_shortcut))
            z_ood.append(logits_o.gather(1, y.unsqueeze(1)).squeeze(1))
        z_ood_mean = torch.stack(z_ood, dim=0).mean(dim=0)
    return (z_id - z_ood_mean).mean().item()
