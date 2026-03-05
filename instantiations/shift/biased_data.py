"""
Biased setup for robust vs shortcut: dataset where shortcuts exist.
- Colored MNIST: digit + color; color correlated with label (shortcut).
- Augmentations: change color (OOD) so ID = original color, OOD = different color.
Pass: robust mask tracks object (digit), shortcut mask tracks background/cue (color);
      quantitative: ID-OOD gap (sho_gap) positive.
"""
from __future__ import annotations
from typing import Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from core.types import EnvBatch


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
    # Simple coloring: R/G/B channels as function of color_idx
    hues = color_idx.float() / max(num_colors, 1)
    r = (1 + torch.cos(2 * 3.14159 * hues)) * 0.5
    g = (1 + torch.cos(2 * 3.14159 * hues + 2.094)) * 0.5
    b = (1 + torch.cos(2 * 3.14159 * hues + 4.188)) * 0.5
    rgb = torch.stack([r, g, b], dim=1)
    rgb = rgb.view(B, 3, 1, 1)
    out = im * rgb
    return out.clamp(0, 1)


class ColoredMNIST(Dataset):
    """
    MNIST with color correlated to label (shortcut). Optionally inject correlation strength.
    Each sample: (image, label). Image is (3, H, W) colored by label (or random color for OOD).
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        correlation: float = 1.0,
        num_colors: int = 10,
    ):
        try:
            from torchvision.datasets import MNIST
            from torchvision import transforms
        except ImportError:
            raise ImportError("torchvision required for ColoredMNIST")
        self.mnist = MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())
        self.correlation = correlation
        self.num_colors = num_colors

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        im, label = self.mnist[i]
        if torch.rand(1).item() < self.correlation:
            color_idx = label
        else:
            color_idx = torch.randint(0, self.num_colors, (1,)).item()
        im_c = colorize_mnist(im.unsqueeze(0), color_idx, self.num_colors).squeeze(0)
        return im_c, label


def env_batch_colored_mnist(
    x: torch.Tensor,
    y: torch.Tensor,
    num_colors: int = 10,
) -> EnvBatch:
    """
    x: (B, 3, H, W) colored MNIST (ID color = correlated with y).
    Build env.xs = [x_id, x_ood1, x_ood2] where OOD = same digit, different color (shortcut change).
    """
    x_id = x
    B = x.shape[0]
    device = x.device
    y_np = y.cpu()
    x_ood1 = x.clone()
    x_ood2 = x.clone()
    for b in range(B):
        other = (y_np[b].item() + 1) % num_colors
        other2 = (y_np[b].item() + 2) % num_colors
        # Recolor: preserve luminance, change hue
        g = x[b:b+1].mean(dim=1, keepdim=True)
        x_ood1[b:b+1] = colorize_mnist(g, other, num_colors)
        x_ood2[b:b+1] = colorize_mnist(g, other2, num_colors)
    return EnvBatch(xs=[x_id, x_ood1, x_ood2], env_ids=["id", "ood1", "ood2"])


def compute_id_ood_gap(
    model: torch.nn.Module,
    env: EnvBatch,
    unit_space: Any,
    m_shortcut: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Sho_gap style: logit of predicted class on ID (env.xs[0]) minus mean logit on OOD (env.xs[1:]) when using shortcut mask."""
    with torch.no_grad():
        logits_id = model(unit_space.keep(env.xs[0], m_shortcut))
        z_id = logits_id.gather(1, y.unsqueeze(1)).squeeze(1)
        z_ood = []
        for xe in env.xs[1:]:
            logits_o = model(unit_space.keep(xe, m_shortcut))
            z_ood.append(logits_o.gather(1, y.unsqueeze(1)).squeeze(1))
        z_ood_mean = torch.stack(z_ood, dim=0).mean(dim=0)
    return (z_id - z_ood_mean).mean().item()
