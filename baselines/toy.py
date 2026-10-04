"""A small classifier whose class evidence sits in known squares.

Class 0 is a red square in the top-left. Class 1 is a blue square in the
bottom-right. The fit is Adam on the cross-entropy for a fixed number of
steps, then the module is left in eval mode. There is no learning-rate
schedule: the run is short and the learning rate stays constant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 32
BOX = 8
_RED = (slice(2, 2 + BOX), slice(2, 2 + BOX))
_BLUE = (slice(SIZE - 2 - BOX, SIZE - 2), slice(SIZE - 2 - BOX, SIZE - 2))


class TinyRegionNet(nn.Module):
    """1x1 color filters, then a global pool. The class cue is the square's color.

    The convolution has no bias, so a gray pixel does not light up a channel on
    its own. Grad-CAM of the logit then has to sit on the square.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(x))
        return self.fc(h.mean(dim=(2, 3)))


def region_boxes() -> tuple[torch.Tensor, torch.Tensor]:
    """``(1, H, W)`` masks of the red square and the blue square."""
    red = torch.zeros(1, SIZE, SIZE)
    blue = torch.zeros(1, SIZE, SIZE)
    red[:, _RED[0], _RED[1]] = 1
    blue[:, _BLUE[0], _BLUE[1]] = 1
    return red, blue


def _paint(x: torch.Tensor, y: torch.Tensor) -> None:
    for i, label in enumerate(y.tolist()):
        if int(label) == 0:
            x[i, :, _RED[0], _RED[1]] = 0
            x[i, 0, _RED[0], _RED[1]] = 1
        else:
            x[i, :, _BLUE[0], _BLUE[1]] = 0
            x[i, 2, _BLUE[0], _BLUE[1]] = 1


def make_region_batch(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gray images with one class square. Values stay in ``[0, 1]``."""
    generator = torch.Generator().manual_seed(int(seed))
    y = torch.randint(0, 2, (n,), generator=generator)
    x = torch.full((n, 3, SIZE, SIZE), 0.45)
    x = (x + 0.05 * torch.rand(x.shape, generator=generator)).clamp(0, 1)
    _paint(x, y)
    return x, y


def make_class_batch(n: int, label: int, seed: int) -> torch.Tensor:
    """``n`` images of one class. The square is the only class cue."""
    generator = torch.Generator().manual_seed(int(seed))
    x = torch.full((n, 3, SIZE, SIZE), 0.45)
    x = (x + 0.05 * torch.rand(x.shape, generator=generator)).clamp(0, 1)
    _paint(x, torch.full((n,), int(label), dtype=torch.long))
    return x


def make_both_cues(n: int, seed: int) -> torch.Tensor:
    """Gray images that contain both squares, for a margin between class 0 and class 1."""
    generator = torch.Generator().manual_seed(int(seed))
    x = torch.full((n, 3, SIZE, SIZE), 0.45)
    x = (x + 0.05 * torch.rand(x.shape, generator=generator)).clamp(0, 1)
    x[:, :, _RED[0], _RED[1]] = 0
    x[:, 0, _RED[0], _RED[1]] = 1
    x[:, :, _BLUE[0], _BLUE[1]] = 0
    x[:, 2, _BLUE[0], _BLUE[1]] = 1
    return x


def _color_detector_init(model: TinyRegionNet) -> None:
    """Red and blue filters, with the linear layer reading those two channels."""
    with torch.no_grad():
        model.conv.weight.zero_()
        model.conv.weight[0, :, 0, 0] = torch.tensor([1.0, -0.5, -0.5])
        model.conv.weight[1, :, 0, 0] = torch.tensor([-0.5, -0.5, 1.0])
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        model.fc.weight[0, 0] = 1.0
        model.fc.weight[1, 1] = 1.0


def train_region_model(
    steps: int = 40,
    batch: int = 64,
    lr: float = 1e-2,
    seed: int = 0,
) -> TinyRegionNet:
    """Fit the square cue. Adam, cross-entropy, one step per batch, no scheduler.

    The run starts from a noisy color detector. Gray is the zero of those
    filters, so the loss can only be reduced by using the colored square.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    torch.manual_seed(int(seed))
    model = TinyRegionNet()
    _color_detector_init(model)
    with torch.no_grad():
        model.conv.weight.add_(0.25 * torch.randn_like(model.conv.weight))
        model.fc.weight.add_(0.25 * torch.randn_like(model.fc.weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(steps):
        x, y = make_region_batch(batch, seed=seed + 1 + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def region_accuracy(model: nn.Module, n: int = 64, seed: int = 10_000) -> float:
    x, y = make_region_batch(n, seed=seed)
    pred = model(x).argmax(dim=-1)
    return float((pred == y).float().mean())
