"""Shared test fixtures for GAMBIT tests."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class TinyCNN(nn.Module):
    """Minimal CNN for testing: Conv2d -> ReLU -> AdaptiveAvgPool -> Linear."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))
