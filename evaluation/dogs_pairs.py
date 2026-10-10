"""Stanford Dogs restyled outside the bounding box.

Styles are three k-means centres of the background mean and standard deviation,
measured on train images only. Pixels inside the box stay as they are.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Tuple
from xml.etree import ElementTree as ET

import torch
from torch.utils.data import Dataset

from core.types import EnvBatch

REPO = Path(__file__).resolve().parents[1]
IMAGES = REPO / "data" / "stanford_dogs" / "images" / "Images"
ANNOTS = REPO / "data" / "stanford_dogs" / "annotations" / "Annotation"
IMAGE_SIZE = 224


class DogsBoxDataset(Dataset):
    """Stanford Dogs images paired with the release's bounding box, as a binary mask."""

    def __init__(self, size: int = IMAGE_SIZE):
        from torchvision import transforms

        self.size = size
        self.t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.classes = sorted(os.listdir(IMAGES))
        self.samples: List[Tuple[str, int, Tuple[float, float, float, float]]] = []
        n_missing = 0
        for ci, breed in enumerate(self.classes):
            for f in sorted(glob.glob(str(IMAGES / breed / "*.jpg"))):
                ann = ANNOTS / breed / Path(f).stem
                if not ann.exists():
                    n_missing += 1
                    continue
                try:
                    root = ET.parse(ann).getroot()
                    w = float(root.find("size/width").text)
                    h = float(root.find("size/height").text)
                    b = root.find("object/bndbox")
                    box = (
                        float(b.find("xmin").text) / w,
                        float(b.find("ymin").text) / h,
                        float(b.find("xmax").text) / w,
                        float(b.find("ymax").text) / h,
                    )
                except Exception:
                    n_missing += 1
                    continue
                self.samples.append((f, ci, box))
        if n_missing:
            print(f"WARNING: {n_missing} images had no usable annotation (skipped)")
        if not self.samples:
            raise FileNotFoundError(f"no annotated images under {IMAGES}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        path, label, (x0, y0, x1, y1) = self.samples[i]
        img = self.t(Image.open(path).convert("RGB"))
        m = torch.zeros(self.size, self.size)
        m[
            int(y0 * self.size):max(int(y1 * self.size), int(y0 * self.size) + 1),
            int(x0 * self.size):max(int(x1 * self.size), int(x0 * self.size) + 1),
        ] = 1.0
        return img, label, m


def background_stats(x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """(B, 6) mean and std of RGB over the pixels outside the box."""
    bg = (1.0 - box).unsqueeze(1)
    n = bg.sum(dim=(2, 3)).clamp_min(1.0)
    mu = (x * bg).sum(dim=(2, 3)) / n
    var = ((x - mu[:, :, None, None]) ** 2 * bg).sum(dim=(2, 3)) / n
    return torch.cat([mu, var.clamp_min(1e-8).sqrt()], dim=1)


def background_styles(ds, n_sample: int, k: int, seed: int) -> torch.Tensor:
    """k background styles, from k-means over measured background statistics."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_sample]
    feats = []
    for i in idx.tolist():
        x, _, m = ds[i]
        feats.append(background_stats(x.unsqueeze(0), m.unsqueeze(0))[0])
    X = torch.stack(feats)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(40):
        a = torch.cdist(X, C).argmin(dim=1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(dim=0)
    return C


def make_env_fn(styles: torch.Tensor):
    """Paired background-transfer environments. The foreground inside the box is untouched."""

    def env_fn(x: torch.Tensor, box: torch.Tensor) -> EnvBatch:
        own = background_stats(x, box)
        mu, sd = own[:, :3], own[:, 3:].clamp_min(1e-6)
        bg = (1.0 - box).unsqueeze(1)
        xs = []
        for j in range(styles.shape[0]):
            tm = styles[j, :3].to(x.device).view(1, 3, 1, 1)
            ts = styles[j, 3:].to(x.device).view(1, 3, 1, 1)
            shifted = ((x - mu[:, :, None, None]) / sd[:, :, None, None]) * ts + tm
            xs.append((x * (1 - bg) + shifted.clamp(0, 1) * bg).clamp(0, 1))
        return EnvBatch(xs=xs, env_ids=[f"bgstyle{j}" for j in range(styles.shape[0])])

    return env_fn
