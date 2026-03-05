"""Visualize keep/remove: save a sample image, its keep(mask) and remove(mask) to compare."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from modality.grid_regions import VisionGridUnitSpace


def main():
    B, C, H, W = 1, 3, 56, 56
    grid_h, grid_w = 7, 7
    unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur")
    # Sample image: gradient + one bright region
    x = torch.zeros(B, C, H, W)
    x[:, :, :, :] = torch.linspace(0, 1, W).view(1, 1, 1, -1)
    x[:, :, 20:36, 20:36] = 1.0  # white square
    # Mask: keep top-left 2x2 regions
    R = grid_h * grid_w
    m = torch.zeros(B, R)
    m[:, 0] = 1.0
    m[:, 1] = 1.0
    m[:, 7] = 1.0
    m[:, 8] = 1.0
    x_keep = unit_space.keep(x, m)
    x_remove = unit_space.remove(x, m)
    out_dir = REPO / "modality" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in [("original", x), ("keep", x_keep), ("remove", x_remove)]:
        img = t[0].permute(1, 2, 0).clamp(0, 1)
        try:
            from PIL import Image
            import numpy as np
            arr = (img.detach().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f"{name}.png")
        except ImportError:
            print("PIL not installed; skipping image save for %s. Install with: pip install Pillow" % name)
            break
        except Exception as e:
            print("Failed to save %s: %s" % (name, e))
    print("Visualization: see modality/out/original.png, keep.png, remove.png")
    print("keep = only masked regions visible (rest baseline); remove = masked regions replaced by baseline.")


if __name__ == "__main__":
    main()
