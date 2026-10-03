"""Save a val contact sheet of the shift pairs.

Waterbirds, planted-patch CIFAR-10, and ColoredMNIST are always drawn.
ImageNet-9 is drawn when the training archives have been extracted. The
challenge release under ``bg_challenge`` is not opened.

    PYTHONPATH=. python scripts/preview_shift_pairs.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from instantiations.shift.biased_data import ColoredMNIST, env_batch_colored_mnist
from instantiations.shift.planted_patch import PlantedPatchCIFAR
from instantiations.shift.waterbirds import WaterbirdsPairs

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "paper" / "shift_pairs" / "preview.png"


def _pil(image: torch.Tensor, size: int = 96) -> Image.Image:
    if image.dim() == 4:
        image = image[0]
    array = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(array).resize((size, size), Image.Resampling.NEAREST)


def _row(cells: list[Image.Image], title: str, tile: int) -> Image.Image:
    width = tile * len(cells)
    canvas = Image.new("RGB", (width, tile + 16), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 1), title, fill=(0, 0, 0))
    for i, cell in enumerate(cells):
        canvas.paste(cell.resize((tile, tile)), (i * tile, 16))
    return canvas


def main() -> None:
    tile = 96
    rows = []

    birds = WaterbirdsPairs(split="val", image_size=224)
    item = birds[0]
    rows.append(_row([
        _pil(item["land"], tile),
        _pil(item["water"], tile),
        _pil(item["mask"].expand(3, -1, -1), tile),
    ], "waterbirds val: land, water, bird mask", tile))

    planted = PlantedPatchCIFAR(split="val", seed=0)
    item = planted[0]
    rows.append(_row([
        _pil(item["present"], tile),
        _pil(item["moved"], tile),
        _pil(item["removed"], tile),
    ], "cifar planted val: present, moved, removed", tile))

    mnist = ColoredMNIST(root=str(REPO / "data"), train=True, download=False, correlation=1.0)
    # Label 0 and the next two hues are far apart, so the recolor is obvious.
    index = next(i for i in range(len(mnist)) if mnist[i][1] == 0)
    image, label = mnist[index]
    env = env_batch_colored_mnist(image.unsqueeze(0), torch.tensor([label]))
    rows.append(_row([_pil(view, tile) for view in env.xs], "colored mnist: id, ood+1, ood+2", tile))

    in9_root = REPO / "data" / "imagenet9"
    if (in9_root / "original").is_dir():
        from instantiations.shift.imagenet9 import ImageNet9Pairs
        in9 = ImageNet9Pairs(root=in9_root, split="val", image_size=224)
        item = in9[0]
        rows.append(_row([
            _pil(item["original"], tile),
            _pil(item["mixed_same"], tile),
            _pil(item["mixed_rand"], tile),
        ], "imagenet-9 val: original, mixed_same, mixed_rand", tile))

    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()