"""S08 shift pairs. On-disk checks skip when the download is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from evaluation.splits import SplitLockedError
from instantiations.shift.biased_data import (
    colorize_mnist,
    env_batch_colored_mnist,
    recover_mnist_gray,
)
from instantiations.shift.imagenet9 import ImageNet9Pairs, foreground_key, pair_variant_files
from instantiations.shift.planted_patch import PlantedPatchCIFAR, patch_layout
from instantiations.shift.waterbirds import (
    composite_foreground,
    is_waterbird,
    split_backgrounds,
)

REPO = Path(__file__).resolve().parents[1]


def test_recolor_keeps_the_digit_and_changes_the_hue():
    digit = torch.zeros(1, 28, 28)
    digit[:, 4:24, 10:18] = torch.linspace(0.2, 1.0, 20).view(20, 1)
    colored = colorize_mnist(digit, 1, 10).squeeze(0)
    env = env_batch_colored_mnist(colored.unsqueeze(0), torch.tensor([1]), num_colors=10)
    assert env.env_ids == ["id", "ood1", "ood2"]
    colors = []
    for view, color_idx in zip(env.xs, (1, 2, 3)):
        image = view[0] if view.dim() == 4 else view
        gray = recover_mnist_gray(image, color_idx, 10)
        assert torch.allclose(gray, digit, atol=1e-5)
        mask = gray.squeeze() > 0.5
        colors.append(image[:, mask].mean(dim=1))
    assert (colors[0] - colors[1]).abs().sum() > 0.2
    assert (colors[0] - colors[2]).abs().sum() > 0.2


def test_waterbird_tokens_follow_the_official_substring_list():
    assert is_waterbird(Path("001.Black_footed_Albatross/x.jpg"))
    assert not is_waterbird(Path("010.Red_winged_Blackbird/x.jpg"))
    # group_DRO matches "tern" as a substring, so bittern is marked water.
    assert is_waterbird(Path("006.American_Bittern/x.jpg"))


def test_background_splits_are_disjoint_and_stable():
    names = [f"p{i:03d}.jpg" for i in range(200)]
    parts = split_backgrounds(names, 43)
    assert len(parts["train"]) == 160
    assert len(parts["val"]) == 20
    assert len(parts["test"]) == 20
    assert set(parts["train"]).isdisjoint(parts["val"])
    assert set(parts["train"]).isdisjoint(parts["test"])
    assert set(parts["val"]).isdisjoint(parts["test"])
    assert split_backgrounds(names, 43) == parts
    assert split_backgrounds(list(reversed(names)), 43) == parts


def test_composite_keeps_the_bird_and_swaps_the_background():
    bird = Image.new("RGB", (20, 16), (255, 0, 0))
    mask = np.zeros((16, 20), np.float32)
    mask[4:12, 6:14] = 1.0
    land = composite_foreground(bird, mask, Image.new("RGB", (30, 30), (0, 255, 0)))[0]
    water = composite_foreground(bird, mask, Image.new("RGB", (8, 40), (0, 0, 255)))[0]
    bird_px = mask > 0.5
    bg_px = mask < 0.5
    assert torch.allclose(land[:, bird_px], water[:, bird_px], atol=1e-5)
    assert land[0, bird_px].mean() > 0.99
    assert (land[:, bg_px] - water[:, bg_px]).abs().mean() > 0.5


def _boxes_disjoint(origins, patch: int = 32) -> bool:
    boxes = [(y, x, y + patch, x + patch) for y, x in origins]
    return all(
        boxes[i][2] <= boxes[j][0]
        or boxes[j][2] <= boxes[i][0]
        or boxes[i][3] <= boxes[j][1]
        or boxes[j][3] <= boxes[i][1]
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
    )


def test_two_patches_move_without_overlapping_and_remove_cleanly():
    for index in range(30):
        a, b, a_moved, b_moved = patch_layout(index, seed=0)
        assert len({a, b, a_moved, b_moved}) == 4
        assert _boxes_disjoint((a, b, a_moved, b_moved))
    # The half-span search cannot separate this draw. The full offset search must.
    hard = patch_layout(32864, seed=0)
    assert len(set(hard)) == 4
    assert _boxes_disjoint(hard)
    root = REPO / "data"
    if not (root / "cifar-10-batches-py").is_dir():
        pytest.skip("CIFAR-10 is not on disk")
    item = PlantedPatchCIFAR(split="val", root=root, seed=0)[0]
    present = item["mask_a"] + item["mask_b"]
    moved = item["mask_a_moved"] + item["mask_b_moved"]
    assert torch.equal(present * moved, torch.zeros_like(present))
    outside = present < 0.5
    assert torch.allclose(item["present"][:, outside], item["removed"][:, outside])
    inside = present > 0.5
    assert (item["present"] - item["removed"]).abs()[:, inside].mean() > 0.2
    assert item["mask_a"].sum() == 32 * 32
    assert item["mask_b"].sum() == 32 * 32


def test_imagenet9_pairs_by_foreground_id(tmp_path: Path):
    classes = ("00_dog", "01_bird")
    original = {
        "00_dog/n0001_1.JPEG": "orig-a",
        "01_bird/n0002_2.JPEG": "orig-b",
    }
    mixed_same = {
        "00_dog/fg_n0001_1_bg_n9999_9.JPEG": "same-a",
        "01_bird/fg_n0002_2_bg_n8888_8.JPEG": "same-b",
    }
    mixed_rand = {
        "00_dog/fg_n0001_1_bg_n7777_7.JPEG": "rand-a",
        "01_bird/fg_n0002_2_bg_n6666_6.JPEG": "rand-b",
    }
    for variant, files in (
        ("original", original),
        ("mixed_same", mixed_same),
        ("mixed_rand", mixed_rand),
    ):
        for rel, color_name in files.items():
            path = tmp_path / variant / "val" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            color = {"orig-a": (255, 0, 0), "orig-b": (0, 255, 0),
                     "same-a": (0, 0, 255), "same-b": (255, 255, 0),
                     "rand-a": (255, 0, 255), "rand-b": (0, 255, 255)}[color_name]
            Image.new("RGB", (16, 16), color).save(path)
    assert foreground_key(Path("00_dog/fg_n0001_1_bg_n9999_9.JPEG")) == "00_dog/n0001_1"
    ds = ImageNet9Pairs(root=tmp_path, split="val", image_size=16)
    assert len(ds) == 2
    item = ds[0]
    assert item["label"] == 0
    assert item["original"].shape == (3, 16, 16)
    # original is red, mixed_same is blue, mixed_rand is magenta
    assert item["original"][0].mean() > 0.9
    assert item["mixed_same"][2].mean() > 0.9
    assert item["mixed_rand"][0].mean() > 0.9 and item["mixed_rand"][2].mean() > 0.9
    directories = {name: tmp_path / name / "val" for name in ("original", "mixed_same", "mixed_rand")}
    assert len(pair_variant_files(directories)) == 2


def test_imagenet9_challenge_release_stays_locked(tmp_path: Path):
    locked = tmp_path / "bg_challenge"
    locked.mkdir()
    with pytest.raises(SplitLockedError):
        ImageNet9Pairs(root=locked, split="val")


def test_dogs_restyle_leaves_the_foreground_in_place():
    from scripts.eval_robust_shortcut_dogs import make_env_fn

    styles = torch.tensor([
        [0.2, 0.4, 0.6, 0.1, 0.1, 0.1],
        [0.8, 0.1, 0.1, 0.2, 0.2, 0.2],
    ])
    image = torch.rand(2, 3, 16, 16)
    box = torch.zeros(2, 16, 16)
    box[:, 4:12, 4:12] = 1
    env = make_env_fn(styles)(image, box)
    assert env.env_ids == ["bgstyle0", "bgstyle1"]
    assert torch.allclose(env.xs[0][:, :, 4:12, 4:12], image[:, :, 4:12, 4:12])
    assert torch.allclose(env.xs[1][:, :, 4:12, 4:12], image[:, :, 4:12, 4:12])


@pytest.mark.skipif(
    not (REPO / "data" / "imagenet9" / "original" / "val").is_dir(),
    reason="ImageNet-9 training archives are not extracted",
)
def test_imagenet9_val_pairs_are_three_views():
    root = REPO / "data" / "imagenet9"
    ds = ImageNet9Pairs(root=root, split="val", image_size=64)
    n_original = sum(1 for p in (root / "original" / "val").rglob("*") if p.suffix == ".JPEG")
    assert len(ds) == n_original
    changed = 0
    for i in range(4):
        item = ds[i]
        assert item["original"].shape == (3, 64, 64)
        assert item["mixed_same"].shape == item["original"].shape
        assert item["mixed_rand"].shape == item["original"].shape
        if not torch.equal(item["original"], item["mixed_rand"]):
            changed += 1
    assert changed == 4


@pytest.mark.skipif(
    not (REPO / "data" / "places365" / "backgrounds" / "val_256").is_dir()
    or not (REPO / "data" / "CUB_200_2011" / "segmentations").is_dir(),
    reason="Places backgrounds or CUB masks are not on disk",
)
def test_waterbirds_val_pair_changes_only_the_background():
    from instantiations.shift.waterbirds import WaterbirdsPairs

    ds = WaterbirdsPairs(split="val", image_size=64)
    assert len(ds) == 600
    bird_delta = []
    bg_delta = []
    labels = set()
    for i in range(4):
        item = ds[i]
        labels.add(item["label"])
        bird = item["mask"] > 0.8
        bg = item["mask"] < 0.2
        assert bird.any() and bg.any()
        bird_delta.append((item["land"][:, bird] - item["water"][:, bird]).abs().mean())
        bg_delta.append((item["land"][:, bg] - item["water"][:, bg]).abs().mean())
    assert torch.stack(bird_delta).mean() < 0.08
    assert torch.stack(bg_delta).mean() > 0.05
    with pytest.raises(SplitLockedError):
        WaterbirdsPairs(split="test", image_size=64)
