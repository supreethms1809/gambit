"""The single deletion operator: hard cells and one blur."""

from __future__ import annotations

import torch

from core.grid import delete, deletion_baseline, upsample_units


def test_constant_image_is_unchanged_by_blur() -> None:
    image = torch.ones(1, 3, 224, 224)
    blurred = deletion_baseline(image, 7, 7)
    assert torch.allclose(blurred, image, atol=1e-5)
    fine = deletion_baseline(image, 14, 14)
    assert torch.allclose(fine, image, atol=1e-5)


def test_deleting_one_cell_leaves_every_other_pixel() -> None:
    torch.manual_seed(0)
    image = torch.rand(2, 3, 224, 224)
    mask = torch.zeros(2, 49)
    mask[:, 0] = 1
    edited = delete(image, mask, 7, 7)
    pixel = upsample_units(mask, 7, 7, 224, 224)
    outside = pixel < 0.5
    assert torch.equal(outside.expand_as(image), outside.expand_as(image))
    assert torch.allclose(edited[outside.expand_as(image)], image[outside.expand_as(image)])
    assert not torch.allclose(edited, image)


def test_blur_sigma_matches_half_a_unit() -> None:
    from core.grid import blur_kernel_size, blur_sigma

    assert blur_sigma(224, 7) == 16
    assert blur_sigma(224, 14) == 8
    assert blur_kernel_size(16) == 2 * 48 + 1
    assert blur_kernel_size(8) == 2 * 24 + 1
