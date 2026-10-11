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


def test_shifted_mask_moves_without_wrapping():
    from core.grid import boundary_offset, shift_pixels

    pixel = torch.zeros(1, 1, 8, 8)
    pixel[..., 0:2, 6:8] = 1.0  # top-right corner block
    assert torch.equal(shift_pixels(pixel, 0, 0), pixel)
    moved = shift_pixels(pixel, 3, -2)
    assert torch.equal(moved[..., 3:5, 4:6], torch.ones(1, 1, 2, 2))
    assert float(moved.sum()) == 4.0
    # Pushed past the edge, the block is cut, not wrapped to the other side.
    cut = shift_pixels(pixel, 0, 1)
    assert float(cut.sum()) == 2.0 and float(cut[..., :, 0].sum()) == 0.0
    assert boundary_offset(224, 224, 7, 7) == (16, 16)
    assert boundary_offset(224, 224, 14, 14) == (8, 8)


def test_delete_with_an_offset_deletes_the_moved_cells():
    from core.grid import shift_pixels

    torch.manual_seed(0)
    x = torch.rand(1, 3, 28, 28)
    mask = torch.zeros(1, 49)
    mask[0, 24] = 1.0
    pixel = shift_pixels(upsample_units(mask, 7, 7, 28, 28), 2, -1)
    expected = (1 - pixel) * x + pixel * deletion_baseline(x, 7, 7)
    assert torch.allclose(delete(x, mask, 7, 7, offset=(2, -1)), expected)
    assert torch.equal(delete(x, mask, 7, 7, offset=(0, 0)), delete(x, mask, 7, 7))


def _accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_cell_upsampling_matches_indexing_and_pooling_matches_scatter():
    from core.grid import _cell_index, pool_sum

    torch.manual_seed(0)
    for height, cells in ((224, 7), (224, 14), (30, 7)):
        mask = torch.rand(3, cells * cells)
        iy, ix = _cell_index(height, cells, "cpu"), _cell_index(height, cells, "cpu")
        reference = mask.reshape(3, cells, cells)[:, iy[:, None], ix[None, :]].unsqueeze(1)
        assert torch.equal(upsample_units(mask, cells, cells, height, height), reference)
        pixel = torch.rand(3, height, height)
        flat = (iy[:, None] * cells + ix[None, :]).reshape(1, -1).expand(3, -1)
        summed = torch.zeros(3, cells * cells).scatter_add(1, flat, pixel.reshape(3, -1))
        assert torch.allclose(pool_sum(pixel, cells, cells), summed, rtol=1e-5)


def test_cell_upsampling_backward_repeats_exactly_on_an_accelerator():
    """Indexing's backward accumulates in a varying order on GPUs; Adam amplified it."""
    import pytest

    device = _accelerator()
    if device is None:
        pytest.skip("no GPU")
    torch.manual_seed(0)
    mask = torch.rand(8, 49, device=device)
    weight = torch.rand(8, 224, 224, device=device)
    grads = []
    for _ in range(3):
        leaf = mask.clone().requires_grad_(True)
        (upsample_units(leaf, 7, 7, 224, 224).squeeze(1) * weight).sum().backward()
        grads.append(leaf.grad.clone())
    assert torch.equal(grads[0], grads[1]) and torch.equal(grads[0], grads[2])
