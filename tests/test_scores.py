"""CD@a, the K×K deletion matrix, ΔD, and two-patch recovery.

The images are known squares. ROAD uses no noise so the score is fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.scores import (
    contrastive_deletion,
    deletion_matrix,
    deletion_specificity,
    disagreement_reduction,
    two_patch_recovery,
)


class _ColorMean(nn.Module):
    """Class 0 is mean red. Class 1 is mean blue."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [x[:, 0].mean(dim=(1, 2)), x[:, 2].mean(dim=(1, 2))],
            dim=1,
        )


def _scene() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.full((2, 3, 16, 16), 0.2)
    image[:, 0, 0:8, 0:8] = 1.0
    image[:, 2, 8:16, 8:16] = 1.0
    red = torch.zeros(2, 16, 16)
    blue = torch.zeros(2, 16, 16)
    red[:, 0:8, 0:8] = 1.0
    blue[:, 8:16, 8:16] = 1.0
    return image, red, blue


def test_contrastive_deletion_prefers_removing_the_foil() -> None:
    image, red, blue = _scene()
    model = _ColorMean()
    model.train()
    score = contrastive_deletion(
        model,
        image,
        red,
        blue,
        class_k=torch.zeros(2, dtype=torch.long),
        class_l=torch.ones(2, dtype=torch.long),
        iters=8,
        noise=0.0,
        seed=0,
    )
    assert model.training
    assert score.shape == (2,)
    assert bool((score > 0).all()), score.tolist()


def test_deletion_matrix_is_class_specific() -> None:
    image, red, blue = _scene()
    # Zero iterations leaves removed pixels at 0, so the class cue cannot be
    # rebuilt from its neighbours.
    matrix = deletion_matrix(
        _ColorMean(),
        image,
        torch.stack([red, blue], dim=1),
        torch.tensor([[0, 1], [0, 1]]),
        iters=0,
        noise=0.0,
        seed=0,
    )
    assert matrix.shape == (2, 2, 2)
    gap, ratio = deletion_specificity(matrix)
    assert bool((gap > 0).all()), gap.tolist()
    assert bool((ratio > 1).all()), ratio.tolist()
    diffused = deletion_matrix(
        _ColorMean(),
        image,
        torch.stack([red, blue], dim=1),
        torch.tensor([[0, 1], [0, 1]]),
        iters=8,
        noise=0.0,
        seed=0,
    )
    assert torch.isfinite(diffused).all()


def test_shortcut_removal_reduces_disagreement_more_than_a_random_mask() -> None:
    _image, red, _blue = _scene()
    identified = torch.full((2, 3, 16, 16), 0.2)
    identified[:, 0, 0:8, 0:8] = 1.0
    ood = torch.full_like(identified, 0.2)
    other = torch.zeros_like(red)
    other[:, 0:8, 8:16] = 1.0
    delta = disagreement_reduction(
        _ColorMean(),
        identified,
        [ood],
        y=torch.zeros(2, dtype=torch.long),
        shortcut=red,
        random_mask=other,
        iters=0,
        noise=0.0,
        seed=0,
    )
    assert bool((delta > 0).all()), delta.tolist()


def test_two_patch_recovery_is_one_when_the_mask_is_the_patch() -> None:
    _image, red, blue = _scene()
    on_a, on_b = two_patch_recovery(red, blue, red, blue)
    assert torch.allclose(on_a, torch.ones(2))
    assert torch.allclose(on_b, torch.ones(2))
    swapped_a, swapped_b = two_patch_recovery(blue, red, red, blue)
    assert bool((swapped_a < 0.1).all())
    assert bool((swapped_b < 0.1).all())
