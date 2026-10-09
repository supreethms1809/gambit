"""SpRAy: two relevance prototypes stay separated, and Zennit returns a finite map."""

from __future__ import annotations

import pytest
import torch

from baselines.spray import cluster_mean_relevance, lrp_maps
from baselines.toy import make_class_batch, region_boxes, train_region_model
from evaluation.masks import mass_in


def _two_prototypes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    red, blue = region_boxes()
    generator = torch.Generator().manual_seed(0)
    red_maps = red + 0.01 * torch.randn(20, 32, 32, generator=generator)
    blue_maps = blue + 0.01 * torch.randn(20, 32, 32, generator=generator)
    return torch.cat((red_maps, blue_maps), dim=0), red, blue


def test_each_prototype_keeps_its_own_cluster_mean() -> None:
    maps, red, blue = _two_prototypes()
    means, labels = cluster_mean_relevance(maps, n_clusters=2, n_eigval=2, seed=0)
    assert int(labels[:20].unique().numel()) == 1
    assert int(labels[20:].unique().numel()) == 1
    assert int(labels[0].item()) != int(labels[20].item())
    red_mean = means[0:1]
    blue_mean = means[20:21]
    assert float(mass_in(red_mean, red)) > 0.9
    assert float(mass_in(blue_mean, blue)) > 0.9
    again, again_labels = cluster_mean_relevance(maps, n_clusters=2, n_eigval=2, seed=0)
    assert torch.equal(labels, again_labels)
    assert torch.allclose(means, again)


def test_bad_stacks_are_rejected() -> None:
    maps, _red, _blue = _two_prototypes()
    with pytest.raises(ValueError, match="n_eigval"):
        cluster_mean_relevance(maps[:4], n_eigval=4)
    with pytest.raises(ValueError, match="at least 3"):
        cluster_mean_relevance(maps[:2], n_eigval=1)
    bad = maps.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        cluster_mean_relevance(bad, n_eigval=2)


def test_epsilon_plus_relevance_matches_the_image() -> None:
    model = train_region_model(steps=2, seed=0)
    image = make_class_batch(2, label=0, seed=1)
    labels = torch.zeros(2, dtype=torch.long)
    relevance = lrp_maps(model, image, labels)
    assert relevance.shape == (2, 32, 32)
    assert torch.isfinite(relevance).all()
    assert torch.allclose(relevance, lrp_maps(model, image, labels, batch_size=1))
    assert model(image).shape == (2, 2)


def test_cluster_mean_relevance_accepts_an_mps_tensor():
    import pytest

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    maps = torch.rand(6, 4, 4).to("mps")
    means, labels = cluster_mean_relevance(maps, n_clusters=2, n_eigval=2, n_neighbors=3, seed=0)
    assert means.shape == maps.shape
    assert labels.shape == (6,)
