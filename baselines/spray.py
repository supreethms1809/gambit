"""Spectral Relevance Analysis on a stack of relevance maps.

Clustering is CoRelAy's ``SpectralClustering``: euclidean distances, a
symmetric sparse k-nearest-neighbor graph, the symmetric normalized
Laplacian, an eigendecomposition, and k-means. This module chooses the
eigenvalue count, the neighbor count, the cluster count, and the k-means
seed. It does not change those steps.

Each image is then given the mean relevance map of its cluster. That is
the per-image conversion of the cluster.

Relevance maps for a module come from Zennit's ``EpsilonPlus`` composite.
The Nature paper's Fisher-vector analysis used a different attribution
implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1] / "third_party"
_CORELAY = _ROOT / "corelay" / "src"
_ZENNIT = _ROOT / "zennit" / "src"


def cluster_mean_relevance(
    maps: torch.Tensor,
    n_clusters: int = 2,
    n_eigval: int = 32,
    n_neighbors: int = 10,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster ``(N, H, W)`` maps and replace each one by its cluster mean.

    Returns ``(mean_maps, labels)``. ``labels`` is ``(N,)``. The CoRelAy
    default of 32 eigenvalues needs more maps than that. k-means uses
    ``random_state=seed`` and ``n_init=10``.
    """
    if maps.ndim != 3:
        raise ValueError("relevance maps are (N, H, W)")
    if not torch.isfinite(maps).all():
        raise ValueError("a relevance map has a non-finite value")
    count = int(maps.shape[0])
    if count < 3:
        raise ValueError("spectral clustering needs at least 3 maps")
    eigenvalues = int(n_eigval)
    if eigenvalues < 1 or eigenvalues >= count:
        raise ValueError("n_eigval must be at least 1 and smaller than the number of maps")
    if int(n_clusters) < 2:
        raise ValueError("n_clusters must be at least 2")

    SpectralClustering, EigenDecomposition, SparseKNN, KMeans = _import_corelay()
    # CPU before float64: MPS has no float64.
    data = maps.detach().cpu().to(dtype=torch.float64).reshape(count, -1).numpy()
    pipeline = SpectralClustering(
        embedding=EigenDecomposition(n_eigval=eigenvalues),
        affinity=SparseKNN(n_neighbors=int(n_neighbors), symmetric=True),
        clustering=KMeans(
            n_clusters=int(n_clusters),
            kwargs={"random_state": int(seed), "n_init": 10},
        ),
    )
    labels_np = pipeline(data)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=maps.device).reshape(count)
    means = torch.empty_like(maps, dtype=torch.float32)
    for cluster in labels.unique().tolist():
        member = labels == int(cluster)
        means[member] = maps[member].to(dtype=torch.float32).mean(dim=0)
    return means, labels


def _import_corelay():
    added = str(_CORELAY) not in sys.path
    if added:
        sys.path.append(str(_CORELAY))
    try:
        from corelay.pipeline.spectral import SpectralClustering
        from corelay.processor.affinity import SparseKNN
        from corelay.processor.clustering import KMeans
        from corelay.processor.embedding import EigenDecomposition
    finally:
        if added and str(_CORELAY) in sys.path:
            sys.path.remove(str(_CORELAY))
    return SpectralClustering, EigenDecomposition, SparseKNN, KMeans


def lrp_maps(model: nn.Module, x: torch.Tensor, class_idx: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    """``(N, H, W)`` channel-sum of Zennit ``EpsilonPlus`` relevance for ``class_idx``."""
    if x.ndim != 4:
        raise ValueError("lrp_maps expects an image batch (N, C, H, W)")
    ids = class_idx.detach().to(dtype=torch.long, device=x.device).reshape(-1)
    if ids.shape[0] != x.shape[0]:
        raise ValueError("class_idx must have one entry per image")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    Gradient, EpsilonPlus = _import_zennit()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            classes = int(model(x[:1]).shape[-1])
        eye = torch.eye(classes, device=x.device, dtype=x.dtype)
        step = x.shape[0] if not batch_size else min(int(batch_size), x.shape[0])
        parts = []
        for start in range(0, x.shape[0], step):
            sl = slice(start, start + step)
            with Gradient(model=model, composite=EpsilonPlus()) as attributor:
                _output, relevance = attributor(x[sl], eye[ids[sl]])
            parts.append(relevance.sum(dim=1))
    finally:
        model.train(was_training)
    return torch.cat(parts, dim=0)


def _import_zennit():
    added = str(_ZENNIT) not in sys.path
    if added:
        sys.path.append(str(_ZENNIT))
    try:
        from zennit.attribution import Gradient
        from zennit.composites import EpsilonPlus
    finally:
        if added and str(_ZENNIT) in sys.path:
            sys.path.remove(str(_ZENNIT))
    return Gradient, EpsilonPlus
