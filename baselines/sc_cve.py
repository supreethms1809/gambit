"""SC-CVE adapter (Vandenhende et al., ECCV 2022) over the vendored snapshot.

The search is the vendored ``compute_counterfactual`` at
``third_party/sc_cve`` (pin ``cd879d6``). This module chooses the distractor
class, the auxiliary features, and the map conversion. It does not change the
edit objective: argmax over candidate edits of
``log p(distractor | edit) + lambd * log semantic_prob(edit)``.

Author defaults from ``counterfactuals_ours_cub_res50.yaml``: ``lambd=0.4``,
``temperature=0.1``, ``topk=0.2`` prefilter, up to 20 distractors. The Goyal
configs in the same repo set ``lambd=0.0`` with no ``topk`` key and 1
distractor, which is the same function without the semantic term.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Author defaults (counterfactuals_ours_cub_res50.yaml).
OURS_LAMBD = 0.4
OURS_TEMPERATURE = 0.1
OURS_TOPK = 0.2
OURS_MAX_DISTRACTORS = 20

# SwAV auxiliary-model pin. The weights file (113.7 MB) is downloaded with
# approval at repro time, hash-recorded, and loaded offline; the loader takes
# a local path so no network call sits in the library path. SwAV is an
# ImageNet self-supervised model, so its semantic prior is weak on HAM10000
# and brain MRI (dossier failure mode).
SWAV_REPO = "facebookresearch/swav"
SWAV_REF = "06b1b7c"
SWAV_DIM = 2048
SWAV_N_PIX = 7
SWAV_WEIGHTS_FILENAME = "swav_800ep_pretrain.pth.tar"

_VENDOR = Path(__file__).resolve().parents[1] / "third_party" / "sc_cve"


def _import_sc_cve():
    """Import the vendored explainer, then drop its directory from ``sys.path``."""
    added = str(_VENDOR) not in sys.path
    if added:
        sys.path.append(str(_VENDOR))
    try:
        from counterfactuals.explainer.counterfactuals import compute_counterfactual
    finally:
        if added and str(_VENDOR) in sys.path:
            sys.path.remove(str(_VENDOR))
    return compute_counterfactual


def check_finite_features(*feats: torch.Tensor) -> None:
    """Raise on a non-finite feature map, mirroring ``baselines.cve``."""
    for feat in feats:
        if feat.ndim not in (3, 4):
            raise ValueError("a feature map is (C, H, W) or (N, C, H, W)")
        if not torch.isfinite(feat).all():
            raise ValueError("a feature map has a non-finite value")


class NoFlipError(ValueError):
    """The vendored search ran out of edits without flipping the prediction.

    Upstream has no exit for this case: ``_find_single_best_edit`` calls
    ``argmax`` on an empty candidate set and the driver's ``except
    BaseException`` skips the image. The wrapper catches this per image and
    reports ``flipped=False``.
    """


def edits_to_rank(edits: list[tuple[int, int]], height: int, width: int, device) -> torch.Tensor:
    """Rank map from an edit list. Entry 1 is the first query cell replaced."""
    rank = torch.zeros(height, width, dtype=torch.long, device=device)
    for step, (query_cell, _source) in enumerate(edits, start=1):
        y, x = divmod(int(query_cell), width)
        rank[y, x] = step
    return rank


def edits_to_scores(rank: torch.Tensor) -> torch.Tensor:
    """Earliest edit scores highest; untouched cells score 0 (as in ``cve_pair``)."""
    top = rank.max().clamp_min(1).to(torch.float32)
    return torch.where(rank > 0, top - rank.float() + 1.0, torch.zeros_like(top))


def run_sc_cve_edits(
    query: torch.Tensor,
    distractors: torch.Tensor,
    decision,
    distractor_class: int,
    *,
    lambd: float = OURS_LAMBD,
    temperature: Optional[float] = OURS_TEMPERATURE,
    topk: Optional[float] = OURS_TOPK,
    query_aux: Optional[torch.Tensor] = None,
    distractor_aux: Optional[torch.Tensor] = None,
    device=None,
) -> list[tuple[int, int]]:
    """One query against N distractor maps through the vendored search.

    ``query`` is ``(C, H, W)``; ``distractors`` is ``(N, C, H, W)``. When
    ``topk`` is not None the kNN prefilter needs auxiliary features, as
    upstream. Returns ``[(query_cell, distractor_cell)]`` in edit order, where
    the distractor cell indexes the flattened ``N * H * W`` stack.
    """
    check_finite_features(query, distractors)
    compute_counterfactual = _import_sc_cve()
    try:
        raw = compute_counterfactual(
            query=query,
            distractor=distractors,
            classification_head=decision,
            distractor_class=int(distractor_class),
            query_aux_features=query_aux,
            distractor_aux_features=distractor_aux,
            lambd=float(lambd),
            temperature=temperature,
            topk=topk,
            device=device,
        )
    except ValueError as exc:
        if "empty sequence" in str(exc):
            raise NoFlipError("the search ran out of edits without flipping") from exc
        raise
    return [(int(q), int(s)) for q, s in raw]


def load_swav_backbone(weights_path: str | Path, device) -> nn.Module:
    """SwAV ResNet-50 trunk with upstream's surgery, from a local weights file.

    Mirrors ``auxiliary_model.get_auxiliary_model`` but never touches the
    network: the checkpoint must already be on disk (downloaded with approval
    and hash-recorded at repro time). Surgery: ``avgpool`` and ``fc`` become
    identity, ``layer4`` is cut to its first block, so the output is
    ``(B, 2048, 7, 7)`` semantic features.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"SwAV weights not found: {path}")
    import torchvision.models

    blob = torch.load(str(path), map_location="cpu")
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}
    model = torchvision.models.resnet50()
    load = model.load_state_dict(state, strict=False)
    if load.missing_keys or load.unexpected_keys:
        raise ValueError(
            f"SwAV state did not match resnet50: missing={load.missing_keys} "
            f"unexpected={load.unexpected_keys}"
        )
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    model.layer4 = model.layer4[0]
    model.eval()
    return model.to(device)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def swav_features(images: torch.Tensor, model: nn.Module, device) -> torch.Tensor:
    """ImageNet-normalised SwAV trunk features ``(B, 2048, 7, 7)`` for raw ``[0, 1]`` input."""
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        out = model((images.to(device) - mean) / std)
    return out.reshape(images.shape[0], SWAV_DIM, SWAV_N_PIX, SWAV_N_PIX).detach().cpu()
