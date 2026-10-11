"""Every method's maps for one batch, as EVAL_PLAN.md section 4 defines them.

A contrastive method returns ``PairMaps``: one score map for the kept class k
(rank 0) and one for the foil l (rank 1), on the same image, plus per-class
maps when the method has them (for the K×K deletion matrix). A shift method
returns ``ShiftMaps``: a robust map and a shortcut map.

Maps are either region maps ``(B, R)`` on the backbone grid or pixel maps
``(B, H, W)``. ``grid`` says which; the scorer converts both through the one
budget adapter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from core.types import HypothesisSet


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ShiftMaps:
    robust: torch.Tensor
    shortcut: torch.Tensor
    grid: Optional[tuple[int, int]]


@dataclass
class PairMaps:
    k: torch.Tensor
    l: Optional[torch.Tensor]                 # None: the method has no foil map (CVE)
    grid: Optional[tuple[int, int]]           # None: pixel maps
    per_class: Optional[torch.Tensor] = None  # (B, K, R) or (B, K, H, W)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knobs. ``fast`` is a smoke setting: it shortens the expensive loops so the
# whole grid can be exercised; records carry it, and it is never a paper run.
# ---------------------------------------------------------------------------

@dataclass
class Knobs:
    ig_steps: int = 16
    extremal_max_iter: int = 800
    rise_masks: int = 4000
    rise_cell: int = 7
    cdea_steps: int = 100
    cve_distractor_tries: int = 32
    sc_cve_distractors: int = 20       # authors' max_num_distractors (ours config)
    sc_cve_swav_weights: Optional[str] = None  # local SwAV file; required off the fast path
    extremal_smooth: float = 0.0
    fast: bool = False


FAST = Knobs(ig_steps=4, extremal_max_iter=40, rise_masks=200, rise_cell=7,
             cdea_steps=8, cve_distractor_tries=8, fast=True,
             sc_cve_distractors=1)


@dataclass
class CdeaConfig:
    """One CDEA configuration. Defaults are the G1 gate: Grad-CAM, T = 100, lr = 0.1."""

    backend: str = "gradcam"           # gradcam | ig | library:<method>
    lr: float = 0.1
    steps: Optional[int] = None        # None: Knobs.cdea_steps
    projection: str = "sinkhorn"       # sinkhorn | hard_top_mass
    shared: bool = True
    independent: bool = False
    preserve: bool = False
    init: str = "evidence"             # evidence | uniform
    pair_only: bool = False
    kind: str = "allocate"             # allocate | first_order | precomputed
    precomputed_dir: Optional[str] = None
    boundary_shift: bool = False       # FORMULATION.md section 4.1
    offset_seed: int = 0


ABLATIONS: dict[str, Callable[[CdeaConfig], CdeaConfig]] = {
    "A1_independent": lambda c: replace(c, independent=True, shared=False),
    "A2_no_shared": lambda c: replace(c, shared=False),
    "A3_preserve": lambda c: replace(c, preserve=True),
    "A4_pair": lambda c: replace(c, pair_only=True),
    "A5_uniform": lambda c: replace(c, init="uniform"),
    "A5_evidence": lambda c: replace(c, init="evidence"),
    "A5_backend_ig": lambda c: replace(c, backend="ig"),
    "A5_backend_layercam": lambda c: replace(c, backend="library:layercam"),
    "A5_backend_xgradcam": lambda c: replace(c, backend="library:xgradcam"),
    "A5_backend_gradcam++": lambda c: replace(c, backend="library:gradcam++"),
    "A5_backend_inputxgradient": lambda c: replace(c, backend="library:inputxgradient"),
    "A5_backend_gradientshap": lambda c: replace(c, backend="library:gradientshap"),
    "A6_hard_top_mass": lambda c: replace(c, projection="hard_top_mass"),
    "A7_steps_25": lambda c: replace(c, steps=25),
    "A7_steps_50": lambda c: replace(c, steps=50),
    "A7_steps_100": lambda c: replace(c, steps=100),
    "A7_steps_200": lambda c: replace(c, steps=200),
    "A8_precomputed": lambda c: replace(c, kind="precomputed"),
    "A9_first_order": lambda c: replace(c, kind="first_order"),
    "A10_boundary_shift": lambda c: replace(c, boundary_shift=True),
    # Sensitivity of A10 to its offset seed. Every seed is reported; none is selected.
    "A10_boundary_shift_o1": lambda c: replace(c, boundary_shift=True, offset_seed=1),
    "A10_boundary_shift_o2": lambda c: replace(c, boundary_shift=True, offset_seed=2),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grid_of(backbone: str) -> tuple[int, int]:
    from core.grid import grid_of as _grid_of

    return _grid_of(backbone)


def hypotheses_for(model: nn.Module, x: torch.Tensor, k: int) -> HypothesisSet:
    from baselines.hypotheses import shared_hypotheses

    with torch.no_grad():
        logits = model(x)
    return shared_hypotheses(logits, min(k, logits.shape[-1]))


def swapped(h: HypothesisSet) -> HypothesisSet:
    """Rank 0 and rank 1 exchanged, so a k-map function yields the l-map."""
    ids = h.ids.clone()
    ids[:, [0, 1]] = ids[:, [1, 0]]
    mask = h.mask.clone()
    mask[:, [0, 1]] = mask[:, [1, 0]]
    return HypothesisSet(ids=ids, mask=mask)


def _single(class_idx: torch.Tensor) -> HypothesisSet:
    ids = class_idx.long().view(-1, 1)
    return HypothesisSet(ids=ids, mask=torch.ones_like(ids, dtype=torch.bool))


def evidence_provider(backend: str, backbone: str, knobs: Knobs):
    from base_evidence.gradcam_regions import GradCAMRegionsProvider
    from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider

    gh, gw = grid_of(backbone)
    if backend == "gradcam":
        return GradCAMRegionsProvider(gh, gw)
    if backend == "ig":
        return IntegratedGradientsRegionsProvider(gh, gw, steps=knobs.ig_steps)
    if backend.startswith("library:"):
        from base_evidence.library_adapters import CAM_METHODS, CamLibraryProvider, CaptumRegionsProvider

        method = backend.split(":", 1)[1]
        if method in CAM_METHODS:
            return CamLibraryProvider(method, gh, gw)
        return CaptumRegionsProvider(method, gh, gw)
    raise ValueError(f"unknown evidence backend {backend!r}")


def default_backend(backbone: str) -> str:
    """Grad-CAM degenerates on LayerNorm'd ViT tokens, so ViT uses IG (EVAL_PLAN 4.2)."""
    return "ig" if backbone.startswith("vit") else "gradcam"


# ---------------------------------------------------------------------------
# CDEA
# ---------------------------------------------------------------------------

def _allocate_config(cfg: CdeaConfig, knobs: Knobs):
    from cdea.allocation import AllocateConfig

    return AllocateConfig(
        steps=knobs.cdea_steps if cfg.steps is None else int(cfg.steps),
        lr=float(cfg.lr),
        projection=cfg.projection,
        shared=cfg.shared,
        independent=cfg.independent,
        preserve=cfg.preserve,
        init=cfg.init,
        pair_only=cfg.pair_only,
        backend=cfg.backend,
        ig_steps=knobs.ig_steps,
        boundary_shift=cfg.boundary_shift,
        offset_seed=cfg.offset_seed,
    )


def cdea_pair(model, x, h, backbone, knobs, device, cfg: Optional[CdeaConfig] = None, area: float = 0.05, index=None) -> PairMaps:
    """Unique allocations of rank 0 and rank 1. The shared row is kept for the sensitivity check."""
    cfg = cfg or CdeaConfig(backend=default_backend(backbone))
    grid = grid_of(backbone)
    if cfg.kind == "precomputed":
        from baselines.precomputed import load_pair

        return load_pair(cfg.precomputed_dir, h, grid, index=index, area=area)
    if cfg.kind == "first_order":
        from cdea.first_order import unit_gradient

        scores = unit_gradient(model, x, h, *grid)
        return PairMaps(k=scores[:, 0], l=scores[:, 1], grid=grid, per_class=scores)
    from cdea.allocation import allocate

    out = allocate(model, x, h, grid[0], grid[1], float(area), _allocate_config(cfg, knobs))
    # The hypotheses the payoffs were optimised against, for the D2 stage payoffs.
    extra = {"payoff_ids": h.ids[:, :2] if cfg.pair_only else h.ids}
    if out.shared is not None:
        extra["shared"] = out.shared
    return PairMaps(k=out.unique[:, 0], l=out.unique[:, 1], grid=grid, per_class=out.unique, extra=extra)


# ---------------------------------------------------------------------------
# Contrastive baselines
# ---------------------------------------------------------------------------

def base_evidence_pair(model, x, h, backbone, knobs, device, backend: Optional[str] = None) -> PairMaps:
    provider = evidence_provider(backend or default_backend(backbone), backbone, knobs)
    E = provider.explain(x, model, h)
    return PairMaps(k=E[:, 0], l=E[:, 1], grid=grid_of(backbone), per_class=E)


def analytic_pair(model, x, h, backbone, knobs, device, backend: Optional[str] = None) -> PairMaps:
    """``E_k - min_j E_j``, with shared map ``min_j E_j``."""
    provider = evidence_provider(backend or default_backend(backbone), backbone, knobs)
    evidence = provider.explain(x, model, h)
    shared = evidence.min(dim=1).values
    unique = evidence - shared.unsqueeze(1)
    return PairMaps(k=unique[:, 0], l=unique[:, 1], grid=grid_of(backbone), per_class=unique,
                    extra={"shared": shared})


def _rival_pairs(model, x, h: HypothesisSet) -> list[HypothesisSet]:
    """For each member j of H, rank 0 is j and rank 1 is the strongest other member."""
    with torch.no_grad():
        scores = model(x).gather(1, h.ids.long())
    pairs = []
    for j in range(h.ids.shape[1]):
        others = scores.clone()
        others[:, j] = float("-inf")
        rival = others.argmax(dim=-1)
        foil = h.ids.gather(1, rival.unsqueeze(1)).squeeze(1)
        ids = torch.stack([h.ids[:, j], foil], dim=1)
        pairs.append(HypothesisSet(ids=ids, mask=torch.ones_like(ids, dtype=torch.bool)))
    return pairs


def margin_gradcam_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    from baselines.margin import margin_gradcam

    columns = [margin_gradcam(model, x, pair) for pair in _rival_pairs(model, x, h)]
    return PairMaps(k=columns[0], l=columns[1], grid=None, per_class=torch.stack(columns, dim=1))


def margin_ig_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    from baselines.margin import margin_integrated_gradients

    pairs = _rival_pairs(model, x, h)
    steps = max(2, knobs.ig_steps)
    # IG is linear in the target, so the rank-1 map is the negated rank-0 map.
    k_map = margin_integrated_gradients(model, x, pairs[0], steps=steps)
    columns = [k_map, -k_map]
    columns += [margin_integrated_gradients(model, x, pair, steps=steps) for pair in pairs[2:]]
    return PairMaps(k=k_map, l=-k_map, grid=None, per_class=torch.stack(columns, dim=1))


def extremal_contrastive_pair(model, x, h, backbone, knobs, device, area: float = 0.05, deletion: bool = True) -> PairMaps:
    from baselines.extremal import margin_masks

    kw = dict(area=area, max_iter=knobs.extremal_max_iter, smooth=knobs.extremal_smooth, deletion=deletion)
    return PairMaps(k=margin_masks(model, x, h, **kw), l=margin_masks(model, x, swapped(h), **kw), grid=None)


def extremal_preserve_pair(model, x, h, backbone, knobs, device, area: float = 0.05) -> PairMaps:
    return extremal_contrastive_pair(model, x, h, backbone, knobs, device, area=area, deletion=False)


def extremal_class_pair(model, x, h, backbone, knobs, device, area: float = 0.05) -> PairMaps:
    from baselines.extremal import class_masks

    kw = dict(area=area, max_iter=knobs.extremal_max_iter, smooth=knobs.extremal_smooth)
    return PairMaps(k=class_masks(model, x, h.ids[:, 0], **kw), l=class_masks(model, x, h.ids[:, 1], **kw), grid=None)


def rise_margin_pair(model, x, h, backbone, knobs, device, seed: int = 0) -> PairMaps:
    from baselines.rise import class_maps, margin_maps

    kw = dict(n_masks=knobs.rise_masks, s=knobs.rise_cell, seed=seed)
    columns = []
    for j in range(h.ids.shape[1]):
        per_image = [
            class_maps(model, x[i:i + 1], int(h.ids[i, j]), **kw)[0]
            for i in range(x.shape[0])
        ]
        columns.append(torch.stack(per_image, dim=0))
    return PairMaps(k=margin_maps(model, x, h, **kw), l=margin_maps(model, x, swapped(h), **kw),
                    grid=None, per_class=torch.stack(columns, dim=1))


def contrastive_gradcam_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    from baselines.contrastive_gradcam import contrastive_gradcam

    return PairMaps(k=contrastive_gradcam(model, x, h), l=contrastive_gradcam(model, x, swapped(h)), grid=None)


def chefer_class_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    """Author-default class maps for k and l: two relprops (EVAL_PLAN 4.2).

    ViT only; the dossier records non-applicability to ResNet. Output is
    14x14 like extremal_class_pair. Normalisation follows the
    ``NormalizedModel`` convention: the converted model has no wrapper, so
    raw input is normalised here when the caller passes one.
    """
    from baselines.chefer import class_relprop, from_torchvision
    from models.wrapper import NormalizedModel, unwrap

    converted = from_torchvision(unwrap(model))
    dev = next(converted.parameters()).device
    if isinstance(model, NormalizedModel):
        x = (x - model.mean.to(x.device)) / model.std.to(x.device)
    columns = []
    with torch.enable_grad():
        for j in range(h.ids.shape[1]):
            maps = []
            for i in range(x.shape[0]):
                img = x[i:i + 1].to(dev)
                maps.append(class_relprop(converted, img, int(h.ids[i, j])).to(x.device))
            columns.append(torch.stack(maps))
    per = torch.stack(columns, dim=1)
    gh, gw = 14, 14
    flat = per.reshape(x.shape[0], h.ids.shape[1], -1)
    return PairMaps(k=flat[:, 0], l=flat[:, 1], grid=(gh, gw), per_class=flat)


def random_pair(model, x, h, backbone, knobs, device, seed: int = 0) -> PairMaps:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    gh, gw = grid_of(backbone)
    per = torch.rand(x.shape[0], h.ids.shape[1], gh * gw, generator=g).to(x.device)
    return PairMaps(k=per[:, 0], l=per[:, 1], grid=(gh, gw), per_class=per)


# ---- CVE -------------------------------------------------------------------

def _resnet_split(model: nn.Module):
    """``features(x)`` = layer4 output on raw input; ``fc`` is the decision layer."""
    from models.wrapper import NormalizedModel, unwrap

    inner = unwrap(model)
    if not all(hasattr(inner, a) for a in ("layer4", "avgpool", "fc")):
        raise ValueError("CVE needs a ResNet-style backbone (EVAL_PLAN 4.2)")
    trunk = nn.Sequential(inner.conv1, inner.bn1, inner.relu, inner.maxpool,
                          inner.layer1, inner.layer2, inner.layer3, inner.layer4)

    def features(x):
        if isinstance(model, NormalizedModel):
            x = (x - model.mean.to(x.device)) / model.std.to(x.device)
        return trunk(x)

    def decision(maps):
        return inner.fc(torch.flatten(inner.avgpool(maps), 1))

    return features, decision, inner.fc


def cve_pair(model, x, h, backbone, knobs, device, dataset: str, seed: int = 0) -> PairMaps:
    """One-sided: query cells of x in edit order while flipping k -> l (CD1 only)."""
    from baselines.cve import gap_linear_log_probs, greedy_edits

    features, decision, fc = _resnet_split(model)
    kept = h.ids[:, 0]
    foil = h.ids[:, 1]
    maps, flipped, found = [], [], []
    with torch.no_grad():
        query_feats = features(x)
    for i in range(x.shape[0]):
        distractor = find_distractor(model, dataset, int(foil[i]), seed + i, knobs.cve_distractor_tries, device)
        if distractor is None:
            maps.append(torch.full(query_feats.shape[-2:], float("nan"), device=x.device))
            flipped.append(False)
            found.append(False)
            continue
        with torch.no_grad():
            d_feat = features(distractor.unsqueeze(0).to(x.device))[0]
        edits = greedy_edits(query_feats[i], d_feat, decision, int(foil[i]),
                             gap_linear_log_probs(d_feat, fc, int(foil[i])))
        rank = edits.rank.to(torch.float32)
        top = rank.max().clamp_min(1.0)
        # Earliest edit scores highest; untouched cells score 0.
        maps.append(torch.where(rank > 0, top - rank + 1.0, torch.zeros_like(rank)).to(x.device))
        flipped.append(bool(edits.flipped))
        found.append(True)
    k_map = torch.stack(maps).reshape(x.shape[0], -1)
    gh, gw = query_feats.shape[-2:]
    return PairMaps(k=k_map, l=None, grid=(int(gh), int(gw)),
                    extra={"cve_flipped": flipped, "cve_distractor_found": found})


# ---- SC-CVE -----------------------------------------------------------------

_AUX_CACHE: dict = {}
_SWAV: dict = {}


def _swav_model(device, weights_path: Optional[str] = None):
    """Pinned SwAV trunk, loaded once per device from a local weights file."""
    from baselines.sc_cve import load_swav_backbone

    key = str(torch.device(device))
    if key not in _SWAV:
        if not weights_path:
            raise ValueError(
                "SC-CVE needs the SwAV weights file (downloaded with approval and "
                "hash-recorded at repro time); pass knobs.sc_cve_swav_weights"
            )
        _SWAV[key] = load_swav_backbone(weights_path, device)
    return _SWAV[key]


def _aux_features(images: torch.Tensor, model, device, dataset: str, split: str,
                  indices: Optional[list[int]], weights_path: Optional[str] = None) -> torch.Tensor:
    """SwAV trunk features for raw ``[0, 1]`` images, cached per pool index.

    ``indices=None`` means batch images outside the pool (the query): computed
    without caching.
    """
    from baselines.sc_cve import swav_features

    swav = _swav_model(device, weights_path)
    if indices is None:
        return swav_features(images, swav, device)
    out = []
    for idx, image in zip(indices, images):
        key = (dataset, split, int(idx))
        if key not in _AUX_CACHE:
            _AUX_CACHE[key] = swav_features(image.unsqueeze(0), swav, device)[0]
        out.append(_AUX_CACHE[key])
    return torch.stack(out, dim=0)


def sc_cve_pair(model, x, h, backbone, knobs, device, dataset: str, seed: int = 0) -> PairMaps:
    """One-sided SC-CVE map: query cells in joint-search edit order (CD1 only).

    The distractor class is the foil (rank 1), not the confusion-matrix class
    upstream uses; the dossier names this shared rule. Fast knobs take the
    Goyal path (``lambd=0``, no prefilter, 1 distractor) so smoke runs never
    touch the SwAV weights; paper runs use the authors' defaults.
    """
    from baselines.sc_cve import (
        OURS_LAMBD,
        OURS_TEMPERATURE,
        OURS_TOPK,
        NoFlipError,
        edits_to_rank,
        edits_to_scores,
        run_sc_cve_edits,
    )

    features, decision, _fc = _resnet_split(model)
    foil = h.ids[:, 1]
    if knobs.fast:
        lambd, temperature, topk, n_dist = 0.0, None, None, 1
    else:
        lambd, temperature, topk = OURS_LAMBD, OURS_TEMPERATURE, OURS_TOPK
        n_dist = knobs.sc_cve_distractors
    pool_split = "val" if dataset == "imagenet" else "train"
    with torch.no_grad():
        query_feats = features(x)
    maps, flipped, found, n_edits = [], [], [], []
    for i in range(x.shape[0]):
        cands = find_distractors(model, dataset, int(foil[i]), seed + i,
                                 knobs.cve_distractor_tries, device, n=n_dist)
        if not cands:
            maps.append(torch.full(query_feats.shape[-2:], float("nan"), device=x.device))
            flipped.append(False)
            found.append(False)
            n_edits.append(0)
            continue
        pool_idx, d_imgs = zip(*cands)
        d_batch = torch.stack(list(d_imgs), dim=0).to(x.device)
        with torch.no_grad():
            d_feat = features(d_batch)
        if topk is None and lambd == 0.0:
            query_aux = distractor_aux = None
        else:
            query_aux = _aux_features(x[i:i + 1], model, device, dataset, pool_split,
                                      None, knobs.sc_cve_swav_weights)
            distractor_aux = _aux_features(d_batch, model, device, dataset, pool_split,
                                           list(pool_idx), knobs.sc_cve_swav_weights)
        try:
            edits = run_sc_cve_edits(query_feats[i].detach(), d_feat.detach(),
                                     decision, int(foil[i]), lambd=lambd,
                                     temperature=temperature, topk=topk,
                                     query_aux=query_aux, distractor_aux=distractor_aux,
                                     device=x.device)
        except NoFlipError:
            maps.append(torch.full(query_feats.shape[-2:], float("nan"), device=x.device))
            flipped.append(False)
            found.append(True)
            n_edits.append(0)
            continue
        rank = edits_to_rank(edits, *query_feats.shape[-2:], x.device)
        maps.append(edits_to_scores(rank))
        flipped.append(True)
        found.append(True)
        n_edits.append(len(edits))
    k_map = torch.stack(maps).reshape(x.shape[0], -1)
    gh, gw = query_feats.shape[-2:]
    return PairMaps(k=k_map, l=None, grid=(int(gh), int(gw)),
                    extra={"sc_cve_flipped": flipped, "sc_cve_distractor_found": found,
                           "sc_cve_n_edits": n_edits})


_POOLS: dict = {}


def find_distractors(model, dataset: str, class_idx: int, seed: int, tries: int, device,
                     n: int = 1) -> list[tuple[int, torch.Tensor]]:
    """Up to ``n`` seeded ``(pool_index, image)`` pairs of ``class_idx``.

    Each image is also predicted as ``class_idx`` by the model. Drawn from the
    train split (ImageNet: our val carve, since it has no train split here).
    CVE keeps ``n=1``; SC-CVE takes up to ``n`` for the joint search.
    """
    pool_split = "val" if dataset == "imagenet" else "train"
    key = (dataset, pool_split)
    if key not in _POOLS:
        from evaluation.datasets import get_eval_loader
        from evaluation.run_data import DATA_ROOT, IMAGE_SIZE

        loader, _ = get_eval_loader(dataset, 1, DATA_ROOT, image_size=IMAGE_SIZE, split=pool_split)
        ds = loader.dataset
        targets = _targets(ds)
        _POOLS[key] = (ds, targets)
    ds, targets = _POOLS[key]
    members = [i for i, t in enumerate(targets) if t == class_idx]
    if not members:
        return []
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(members), generator=g)[:tries].tolist()
    found = []
    for j in order:
        image, _ = ds[members[j]]
        with torch.no_grad():
            pred = int(model(image.unsqueeze(0).to(device)).argmax(-1))
        if pred == class_idx:
            found.append((members[j], image))
            if len(found) >= max(1, int(n)):
                break
    return found


def find_distractor(model, dataset: str, class_idx: int, seed: int, tries: int, device) -> Optional[torch.Tensor]:
    """A seeded image of ``class_idx`` that the model also predicts as ``class_idx``.

    Drawn from the train split (ImageNet: our val carve, since it has no train
    split here). Returns ``None`` when no such image is found within ``tries``.
    """
    found = find_distractors(model, dataset, class_idx, seed, tries, device, n=1)
    return found[0][1] if found else None


def _targets(ds) -> list[int]:
    from torch.utils.data import Subset

    if isinstance(ds, Subset):
        inner = _targets(ds.dataset)
        return [inner[i] for i in ds.indices]
    targets = getattr(ds, "targets", None)
    if targets is None:
        targets = [int(ds[i][1]) for i in range(len(ds))]
    return [int(t) for t in targets]


CONTRASTIVE_CORE = ("cdea", "base_evidence", "margin_gradcam", "margin_ig", "extremal", "cve", "random_floor")
CONTRASTIVE_EXTENDED = ("extremal_preserve", "rise_margin", "contrastive_gradcam", "analytic", "sc_cve", "chefer")

WHOLE_SAMPLE = ("cve", "sc_cve", "random_floor")


def contrastive_method(name: str):
    return {
        "cdea": cdea_pair,
        "base_evidence": base_evidence_pair,
        "margin_gradcam": margin_gradcam_pair,
        "margin_ig": margin_ig_pair,
        "extremal": extremal_contrastive_pair,
        "cve": cve_pair,
        "random_floor": random_pair,
        "extremal_preserve": extremal_preserve_pair,
        "rise_margin": rise_margin_pair,
        "contrastive_gradcam": contrastive_gradcam_pair,
        "analytic": analytic_pair,
        "sc_cve": sc_cve_pair,
        "chefer": chefer_class_pair,
    }[name]


# ---------------------------------------------------------------------------
# Val selection candidates (EVAL_PLAN.md section 6.2). Each name is
# ``method@variant``. A CdeaConfig replaces the method's config; a dict
# replaces Knobs fields.
# ---------------------------------------------------------------------------

def _cdea_grid() -> dict:
    out = {}
    for backend in ("gradcam", "ig"):
        for lr in (0.05, 0.2):
            out[f"cdea@{backend}_lr{lr:g}_t100"] = CdeaConfig(backend=backend, lr=lr, steps=100)
    for lr in (0.05, 0.2):
        out[f"cdea@ig_lr{lr:g}_t50"] = CdeaConfig(backend="ig", lr=lr, steps=50)
    return out


CONTRASTIVE_CANDIDATES: dict = {
    **_cdea_grid(),
    "margin_gradcam@default": {},
    "margin_ig@ig16": {"ig_steps": 16},
    "margin_ig@ig32": {"ig_steps": 32},
    **{f"extremal@it{it}_sm{sm:g}": {"extremal_max_iter": it, "extremal_smooth": sm}
       for it in (300, 800) for sm in (0.0, 0.1)},
    **{f"extremal_preserve@it{it}_sm{sm:g}": {"extremal_max_iter": it, "extremal_smooth": sm}
       for it in (300, 800) for sm in (0.0, 0.1)},
    **{f"rise_margin@m{n}_s{s}": {"rise_masks": n, "rise_cell": s} for n in (2000, 4000) for s in (7, 8)},
    "base_evidence@gradcam": {"backend": "gradcam"},
    "base_evidence@ig": {"backend": "ig"},
    "analytic@gradcam": {"backend": "gradcam"},
    "analytic@ig": {"backend": "ig"},
    "chefer@start0": {},
    "chefer@start1": {},
}


SHIFT_CORE = (
    "cdea_shift",
    "gap_attribution",
    "attribution_difference",
    "spray",
    "extremal_shift",
    "random_floor",
)
SHIFT_EXTENDED = ("r2r",)


def _shift_config(cfg, knobs: Knobs):
    from cdea.shift import ShiftConfig

    cfg = cfg or ShiftConfig()
    return ShiftConfig(
        steps=knobs.cdea_steps if cfg.steps is None else int(cfg.steps),
        lr=float(cfg.lr),
        projection=cfg.projection,
        independent=cfg.independent,
        robust=cfg.robust,
        shortcut=cfg.shortcut,
        init=cfg.init,
        backend=cfg.backend,
        ig_steps=knobs.ig_steps,
        kind=cfg.kind,
    )


def cdea_shift_maps(model, xs, backbone, knobs, device, cfg=None, area: float = 0.05) -> ShiftMaps:
    from cdea.shift import allocate_shift

    grid = grid_of(backbone)
    resolved = _shift_config(cfg, knobs)
    if backbone.startswith("vit") and resolved.backend == "gradcam":
        resolved = replace(resolved, backend="ig")
    out = allocate_shift(model, xs, grid[0], grid[1], area, resolved)
    return ShiftMaps(robust=out.robust, shortcut=out.shortcut, grid=grid)


def gap_attribution_maps(model, xs, backbone, knobs, device) -> ShiftMaps:
    from baselines.gap_attribution import gap_attribution

    grid = grid_of(backbone)
    robust, shortcut = gap_attribution(model, xs, grid[0], grid[1], steps=knobs.ig_steps)
    return ShiftMaps(robust=robust, shortcut=shortcut, grid=grid)


def attribution_difference_maps(model, xs, backbone, knobs, device) -> ShiftMaps:
    from base_evidence.gradcam_regions import GradCAMRegionsProvider
    from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
    from baselines.shift_maps import environment_maps

    grid = grid_of(backbone)
    y = model(xs[0]).argmax(dim=-1)
    hyp = HypothesisSet(
        ids=y.view(-1, 1),
        mask=torch.ones(y.shape[0], 1, dtype=torch.bool, device=y.device),
    )
    if default_backend(backbone) == "ig":
        provider = IntegratedGradientsRegionsProvider(grid[0], grid[1], steps=knobs.ig_steps, baseline="blur")
    else:
        provider = GradCAMRegionsProvider(grid[0], grid[1])
    maps = []
    for view in xs:
        evidence = provider.explain(view, model, hyp)[:, 0]
        maps.append(evidence.reshape(evidence.shape[0], grid[0], grid[1]))
    robust, shortcut = environment_maps(maps)
    return ShiftMaps(
        robust=robust.reshape(robust.shape[0], -1),
        shortcut=shortcut.reshape(shortcut.shape[0], -1),
        grid=grid,
    )


def spray_maps(model, xs, backbone, knobs, device, seed: int = 0) -> ShiftMaps:
    from baselines.shift_maps import environment_maps
    from baselines.spray import cluster_mean_relevance, lrp_maps

    y = model(xs[0]).argmax(dim=-1)
    reduced = []
    for view in xs:
        relevance = lrp_maps(model, view, y)
        count = int(relevance.shape[0])
        if count >= 3:
            eigenvalues = min(32, count - 1)
            neighbors = min(10, count - 1)
            means, _labels = cluster_mean_relevance(
                relevance, n_clusters=2, n_eigval=eigenvalues, n_neighbors=neighbors, seed=seed,
            )
            reduced.append(means)
        else:
            reduced.append(relevance)
    robust, shortcut = environment_maps(reduced)
    return ShiftMaps(robust=robust, shortcut=shortcut, grid=None)


def extremal_shift_maps(model, xs, backbone, knobs, device, area: float = 0.05) -> ShiftMaps:
    from baselines.shift_maps import per_environment_extremal

    y = model(xs[0]).argmax(dim=-1)
    robust, shortcut = per_environment_extremal(
        model, xs, y, area=area, max_iter=knobs.extremal_max_iter, smooth=knobs.extremal_smooth,
    )
    return ShiftMaps(robust=robust, shortcut=shortcut, grid=None)


def random_shift_maps(model, xs, backbone, knobs, device, seed: int = 0) -> ShiftMaps:
    grid = grid_of(backbone)
    count = xs[0].shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scores = torch.rand(count, grid[0] * grid[1], generator=generator).to(device=xs[0].device)
    return ShiftMaps(robust=scores, shortcut=scores, grid=grid)


def r2r_shift_maps(model, xs, backbone, knobs, device) -> ShiftMaps:
    root = Path(__file__).resolve().parents[1] / "third_party" / "r2r"
    if not root.is_dir():
        raise FileNotFoundError("R2R is fetched by scripts/fetch_r2r.sh and is not vendored")
    raise FileNotFoundError("R2R was fetched but the adapter has no weights for this cell")


def shift_method(name: str):
    return {
        "cdea_shift": cdea_shift_maps,
        "gap_attribution": gap_attribution_maps,
        "attribution_difference": attribution_difference_maps,
        "spray": spray_maps,
        "extremal_shift": extremal_shift_maps,
        "random_floor": random_shift_maps,
        "r2r": r2r_shift_maps,
    }[name]


def _shift_grid() -> dict:
    from cdea.shift import ShiftConfig

    out = {}
    for backend in ("gradcam", "ig"):
        for lr in (0.05, 0.2):
            out[f"cdea_shift@{backend}_lr{lr:g}_t100"] = ShiftConfig(backend=backend, lr=lr, steps=100)
    return out


SHIFT_ABLATIONS = {
    "S1_deletion": None,
    "S2_independent": None,
    "S3_mean": None,
    "S4_uniform": None,
    "S5_first_order": None,
    "S6_precomputed": "precomputed",
}


def shift_ablation(name: str):
    from cdea.shift import ShiftConfig

    return {
        "S1_deletion": ShiftConfig(shortcut="deletion"),
        "S2_independent": ShiftConfig(independent=True),
        "S3_mean": ShiftConfig(robust="mean"),
        "S4_uniform": ShiftConfig(init="uniform"),
        "S5_first_order": ShiftConfig(kind="first_order"),
        "S6_precomputed": "precomputed",
    }[name]


SHIFT_CANDIDATES = _shift_grid()


def candidate_applies(name: str, backbone: str) -> bool:
    """ViT uses IG. ResNet's CDEA grid is T = 100. ViT's CDEA grid is IG only."""
    vit = backbone.startswith("vit")
    if vit and "gradcam" in name:
        return False
    if name.startswith("cdea@"):
        if vit:
            return name.startswith("cdea@ig_")
        return "_t50" not in name
    if name.startswith("chefer"):
        return vit
    return True
