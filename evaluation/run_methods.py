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
from typing import Callable, Optional

import torch
import torch.nn as nn

from core.types import EnvBatch, HypothesisSet


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class PairMaps:
    k: torch.Tensor
    l: Optional[torch.Tensor]                 # None: the method has no foil map (CVE)
    grid: Optional[tuple[int, int]]           # None: pixel maps
    per_class: Optional[torch.Tensor] = None  # (B, K, R) or (B, K, H, W)
    extra: dict = field(default_factory=dict)


@dataclass
class ShiftMaps:
    robust: torch.Tensor
    shortcut: torch.Tensor
    grid: Optional[tuple[int, int]]
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
    cdea_steps: int = 50
    shift_steps: int = 50
    cve_distractor_tries: int = 32
    extremal_smooth: float = 0.0
    fast: bool = False


FAST = Knobs(ig_steps=4, extremal_max_iter=40, rise_masks=200, rise_cell=7,
             cdea_steps=8, shift_steps=8, cve_distractor_tries=8, fast=True)


@dataclass
class CdeaConfig:
    """One CDEA configuration. Defaults are the plan's preset before selection."""

    backend: str = "gradcam"           # gradcam | ig | library:<method>
    preset: str = "mixed"
    lambda_margin: float = 1.0
    lambda_overlap: float = 0.2
    lambda_shared_sparse: float = 0.25
    use_shared: Optional[bool] = None  # None: the preset's choice
    lr: float = 0.2
    steps: Optional[int] = None        # None: Knobs.cdea_steps
    init_from_evidence: bool = True
    interaction: str = "none"          # none | attention | transformer
    attn_mix: float = 0.0
    independent: bool = False          # A1: K separate single-class runs
    top_k: int = 5


ABLATIONS: dict[str, Callable[[CdeaConfig], CdeaConfig]] = {
    "A1_independent": lambda c: replace(c, independent=True, use_shared=False),
    "A2_no_margin": lambda c: replace(c, lambda_margin=0.0),
    "A3_no_overlap": lambda c: replace(c, lambda_overlap=0.0),
    "A4_no_shared": lambda c: replace(c, use_shared=False),
    "A5_init_zero": lambda c: replace(c, init_from_evidence=False),
    "A5_init_evidence": lambda c: replace(c, init_from_evidence=True),
    "A5_backend_ig": lambda c: replace(c, backend="ig"),
    "A5_backend_layercam": lambda c: replace(c, backend="library:layercam"),
    "A5_backend_xgradcam": lambda c: replace(c, backend="library:xgradcam"),
    "A5_backend_gradcam++": lambda c: replace(c, backend="library:gradcam++"),
    "A5_backend_inputxgradient": lambda c: replace(c, backend="library:inputxgradient"),
    "A5_backend_gradientshap": lambda c: replace(c, backend="library:gradientshap"),
    "A6_interaction_none": lambda c: replace(c, interaction="none", attn_mix=0.0),
    "A6_interaction_attention": lambda c: replace(c, interaction="attention", attn_mix=0.35),
    "A6_interaction_transformer": lambda c: replace(c, interaction="transformer", attn_mix=0.35),
    "A7_steps_10": lambda c: replace(c, steps=10),
    "A7_steps_25": lambda c: replace(c, steps=25),
    "A7_steps_50": lambda c: replace(c, steps=50),
    "A7_steps_100": lambda c: replace(c, steps=100),
    "A8_preset_mixed": lambda c: replace(c, preset="mixed"),
    "A8_preset_cooperative": lambda c: replace(c, preset="cooperative"),
    "A8_preset_competitive": lambda c: replace(c, preset="competitive"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grid_of(backbone: str) -> tuple[int, int]:
    from scripts.train_backbone import model_grid_size

    return tuple(model_grid_size(backbone))


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

class _OnlyRank:
    """A1: the top-K set with every hypothesis but one rank masked out."""

    def __init__(self, base, rank: int):
        self.base = base
        self.rank = rank

    def select(self, logits, probs):
        h = self.base.select(logits, probs)
        keep = torch.zeros_like(h.mask)
        keep[:, self.rank] = h.mask[:, self.rank]
        return HypothesisSet(ids=h.ids, mask=keep)


def build_cdea(model: nn.Module, backbone: str, cfg: CdeaConfig, knobs: Knobs, device, selector=None,
               loss_scale: float = 1.0):
    from core.game_modes import resolve_contrastive_game
    from core.hypotheses import TopMSelector
    from core.interaction import AttentionOnlyInteraction, Transformer1LayerInteraction
    from core.runner import CDEAExplainer
    from instantiations.contrastive.allocator import OptimizationAllocator
    from instantiations.contrastive.objective import ContrastiveObjective
    from modality.grid_regions import VisionGridUnitSpace

    gh, gw = grid_of(backbone)
    game = resolve_contrastive_game(cfg.preset)
    use_shared = game.use_shared if cfg.use_shared is None else cfg.use_shared
    margin = cfg.lambda_margin if cfg.preset == "mixed" else game.lambda_margin
    overlap = cfg.lambda_overlap if cfg.preset == "mixed" else game.lambda_overlap
    objective = ContrastiveObjective(
        lambda_margin=margin,
        lambda_overlap=overlap,
        lambda_shared_sparse=cfg.lambda_shared_sparse,
        mass_ref_regions=49,
    )
    allocator = OptimizationAllocator(
        objective,
        num_steps=cfg.steps if cfg.steps is not None else knobs.cdea_steps,
        lr=cfg.lr,
        use_shared=use_shared,
        lambda_partition=game.lambda_partition,
        init_from_evidence=cfg.init_from_evidence,
        attn_mix=cfg.attn_mix,
        loss_scale=loss_scale,
    )
    interaction = None
    embed_dim = None
    if cfg.interaction != "none":
        embed_dim = 32
        interaction = (AttentionOnlyInteraction(embed_dim) if cfg.interaction == "attention"
                       else Transformer1LayerInteraction(embed_dim))
    return CDEAExplainer(
        model=model,
        unit_space=VisionGridUnitSpace(gh, gw, embed_dim=embed_dim),
        selector=selector or TopMSelector(m=cfg.top_k),
        base_evidence=evidence_provider(cfg.backend, backbone, knobs),
        allocator=allocator,
        objective=objective,
        interaction=interaction,
        normalize_evidence=True,
        device=device,
    )


def cdea_pair(model, x, h, backbone, knobs, device, cfg: Optional[CdeaConfig] = None,
              loss_scale: float = 1.0) -> PairMaps:
    """Unique masks of rank 0 and rank 1 (EVAL_PLAN 4.2). Shared is kept for the sensitivity check."""
    from core.hypotheses import TopMSelector

    cfg = cfg or CdeaConfig(backend=default_backend(backbone))
    grid = grid_of(backbone)
    if cfg.independent:
        base = TopMSelector(m=cfg.top_k)
        columns = []
        for rank in range(min(cfg.top_k, h.ids.shape[1])):
            explainer = build_cdea(model, backbone, cfg, knobs, device, selector=_OnlyRank(base, rank),
                                   loss_scale=loss_scale)
            columns.append(explainer.explain(x).masks["unique"][:, rank])
        unique = torch.stack(columns, dim=1)
        shared = None
    else:
        out = build_cdea(model, backbone, cfg, knobs, device, loss_scale=loss_scale).explain(x)
        unique = out.masks["unique"]
        shared = out.masks.get("shared")
    unique = unique.reshape(unique.shape[0], unique.shape[1], -1)
    extra = {}
    if shared is not None:
        extra["shared"] = shared.reshape(shared.shape[0], -1)
    return PairMaps(k=unique[:, 0], l=unique[:, 1], grid=grid, per_class=unique, extra=extra)


# ---------------------------------------------------------------------------
# Contrastive baselines
# ---------------------------------------------------------------------------

def base_evidence_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    provider = evidence_provider(default_backend(backbone), backbone, knobs)
    E = provider.explain(x, model, h)
    return PairMaps(k=E[:, 0], l=E[:, 1], grid=grid_of(backbone), per_class=E)


def naive_contrastive_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    provider = evidence_provider(default_backend(backbone), backbone, knobs)
    E = provider.explain(x, model, h)
    valid = h.mask.to(E.dtype).unsqueeze(-1)
    total = (E * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp_min(2.0)

    def contrast(j):
        others = (total - E[:, j] * valid[:, j]) / (count - 1)
        return E[:, j] - others

    return PairMaps(k=contrast(0), l=contrast(1), grid=grid_of(backbone))


def margin_gradcam_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    from baselines.margin import margin_gradcam

    # A separate call for l: with the ReLU, the l-map is not the negated k-map.
    return PairMaps(k=margin_gradcam(model, x, h), l=margin_gradcam(model, x, swapped(h)), grid=None)


def margin_ig_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    from baselines.margin import margin_integrated_gradients

    # IG is linear in the target, so attr(z_l - z_k) = -attr(z_k - z_l).
    k_map = margin_integrated_gradients(model, x, h, steps=max(2, knobs.ig_steps))
    return PairMaps(k=k_map, l=-k_map, grid=None)


def extremal_contrastive_pair(model, x, h, backbone, knobs, device, area: float = 0.05) -> PairMaps:
    from baselines.extremal import margin_masks

    kw = dict(area=area, max_iter=knobs.extremal_max_iter, smooth=knobs.extremal_smooth)
    return PairMaps(k=margin_masks(model, x, h, **kw), l=margin_masks(model, x, swapped(h), **kw), grid=None)


def extremal_class_pair(model, x, h, backbone, knobs, device, area: float = 0.05) -> PairMaps:
    from baselines.extremal import class_masks

    kw = dict(area=area, max_iter=knobs.extremal_max_iter, smooth=knobs.extremal_smooth)
    return PairMaps(k=class_masks(model, x, h.ids[:, 0], **kw), l=class_masks(model, x, h.ids[:, 1], **kw), grid=None)


def rise_margin_pair(model, x, h, backbone, knobs, device, seed: int = 0) -> PairMaps:
    from baselines.rise import margin_maps

    kw = dict(n_masks=knobs.rise_masks, s=knobs.rise_cell, seed=seed)
    return PairMaps(k=margin_maps(model, x, h, **kw), l=margin_maps(model, x, swapped(h), **kw), grid=None)


def contrastive_gradcam_pair(model, x, h, backbone, knobs, device) -> PairMaps:
    from baselines.contrastive_gradcam import contrastive_gradcam

    return PairMaps(k=contrastive_gradcam(model, x, h), l=contrastive_gradcam(model, x, swapped(h)), grid=None)


def random_pair(model, x, h, backbone, knobs, device, seed: int = 0) -> PairMaps:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    gh, gw = grid_of(backbone)
    k_map = torch.rand(x.shape[0], gh * gw, generator=g)
    l_map = torch.rand(x.shape[0], gh * gw, generator=g)
    return PairMaps(k=k_map.to(x.device), l=l_map.to(x.device), grid=(gh, gw))


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


_POOLS: dict = {}


def find_distractor(model, dataset: str, class_idx: int, seed: int, tries: int, device) -> Optional[torch.Tensor]:
    """A seeded image of ``class_idx`` that the model also predicts as ``class_idx``.

    Drawn from the train split (ImageNet: our val carve, since it has no train
    split here). Returns ``None`` when no such image is found within ``tries``.
    """
    pool_split = "val" if dataset == "imagenet" else "train"
    key = (dataset, pool_split)
    if key not in _POOLS:
        from scripts.ablation_contrastive import _get_eval_loader
        from evaluation.run_data import DATA_ROOT, IMAGE_SIZE

        loader, _ = _get_eval_loader(dataset, 1, DATA_ROOT, image_size=IMAGE_SIZE, split=pool_split)
        ds = loader.dataset
        targets = _targets(ds)
        _POOLS[key] = (ds, targets)
    ds, targets = _POOLS[key]
    members = [i for i, t in enumerate(targets) if t == class_idx]
    if not members:
        return None
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(members), generator=g)[:tries].tolist()
    for j in order:
        image, _ = ds[members[j]]
        with torch.no_grad():
            pred = int(model(image.unsqueeze(0).to(device)).argmax(-1))
        if pred == class_idx:
            return image
    return None


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
CONTRASTIVE_EXTENDED = ("extremal_class", "rise_margin", "contrastive_gradcam", "naive_contrastive")


def contrastive_method(name: str):
    return {
        "cdea": cdea_pair,
        "base_evidence": base_evidence_pair,
        "margin_gradcam": margin_gradcam_pair,
        "margin_ig": margin_ig_pair,
        "extremal": extremal_contrastive_pair,
        "cve": cve_pair,
        "random_floor": random_pair,
        "extremal_class": extremal_class_pair,
        "rise_margin": rise_margin_pair,
        "contrastive_gradcam": contrastive_gradcam_pair,
        "naive_contrastive": naive_contrastive_pair,
    }[name]


# ---------------------------------------------------------------------------
# Shift methods
# ---------------------------------------------------------------------------

@dataclass
class ShiftConfig:
    backend: str = "gradcam"
    preset: str = "mixed"
    lambda_gap: Optional[float] = None
    lambda_mass: float = 0.1
    lambda_disjoint: Optional[float] = None
    lr: float = 0.5
    steps: Optional[int] = None
    objective: str = "paired"        # paired | unpaired (AS1)


SHIFT_ABLATIONS: dict[str, Callable[[ShiftConfig], ShiftConfig]] = {
    "AS1_paired_mass": lambda c: replace(c, objective="paired", lambda_mass=0.1),
    "AS1_paired_nomass": lambda c: replace(c, objective="paired", lambda_mass=0.0),
    "AS1_unpaired_mass": lambda c: replace(c, objective="unpaired", lambda_mass=0.1),
    "AS1_unpaired_nomass": lambda c: replace(c, objective="unpaired", lambda_mass=0.0),
}


def _predicted(model, x) -> torch.Tensor:
    with torch.no_grad():
        return model(x).argmax(-1)


def cdea_shift_maps(model, sample, backbone, knobs, device, cfg: Optional[ShiftConfig] = None, seed: int = 0,
                    loss_scale: float = 1.0, init_offset: int = 0) -> ShiftMaps:
    from core.game_modes import resolve_shift_game
    from core.hypotheses import TopMSelector
    from core.runner import CDEAExplainer
    from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
    from instantiations.shift.objective import GroupStatisticsObjective, RobustShortcutObjective
    from modality.grid_regions import VisionGridUnitSpace

    cfg = cfg or ShiftConfig(backend=default_backend(backbone))
    game = resolve_shift_game(cfg.preset)
    gap = game.lambda_gap if cfg.lambda_gap is None else cfg.lambda_gap
    disjoint = game.lambda_disjoint if cfg.lambda_disjoint is None else cfg.lambda_disjoint
    gh, gw = grid_of(backbone)
    common = dict(lambda_mean=game.lambda_mean, lambda_var=game.lambda_var, lambda_gap=gap,
                  lambda_disjoint=disjoint, lambda_sparse=game.lambda_sparse,
                  lambda_mass=cfg.lambda_mass, target="pred")
    steps = cfg.steps if cfg.steps is not None else knobs.shift_steps
    if cfg.objective == "unpaired":
        # Each image belongs to one group: the ID and OOD views become separate
        # images, grouped by (label, view).
        objective = GroupStatisticsObjective(**common)
        x = torch.cat(sample.env.xs[:2], dim=0)
        labels = sample.labels.repeat(2)
        view = torch.cat([torch.zeros_like(sample.labels), torch.ones_like(sample.labels)])
        group = (labels * 2 + view).to(device)
        allocator = RobustShortcutOptimizationAllocator(objective, num_steps=steps, lr=cfg.lr,
                                                        lambda_disjoint=disjoint, init_seed=seed,
                                                        loss_scale=loss_scale, init_offset=init_offset)
        explainer = CDEAExplainer(model=model, unit_space=VisionGridUnitSpace(gh, gw),
                                  selector=TopMSelector(m=1),
                                  base_evidence=evidence_provider(cfg.backend, backbone, knobs),
                                  allocator=_GroupAllocator(allocator, group), objective=_GroupObjective(objective, group),
                                  device=device)
        # The allocator requires an EnvBatch; the group objective ignores it.
        out = explainer.explain(x, env=EnvBatch(xs=[x], env_ids=["pooled"]))
        n = sample.labels.shape[0]
        robust = out.masks["robust"].reshape(2 * n, -1)[:n]
        shortcut = out.masks["shortcut"].reshape(2 * n, -1)[:n]
        return ShiftMaps(robust=robust, shortcut=shortcut, grid=(gh, gw))
    objective = RobustShortcutObjective(**common, lambda_shortcut=game.lambda_shortcut)
    allocator = RobustShortcutOptimizationAllocator(objective, num_steps=steps, lr=cfg.lr,
                                                    lambda_disjoint=disjoint, init_seed=seed,
                                                    loss_scale=loss_scale, init_offset=init_offset)
    explainer = CDEAExplainer(model=model, unit_space=VisionGridUnitSpace(gh, gw),
                              selector=TopMSelector(m=1),
                              base_evidence=evidence_provider(cfg.backend, backbone, knobs),
                              allocator=allocator, objective=objective, device=device)
    out = explainer.explain(sample.x_id, env=sample.env)
    return ShiftMaps(robust=out.masks["robust"].reshape(sample.x_id.shape[0], -1),
                     shortcut=out.masks["shortcut"].reshape(sample.x_id.shape[0], -1), grid=(gh, gw))


class _GroupObjective:
    """Passes the fixed ``group`` tensor to GroupStatisticsObjective.compute."""

    def __init__(self, inner, group):
        self.inner = inner
        self.group = group
        for name in ("lambda_mass", "mass_ref_regions", "ld"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    def compute(self, *args, **kwargs):
        kwargs.setdefault("group", self.group)
        return self.inner.compute(*args, **kwargs)


class _GroupAllocator:
    def __init__(self, inner, group):
        self.inner = inner
        self.inner.objective = _GroupObjective(inner.objective, group)

    def allocate(self, **kwargs):
        return self.inner.allocate(**kwargs)


def attribution_difference_maps(model, sample, backbone, knobs, device, backend: Optional[str] = None) -> ShiftMaps:
    from baselines.shift_maps import environment_maps

    provider = evidence_provider(backend or default_backend(backbone), backbone, knobs)
    gh, gw = grid_of(backbone)
    h = _single(_predicted(model, sample.x_id))
    per_env = [provider.explain(xe, model, h)[:, 0].reshape(-1, gh, gw) for xe in sample.env.xs]
    robust, shortcut = environment_maps(per_env)
    b = sample.x_id.shape[0]
    return ShiftMaps(robust=robust.reshape(b, -1), shortcut=shortcut.reshape(b, -1), grid=(gh, gw))


def spray_maps(model, sample, backbone, knobs, device, seed: int = 0) -> ShiftMaps:
    """Zennit EpsilonPlus relevance on x_id, clustered with CoRelAy (EVAL_PLAN 4.3).

    Clustering needs at least 3 images. The shortcut map is each image's
    cluster-mean relevance; the robust map is the image's own relevance.
    """
    from baselines.spray import cluster_mean_relevance, lrp_maps

    pred = _predicted(model, sample.x_id)
    relevance = lrp_maps(model, sample.x_id, pred, batch_size=4).detach()
    n = relevance.shape[0]
    if n < 3:
        raise ValueError("SpRAy clusters relevance maps and needs at least 3 images in the cell")
    means, labels = cluster_mean_relevance(relevance, n_clusters=2, n_eigval=min(32, n - 1),
                                           n_neighbors=min(10, n - 1), seed=seed)
    return ShiftMaps(robust=relevance, shortcut=means, grid=None,
                     extra={"spray_clusters": labels.tolist()})


def extremal_per_env_maps(model, sample, backbone, knobs, device, area: float = 0.05) -> ShiftMaps:
    from baselines.shift_maps import per_environment_extremal

    pred = _predicted(model, sample.x_id)
    robust, shortcut = per_environment_extremal(model, list(sample.env.xs), pred, area=area,
                                                max_iter=knobs.extremal_max_iter, smooth=knobs.extremal_smooth)
    return ShiftMaps(robust=robust, shortcut=shortcut, grid=None)


def random_shift_maps(model, sample, backbone, knobs, device, seed: int = 0) -> ShiftMaps:
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
    gh, gw = grid_of(backbone)
    b = sample.x_id.shape[0]
    return ShiftMaps(robust=torch.rand(b, gh * gw, generator=g).to(sample.x_id.device),
                     shortcut=torch.rand(b, gh * gw, generator=g).to(sample.x_id.device), grid=(gh, gw))


SHIFT_CORE = ("cdea_shift", "attribution_difference", "spray", "extremal_per_env", "random_floor")


def shift_method(name: str):
    return {
        "cdea_shift": cdea_shift_maps,
        "attribution_difference": attribution_difference_maps,
        "spray": spray_maps,
        "extremal_per_env": extremal_per_env_maps,
        "random_floor": random_shift_maps,
    }[name]


# ---------------------------------------------------------------------------
# Val selection candidates (EVAL_PLAN.md section 6.2). Each name is
# ``method@variant``. A CdeaConfig or ShiftConfig replaces the method's config;
# a dict replaces Knobs fields; {"backend": ...} picks the evidence backend.
# ---------------------------------------------------------------------------

def _cdea_grid() -> dict:
    out = {}
    for backend in ("gradcam", "ig"):
        for margin in (0.5, 1.0, 2.0):
            for overlap in (0.1, 0.2, 0.4):
                out[f"cdea@{backend}_m{margin:g}_o{overlap:g}"] = CdeaConfig(
                    backend=backend, lambda_margin=margin, lambda_overlap=overlap)
    return out


CONTRASTIVE_CANDIDATES: dict = {
    **_cdea_grid(),
    "margin_gradcam@default": {},
    "margin_ig@ig16": {"ig_steps": 16},
    "margin_ig@ig32": {"ig_steps": 32},
    **{f"extremal@it{it}_sm{sm:g}": {"extremal_max_iter": it, "extremal_smooth": sm}
       for it in (300, 800) for sm in (0.0, 0.1)},
    **{f"rise_margin@m{n}_s{s}": {"rise_masks": n, "rise_cell": s} for n in (2000, 4000) for s in (7, 8)},
}

SHIFT_CANDIDATES: dict = {
    **{f"cdea_shift@g{g:g}_mass{m:g}_d{d:g}": ShiftConfig(lambda_gap=g, lambda_mass=m, lambda_disjoint=d)
       for g in (0.5, 1.0, 1.5) for m in (0.1, 0.5) for d in (0.2, 0.4)},
    "attribution_difference@gradcam": {"backend": "gradcam"},
    "attribution_difference@ig": {"backend": "ig"},
}


def candidate_applies(name: str, backbone: str) -> bool:
    """ViT has no Grad-CAM evidence (EVAL_PLAN 4.2); CVE has no candidates."""
    if backbone.startswith("vit") and ("@gradcam" in name or name == "margin_gradcam@default"):
        return False
    return True
