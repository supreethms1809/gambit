"""
Ablation: 100-500 images, compare
1. Base evidence alone (no allocation)
2. Naïve contrastive baseline (E_k - mean(E_foils))
3. OptimizationAllocator + ContrastiveObjective

Pass: report at least one quantitative improvement (e.g. overlap reduction at comparable sufficiency).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.types import Tensor, HypothesisSet
from core.hypotheses import TopMSelector
from core.device import get_device
from core.reporting import config_hash, refuse_final_if_dirty, save_json, save_rows_csv
from evaluation.splits import MissingSplitError, SplitLockedError
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider

try:
    from tqdm.auto import tqdm as tqdm_auto
except ImportError:  # pragma: no cover
    tqdm_auto = None  # type: ignore[misc, assignment]


# ----- Small CNN for CIFAR-10 -----
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


TV_INPUT_SIZE = 224
MODEL_CHOICES = ["smallcnn", "resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0",
                 "efficientnet_v2_s", "vit_b_16", "vit_b_32"]
DATASET_CHOICES = [
    "mnist", "cifar10", "cifar100", "pets", "oxford_pets", "stanford_dogs",
    "cub200", "ham10000", "brain_tumor",
]


def _load_state_dict(checkpoint: str):
    """Read either checkpoint format: a raw state_dict (train_backbone.py) or a
    metadata-wrapped dict (examples/contrastive_explanation.py save_checkpoint)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state

# Medical datasets ship pre-split; evaluate on the held-out split (relative to data_root).
# Keep in sync with MEDICAL_SPLIT_ROOTS in examples/contrastive_explanation.py.
MEDICAL_EVAL_ROOTS = {
    "ham10000": Path("ham10000") / "val",
    "brain_tumor": Path("brain_tumor") / "Testing",
}
DATASET_NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "oxford_pets": 37,
    "cub200": 200,
    "pets": 2,
    "stanford_dogs": 120,
    "ham10000": 7,
    "brain_tumor": 3,
}


def _build_model(model_name: str, num_classes: int, pretrained: bool = False,
                 checkpoint: Optional[str] = None) -> nn.Module:
    if model_name == "smallcnn":
        model = SmallCNN(num_classes=num_classes)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
            print(f"  [model] loaded checkpoint: {checkpoint}")
        return model
    try:
        from torchvision import models
    except ImportError as e:
        raise ImportError("torchvision is required for torchvision model backbones") from e

    weights = "IMAGENET1K_V1" if pretrained else None
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        # Its last Conv2d is the final 1x1 projection with a 7x7 spatial output, so
        # GradCAM's automatic target-layer pick is the canonical one and the grid
        # stays 7x7 — comparable to the resnet baselines. Mirrors the builder in
        # examples/contrastive_explanation.py, which trained these checkpoints.
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"model_name must be one of: {', '.join(MODEL_CHOICES)}")
    if checkpoint is not None:
        state = _load_state_dict(checkpoint)
        model.load_state_dict(state)
        print(f"  [model] loaded checkpoint: {checkpoint}")
    return model


def _finalize_eval_dataset(
    dataset_name: str,
    ds,
    *,
    split: str,
    final: bool,
    config_hash: Optional[str],
    num_images: Optional[int],
    seed: int,
):
    """Apply the recorded split, then a seeded subset. Never a class-ordered prefix."""
    from torch.utils.data import Subset

    from evaluation.sampling import seeded_subset
    from evaluation.splits import load_indices

    indices = load_indices(
        dataset_name,
        split,
        final=final,
        config_hash=config_hash,
        n_items=len(ds),
    )
    ds = Subset(ds, indices)
    if num_images is not None:
        ds = seeded_subset(ds, num_images, seed)
    return ds


def _get_eval_loader(
    dataset: str,
    batch_size: int,
    data_root: Path,
    image_size: Optional[int] = None,
    *,
    seed: int = 0,
    num_images: Optional[int] = None,
    split: str = "val",
    final: bool = False,
    config_hash: Optional[str] = None,
):
    try:
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import MNIST, CIFAR10, ImageFolder
    except ImportError as e:
        raise ImportError("torchvision is required for dataset loading") from e

    from evaluation.splits import load_spec

    resize = transforms.Resize((image_size, image_size)) if image_size is not None else None
    spec = load_spec(dataset)
    root_rel = spec["roots"][split]

    if dataset == "mnist":
        ops = [transforms.ToTensor(), transforms.Lambda(lambda t: t.repeat(3, 1, 1))]
        if resize is not None:
            ops.append(resize)
        t = transforms.Compose(ops)
        ds = MNIST(root=str(data_root), train=root_rel == "train", download=False, transform=t)
        num_classes = 10
    elif dataset == "cifar10":
        ops = [transforms.ToTensor()]
        if resize is not None:
            ops.append(resize)
        t = transforms.Compose(ops)
        ds = CIFAR10(root=str(data_root), train=root_rel == "train", download=False, transform=t)
        num_classes = 10
    elif dataset == "pets":
        target_size = image_size if image_size is not None else 64
        t = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor()])
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
        num_classes = len(ds.classes)
    elif dataset == "stanford_dogs":
        target_size = image_size if image_size is not None else 224
        t = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor()])
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
        num_classes = len(ds.classes)
    elif dataset in {"cifar100", "oxford_pets", "cub200"}:
        from evaluation.paper_datasets import open_unsplit

        target_size = image_size if image_size is not None else 224
        t = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        ds = open_unsplit(dataset, split, data_root, transform=t)
        num_classes = len(set(ds.targets))
    elif dataset in MEDICAL_EVAL_ROOTS:
        target_size = image_size if image_size is not None else 224
        t = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor()])
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
        num_classes = len(ds.classes)
    else:
        raise ValueError("dataset must be one of: " + ", ".join(DATASET_CHOICES))

    ds = _finalize_eval_dataset(
        dataset, ds, split=split, final=final, config_hash=config_hash,
        num_images=num_images, seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), num_classes


def _fallback_random_loader(
    dataset: str,
    batch_size: int,
    num_images: int,
    image_size: Optional[int] = None,
):
    class _RandomDS(torch.utils.data.Dataset):
        def __init__(self, n: int, h: int, w: int, c: int, num_classes: int):
            self.n = n
            self.h = h
            self.w = w
            self.c = c
            self.num_classes = num_classes

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            x = torch.rand(self.c, self.h, self.w)
            y = torch.tensor(idx % self.num_classes, dtype=torch.long)
            return x, y

    if image_size is not None:
        h, w = image_size, image_size
        num_classes = DATASET_NUM_CLASSES.get(dataset, 10)
    elif dataset == "mnist":
        h, w, num_classes = 28, 28, 10
    elif dataset == "pets":
        h, w, num_classes = 64, 64, 2
    elif dataset in DATASET_NUM_CLASSES and dataset != "cifar10":
        h, w, num_classes = 224, 224, DATASET_NUM_CLASSES[dataset]
    else:
        h, w, num_classes = 32, 32, 10

    ds = _RandomDS(max(num_images, batch_size), h=h, w=w, c=3, num_classes=num_classes)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), num_classes


def compute_metrics(
    x: Tensor,
    model: nn.Module,
    unit_space: VisionGridUnitSpace,
    hypotheses: HypothesisSet,
    masks_unique: Tensor,
    masks_shared: Tensor | None = None,
) -> Dict[str, float]:
    """Kept logit, margin, overlap, and sparsity, using the objective's overlap formula.

    ``kept_logit`` / ``suff`` is the raw logit on the kept image, matching
    ContrastiveObjective. ``baseline_subtracted_kept_logit`` subtracts the logit
    of an all-zero input. That second quantity is what the shift game reports.
    """
    from instantiations.contrastive.objective import pairwise_overlap

    B, K, R = masks_unique.shape
    valid = hypotheses.mask
    h_ids = hypotheses.ids
    m_tot = masks_unique + (masks_shared[:, None, :] if masks_shared is not None else 0.0)

    with torch.no_grad():
        logits_zeros = model(torch.zeros_like(x))

    kept = torch.zeros(B, K, device=x.device)
    subtracted = torch.zeros(B, K, device=x.device)
    margin = torch.zeros(B, K, device=x.device)
    for k in range(K):
        mk = m_tot[:, k, :]
        x_keep = unit_space.keep(x, mk)
        logits = model(x_keep)
        cls_k = h_ids[:, k].clamp_min(0)
        z_k = logits.gather(1, cls_k.unsqueeze(1)).squeeze(1)
        z_base = logits_zeros.gather(1, cls_k.unsqueeze(1)).squeeze(1)
        kept[:, k] = z_k
        subtracted[:, k] = z_k - z_base
        z_all = logits.gather(1, h_ids.clamp_min(0))
        z_all = z_all.masked_fill(~valid, float("-inf"))
        z_foil = z_all.clone()
        z_foil[:, k] = float("-inf")
        margin[:, k] = z_k - z_foil.max(dim=1).values

    def _mean_valid(values: Tensor) -> float:
        return (values.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)).mean().item()

    kept_logit = _mean_valid(kept)
    overlap = pairwise_overlap(masks_unique).mean().item()
    sparse = masks_unique.abs().sum(dim=-1).mean(dim=1).mean().item()
    return {
        "kept_logit": kept_logit,
        "suff": kept_logit,
        "baseline_subtracted_kept_logit": _mean_valid(subtracted),
        "margin": _mean_valid(margin),
        "overlap": overlap,
        "sparse": sparse,
    }


def run_ablation(
    num_images: Optional[int] = None,
    batch_size: int = 16,
    dataset: str = "cifar10",
    evidence: str = "gradcam",
    model_name: str = "resnet18",
    pretrained: bool = False,
    ig_steps: int = 8,
    lambda_disjoint: float = 0.0,
    lambda_mass: float = 2.0,
    lambda_shared_sparse: float = 0.25,
    data_root: str | None = None,
    export_prefix: str | None = None,
    checkpoint: Optional[str] = None,
    game_mode: str | None = None,
    num_steps: int = 40,
    lr: float = 0.3,
    out_dir: str | Path | None = None,
    seed: int = 0,
    split: str = "val",
    final: bool = False,
):
    from core.runner import CDEAExplainer
    from core.allocator import EvidenceAsMaskAllocator
    from core.game_modes import resolve_contrastive_game
    from instantiations.contrastive.objective import ContrastiveObjective
    from instantiations.contrastive.allocator import OptimizationAllocator

    if dataset not in set(DATASET_CHOICES):
        raise ValueError("dataset must be one of: " + ", ".join(DATASET_CHOICES))
    if evidence not in {"gradcam", "ig"}:
        raise ValueError("evidence must be 'gradcam' or 'ig'")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"model_name must be one of: {', '.join(MODEL_CHOICES)}")
    if ig_steps <= 0:
        raise ValueError("ig_steps must be > 0")
    if lambda_disjoint < 0:
        raise ValueError("lambda_disjoint must be >= 0")
    if lambda_mass < 0:
        raise ValueError("lambda_mass must be >= 0")
    evidence_kind = evidence

    data_root_path = Path(data_root) if data_root is not None else (REPO / "data")
    refuse_final_if_dirty(final)
    run_config = {
        "dataset": dataset,
        "evidence": evidence,
        "model_name": model_name,
        "split": split,
        "seed": seed,
        "num_images": num_images,
        "lambda_disjoint": lambda_disjoint,
        "lambda_mass": lambda_mass,
        "lambda_shared_sparse": lambda_shared_sparse,
        "game_mode": game_mode,
        "num_steps": num_steps,
        "lr": lr,
    }
    run_hash = config_hash(run_config)
    device = get_device()
    use_torchvision_backbone = model_name != "smallcnn"
    input_size = TV_INPUT_SIZE if use_torchvision_backbone else None
    if not use_torchvision_backbone:
        grid_h, grid_w = 4, 4
    elif model_name == "vit_b_16":
        grid_h, grid_w = 14, 14
    else:
        grid_h, grid_w = 7, 7
    unit_space = VisionGridUnitSpace(grid_h, grid_w)
    try:
        loader, num_classes = _get_eval_loader(
            dataset, batch_size, data_root_path, image_size=input_size,
            seed=seed, num_images=num_images, split=split, final=final, config_hash=run_hash,
        )
    except (SplitLockedError, MissingSplitError, RuntimeError):
        raise
    except Exception as e:
        print("Dataset load failed for", dataset, "using random fallback:", e)
        loader, num_classes = _fallback_random_loader(dataset, batch_size, num_images or 200, image_size=input_size)

    model = _build_model(model_name, num_classes, pretrained=pretrained,
                         checkpoint=checkpoint).to(device).eval()
    selector = TopMSelector(m=min(5, num_classes))
    if evidence_kind == "gradcam":
        provider = GradCAMRegionsProvider(grid_h, grid_w)
    else:
        provider = IntegratedGradientsRegionsProvider(grid_h, grid_w, steps=ig_steps, baseline="zero")

    # Game preset drives use_shared / margin / overlap / partition. Defaulting to
    # None keeps this script's historical behaviour (no shared mask, no partition
    # term); passing --game_mode mixed makes it match eval_localization.py so the
    # separation and validation results describe one configuration rather than two.
    game_cfg = resolve_contrastive_game(game_mode) if game_mode else None
    objective = ContrastiveObjective(
        lambda_suff=1.0,
        lambda_margin=game_cfg.lambda_margin if game_cfg else 1.0,
        lambda_sparse=0.05,
        lambda_overlap=game_cfg.lambda_overlap if game_cfg else 0.2,
        lambda_mass=lambda_mass,
        lambda_shared_sparse=lambda_shared_sparse,
    )
    opt_allocator = OptimizationAllocator(
        objective,
        num_steps=num_steps,
        lr=lr,
        use_shared=game_cfg.use_shared if game_cfg else False,
        lambda_disjoint=game_cfg.lambda_disjoint if game_cfg else lambda_disjoint,
        lambda_partition=game_cfg.lambda_partition if game_cfg else 0.0,
    )
    base_allocator = EvidenceAsMaskAllocator()

    try:
        n_batches_hint: int | None = len(loader)
    except (TypeError, NotImplementedError):
        n_batches_hint = None
    if num_images is not None:
        n_batches_hint = max(1, math.ceil(num_images / max(batch_size, 1)))

    use_tqdm = tqdm_auto is not None
    if use_tqdm:
        batch_iter = tqdm_auto(
            loader,
            total=n_batches_hint,
            desc=f"ablation {dataset}/{evidence_kind}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
    else:
        batch_iter = loader
        print("  [ablation] tqdm not installed; `pip install tqdm` for a progress bar.", flush=True)

    results: Dict[str, List[Dict[str, float]]] = {"base_evidence": [], "naive_contrastive": [], "optimized": []}
    count = 0
    for x, _ in batch_iter:
        if num_images is not None and count >= num_images:
            break
        x = x.to(device)
        if num_images is not None:
            x = x[: min(batch_size, num_images - count)]
        count += x.shape[0]
        with torch.no_grad():
            logits = model(x)
            hypotheses = selector.select(logits, torch.softmax(logits, dim=-1))
        # Grad-CAM needs a forward/backward through the model; use input that requires grad
        x_grad = x.detach().clone().requires_grad_(True)
        evidence_map = provider.explain(x_grad, model, hypotheses)
        evidence_map = evidence_map.detach() / (evidence_map.sum(dim=-1, keepdim=True).clamp_min(1e-8))

        # 1) Base evidence alone
        m_base = evidence_map
        metrics_base = compute_metrics(x, model, unit_space, hypotheses, m_base)
        results["base_evidence"].append(metrics_base)

        # 2) Naïve contrastive: E_k - mean(E_foils); E_foil[b,k] = mean over j!=k of evidence[b,j]
        B, K, R = evidence_map.shape
        valid = hypotheses.mask.float()  # (B, K)
        eye = torch.eye(K, device=evidence_map.device, dtype=evidence_map.dtype).unsqueeze(0)  # (1, K, K)
        mask_other = (1 - eye) * valid.unsqueeze(2)  # (B, K, K): [b,k,j]=1 if j!=k and valid[b,j]
        E_foil = (evidence_map.unsqueeze(1) * mask_other.unsqueeze(-1)).sum(dim=2) / (mask_other.sum(dim=2, keepdim=True).clamp_min(1e-8))
        m_naive = (evidence_map - E_foil).clamp(0.0, None)
        m_naive = m_naive / (m_naive.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        metrics_naive = compute_metrics(x, model, unit_space, hypotheses, m_naive)
        results["naive_contrastive"].append(metrics_naive)

        # 3) Optimized masks
        masks_opt = opt_allocator.allocate(
            x=x,
            model=model,
            unit_space=unit_space,
            hypotheses=hypotheses,
            evidence=evidence_map,
        )
        metrics_opt = compute_metrics(x, model, unit_space, hypotheses, masks_opt["unique"], masks_opt.get("shared"))
        results["optimized"].append(metrics_opt)

        if use_tqdm:
            batch_iter.set_postfix_str(f"{count} imgs", refresh=False)

    def agg(name: str) -> Dict[str, float]:
        L = results[name]
        if not L:
            return {}
        return {k: sum(d[k] for d in L) / len(L) for k in L[0].keys()}

    base_agg = agg("base_evidence")
    naive_agg = agg("naive_contrastive")
    opt_agg = agg("optimized")

    print("--- Ablation (mean over batches) ---")
    print(
        "dataset=%s evidence=%s model=%s pretrained=%s ig_steps=%d lambda_disjoint=%.3f lambda_mass=%.3f num_images=%s batch_size=%d"
        % (
            dataset,
            evidence_kind,
            model_name,
            str(bool(pretrained)).lower(),
            ig_steps,
            lambda_disjoint,
            lambda_mass,
            "all" if num_images is None else str(num_images),
            batch_size,
        )
    )
    print(f"{'method':<22} {'suff':>8} {'margin':>8} {'overlap':>8} {'sparse':>8}")
    print(f"{'base_evidence':<22} {base_agg.get('suff', 0):>8.4f} {base_agg.get('margin', 0):>8.4f} {base_agg.get('overlap', 0):>8.4f} {base_agg.get('sparse', 0):>8.4f}")
    print(f"{'naive_contrastive':<22} {naive_agg.get('suff', 0):>8.4f} {naive_agg.get('margin', 0):>8.4f} {naive_agg.get('overlap', 0):>8.4f} {naive_agg.get('sparse', 0):>8.4f}")
    print(f"{'optimized':<22} {opt_agg.get('suff', 0):>8.4f} {opt_agg.get('margin', 0):>8.4f} {opt_agg.get('overlap', 0):>8.4f} {opt_agg.get('sparse', 0):>8.4f}")

    # Quantitative improvement: overlap reduction, reported alongside what happened
    # to sufficiency and to the mass budget.
    if base_agg and opt_agg:
        overlap_red = base_agg["overlap"] - opt_agg["overlap"]
        suff_delta = opt_agg["suff"] - base_agg["suff"]
        sparse_ratio = opt_agg.get("sparse", 0.0) / max(base_agg.get("sparse", 0.0), 1e-8)
        print(f"\nOverlap reduction (optimized vs base): {overlap_red:.4f} "
              f"({base_agg['overlap']:.4f} -> {opt_agg['overlap']:.4f})")
        # An overlap drop is only meaningful if sufficiency did not fall and the mask
        # budget did not grow — otherwise the masks got cleaner by getting weaker or
        # by simply spending more highlight. Report both rather than asserting
        # "comparable sufficiency", which is false whenever the delta is large.
        print(f"Sufficiency change: {suff_delta:+.4f} "
              f"({base_agg['suff']:.4f} -> {opt_agg['suff']:.4f})")
        print(f"Mask budget ratio (optimized/base sparse): {sparse_ratio:.4f} "
              f"(~1.0 means the budget was held)")
        verdict = "PASS" if (suff_delta >= 0 and sparse_ratio <= 1.05) else "CHECK"
        print(f"\n--- {verdict}: overlap fell by {overlap_red:.4f} with sufficiency "
              f"{suff_delta:+.4f} at {sparse_ratio:.2f}x budget ---")

    out_dir = Path(out_dir) if out_dir is not None else (REPO / "scripts" / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    if export_prefix is None:
        if dataset == "cifar10" and evidence_kind == "gradcam":
            export_prefix = "ablation_contrastive"
        else:
            export_prefix = f"ablation_contrastive_{dataset}_{evidence_kind}"
    summary_json = out_dir / f"{export_prefix}_metrics.json"
    summary_csv = out_dir / f"{export_prefix}_metrics.csv"
    per_batch_csv = out_dir / f"{export_prefix}_per_batch.csv"

    summary = {
        "run_name": export_prefix,
        "dataset": dataset,
        "evidence": evidence_kind,
        "model": model_name,
        "pretrained": bool(pretrained),
        "input_size": int(input_size) if input_size is not None else None,
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "ig_steps": int(ig_steps) if evidence_kind == "ig" else None,
        "lambda_disjoint": float(lambda_disjoint),
        "lambda_mass": float(lambda_mass),
        "lambda_shared_sparse": float(lambda_shared_sparse),
        "num_images": count,
        "batch_size": int(batch_size),
        # Recorded so a run can be checked against eval_localization.py's config
        # rather than assumed to match it.
        "game_mode": game_mode,
        "use_shared": bool(opt_allocator.use_shared),
        "lambda_margin": float(objective.lambda_margin),
        "lambda_overlap": float(objective.lambda_overlap),
        "lambda_partition": float(opt_allocator.lambda_partition),
        "num_steps": int(num_steps),
        "lr": float(lr),
        "aggregates": {
            "base_evidence": base_agg,
            "naive_contrastive": naive_agg,
            "optimized": opt_agg,
        },
    }
    save_json(summary_json, summary, config_hash=run_hash, device=device)

    agg_rows = []
    for method, agg in [("base_evidence", base_agg), ("naive_contrastive", naive_agg), ("optimized", opt_agg)]:
        row = {"method": method}
        row.update(agg)
        agg_rows.append(row)
    save_rows_csv(summary_csv, agg_rows)

    per_batch_rows = []
    for method, metrics_list in results.items():
        for idx, metrics in enumerate(metrics_list):
            row = {"method": method, "batch_idx": idx}
            row.update(metrics)
            per_batch_rows.append(row)
    save_rows_csv(per_batch_csv, per_batch_rows)

    print("Saved metrics summary to", summary_json)
    print("Saved aggregate metrics table to", summary_csv)
    print("Saved per-batch metrics table to", per_batch_csv)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CDEA contrastive ablation")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=DATASET_CHOICES)
    parser.add_argument("--evidence", type=str, default="gradcam", choices=["gradcam", "ig"])
    parser.add_argument("--model", dest="model_name", type=str, default="resnet18", choices=MODEL_CHOICES)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the eval subset. Does not change the train/val/test split.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="val is the only split used for decisions. test requires --final.")
    parser.add_argument("--final", action="store_true",
                        help="Allow the test split. Refuses a dirty tree and an unfrozen eval plan.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ig_steps", type=int, default=8)
    parser.add_argument("--lambda_disjoint", type=float, default=0.0,
                        help="Retired. Must be 0. Overlap has one weight, lambda_overlap.")
    parser.add_argument("--lambda_mass", type=float, default=2.0)
    parser.add_argument("--lambda_shared_sparse", type=float, default=0.25,
                        help="L1 penalty on the shared mask. At 0.0 it is in no penalty "
                             "term at all and inflates to blanket ~46%% of the grid. Note "
                             "that sufficiency is measured on keep(x, unique + shared), so "
                             "a blanket inflates it while the reported `sparse` budget "
                             "counts unique mass only. See docs/MEDICAL_RESULTS.md 9a.")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--export_prefix", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Trained weights to load; without this the model is untrained "
                             "and the evidence (and so the ablation) is uninformative")
    parser.add_argument("--game_mode", type=str, default=None,
                        choices=["cooperative", "mixed", "competitive"],
                        help="Contrastive game preset. Omit for this script's historical "
                             "config (no shared mask); use 'mixed' to match eval_localization.py")
    parser.add_argument("--num_steps", type=int, default=40,
                        help="Allocator optimization steps (eval_localization.py uses 50)")
    parser.add_argument("--lr", type=float, default=0.3,
                        help="Allocator learning rate (eval_localization.py uses 0.2)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory for result CSV/JSON (default: scripts/out)")
    args = parser.parse_args()

    run_ablation(
        checkpoint=args.checkpoint,
        num_images=args.num_images,
        batch_size=args.batch_size,
        dataset=args.dataset,
        evidence=args.evidence,
        model_name=args.model_name,
        pretrained=bool(args.pretrained),
        ig_steps=args.ig_steps,
        lambda_disjoint=args.lambda_disjoint,
        lambda_mass=args.lambda_mass,
        lambda_shared_sparse=args.lambda_shared_sparse,
        data_root=args.data_root,
        export_prefix=args.export_prefix,
        game_mode=args.game_mode,
        num_steps=args.num_steps,
        lr=args.lr,
        out_dir=args.out_dir,
        seed=args.seed,
        split=args.split,
        final=bool(args.final),
    )
