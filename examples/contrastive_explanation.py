"""
Contrastive explanation example.

Uses the full CDEA pipeline for contrastive explanations:
  - Select top-K hypotheses (classes) per sample
  - Compute evidence (Grad-CAM or Integrated Gradients pooled to grid regions)
  - Optimize unique masks per hypothesis (disjoint, sparse) with ContrastiveObjective
  - Return explanation with masks and metrics

Visualization: for one sample, shows input image, per-hypothesis mask heatmaps,
and "keep" views (what the model sees when retaining each hypothesis's regions).

Datasets (from data/): mnist, cifar10, pets (PetImages: Cat vs Dog), stanford_dogs (120 breeds).

Run from repo root:
  # Explain with Grad-CAM evidence (default):
  PYTHONPATH=. python examples/contrastive_explanation.py [--dataset mnist|cifar10|pets|stanford_dogs] [--model resnet18|...] [--pretrained]

  # Explain with Integrated Gradients evidence:
  PYTHONPATH=. python examples/contrastive_explanation.py --evidence ig --ig_steps 24 --ig_baseline zero

  # Train the model on the dataset, then run contrastive explanation (recommended for interpretable masks):
  PYTHONPATH=. python examples/contrastive_explanation.py --dataset cifar10 --train --epochs 10 [--checkpoint path.pt]

  # Explain using a previously trained checkpoint:
  PYTHONPATH=. python examples/contrastive_explanation.py --dataset cifar10 --checkpoint examples/out/checkpoints/cifar10_resnet18.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add repo root so core, modality, base_evidence, instantiations are importable
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.runner import CDEAExplainer
from core.types import Explanation
from core.hypotheses import TopMSelector
from core.device import get_device
from core.interaction import get_interaction
from core.game_modes import contrastive_game_modes, resolve_contrastive_game
from core.reporting import extract_metric_shapes, extract_scalar_metrics, save_json, save_rows_csv
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator

DATA_ROOT = REPO / "data"
STANFORD_DOGS_IMAGES_ROOT = DATA_ROOT / "stanford_dogs" / "images" / "Images"

# Default input size for torchvision models (e.g. ResNet expects 224x224 for ImageNet-style)
TV_INPUT_SIZE = 224


def _format_stanford_dogs_label(name: str) -> str:
    """Convert folder names like 'n02085620-Chihuahua' to a readable label."""
    if "-" in name:
        name = name.split("-", 1)[1]
    return name.replace("_", " ")


def get_torchvision_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a torchvision model with the correct number of output classes."""
    try:
        from torchvision import models
    except ImportError:
        raise ImportError("torchvision required. Install with: pip install torchvision")

    if name == "resnet18":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet34":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif name == "efficientnet_b0":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("model must be one of: resnet18, resnet34, mobilenet_v2, efficientnet_b0")
    return model


def _load_batch(
    dataset_name: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, list[str], int]:
    x, _y, class_names, num_classes = _load_batch_with_labels(dataset_name, batch_size, device)
    return x, class_names, num_classes


def _load_batch_with_labels(
    dataset_name: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, list[str], int]:
    """
    Load one batch from the chosen dataset in data/.
    Returns (x, y, class_names, num_classes).
    """
    try:
        from torchvision import transforms
        from torchvision.datasets import MNIST, CIFAR10, ImageFolder
    except ImportError:
        raise ImportError("torchvision required for dataset loading. Install with: pip install torchvision")

    if dataset_name == "mnist":
        t = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.repeat(3, 1, 1))])
        ds = MNIST(root=str(DATA_ROOT), train=False, download=False, transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        class_names = [str(i) for i in range(10)]
        num_classes = 10
    elif dataset_name == "cifar10":
        t = transforms.ToTensor()
        ds = CIFAR10(root=str(DATA_ROOT), train=False, download=False, transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        num_classes = 10
    elif dataset_name == "pets":
        t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        ds = ImageFolder(root=str(DATA_ROOT / "PetImages"), transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        class_names = ds.classes
        num_classes = len(class_names)
    elif dataset_name == "stanford_dogs":
        t = transforms.Compose([
            transforms.Resize((TV_INPUT_SIZE, TV_INPUT_SIZE)),
            transforms.ToTensor(),
        ])
        ds = ImageFolder(root=str(STANFORD_DOGS_IMAGES_ROOT), transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        class_names = [_format_stanford_dogs_label(c) for c in ds.classes]
        num_classes = len(class_names)
    else:
        raise ValueError("dataset must be one of: mnist, cifar10, pets, stanford_dogs")

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    return x, y, class_names, num_classes


def _get_dataloaders(
    dataset_name: str,
    batch_size: int,
    tv_size: int = TV_INPUT_SIZE,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, list[str], int]:
    """
    Train and validation dataloaders for the chosen dataset.
    Images are resized to tv_size x tv_size (e.g. 224 for torchvision models).
    Returns (train_loader, val_loader, class_names, num_classes).
    """
    try:
        from torchvision import transforms
        from torchvision.datasets import MNIST, CIFAR10, ImageFolder
    except ImportError:
        raise ImportError("torchvision required for dataset loading. Install with: pip install torchvision")

    resize = transforms.Resize((tv_size, tv_size))
    if dataset_name == "mnist":
        t_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
            resize,
        ])
        t_val = t_train
        train_ds = MNIST(root=str(DATA_ROOT), train=True, download=False, transform=t_train)
        val_ds = MNIST(root=str(DATA_ROOT), train=False, download=False, transform=t_val)
        class_names = [str(i) for i in range(10)]
        num_classes = 10
    elif dataset_name == "cifar10":
        t_train = transforms.Compose([
            transforms.ToTensor(),
            resize,
        ])
        t_val = t_train
        train_ds = CIFAR10(root=str(DATA_ROOT), train=True, download=False, transform=t_train)
        val_ds = CIFAR10(root=str(DATA_ROOT), train=False, download=False, transform=t_val)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        num_classes = 10
    elif dataset_name == "pets":
        t_train = transforms.Compose([
            transforms.Resize((tv_size, tv_size)),
            transforms.ToTensor(),
        ])
        t_val = t_train
        root = str(DATA_ROOT / "PetImages")
        full_ds = ImageFolder(root=root, transform=t_train)
        class_names = full_ds.classes
        num_classes = len(class_names)
        n = len(full_ds)
        n_val = max(1, n // 5)
        n_train = n - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    elif dataset_name == "stanford_dogs":
        t_train = transforms.Compose([
            transforms.Resize((tv_size, tv_size)),
            transforms.ToTensor(),
        ])
        t_val = t_train
        full_ds = ImageFolder(root=str(STANFORD_DOGS_IMAGES_ROOT), transform=t_train)
        class_names = [_format_stanford_dogs_label(c) for c in full_ds.classes]
        num_classes = len(class_names)
        n = len(full_ds)
        n_val = max(1, n // 5)
        n_train = n - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    else:
        raise ValueError("dataset must be one of: mnist, cifar10, pets, stanford_dogs")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_names, num_classes


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
) -> Tuple[nn.Module, float]:
    """
    Train model with cross-entropy; report train loss and val accuracy each epoch.
    Returns (model with best val accuracy weights, best_val_acc).
    """
    model = model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print("Epoch %d/%d  train_loss=%.4f  val_acc=%.4f" % (epoch + 1, epochs, train_loss, val_acc))

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device).eval()
    return model, best_val_acc


def save_checkpoint(
    path: Path,
    model: nn.Module,
    dataset_name: str,
    model_name: str,
    num_classes: int,
) -> None:
    """Save checkpoint for later loading (state_dict + metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "dataset": dataset_name,
        "model_name": model_name,
        "num_classes": num_classes,
    }, path)
    print("Saved checkpoint to", path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[nn.Module, str, list[str], int]:
    """
    Load checkpoint; build model from metadata and load state_dict.
    Returns (model, dataset_name, class_names, num_classes).
    """
    path = Path(path)
    ck = torch.load(path, map_location=device, weights_only=True)
    state_dict = ck["state_dict"]
    dataset_name = ck["dataset"]
    model_name = ck["model_name"]
    num_classes = ck["num_classes"]
    model = get_torchvision_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    _, _, class_names, _ = _get_dataloaders(dataset_name, batch_size=1, tv_size=TV_INPUT_SIZE)
    return model, dataset_name, class_names, num_classes


def _mask_to_image(m: torch.Tensor, grid_h: int, grid_w: int, h: int, w: int) -> torch.Tensor:
    """Upsample region mask (R,) or (K, R) to (H, W) or (K, H, W) for display."""
    if m.dim() == 1:
        m = m.unsqueeze(0)
    # m (K, R) -> (K, 1, grid_h, grid_w)
    R = grid_h * grid_w
    pm = m.view(-1, 1, grid_h, grid_w)
    pm = F.interpolate(pm, size=(h, w), mode="bilinear", align_corners=False)
    return pm.squeeze(1)


def visualize_contrastive(
    sample_idx: int,
    x: torch.Tensor,
    explanation: Explanation,
    unit_space: VisionGridUnitSpace,
    class_names: list[str] | None = None,
    y_true: torch.Tensor | None = None,
    evidence: torch.Tensor | None = None,
    probs: torch.Tensor | None = None,
    max_viz_classes: int = 3,
    out_path: Path | None = None,
) -> None:
    """
    Compact contrastive explanation figure:
    Row 0: Input image + label summary (true class, predicted class, top classes shown)
    Row 1: Evidence maps for top classes
    Row 2: Contrastive overlays (unique/shared) for top classes
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
    except ImportError:
        print("Matplotlib required for visualization. Install with: pip install matplotlib")
        return

    x = x.detach().cpu()
    B, C, H, W = x.shape
    if sample_idx >= B:
        sample_idx = 0
    xi = x[sample_idx : sample_idx + 1]
    m_unique = explanation.masks["unique"][sample_idx].detach().cpu()
    m_shared = explanation.masks.get("shared", None)
    m_shared_i = m_shared[sample_idx].detach().cpu() if m_shared is not None else None
    ids = explanation.hypotheses.ids[sample_idx].detach().cpu()
    K = m_unique.shape[0]
    show_k = K if max_viz_classes <= 0 else min(K, max_viz_classes)
    grid_h, grid_w = unit_space.grid_h, unit_space.grid_w

    if class_names is None:
        class_names = [str(i) for i in range(int(ids.max().item()) + 1)]
    id_vals = [int(v.item()) for v in ids]
    labels = [class_names[c] if 0 <= c < len(class_names) else str(c) for c in id_vals]
    ids_show = id_vals[:show_k]
    labels_show = labels[:show_k]

    n_cols = max(2, show_k)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.0 * n_cols, 6.2))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for ax in axes.flat:
        ax.set_axis_off()

    img = xi[0].permute(1, 2, 0).clamp(0, 1).numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # Row 0: input image + concise label summary
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Input image", fontsize=11)
    axes[0, 0].set_axis_on()

    pred_idx = ids_show[0] if ids_show else int(id_vals[0])
    pred_prob = None
    top_lines = []
    if probs is not None:
        p = probs[sample_idx].detach().cpu()
        pred_idx = int(p.argmax().item())
        pred_prob = float(p[pred_idx].item())
        for rank, cls_id in enumerate(ids_show):
            lbl = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            top_lines.append(f"{rank + 1}. {lbl} ({float(p[cls_id].item()):.3f})")
    else:
        for rank, cls_id in enumerate(ids_show):
            lbl = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            top_lines.append(f"{rank + 1}. {lbl}")

    true_idx = None
    if isinstance(y_true, torch.Tensor):
        yt = y_true[sample_idx].detach().cpu()
        true_idx = int(yt.item()) if yt.numel() > 0 else None
    true_lbl = class_names[true_idx] if isinstance(true_idx, int) and 0 <= true_idx < len(class_names) else (
        str(true_idx) if true_idx is not None else "N/A"
    )
    pred_lbl = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
    pred_text = f"{pred_lbl} ({pred_prob:.3f})" if pred_prob is not None else pred_lbl

    summary_lines = [
        f"True class: {true_lbl}",
        f"Predicted: {pred_text}",
        f"Top-{show_k} shown:",
    ] + top_lines
    axes[0, 1].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=axes[0, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        wrap=True,
    )
    axes[0, 1].set_title("Label summary", fontsize=11)
    axes[0, 1].set_axis_off()
    for c in range(2, n_cols):
        axes[0, c].set_axis_off()

    # Row 1: evidence maps for shown classes
    ev = evidence[sample_idx].detach().cpu() if evidence is not None else None
    for k in range(show_k):
        ax = axes[1, k]
        ax.imshow(img)
        if ev is not None:
            ev_k = ev[k]
            heat_ev = _mask_to_image(ev_k, grid_h, grid_w, H, W).numpy()
            if heat_ev.ndim == 3:
                heat_ev = heat_ev[0]
            h_min, h_max = float(heat_ev.min()), float(heat_ev.max())
            if h_max > h_min + 1e-8:
                heat_ev = np.clip((heat_ev - h_min) / (h_max - h_min + 1e-8), 0, 1) ** 0.5
            ax.imshow(heat_ev, cmap="hot", alpha=0.45, vmin=0, vmax=1)
        ax.set_title("Evidence: " + labels_show[k], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2: contrastive overlays for shown classes
    masks_upsampled = []
    for k in range(show_k):
        mk = _mask_to_image(m_unique[k], grid_h, grid_w, H, W).numpy()
        if mk.ndim == 3:
            mk = mk[0]
        masks_upsampled.append(mk)
    masks_upsampled = np.stack(masks_upsampled, axis=0)

    shared_explicit_up = None
    if m_shared_i is not None:
        shared_explicit_up = _mask_to_image(m_shared_i, grid_h, grid_w, H, W).numpy()
        if shared_explicit_up.ndim == 3:
            shared_explicit_up = shared_explicit_up[0]
    # Per-panel scale for unique so small differences are visible; shared on common scale
    all_shared = np.max(masks_upsampled, axis=0)
    if shared_explicit_up is not None:
        all_shared = np.maximum(all_shared, shared_explicit_up)
    s_global = float(np.max(all_shared)) + 1e-8
    for k in range(show_k):
        ax = axes[2, k]
        m0 = masks_upsampled[k]
        if shared_explicit_up is not None:
            unique_k = np.clip(m0, 0, 1).astype(np.float32)
            shared_k = np.clip(shared_explicit_up, 0, 1).astype(np.float32)
        else:
            others = np.delete(masks_upsampled, k, axis=0)
            m_others = np.max(others, axis=0) if others.size > 0 else np.zeros_like(m0)
            unique_k = np.clip(m0 - m_others, 0, 1).astype(np.float32)
            shared_k = np.minimum(m0, np.maximum(m_others, 0)).astype(np.float32)
        u_max = float(np.max(unique_k)) + 1e-8
        unique_norm = (np.clip(unique_k / u_max, 0, 1) ** 0.5).astype(np.float32)
        shared_norm = (np.clip(shared_k / s_global, 0, 1) ** 0.5).astype(np.float32)
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[:, :, 0] = shared_norm
        overlay[:, :, 1] = unique_norm
        overlay[:, :, 2] = 0
        overlay[:, :, 3] = 0.5 * (unique_norm + shared_norm)
        ax.imshow(img)
        ax.imshow(overlay)
        title = "Contrastive: " + labels_show[k]
        if shared_explicit_up is not None:
            title += " (explicit shared)"
        if u_max < 1e-6:
            title += " (≈shared only)"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for c in range(show_k, n_cols):
        axes[1, c].set_axis_off()
        axes[2, c].set_axis_off()

    legend_handles = [
        Patch(facecolor="#ff4500", alpha=0.55, label="Evidence heatmap (brighter = higher)"),
        Patch(facecolor=(0.0, 1.0, 0.0, 0.55), label="Unique evidence (green)"),
        Patch(facecolor=(1.0, 0.0, 0.0, 0.55), label="Shared evidence (red)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    plt.tight_layout(rect=[0.0, 0.07, 1.0, 1.0])
    if out_path is None:
        out_path = REPO / "examples" / "out" / "contrastive_explanation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved visualization to", out_path)


def save_contrastive_reports(
    explanation: Explanation,
    class_names: list[str],
    dataset: str,
    out_dir: Path,
    run_name: str,
    config: Dict[str, Any],
) -> None:
    """Save compact scalar summary + per-hypothesis split table for downstream analysis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scalar_metrics = extract_scalar_metrics(explanation.metrics)
    metric_shapes = extract_metric_shapes(explanation.metrics)
    summary = {
        "run_name": run_name,
        "dataset": dataset,
        "batch_size": int(explanation.hypotheses.ids.shape[0]),
        "top_k": int(explanation.hypotheses.ids.shape[1]),
        "config": config,
        "scalar_metrics": scalar_metrics,
        "metric_tensor_shapes": metric_shapes,
    }
    summary_json = out_dir / f"{run_name}_metrics.json"
    summary_csv = out_dir / f"{run_name}_metrics.csv"
    save_json(summary_json, summary)

    row = {"run_name": run_name, "dataset": dataset}
    row.update(scalar_metrics)
    save_rows_csv(summary_csv, [row])

    split_rows = []
    h_ids = explanation.hypotheses.ids.detach().cpu()
    h_valid = explanation.hypotheses.mask.detach().cpu()
    split_shared = explanation.metrics.get("split_shared_only_probs_topm")
    split_plus = explanation.metrics.get("split_shared_plus_unique_probs_topm")
    h_weights = explanation.metrics.get("hypothesis_weights_topm")
    probs = explanation.extras.get("probs")
    if isinstance(split_shared, torch.Tensor) and isinstance(split_plus, torch.Tensor):
        split_shared = split_shared.detach().cpu()
        split_plus = split_plus.detach().cpu()
        if isinstance(h_weights, torch.Tensor):
            h_weights = h_weights.detach().cpu()
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu()
        B, K = h_ids.shape
        for b in range(B):
            for k in range(K):
                if not bool(h_valid[b, k].item()):
                    continue
                cls_id = int(h_ids[b, k].item())
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                p_shared = float(split_shared[b, k].item())
                p_plus = float(split_plus[b, k].item())
                p_full = float(probs[b, cls_id].item()) if isinstance(probs, torch.Tensor) else None
                row = {
                    "sample_idx": b,
                    "rank": k,
                    "class_id": cls_id,
                    "label": label,
                    "p_full": p_full,
                    "p_shared_only": p_shared,
                    "p_shared_plus_unique": p_plus,
                    "delta": p_plus - p_shared,
                }
                if isinstance(h_weights, torch.Tensor):
                    row["hypothesis_weight"] = float(h_weights[b, k].item())
                split_rows.append(row)
    split_csv = out_dir / f"{run_name}_split.csv"
    save_rows_csv(split_csv, split_rows)

    pairwise_rows = []
    pair_shared_log = explanation.metrics.get("pairwise_margin_shared_only_logits_topm")
    pair_plus_log = explanation.metrics.get("pairwise_margin_shared_plus_unique_logits_topm")
    pair_delta_log = explanation.metrics.get("pairwise_margin_delta_logits_topm")
    pair_shared_prob = explanation.metrics.get("pairwise_margin_shared_only_probs_topm")
    pair_plus_prob = explanation.metrics.get("pairwise_margin_shared_plus_unique_probs_topm")
    pair_delta_prob = explanation.metrics.get("pairwise_margin_delta_probs_topm")
    if (
        isinstance(pair_shared_log, torch.Tensor)
        and isinstance(pair_plus_log, torch.Tensor)
        and isinstance(pair_delta_log, torch.Tensor)
        and isinstance(pair_shared_prob, torch.Tensor)
        and isinstance(pair_plus_prob, torch.Tensor)
        and isinstance(pair_delta_prob, torch.Tensor)
    ):
        pair_shared_log = pair_shared_log.detach().cpu()
        pair_plus_log = pair_plus_log.detach().cpu()
        pair_delta_log = pair_delta_log.detach().cpu()
        pair_shared_prob = pair_shared_prob.detach().cpu()
        pair_plus_prob = pair_plus_prob.detach().cpu()
        pair_delta_prob = pair_delta_prob.detach().cpu()
        B, K = h_ids.shape
        for b in range(B):
            for k in range(K):
                if not bool(h_valid[b, k].item()):
                    continue
                cls_k = int(h_ids[b, k].item())
                lbl_k = class_names[cls_k] if cls_k < len(class_names) else str(cls_k)
                for l in range(K):
                    if k == l or not bool(h_valid[b, l].item()):
                        continue
                    cls_l = int(h_ids[b, l].item())
                    lbl_l = class_names[cls_l] if cls_l < len(class_names) else str(cls_l)
                    pairwise_rows.append(
                        {
                            "sample_idx": b,
                            "k_rank": k,
                            "l_rank": l,
                            "k_class_id": cls_k,
                            "k_label": lbl_k,
                            "l_class_id": cls_l,
                            "l_label": lbl_l,
                            "shared_only_margin_logit": float(pair_shared_log[b, k, l].item()),
                            "shared_plus_unique_margin_logit": float(pair_plus_log[b, k, l].item()),
                            "delta_margin_logit": float(pair_delta_log[b, k, l].item()),
                            "shared_only_margin_prob": float(pair_shared_prob[b, k, l].item()),
                            "shared_plus_unique_margin_prob": float(pair_plus_prob[b, k, l].item()),
                            "delta_margin_prob": float(pair_delta_prob[b, k, l].item()),
                        }
                    )
    pairwise_csv = out_dir / f"{run_name}_pairwise.csv"
    save_rows_csv(pairwise_csv, pairwise_rows)

    print("Saved metrics summary to", summary_json)
    print("Saved scalar metrics table to", summary_csv)
    print("Saved split table to", split_csv)
    print("Saved pairwise table to", pairwise_csv)


def main():
    parser = argparse.ArgumentParser(description="Contrastive explanation example")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "pets", "stanford_dogs"],
        default="cifar10",
        help="Dataset in data/: mnist, cifar10, pets (PetImages), or stanford_dogs",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_viz_classes", type=int, default=3,
                        help="Maximum top classes to display in the figure (reduces visual clutter)")
    parser.add_argument("--num_alloc_steps", type=int, default=25, help="Allocation optimization steps (fewer preserves evidence structure with untrained model)")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0"],
                        help="Torchvision model name")
    parser.add_argument("--interaction", type=str, default="none", choices=["none", "attention", "transformer"],
                        help="Hypothesis interaction module over top-K tokens")
    parser.add_argument("--interaction_dim", type=int, default=64,
                        help="Token embedding dimension for interaction (used when --interaction != none)")
    parser.add_argument("--interaction_heads", type=int, default=4,
                        help="Attention heads for interaction")
    parser.add_argument("--interaction_ffn", type=int, default=256,
                        help="Feed-forward width for transformer interaction")
    parser.add_argument("--interaction_dropout", type=float, default=0.0,
                        help="Dropout for interaction module")
    parser.add_argument(
        "--game_mode",
        type=str,
        choices=list(contrastive_game_modes()),
        default="mixed",
        help="Contrastive game mode: cooperative, competitive, mixed, or manual (explicit lambdas)",
    )
    parser.add_argument(
        "--interaction_attn_mix",
        type=float,
        default=0.35,
        help="Blend ratio in [0,1] for attention-conditioned evidence init in allocator",
    )
    parser.add_argument(
        "--interaction_weight_blend",
        type=float,
        default=0.5,
        help="Blend ratio in [0,1] for attention-conditioned hypothesis weighting in objective",
    )
    parser.add_argument(
        "--use_shared",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an explicit shared mask in contrastive allocation",
    )
    parser.add_argument(
        "--lambda_partition",
        type=float,
        default=0.1,
        help="Partition penalty weight when shared mask is enabled",
    )
    parser.add_argument(
        "--lambda_margin",
        type=float,
        default=1.0,
        help="Contrastive margin weight (used when --game_mode manual)",
    )
    parser.add_argument(
        "--lambda_overlap",
        type=float,
        default=0.2,
        help="Unique-mask overlap penalty (used when --game_mode manual)",
    )
    parser.add_argument(
        "--lambda_disjoint",
        type=float,
        default=0.1,
        help="Allocator disjointness penalty (used when --game_mode manual)",
    )
    parser.add_argument(
        "--lambda_mass",
        type=float,
        default=2.0,
        help="Penalty weight to keep optimized mask mass close to base evidence mass",
    )
    parser.add_argument("--evidence", type=str, choices=["gradcam", "ig"], default="gradcam",
                        help="Base evidence provider: gradcam (Grad-CAM) or ig (Integrated Gradients)")
    parser.add_argument("--ig_steps", type=int, default=24, help="Integrated Gradients interpolation steps (only used with --evidence ig)")
    parser.add_argument("--ig_baseline", type=str, default="zero", choices=["zero", "mean"],
                        help="Baseline for Integrated Gradients path integral (only used with --evidence ig)")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights (final layer still replaced for num_classes)")
    parser.add_argument("--train", action="store_true", help="Train the model on the dataset first; save checkpoint then run explanation")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs when --train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate when --train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save (when --train) or load model checkpoint; default when training: examples/out/checkpoints/<dataset>_<model>.pt")
    args = parser.parse_args()
    if args.interaction != "none":
        if args.interaction_dim <= 0:
            parser.error("--interaction_dim must be > 0 when --interaction is enabled")
        if args.interaction_heads <= 0:
            parser.error("--interaction_heads must be > 0 when --interaction is enabled")
        if args.interaction_dim % args.interaction_heads != 0:
            parser.error("--interaction_dim must be divisible by --interaction_heads")
        if args.interaction == "transformer" and args.interaction_ffn <= 0:
            parser.error("--interaction_ffn must be > 0 for --interaction transformer")
    if args.lambda_partition < 0:
        parser.error("--lambda_partition must be >= 0")
    if args.lambda_margin < 0:
        parser.error("--lambda_margin must be >= 0")
    if args.lambda_overlap < 0:
        parser.error("--lambda_overlap must be >= 0")
    if args.lambda_disjoint < 0:
        parser.error("--lambda_disjoint must be >= 0")
    if args.lambda_mass < 0:
        parser.error("--lambda_mass must be >= 0")
    if args.max_viz_classes <= 0:
        parser.error("--max_viz_classes must be > 0")
    if not (0.0 <= args.interaction_attn_mix <= 1.0):
        parser.error("--interaction_attn_mix must be in [0, 1]")
    if not (0.0 <= args.interaction_weight_blend <= 1.0):
        parser.error("--interaction_weight_blend must be in [0, 1]")
    if args.evidence == "ig" and args.ig_steps <= 0:
        parser.error("--ig_steps must be > 0")

    manual_flags = {"--use_shared", "--no-use_shared", "--lambda_margin", "--lambda_partition", "--lambda_overlap", "--lambda_disjoint"}
    used_manual_flags = sorted(f for f in manual_flags if f in sys.argv[1:])
    effective_game_mode = args.game_mode
    if args.game_mode != "manual" and used_manual_flags:
        effective_game_mode = "manual"
        print(
            "Detected explicit shared/lambda flags (%s); switching game_mode to manual."
            % ", ".join(used_manual_flags)
        )

    if effective_game_mode == "manual":
        game_cfg = resolve_contrastive_game(
            "manual",
            use_shared=bool(args.use_shared),
            lambda_margin=float(args.lambda_margin),
            lambda_overlap=float(args.lambda_overlap),
            lambda_disjoint=float(args.lambda_disjoint),
            lambda_partition=float(args.lambda_partition),
        )
    else:
        game_cfg = resolve_contrastive_game(effective_game_mode)

    device = get_device()
    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Model:", args.model, "(pretrained)" if args.pretrained else "")
    if args.evidence == "ig":
        print("Evidence: integrated_gradients (steps=%d baseline=%s)" % (args.ig_steps, args.ig_baseline))
    else:
        print("Evidence: gradcam")
    if args.interaction == "none":
        print("Interaction: none")
    else:
        print(
            "Interaction: %s (dim=%d heads=%d ffn=%d dropout=%.3f attn_mix=%.2f weight_blend=%.2f)"
            % (
                args.interaction,
                args.interaction_dim,
                args.interaction_heads,
                args.interaction_ffn,
                args.interaction_dropout,
                args.interaction_attn_mix,
                args.interaction_weight_blend,
            )
        )
    print(
        "Game mode: %s (shared=%s margin=%.3f overlap=%.3f disjoint=%.3f partition=%.3f)"
        % (
            game_cfg.mode,
            "on" if game_cfg.use_shared else "off",
            game_cfg.lambda_margin,
            game_cfg.lambda_overlap,
            game_cfg.lambda_disjoint,
            game_cfg.lambda_partition,
        )
    )

    # Resolve checkpoint path: default when training
    if args.checkpoint is None and args.train:
        args.checkpoint = str(REPO / "examples" / "out" / "checkpoints" / f"{args.dataset}_{args.model}.pt")
    elif args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)

    model = None
    class_names = None
    num_classes = None
    dataset_for_batch = args.dataset

    if args.train:
        # Train then use trained model for explanation
        print("Training...")
        train_loader, val_loader, class_names, num_classes = _get_dataloaders(
            args.dataset, args.batch_size, tv_size=TV_INPUT_SIZE
        )
        model = get_torchvision_model(args.model, num_classes, pretrained=args.pretrained)
        model, best_acc = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
        print("Best val accuracy: %.4f" % best_acc)
        save_checkpoint(Path(args.checkpoint), model, args.dataset, args.model, num_classes)
    elif args.checkpoint is not None:
        # Load trained model from checkpoint
        model, dataset_for_batch, class_names, num_classes = load_checkpoint(args.checkpoint, device)
        print("Loaded model from", args.checkpoint)
    else:
        # Fresh or pretrained model (no checkpoint)
        try:
            _, class_names, num_classes = _load_batch(args.dataset, args.batch_size, device)
        except Exception:
            if args.dataset == "pets":
                num_classes = 2
            elif args.dataset == "stanford_dogs":
                num_classes = 120
            else:
                num_classes = 10
            class_names = [str(i) for i in range(num_classes)]
        model = get_torchvision_model(args.model, num_classes, pretrained=args.pretrained)

    if model is None:
        raise RuntimeError("Model not set")
    model = model.to(device).eval()
    if class_names is None or num_classes is None:
        _, class_names, num_classes = _load_batch(dataset_for_batch, args.batch_size, device)

    # Load one batch for explanation (same dataset as model), including true labels
    try:
        x, y_true, class_names, num_classes = _load_batch_with_labels(dataset_for_batch, args.batch_size, device)
    except FileNotFoundError as e:
        print("Dataset not found in data/. Using random batch. Error:", e)
        if dataset_for_batch == "mnist":
            hw = 28
            num_classes = 10
        elif dataset_for_batch == "cifar10":
            hw = 32
            num_classes = 10
        elif dataset_for_batch == "pets":
            hw = 64
            num_classes = 2
        else:
            hw = TV_INPUT_SIZE
            num_classes = 120
        x = torch.rand(args.batch_size, 3, hw, hw, device=device)
        y_true = torch.randint(0, num_classes, (args.batch_size,), device=device)
        class_names = [str(i) for i in range(num_classes)]
    except Exception as e:
        print("Dataset load failed:", e)
        raise

    # Torchvision models expect 224x224; resize so evidence and masks align
    if x.shape[2] != TV_INPUT_SIZE or x.shape[3] != TV_INPUT_SIZE:
        x = F.interpolate(x, size=(TV_INPUT_SIZE, TV_INPUT_SIZE), mode="bilinear", align_corners=False)

    top_k = min(5, num_classes)
    # ResNet-style backbones output 7x7 for 224 input; use 7x7 grid so MPS adaptive_pool is valid
    grid_h, grid_w = 7, 7

    # Unit space (vision grid)
    embed_dim = args.interaction_dim if args.interaction != "none" else None
    unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur", embed_dim=embed_dim)
    # Hypothesis selector: top-K classes per sample
    selector = TopMSelector(m=top_k)
    # Evidence: Grad-CAM or Integrated Gradients pooled to grid regions (B, K, R)
    if args.evidence == "ig":
        base_evidence = IntegratedGradientsRegionsProvider(
            grid_h=grid_h,
            grid_w=grid_w,
            steps=args.ig_steps,
            baseline=args.ig_baseline,
        )
    else:
        base_evidence = GradCAMRegionsProvider(grid_h, grid_w)
    interaction = get_interaction(
        flag=args.interaction,
        d_model=(args.interaction_dim if args.interaction != "none" else None),
        num_heads=args.interaction_heads,
        dim_feedforward=args.interaction_ffn,
        dropout=args.interaction_dropout,
    )
    # Objective: sufficiency + contrastive margin - overlap - sparsity
    objective = ContrastiveObjective(
        lambda_suff=1.0,
        lambda_margin=game_cfg.lambda_margin,
        lambda_sparse=0.05,
        lambda_overlap=game_cfg.lambda_overlap,
        lambda_mass=args.lambda_mass,
        attn_weight_blend=args.interaction_weight_blend,
    )
    # Allocator: optimize unique masks from evidence. With an untrained model, gradients are
    # uninformative so strong optimization can flatten masks to uniform; use moderate steps
    # and penalty so evidence-driven structure is preserved. For interpretable masks, use a trained model.
    allocator = OptimizationAllocator(
        objective,
        num_steps=args.num_alloc_steps,
        lr=0.2,
        use_shared=game_cfg.use_shared,
        lambda_disjoint=game_cfg.lambda_disjoint,
        lambda_partition=game_cfg.lambda_partition,
        attn_mix=args.interaction_attn_mix,
    )

    explainer = CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=selector,
        base_evidence=base_evidence,
        allocator=allocator,
        objective=objective,
        interaction=interaction,
        normalize_evidence=True,
        device=device,
    )

    explanation = explainer.explain(x)

    print("\n--- Contrastive explanation ---")
    print("Hypotheses (top-%d class ids) shape:" % top_k, explanation.hypotheses.ids.shape)
    print("Unique masks shape:", explanation.masks["unique"].shape)
    if "shared" in explanation.masks:
        print("Shared mask shape:", explanation.masks["shared"].shape)
    probs_top = explanation.extras.get("probs")
    if isinstance(probs_top, torch.Tensor):
        pred_ids = probs_top.argmax(dim=-1).detach().cpu()
    else:
        pred_ids = explanation.hypotheses.ids[:, 0].detach().cpu()
    y0 = int(y_true[0].detach().cpu().item())
    p0 = int(pred_ids[0].item())
    y0_lbl = class_names[y0] if 0 <= y0 < len(class_names) else str(y0)
    p0_lbl = class_names[p0] if 0 <= p0 < len(class_names) else str(p0)
    print("Sample 0: true=%s (%d)  predicted=%s (%d)" % (y0_lbl, y0, p0_lbl, p0))
    print("Metrics: suff=%.4f margin=%.4f overlap=%.4f sparse=%.4f" % (
        explanation.metrics["suff"].item(),
        explanation.metrics["margin"].item(),
        explanation.metrics["overlap"].item(),
        explanation.metrics["sparse"].item(),
    ))
    split_shared = explanation.metrics.get("split_shared_only_probs_topm")
    split_plus = explanation.metrics.get("split_shared_plus_unique_probs_topm")
    if isinstance(split_shared, torch.Tensor) and isinstance(split_plus, torch.Tensor):
        ids0 = explanation.hypotheses.ids[0].detach().cpu()
        valid0 = explanation.hypotheses.mask[0].detach().cpu()
        split_shared0 = split_shared[0].detach().cpu()
        split_plus0 = split_plus[0].detach().cpu()
        print("Probability split report (sample 0, top-m):")
        for k in range(int(valid0.sum().item())):
            cls_id = int(ids0[k].item())
            lbl = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            p_sh = float(split_shared0[k].item())
            p_pl = float(split_plus0[k].item())
            print("  %s: shared-only=%.4f shared+unique=%.4f delta=%+.4f" % (lbl, p_sh, p_pl, p_pl - p_sh))
    m0 = explanation.masks["unique"][0]
    if m0.std().item() < 0.03:
        print("(Mask variance is low; explanations may look diffuse. Check checkpoint quality or adjust allocation settings.)")

    x_cpu = x.detach().cpu()
    ev_suffix = "_ig" if args.evidence == "ig" else ""
    out_path = REPO / "examples" / "out" / f"contrastive_explanation{ev_suffix}_{args.dataset}.png"
    evidence_for_viz = explanation.extras.get("evidence")
    probs_for_viz = explanation.extras.get("probs")
    visualize_contrastive(
        sample_idx=0,
        x=x_cpu,
        explanation=explanation,
        unit_space=unit_space,
        class_names=class_names,
        y_true=y_true.detach().cpu(),
        evidence=evidence_for_viz,
        probs=probs_for_viz,
        max_viz_classes=args.max_viz_classes,
        out_path=out_path,
    )
    save_contrastive_reports(
        explanation=explanation,
        class_names=class_names,
        dataset=args.dataset,
        out_dir=REPO / "examples" / "out",
        run_name=f"contrastive{ev_suffix}_{args.dataset}",
        config={
            "evidence": args.evidence,
            "dataset": args.dataset,
            "model": args.model,
            "pretrained": bool(args.pretrained),
            "batch_size": int(args.batch_size),
            "max_viz_classes": int(args.max_viz_classes),
            "num_alloc_steps": int(args.num_alloc_steps),
            "interaction": args.interaction,
            "interaction_dim": int(args.interaction_dim),
            "interaction_heads": int(args.interaction_heads),
            "interaction_ffn": int(args.interaction_ffn),
            "interaction_dropout": float(args.interaction_dropout),
            "interaction_attn_mix": float(args.interaction_attn_mix),
            "interaction_weight_blend": float(args.interaction_weight_blend),
            "game_mode": game_cfg.mode,
            "use_shared": bool(game_cfg.use_shared),
            "lambda_margin": float(game_cfg.lambda_margin),
            "lambda_overlap": float(game_cfg.lambda_overlap),
            "lambda_disjoint": float(game_cfg.lambda_disjoint),
            "lambda_partition": float(game_cfg.lambda_partition),
            "lambda_mass": float(args.lambda_mass),
            **({"ig_steps": int(args.ig_steps), "ig_baseline": args.ig_baseline} if args.evidence == "ig" else {}),
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
