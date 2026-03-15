"""
Evaluate robust vs shortcut on a biased setup (Colored MNIST).
Pass: robust mask tracks object evidence, shortcut mask tracks background cue (qualitative);
      ID-OOD gap (sho_gap) reported (quantitative).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.types import EnvBatch
from core.hypotheses import TopMSelector
from core.device import get_device
from core.game_modes import resolve_shift_game, shift_game_modes
from core.reporting import save_json, save_rows_csv
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from instantiations.shift.objective import RobustShortcutObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from instantiations.shift.biased_data import (
    ColoredMNIST, env_batch_colored_mnist, compute_id_ood_gap,
    build_biased_dataset, get_env_batch_fn, BIASED_DATASETS,
)


TV_MODELS = ["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0", "vit_b_16", "vit_b_32"]
TV_INPUT_SIZE = 224
TV_GRID_H, TV_GRID_W = 7, 7
VIT_B16_GRID_H, VIT_B16_GRID_W = 14, 14


class _SmallCNN(nn.Module):
    """Lightweight backbone — smoke tests only, not for real experiments."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


def _build_backbone(model_name: str, num_classes: int, pretrained: bool,
                    checkpoint: Optional[str] = None) -> nn.Module:
    from torchvision import models
    weights = "IMAGENET1K_V1" if pretrained else None
    if model_name == "resnet18":
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "resnet34":
        m = models.resnet34(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "vit_b_16":
        m = models.vit_b_16(weights=weights)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif model_name == "vit_b_32":
        m = models.vit_b_32(weights=weights)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"model_name must be one of: {TV_MODELS}")
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        m.load_state_dict(state)
        print(f"  [model] loaded checkpoint: {checkpoint}")
    return m


def _mask_to_image(m: torch.Tensor, grid_h: int, grid_w: int, h: int, w: int) -> torch.Tensor:
    if m.dim() == 1:
        m = m.unsqueeze(0)
    pm = m.view(-1, 1, grid_h, grid_w)
    pm = torch.nn.functional.interpolate(pm, size=(h, w), mode="bilinear", align_corners=False)
    return pm.squeeze(1)


def _save_mask_visualization(
    x: torch.Tensor,
    masks: dict[str, torch.Tensor],
    unit_space: VisionGridUnitSpace,
    out_path: Path,
    sample_idx: int = 0,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not found; skipping robust/shortcut visualization save.")
        return

    x = x.detach().cpu()
    m_rob = masks["robust"].detach().cpu()
    m_sho = masks["shortcut"].detach().cpu()
    B, C, H, W = x.shape
    if sample_idx >= B:
        sample_idx = 0
    img = x[sample_idx].permute(1, 2, 0).clamp(0, 1).numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    mk_rob = _mask_to_image(m_rob[sample_idx], unit_space.grid_h, unit_space.grid_w, H, W).numpy()
    mk_sho = _mask_to_image(m_sho[sample_idx], unit_space.grid_h, unit_space.grid_w, H, W).numpy()
    if mk_rob.ndim == 3:
        mk_rob = mk_rob[0]
    if mk_sho.ndim == 3:
        mk_sho = mk_sho[0]

    x_keep_rob = unit_space.keep(x[sample_idx : sample_idx + 1], m_rob[sample_idx : sample_idx + 1])[0]
    x_keep_sho = unit_space.keep(x[sample_idx : sample_idx + 1], m_sho[sample_idx : sample_idx + 1])[0]
    img_keep_rob = x_keep_rob.permute(1, 2, 0).clamp(0, 1).numpy()
    img_keep_sho = x_keep_sho.permute(1, 2, 0).clamp(0, 1).numpy()
    if img_keep_rob.shape[2] == 1:
        img_keep_rob = np.repeat(img_keep_rob, 3, axis=2)
    if img_keep_sho.shape[2] == 1:
        img_keep_sho = np.repeat(img_keep_sho, 3, axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax in axes.flat:
        ax.set_axis_off()

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(img_keep_rob)
    axes[0, 1].set_title("Keep robust")
    axes[0, 2].imshow(img_keep_sho)
    axes[0, 2].set_title("Keep shortcut")

    axes[1, 0].imshow(img)
    axes[1, 0].imshow(mk_rob, cmap="Greens", alpha=0.5, vmin=0, vmax=1)
    axes[1, 0].set_title("Robust mask")
    axes[1, 1].imshow(img)
    axes[1, 1].imshow(mk_sho, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title("Shortcut mask")
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[:, :, 1] = np.clip(mk_rob, 0, 1)
    overlay[:, :, 0] = np.clip(mk_sho, 0, 1)
    overlay[:, :, 3] = 0.5 * (overlay[:, :, 0] + overlay[:, :, 1])
    axes[1, 2].imshow(img)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Combined masks")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved visualization to", out_path)


def run_eval(
    num_images: Optional[int] = None,
    batch_size: int = 16,
    num_steps: int = 40,
    data_root: Optional[str] = None,
    export_prefix: str = "robust_shortcut",
    game_mode: str = "mixed",
    model_name: str = "resnet18",
    pretrained: bool = True,
    checkpoint: Optional[str] = None,
    dataset_name: str = "colored_mnist",
    lambda_mean: Optional[float] = None,
    lambda_var: Optional[float] = None,
    lambda_gap: Optional[float] = None,
    lambda_shortcut: Optional[float] = None,
    lambda_disjoint: Optional[float] = None,
    lambda_sparse: Optional[float] = None,
):
    if game_mode == "manual":
        game_cfg = resolve_shift_game(
            game_mode,
            lambda_mean=lambda_mean,
            lambda_var=lambda_var,
            lambda_gap=lambda_gap,
            lambda_shortcut=lambda_shortcut,
            lambda_disjoint=lambda_disjoint,
            lambda_sparse=lambda_sparse,
        )
    else:
        if any(v is not None for v in (lambda_mean, lambda_var, lambda_gap, lambda_shortcut, lambda_disjoint, lambda_sparse)):
            raise ValueError(
                "Preset game modes do not accept explicit lambda overrides. "
                "Use game_mode='manual' to set custom lambdas."
            )
        game_cfg = resolve_shift_game(game_mode)

    data_root = data_root or str(REPO / "data")
    device = get_device()

    use_tv = model_name in TV_MODELS
    if not use_tv:
        grid_h, grid_w = 4, 4
    elif model_name == "vit_b_16":
        grid_h, grid_w = VIT_B16_GRID_H, VIT_B16_GRID_W
    else:
        grid_h, grid_w = TV_GRID_H, TV_GRID_W
    input_size = TV_INPUT_SIZE if use_tv else None

    try:
        dataset = build_biased_dataset(dataset_name, root=data_root, train=False, download=True, correlation=0.95)
        env_batch_fn = get_env_batch_fn(dataset_name)
    except Exception as e:
        print(f"{dataset_name} failed (need torchvision): {e}")
        print("Falling back to synthetic biased data (color = label mod 3).")
        dataset = _synthetic_biased((num_images or 200) * 2)
        env_batch_fn = _env_synthetic
        if num_images is not None:
            num_images = min(num_images, len(dataset))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    unit_space = VisionGridUnitSpace(grid_h, grid_w)

    if use_tv:
        model = _build_backbone(model_name, num_classes=10, pretrained=pretrained,
                                checkpoint=checkpoint).to(device).eval()
        print(f"Model: {model_name} ({'pretrained ImageNet' if pretrained else 'random init'})")
    else:
        model = _SmallCNN(num_classes=10).to(device).eval()
        print("Model: SmallCNN (smoke test only)")

    selector = TopMSelector(m=5)
    provider = GradCAMRegionsProvider(grid_h, grid_w)
    objective = RobustShortcutObjective(
        lambda_mean=game_cfg.lambda_mean,
        lambda_var=game_cfg.lambda_var,
        lambda_gap=game_cfg.lambda_gap,
        lambda_shortcut=game_cfg.lambda_shortcut,
        lambda_disjoint=game_cfg.lambda_disjoint,
        lambda_sparse=game_cfg.lambda_sparse,
    )
    allocator = RobustShortcutOptimizationAllocator(
        objective,
        num_steps=num_steps,
        lr=0.3,
        lambda_disjoint=game_cfg.lambda_disjoint,
    )

    all_metrics = []
    id_ood_gaps = []
    saved_viz = False
    count = 0
    for x, y in loader:
        if num_images is not None and count >= num_images:
            break
        x = x.to(device)
        y = y.to(device)
        if num_images is not None:
            x = x[: min(batch_size, num_images - count)]
            y = y[: x.shape[0]]
        count += x.shape[0]

        if input_size is not None and x.shape[-1] != input_size:
            x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)

        env = env_batch_fn(x, y)
        with torch.no_grad():
            logits = model(x)
            hypotheses = selector.select(logits, torch.softmax(logits, dim=-1))
        x_grad = x.detach().clone().requires_grad_(True)
        evidence = provider.explain(x_grad, model, hypotheses)
        evidence = evidence.detach() / (evidence.sum(dim=-1, keepdim=True).clamp_min(1e-8))

        masks = allocator.allocate(
            x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, evidence=evidence, env=env
        )
        metrics = objective.compute(
            x=x, model=model, unit_space=unit_space, hypotheses=hypotheses, masks=masks, evidence=evidence, env=env
        )
        all_metrics.append({k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()})
        gap = compute_id_ood_gap(model, env, unit_space, masks["shortcut"], y)
        id_ood_gaps.append(gap)
        if not saved_viz:
            _save_mask_visualization(
                x=x,
                masks=masks,
                unit_space=unit_space,
                out_path=REPO / "scripts" / "out" / "robust_shortcut_masks.png",
                sample_idx=0,
            )
            saved_viz = True

    mean_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    mean_gap = sum(id_ood_gaps) / len(id_ood_gaps) if id_ood_gaps else 0.0
    print("--- Robust vs Shortcut on biased setup ---")
    print(
        "Game mode: %s (mean=%.3f var=%.3f gap=%.3f shortcut=%.3f disjoint=%.3f sparse=%.3f)"
        % (
            game_cfg.mode,
            game_cfg.lambda_mean,
            game_cfg.lambda_var,
            game_cfg.lambda_gap,
            game_cfg.lambda_shortcut,
            game_cfg.lambda_disjoint,
            game_cfg.lambda_sparse,
        )
    )
    print("Metrics: rob_mean=%.4f rob_var=%.4f sho_gap=%.4f disjoint=%.4f sparse=%.4f" % (
        mean_metrics["rob_mean"], mean_metrics["rob_var"], mean_metrics["sho_gap"],
        mean_metrics["disjoint"], mean_metrics["sparse"]))
    print("ID-OOD gap (shortcut mask, quantitative): %.4f" % mean_gap)
    print("Qualitative: robust mask -> object/digit regions; shortcut mask -> color/background cue.")
    print("Pass: robust tracks object, shortcut tracks background; ID-OOD gap reported.")

    out_dir = REPO / "scripts" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / f"{export_prefix}_metrics.json"
    summary_csv = out_dir / f"{export_prefix}_metrics.csv"
    per_batch_csv = out_dir / f"{export_prefix}_per_batch.csv"

    summary = {
        "run_name": export_prefix,
        "game_mode": game_cfg.mode,
        "num_images": count,
        "batch_size": int(batch_size),
        "num_steps": int(num_steps),
        "game_config": {
            "lambda_mean": float(game_cfg.lambda_mean),
            "lambda_var": float(game_cfg.lambda_var),
            "lambda_gap": float(game_cfg.lambda_gap),
            "lambda_shortcut": float(game_cfg.lambda_shortcut),
            "lambda_disjoint": float(game_cfg.lambda_disjoint),
            "lambda_sparse": float(game_cfg.lambda_sparse),
        },
        "scalar_metrics": mean_metrics,
        "id_ood_gap": float(mean_gap),
    }
    save_json(summary_json, summary)

    scalar_row = {
        "run_name": export_prefix,
        "game_mode": game_cfg.mode,
        "id_ood_gap": float(mean_gap),
        "lambda_mean": float(game_cfg.lambda_mean),
        "lambda_var": float(game_cfg.lambda_var),
        "lambda_gap": float(game_cfg.lambda_gap),
        "lambda_shortcut": float(game_cfg.lambda_shortcut),
        "lambda_disjoint": float(game_cfg.lambda_disjoint),
        "lambda_sparse": float(game_cfg.lambda_sparse),
    }
    scalar_row.update(mean_metrics)
    save_rows_csv(summary_csv, [scalar_row])

    per_batch_rows = []
    for idx, m in enumerate(all_metrics):
        row = {"batch_idx": idx}
        row.update(m)
        if idx < len(id_ood_gaps):
            row["id_ood_gap"] = float(id_ood_gaps[idx])
        per_batch_rows.append(row)
    save_rows_csv(per_batch_csv, per_batch_rows)

    print("Saved metrics summary to", summary_json)
    print("Saved scalar metrics table to", summary_csv)
    print("Saved per-batch metrics table to", per_batch_csv)
    return mean_metrics, mean_gap


def _synthetic_biased(n: int):
    from instantiations.shift.biased_data import colorize_mnist
    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return n
        def __getitem__(self, i):
            im = torch.rand(1, 28, 28)
            label = i % 10
            color_idx = label % 3
            im_c = colorize_mnist(im, color_idx, 10).squeeze(0)
            return im_c, label
    return DS()


def _env_synthetic(x: torch.Tensor, y: torch.Tensor) -> EnvBatch:
    from instantiations.shift.biased_data import colorize_mnist
    B = x.shape[0]
    device = x.device
    g = x.mean(dim=1, keepdim=True)
    o1 = torch.stack([colorize_mnist(g[b:b+1], (y[b].item() + 1) % 10, 10).squeeze(0) for b in range(B)])
    o2 = torch.stack([colorize_mnist(g[b:b+1], (y[b].item() + 2) % 10, 10).squeeze(0) for b in range(B)])
    return EnvBatch(xs=[x, o1, o2], env_ids=["id", "ood1", "ood2"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate shift-aware robust/shortcut instantiation")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images to evaluate (default: full dataset)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--export_prefix", type=str, default="robust_shortcut")
    parser.add_argument("--dataset", dest="dataset_name", type=str,
                        default="colored_mnist", choices=BIASED_DATASETS,
                        help="Biased dataset to evaluate on")
    parser.add_argument("--model", dest="model_name", type=str, default="resnet18",
                        choices=TV_MODELS + ["smallcnn"])
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument(
        "--game_mode",
        type=str,
        choices=list(shift_game_modes()),
        default="mixed",
        help="Shift game mode: cooperative, competitive, mixed, or manual (explicit lambdas)",
    )
    parser.add_argument("--lambda_mean", type=float, default=None, help="manual mode only")
    parser.add_argument("--lambda_var", type=float, default=None, help="manual mode only")
    parser.add_argument("--lambda_gap", type=float, default=None, help="manual mode only")
    parser.add_argument("--lambda_shortcut", type=float, default=None, help="manual mode only")
    parser.add_argument("--lambda_disjoint", type=float, default=None, help="manual mode only")
    parser.add_argument("--lambda_sparse", type=float, default=None, help="manual mode only")
    args = parser.parse_args()

    manual_overrides = {
        "lambda_mean": args.lambda_mean,
        "lambda_var": args.lambda_var,
        "lambda_gap": args.lambda_gap,
        "lambda_shortcut": args.lambda_shortcut,
        "lambda_disjoint": args.lambda_disjoint,
        "lambda_sparse": args.lambda_sparse,
    }
    provided = sorted(k for k, v in manual_overrides.items() if v is not None)
    if args.game_mode != "manual" and provided:
        parser.error(
            "--game_mode %s does not accept explicit lambdas (%s). "
            "Use --game_mode manual for explicit values."
            % (args.game_mode, ", ".join(provided))
        )
    if args.game_mode == "manual":
        missing = sorted(k for k, v in manual_overrides.items() if v is None)
        if missing:
            parser.error("manual mode requires all lambdas; missing: %s" % ", ".join(missing))

    try:
        run_eval(
            num_images=args.num_images,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            data_root=args.data_root,
            export_prefix=args.export_prefix,
            game_mode=args.game_mode,
            model_name=args.model_name,
            pretrained=args.pretrained,
            dataset_name=args.dataset_name,
            lambda_mean=args.lambda_mean,
            lambda_var=args.lambda_var,
            lambda_gap=args.lambda_gap,
            lambda_shortcut=args.lambda_shortcut,
            lambda_disjoint=args.lambda_disjoint,
            lambda_sparse=args.lambda_sparse,
        )
    except ValueError as e:
        parser.error(str(e))
