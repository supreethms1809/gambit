"""
scripts/train_backbone.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Train or fine-tune a ResNet backbone for each dataset before running GAMBIT experiments.

Strategy
--------
- **Contrastive datasets** (mnist, cifar10, pets, stanford_dogs):
  Linear probe by default — freeze ImageNet backbone, train only the output head.
  Typically 10 epochs is sufficient for the head to learn the class boundaries.
  Pass ``freeze_backbone=False`` for full fine-tuning (slower, higher accuracy).

- **Shift dataset** (colored_mnist):
  Fine-tune ResNet18 on ColoredMNIST train split with ``correlation=0.9`` so the
  model learns to exploit color as a shortcut.  The GAMBIT shift experiment then
  decomposes robust (digit shape) vs shortcut (color) evidence.

Checkpoints are cached in ``ckpt_dir`` (default ``scripts/out/checkpoints/``).
A run that finds an existing checkpoint skips training unless ``force=True``.

Usage (standalone)::

    PYTHONPATH=. python scripts/train_backbone.py --dataset cifar10
    PYTHONPATH=. python scripts/train_backbone.py --dataset colored_mnist --epochs 15
    PYTHONPATH=. python scripts/train_backbone.py --dataset pets --freeze_backbone
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent

TV_INPUT_SIZE = 224
COLORED_MNIST_CORRELATION = 0.9

# ---------------------------------------------------------------------------
# Data loaders  (train splits)
# ---------------------------------------------------------------------------

def _make_tv_transforms(image_size: int, grayscale_to_rgb: bool = False,
                         augment: bool = True):
    """Return a torchvision transform for training."""
    from torchvision import transforms
    ops = []
    if grayscale_to_rgb:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.Resize((image_size, image_size)))
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


def get_train_loader(
    dataset: str,
    batch_size: int,
    data_root: Path,
    image_size: int = TV_INPUT_SIZE,
) -> Tuple[torch.utils.data.DataLoader, int]:
    """Return (train_loader, num_classes) for the given dataset."""
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision.datasets import MNIST, CIFAR10, ImageFolder

    if dataset == "mnist":
        from torchvision import transforms
        t = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        ds = MNIST(root=str(data_root), train=True, download=True, transform=t)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0), 10

    if dataset == "cifar10":
        t = _make_tv_transforms(image_size, augment=True)
        ds = CIFAR10(root=str(data_root), train=True, download=True, transform=t)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0), 10

    if dataset == "pets":
        t = _make_tv_transforms(image_size, augment=True)
        ds = ImageFolder(root=str(data_root / "PetImages"), transform=t)
        n_train = int(0.8 * len(ds))
        train_ds, _ = random_split(ds, [n_train, len(ds) - n_train],
                                   generator=torch.Generator().manual_seed(42))
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0), len(ds.classes)

    if dataset == "stanford_dogs":
        t = _make_tv_transforms(image_size, augment=True)
        ds = ImageFolder(root=str(data_root / "stanford_dogs" / "images" / "Images"), transform=t)
        n_train = int(0.8 * len(ds))
        train_ds, _ = random_split(ds, [n_train, len(ds) - n_train],
                                   generator=torch.Generator().manual_seed(42))
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0), len(ds.classes)

    if dataset == "colored_mnist":
        from instantiations.shift.biased_data import ColoredMNIST
        from torchvision import transforms
        # ColoredMNIST returns float tensors in [0,1] already; just resize + normalize
        base_ds = ColoredMNIST(root=str(data_root), train=True, download=True,
                               correlation=COLORED_MNIST_CORRELATION)
        # Wrap to apply resize + normalize
        norm = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # applied to PIL, but here tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        class _ResizeWrapper(torch.utils.data.Dataset):
            def __init__(self, inner):
                self.inner = inner

            def __len__(self):
                return len(self.inner)

            def __getitem__(self, i):
                x, y = self.inner[i]
                # x is (3, 28, 28); resize to (3, image_size, image_size)
                x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size),
                                  mode="bilinear", align_corners=False).squeeze(0)
                x = transforms.functional.normalize(
                    x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                return x, y

        ds = _ResizeWrapper(base_ds)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0), 10

    raise ValueError(f"Unknown dataset: {dataset}. "
                     f"Choices: mnist, cifar10, pets, stanford_dogs, colored_mnist")


# ---------------------------------------------------------------------------
# Model building  (mirrors ablation_contrastive._build_model)
# ---------------------------------------------------------------------------

def _build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
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
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return m


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the output head (fc / classifier)."""
    for name, param in model.named_parameters():
        is_head = "fc" in name or "classifier" in name
        param.requires_grad_(is_head)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Fine-tune *model* in-place and return it.

    Args:
        model:           Freshly built model (output head already replaced).
        train_loader:    Training data loader.
        num_epochs:      Training epochs.
        lr:              Learning rate for Adam.
        freeze_backbone: If True, only the output head is trained (linear probe).
        device:          Torch device (auto-detected if None).

    Returns:
        The trained model in eval() mode.
    """
    if device is None:
        from core.device import get_device
        device = get_device()

    print(f"  [train] device: {device}")
    model = model.to(device)

    if freeze_backbone:
        _freeze_backbone(model)
        print("  [train] linear probe — backbone frozen, training output head only")
    else:
        print("  [train] full fine-tune — all layers trainable")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [train] trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        n_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct   += (logits.argmax(1) == y).sum().item()
            n_total   += len(y)
        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        acc = correct / max(n_total, 1)
        print(f"  [train] epoch {epoch + 1:>2}/{num_epochs}  "
              f"loss={avg_loss:.4f}  acc={acc:.3f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Checkpoint cache
# ---------------------------------------------------------------------------

def get_or_train(
    dataset: str,
    model_name: str = "resnet18",
    pretrained: bool = True,
    data_root: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    batch_size: int = 32,
    force: bool = False,
) -> Path:
    """Return path to a trained checkpoint, training if not already cached.

    Checkpoint filename encodes all hyper-parameters so different configs are
    cached separately.

    Args:
        dataset:         One of mnist | cifar10 | pets | stanford_dogs | colored_mnist.
        model_name:      Backbone architecture (resnet18, resnet34, …).
        pretrained:      Start from ImageNet weights.
        data_root:       Root directory for dataset files.
        ckpt_dir:        Directory to save/load checkpoints.
        num_epochs:      Training epochs.
        lr:              Adam learning rate.
        freeze_backbone: Linear probe (True) or full fine-tune (False).
        batch_size:      Training batch size.
        force:           Re-train even if a checkpoint exists.

    Returns:
        Path to the saved ``.pt`` checkpoint file.
    """
    if data_root is None:
        data_root = REPO / "data"
    if ckpt_dir is None:
        ckpt_dir = REPO / "scripts" / "out" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "lp" if freeze_backbone else "ft"   # linear-probe vs fine-tune
    pre_tag  = "pt" if pretrained else "rand"
    ckpt_name = f"{dataset}_{model_name}_{pre_tag}_{mode_tag}_ep{num_epochs}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    if ckpt_path.exists() and not force:
        print(f"  [train] checkpoint found: {ckpt_path} (skipping training)")
        return ckpt_path

    print(f"\n{'='*60}")
    print(f"  Training: dataset={dataset}  model={model_name}  "
          f"pretrained={pretrained}  freeze={freeze_backbone}  epochs={num_epochs}")
    print(f"{'='*60}")

    # Get num_classes first (load one batch from the eval loader if train fails)
    try:
        train_loader, num_classes = get_train_loader(dataset, batch_size, data_root)
    except Exception as e:
        print(f"  [train] WARNING: could not load train split for {dataset}: {e}")
        print(f"  [train] Skipping training — no checkpoint saved.")
        return ckpt_path  # caller should handle missing file

    model = _build_model(model_name, num_classes, pretrained=pretrained)
    model = train_model(model, train_loader, num_epochs=num_epochs, lr=lr,
                        freeze_backbone=freeze_backbone)

    torch.save(model.state_dict(), ckpt_path)
    print(f"  [train] checkpoint saved: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAMBIT backbone checkpoints")
    parser.add_argument("--dataset", required=True,
                        choices=["mnist", "cifar10", "pets", "stanford_dogs", "colored_mnist"])
    parser.add_argument("--model", dest="model_name", default="resnet18",
                        choices=["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Linear probe: freeze backbone, train head only (default)")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="Full fine-tune: train all layers")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    parser.add_argument("--force", action="store_true", help="Re-train even if checkpoint exists")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    ckpt = get_or_train(
        dataset=args.dataset,
        model_name=args.model_name,
        pretrained=args.pretrained,
        data_root=Path(args.data_root) if args.data_root else None,
        num_epochs=args.num_epochs,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(f"\nDone. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
