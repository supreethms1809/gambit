"""
scripts/train_backbone.py

Train or fine-tune a ResNet backbone for each dataset before running GAMBIT experiments.

Strategy
--------
- **Contrastive datasets** (mnist, cifar10, pets, stanford_dogs, ham10000, brain_tumor):
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
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent

TV_INPUT_SIZE = 224


def paper_checkpoint_name(
    dataset: str,
    model_name: str,
    pretrained: bool,
    freeze_backbone: bool,
    num_epochs: int,
    lr: float,
    seed: int,
    convention: str = "raw",
) -> str:
    """Filename for one paper cell. ImageNet normalisation gets its own cache key."""
    mode_tag = "lp" if freeze_backbone else "ft"
    pre_tag = "pt" if pretrained else "rand"
    lr_tag = f"lr{lr:g}"
    conv_tag = "" if convention == "raw" else f"_{convention}"
    return (
        f"{dataset}_{model_name}_{pre_tag}_{mode_tag}_ep{num_epochs}"
        f"_{lr_tag}{conv_tag}_seed{seed}.pt"
    )
COLORED_MNIST_CORRELATION = 0.9

# Medical datasets ship pre-split; train on the train split only (relative to data_root).
# Keep in sync with MEDICAL_SPLIT_ROOTS in examples/contrastive_explanation.py.
MEDICAL_TRAIN_ROOTS = {
    "ham10000": Path("ham10000") / "train",
    "brain_tumor": Path("brain_tumor") / "Training",
}

# These folders are the paper TEST split (the old grouped holdout). Checkpoint
# selection must not read them. Val is a grouped carve inside the train folder,
# recorded in data/splits/.
LEGACY_TEST_ROOTS = {
    "ham10000": Path("ham10000") / "val",
    "brain_tumor": Path("brain_tumor") / "Testing",
}
PAPER_SPLIT_DATASETS = {
    "mnist", "cifar10", "pets", "stanford_dogs", "ham10000", "brain_tumor",
    "cifar100", "oxford_pets", "cub200",
    "colored_mnist", "planted_patch", "imagenet9", "waterbirds",
}
# Fixed color noise and patch positions. Model seeds do not redraw the data.
SHIFT_DATA_SEED = 43


def model_grid_size(model_name: str) -> tuple:
    """Return (grid_h, grid_w) for model at TV_INPUT_SIZE=224.

    vit_b_16 → 14×14 (224/16=14 patches per side)
    all others → 7×7
    """
    if model_name == "vit_b_16":
        return 14, 14
    return 7, 7

# ---------------------------------------------------------------------------
# Data loaders  (train splits)
# ---------------------------------------------------------------------------

def _make_tv_transforms(image_size: int, grayscale_to_rgb: bool = False,
                        augment: bool = True, normalize: bool = False):
    """Return a torchvision transform for training.

    ``normalize`` defaults to **False**, and that is deliberate. Every evaluation and
    explanation path in this repo — ablation_contrastive, eval_localization,
    eval_decomposition, eval_robust_shortcut, and examples/contrastive_explanation —
    feeds raw [0, 1] tensors with no normalization. Training with ImageNet statistics
    while evaluating without them is a distribution shift severe enough to look like a
    broken model: measured here, a fine-tuned ResNet-18 scored 0.337 on brain tumor
    (chance 0.333) evaluated raw versus 0.948 normalized, and a CIFAR-10 linear probe
    scored 0.389 raw versus 0.816 normalized.

    The unnormalized convention is the one to keep, because the interventions the
    objective is built on are defined in [0, 1] pixel space: VisionGridUnitSpace's
    blur/mean baselines and Integrated Gradients' ``baseline="zero"`` all mean
    something different once inputs are standardized. Pass ``normalize=True`` only if
    the consuming pipeline normalizes too.
    """
    from torchvision import transforms
    ops = []
    if grayscale_to_rgb:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.Resize((image_size, image_size)))
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


def _paper_subset(ds, dataset: str, split: str):
    """Indices from data/splits. Test is refused here: training never requests it."""
    from torch.utils.data import Subset

    from evaluation.splits import load_indices

    if split == "test":
        raise RuntimeError("training and checkpoint selection cannot read the test split")
    indices = load_indices(dataset, split, final=False, n_items=len(ds))
    return Subset(ds, indices)


def _subset_targets(ds) -> Optional[list]:
    """Labels of a dataset or a Subset, without a pass over the images."""
    from torch.utils.data import Subset

    if isinstance(ds, Subset):
        inner = _subset_targets(ds.dataset)
        if inner is None:
            return None
        return [inner[i] for i in ds.indices]
    targets = getattr(ds, "targets", None)
    if targets is None:
        return None
    return [int(t) for t in targets]


def _worker_init(_worker_id: int) -> None:
    """One thread per prefetch worker so the pool does not oversubscribe the CPU."""
    torch.set_num_threads(1)


class _ResizeTo(torch.utils.data.Dataset):
    """Bilinear resize of a tensor image. Stays raw ``[0, 1]``; no normalization.

    Defined at module scope so a CUDA prefetch worker can unpickle it.
    """

    def __init__(self, inner, size: int):
        self.inner = inner
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        image, label = self.inner[index]
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return image, label


def _dataloader(ds, batch_size: int, shuffle: bool, seed: int) -> torch.utils.data.DataLoader:
    """Shuffle from ``seed``, separate from the backbone's initialization stream.

    CUDA prefetches batches in worker processes. MPS stays in-process: worker
    processes deadlock against MPS (project notes, B6).
    """
    from torch.utils.data import DataLoader

    generator = None
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    workers = 8 if torch.cuda.is_available() else 0
    kwargs = {}
    if workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
        kwargs["pin_memory"] = True
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["worker_init_fn"] = _worker_init
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        **kwargs,
    )


def get_train_loader(
    dataset: str,
    batch_size: int,
    data_root: Path,
    image_size: int = TV_INPUT_SIZE,
    seed: int = 0,
) -> Tuple[torch.utils.data.DataLoader, int]:
    """Return (train_loader, num_classes) for the given dataset."""
    from torchvision.datasets import MNIST, CIFAR10, ImageFolder

    if dataset == "mnist":
        from torchvision import transforms
        t = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # No Normalize: every eval/explanation path here consumes raw [0,1].
        ])
        ds = MNIST(root=str(data_root), train=True, download=True, transform=t)
        ds = _paper_subset(ds, "mnist", "train")
        return _dataloader(ds, batch_size, True, seed), 10

    if dataset == "cifar10":
        t = _make_tv_transforms(image_size, augment=True)
        ds = CIFAR10(root=str(data_root), train=True, download=True, transform=t)
        ds = _paper_subset(ds, "cifar10", "train")
        return _dataloader(ds, batch_size, True, seed), 10

    if dataset == "pets":
        t = _make_tv_transforms(image_size, augment=True)
        ds = ImageFolder(root=str(data_root / "PetImages"), transform=t)
        num_classes = len(ds.classes)
        ds = _paper_subset(ds, "pets", "train")
        return _dataloader(ds, batch_size, True, seed), num_classes

    if dataset in {"cifar100", "oxford_pets", "cub200"}:
        from evaluation.paper_datasets import open_unsplit

        t = _make_tv_transforms(image_size, augment=True)
        ds = open_unsplit(dataset, "train", data_root, transform=t)
        num_classes = len(set(ds.targets))
        ds = _paper_subset(ds, dataset, "train")
        return _dataloader(ds, batch_size, True, seed), num_classes

    if dataset == "stanford_dogs":
        t = _make_tv_transforms(image_size, augment=True)
        ds = ImageFolder(root=str(data_root / "stanford_dogs" / "images" / "Images"), transform=t)
        num_classes = len(ds.classes)
        ds = _paper_subset(ds, "stanford_dogs", "train")
        return _dataloader(ds, batch_size, True, seed), num_classes

    if dataset in MEDICAL_TRAIN_ROOTS:
        # Pre-split medical datasets: train on the train/Training split only.
        t = _make_tv_transforms(image_size, augment=True)
        ds = ImageFolder(root=str(data_root / MEDICAL_TRAIN_ROOTS[dataset]), transform=t)
        num_classes = len(ds.classes)
        ds = _paper_subset(ds, dataset, "train")
        return _dataloader(ds, batch_size, True, seed), num_classes

    if dataset == "colored_cifar10":
        from instantiations.shift.biased_data import ColoredCIFAR10
        base_ds = ColoredCIFAR10(root=str(data_root), train=True, download=True,
                                 correlation=COLORED_MNIST_CORRELATION)
        ds = _ResizeTo(base_ds, image_size)
        return _dataloader(ds, batch_size, True, seed), 10

    if dataset == "texture_mnist":
        from instantiations.shift.biased_data import TextureBiasedMNIST
        base_ds = TextureBiasedMNIST(root=str(data_root), train=True, download=True,
                                     correlation=COLORED_MNIST_CORRELATION)
        ds = _ResizeTo(base_ds, image_size)
        return _dataloader(ds, batch_size, True, seed), 10

    if dataset == "colored_mnist":
        from instantiations.shift.biased_data import ColoredMNIST
        # Colors are fixed by SHIFT_DATA_SEED. Train indices are the MNIST paper
        # train split, so the val images are held out for checkpoint selection.
        base_ds = ColoredMNIST(root=str(data_root), train=True, download=False,
                               correlation=COLORED_MNIST_CORRELATION, seed=SHIFT_DATA_SEED)
        ds = _paper_subset(_ResizeTo(base_ds, image_size), "mnist", "train")
        return _dataloader(ds, batch_size, True, seed), 10

    if dataset == "planted_patch":
        from instantiations.shift.planted_patch import PlantedPatchClassifier
        ds = PlantedPatchClassifier(
            split="train", root=data_root, image_size=image_size, patch_seed=0,
        )
        return _dataloader(ds, batch_size, True, seed), 10

    if dataset == "imagenet9":
        from instantiations.shift.imagenet9 import ImageNet9Classifier
        ds = ImageNet9Classifier(split="train", root=data_root / "imagenet9", image_size=image_size)
        # Folder prefixes are the class ids. Do not open the images to count them.
        class_ids = []
        for pair in ds.inner.pairs:
            prefix = pair["original"].parent.name.split("_", 1)[0]
            class_ids.append(int(prefix) if prefix.isdigit() else 0)
        num_classes = max(class_ids) + 1
        return _dataloader(ds, batch_size, True, seed), num_classes

    if dataset == "waterbirds":
        from instantiations.shift.waterbirds import WaterbirdsClassifier
        ds = WaterbirdsClassifier(split="train", image_size=image_size)
        return _dataloader(ds, batch_size, True, seed), 2

    raise ValueError(f"Unknown dataset: {dataset}. "
                     f"Choices: mnist, cifar10, pets, stanford_dogs, ham10000, brain_tumor, "
                     f"colored_mnist, colored_cifar10, texture_mnist, "
                     f"planted_patch, imagenet9, waterbirds")


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
    elif model_name == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        # Its last Conv2d is the final 1x1 projection with a 7x7 spatial output, so
        # GradCAM's automatic target-layer pick is the canonical one and the grid stays
        # 7x7 — comparable to the resnet baselines. Mirrors the builder in
        # examples/contrastive_explanation.py, which already offered this backbone.
        m = models.efficientnet_v2_s(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "vit_b_16":
        m = models.vit_b_16(weights=weights)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif model_name == "vit_b_32":
        m = models.vit_b_32(weights=weights)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return m


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the output head (fc / classifier / heads)."""
    for name, param in model.named_parameters():
        is_head = "fc" in name or "classifier" in name or "heads" in name
        param.requires_grad_(is_head)


_HEAD_MODULE_SEGMENTS = ("fc", "classifier", "heads")


def _noise_module_types() -> tuple:
    """BatchNorm, dropout, and stochastic depth: modules that make train != eval."""
    types = [
        nn.modules.batchnorm._BatchNorm,
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
    ]
    try:
        from torchvision.ops import StochasticDepth

        types.append(StochasticDepth)
    except ImportError:
        pass
    return tuple(types)


def _backbone_to_eval(model: nn.Module) -> None:
    """Hold backbone noise modules (BN, dropout) in eval during probe training.

    Freezing ``requires_grad`` stops weight updates, but a backbone left in
    train mode still updates BatchNorm running statistics and samples dropout,
    so the "frozen" features drift every epoch and train/eval disagree. Head
    modules (fc / classifier / heads) stay in whatever mode the caller set.
    Call after every ``model.train()``: that call re-enables the whole tree.
    """
    noise_types = _noise_module_types()
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, noise_types):
            continue
        if any(seg in _HEAD_MODULE_SEGMENTS for seg in mod_name.split(".")):
            continue
        mod.eval()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_val_loader(
    dataset: str,
    batch_size: int,
    data_root: Path,
    image_size: int = TV_INPUT_SIZE,
) -> Optional[torch.utils.data.DataLoader]:
    """Val split used for checkpoint selection. This is not the test split."""
    from torchvision.datasets import CIFAR10, ImageFolder, MNIST

    from evaluation.splits import load_spec

    if dataset not in PAPER_SPLIT_DATASETS:
        return None
    if dataset == "colored_mnist":
        from instantiations.shift.biased_data import ColoredMNIST
        base = ColoredMNIST(
            root=str(data_root), train=True, download=False,
            correlation=COLORED_MNIST_CORRELATION, seed=SHIFT_DATA_SEED,
        )

        class _ResizeWrapper(torch.utils.data.Dataset):
            def __init__(self, inner):
                self.inner = inner

            def __len__(self):
                return len(self.inner)

            def __getitem__(self, i):
                x, y = self.inner[i]
                x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size),
                                  mode="bilinear", align_corners=False).squeeze(0)
                return x, y

        ds = _paper_subset(_ResizeWrapper(base), "mnist", "val")
        return _dataloader(ds, batch_size, False, 0)
    if dataset == "planted_patch":
        from instantiations.shift.planted_patch import PlantedPatchClassifier
        ds = PlantedPatchClassifier(
            split="val", root=data_root, image_size=image_size, patch_seed=0,
        )
        return _dataloader(ds, batch_size, False, 0)
    if dataset == "imagenet9":
        from instantiations.shift.imagenet9 import ImageNet9Classifier
        ds = ImageNet9Classifier(split="val", root=data_root / "imagenet9", image_size=image_size)
        return _dataloader(ds, batch_size, False, 0)
    if dataset == "waterbirds":
        from instantiations.shift.waterbirds import WaterbirdsClassifier
        ds = WaterbirdsClassifier(split="val", image_size=image_size)
        return _dataloader(ds, batch_size, False, 0)
    spec = load_spec(dataset)
    root_rel = spec["roots"]["val"]
    legacy_test = LEGACY_TEST_ROOTS.get(dataset)
    if legacy_test is not None and Path(root_rel) == legacy_test:
        raise RuntimeError(f"{dataset} val root points at the test folder {legacy_test}")
    t = _make_tv_transforms(image_size, augment=False)
    if dataset == "mnist":
        ds = MNIST(root=str(data_root), train=True, download=False, transform=t)
    elif dataset == "cifar10":
        ds = CIFAR10(root=str(data_root), train=True, download=False, transform=t)
    elif dataset in {"cifar100", "oxford_pets", "cub200"}:
        from evaluation.paper_datasets import open_unsplit

        ds = open_unsplit(dataset, "val", data_root, transform=t)
    else:
        ds = ImageFolder(root=str(data_root / root_rel), transform=t)
    ds = _paper_subset(ds, dataset, "val")
    return _dataloader(ds, batch_size, False, 0)


def inverse_frequency_weights(
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.

    HAM10000 is ~67% melanocytic nevus. Unweighted cross-entropy converges to
    predicting the majority class, which makes both the accuracy number and the
    resulting Grad-CAM evidence meaningless.
    """
    counts = torch.zeros(num_classes)
    ds = getattr(loader, "dataset", None)
    targets = _subset_targets(ds)
    if targets is not None:  # ImageFolder / Subset labels, without a data pass
        counts = torch.bincount(torch.as_tensor(targets), minlength=num_classes).float()
    else:
        for _, y in loader:
            counts += torch.bincount(y.cpu(), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    weights = weights * (num_classes / weights.sum())  # mean 1 -> loss scale unchanged
    weights[counts == 0] = 0.0
    return weights.to(device)


def score_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float]:
    """Top-1 accuracy and macro recall. One pass, no gradient."""
    from evaluation.accuracy import top1_and_balanced

    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            targets.append(y.cpu())
    if not preds:
        return 0.0, 0.0
    return top1_and_balanced(torch.cat(preds), torch.cat(targets), num_classes)


def balanced_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> float:
    """Macro-averaged recall over classes present in *loader*."""
    return score_loader(model, loader, device, num_classes)[1]


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    device: Optional[torch.device] = None,
    class_weights: Optional[torch.Tensor] = None,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """Fine-tune *model* in-place and return it.

    Args:
        model:           Freshly built model (output head already replaced).
        train_loader:    Training data loader.
        num_epochs:      Training epochs.
        lr:              Learning rate for Adam.
        freeze_backbone: If True, only the output head is trained (linear probe).
        device:          Torch device (auto-detected if None).
        class_weights:   Per-class cross-entropy weights for imbalanced data.
        val_loader:      If given, the returned model is the epoch with the best
                         balanced accuracy rather than simply the last epoch.
        num_classes:     Required alongside ``val_loader``.

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

    if class_weights is not None:
        class_weights = class_weights.to(device)
        print("  [train] class-weighted loss (inverse frequency)")
    if val_loader is not None and num_classes is None:
        raise ValueError("num_classes is required when val_loader is given")

    best_score = -1.0
    best_state: Optional[dict] = None

    for epoch in range(num_epochs):
        model.train()
        # model.train() re-enables the whole tree. A frozen backbone must not
        # update BN stats or sample dropout (see _backbone_to_eval).
        if freeze_backbone:
            _backbone_to_eval(model)
        total_loss = 0.0
        correct = 0
        n_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct   += (logits.argmax(1) == y).sum().item()
            n_total   += len(y)
        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        acc = correct / max(n_total, 1)
        msg = (f"  [train] epoch {epoch + 1:>2}/{num_epochs}  "
               f"loss={avg_loss:.4f}  acc={acc:.3f}")

        # Select on balanced accuracy, not on the last epoch: with a skewed dataset
        # the final epoch is often not the best model for minority classes, and
        # minority classes are the clinically interesting ones.
        if val_loader is not None:
            val_top1, val_bal = score_loader(model, val_loader, device, num_classes)
            msg += f"  val_top1={val_top1:.4f}  val_balanced_acc={val_bal:.4f}"
            if val_bal > best_score:
                best_score = val_bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                msg += "  *"
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [train] restored best epoch (val_balanced_acc={best_score:.4f})")

        # A model at chance is not a weak model, it is a failed run — and it flows
        # downstream silently, because every explanation method will happily produce
        # confident-looking attributions for a network that learned nothing. The usual
        # cause is too high a learning rate for full fine-tuning of a pretrained
        # backbone: 1e-3 with Adam wipes the pretrained features in the first epoch.
        chance = 1.0 / max(num_classes, 1)
        if best_score <= chance * 1.15:
            warnings.warn(
                f"Training finished at balanced accuracy {best_score:.4f}, at or near "
                f"chance ({chance:.4f}) for {num_classes} classes. Any explanation built "
                f"on this checkpoint is meaningless. If this is a full fine-tune of a "
                f"pretrained backbone, lr={lr:g} is likely too high — try 1e-4.",
                RuntimeWarning,
                stacklevel=2,
            )
            print(f"  [train] *** WARNING: balanced accuracy {best_score:.4f} is at "
                  f"chance ({chance:.4f}) — this checkpoint is unusable ***")

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
    seed: int = 0,
    force: bool = False,
    balanced: Optional[bool] = None,
) -> Path:
    """Return path to a trained checkpoint, training if not already cached.

    Checkpoint filename encodes all hyper-parameters (including seed) so
    different configs are cached separately.

    Args:
        dataset:         One of mnist | cifar10 | pets | stanford_dogs | ham10000 |
                         brain_tumor | colored_mnist.
        model_name:      Backbone architecture (resnet18, resnet34, …).
        pretrained:      Start from ImageNet weights.
        data_root:       Root directory for dataset files.
        ckpt_dir:        Directory to save/load checkpoints.
        num_epochs:      Training epochs.
        lr:              Adam learning rate.
        freeze_backbone: Linear probe (True) or full fine-tune (False).
        batch_size:      Training batch size.
        seed:            Random seed for weight init and data shuffling.
        force:           Re-train even if a checkpoint exists.
        balanced:        Class-weighted loss + balanced-accuracy model selection.
                         Defaults to on for the medical datasets, off elsewhere.

    Returns:
        Path to the saved ``.pt`` checkpoint file.
    """
    if data_root is None:
        data_root = REPO / "data"
    if ckpt_dir is None:
        ckpt_dir = REPO / "scripts" / "out" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from models.wrapper import NormalizedModel, maybe_wrap, read_input_convention

    # The learning rate belongs in the name: it is the difference between a working
    # fine-tune and one that converges to chance, and without it a re-run at a
    # corrected lr silently returns the broken checkpoint from the cache.
    # ImageNet normalisation is a separate cache key so a raw checkpoint is not
    # reused after the convention changes.
    convention = read_input_convention()
    ckpt_name = paper_checkpoint_name(
        dataset, model_name, pretrained, freeze_backbone, num_epochs, lr, seed, convention,
    )
    ckpt_path = ckpt_dir / ckpt_name

    if ckpt_path.exists() and not force:
        print(f"  [train] checkpoint found: {ckpt_path} (skipping training)")
        return ckpt_path

    print(f"\n{'='*60}")
    print(f"  Training: dataset={dataset}  model={model_name}  "
          f"pretrained={pretrained}  freeze={freeze_backbone}  epochs={num_epochs}  seed={seed}")
    print(f"  [train] input convention: {convention}")
    print(f"{'='*60}")

    # Get num_classes first (load one batch from the eval loader if train fails)
    try:
        train_loader, num_classes = get_train_loader(
            dataset, batch_size, data_root, seed=seed,
        )
    except Exception as e:
        print(f"  [train] WARNING: could not load train split for {dataset}: {e}")
        print(f"  [train] Skipping training — no checkpoint saved.")
        return ckpt_path  # caller should handle missing file

    torch.manual_seed(seed)
    model = maybe_wrap(_build_model(model_name, num_classes, pretrained=pretrained), convention)

    # Skewed medical datasets need class-weighted loss and balanced-accuracy model
    # selection, or training collapses onto the majority class (HAM10000 is ~67%
    # melanocytic nevus). Without this a seed sweep would silently produce weaker
    # models than the ones the reported numbers came from.
    if balanced is None:
        balanced = dataset in MEDICAL_TRAIN_ROOTS
    class_weights = None
    # Every paper dataset selects the checkpoint on val. Class weights stay on
    # for the skewed medical sets only. The loss, Adam, and cosine schedule
    # are unchanged; only which images are train versus val changed.
    val_loader = get_val_loader(dataset, batch_size, data_root) if dataset in PAPER_SPLIT_DATASETS else None
    if val_loader is None and dataset in PAPER_SPLIT_DATASETS:
        print(f"  [train] WARNING: no val split for {dataset}; keeping the last epoch")
    if balanced:
        from core.device import get_device
        class_weights = inverse_frequency_weights(train_loader, num_classes, get_device())

    model = train_model(model, train_loader, num_epochs=num_epochs, lr=lr,
                        freeze_backbone=freeze_backbone,
                        class_weights=class_weights, val_loader=val_loader,
                        num_classes=num_classes if val_loader is not None else None)

    # Metadata-wrapped so eval_localization.py / eval_decomposition.py can load these
    # directly. scripts/ablation_contrastive.py reads either format.
    inner = model.model if isinstance(model, NormalizedModel) else model
    torch.save({
        "state_dict": inner.state_dict(),
        "dataset": dataset,
        "model_name": model_name,
        "num_classes": num_classes,
        "seed": seed,
        "balanced": bool(balanced),
        "input_convention": convention,
    }, ckpt_path)
    print(f"  [train] checkpoint saved: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAMBIT backbone checkpoints")
    parser.add_argument("--dataset", required=True,
                        choices=["mnist", "cifar10", "cifar100", "pets", "oxford_pets",
                                 "stanford_dogs", "cub200", "ham10000", "brain_tumor",
                                 "colored_mnist", "colored_cifar10", "texture_mnist",
                                 "planted_patch", "imagenet9", "waterbirds"])
    parser.add_argument("--model", dest="model_name", default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "mobilenet_v2",
                                 "efficientnet_b0", "efficientnet_v2_s",
                                 "vit_b_16", "vit_b_32"])
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Linear probe: freeze backbone, train head only (default)")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="Full fine-tune: train all layers")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    parser.add_argument("--force", action="store_true", help="Re-train even if checkpoint exists")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Directory to save checkpoints (default: scripts/out/checkpoints)")
    parser.add_argument("--balanced", dest="balanced", action="store_true", default=None,
                        help="Class-weighted loss + balanced-accuracy selection "
                             "(default: on for medical datasets)")
    parser.add_argument("--no-balanced", dest="balanced", action="store_false",
                        help="Plain cross-entropy, keep the last epoch")
    args = parser.parse_args()

    ckpt = get_or_train(
        dataset=args.dataset,
        model_name=args.model_name,
        pretrained=args.pretrained,
        data_root=Path(args.data_root) if args.data_root else None,
        ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None,
        num_epochs=args.num_epochs,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        batch_size=args.batch_size,
        seed=args.seed,
        force=args.force,
        balanced=args.balanced,
    )
    print(f"\nDone. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
