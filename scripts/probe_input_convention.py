"""CIFAR-10 linear probe: raw [0, 1] versus ImageNet normalisation inside the model.

The backbone runs once. Both heads train on those cached features, so the
comparison is the input convention and not a second full-image schedule.
Pixels stay in [0, 1]. ``NormalizedModel`` applies ImageNet mean and std
inside the forward. Train and val use the paper splits. The test split is
not read. Augmentation is off so both conventions see the same images.

The winner is val balanced accuracy. A tie keeps raw.

    PYTHONPATH=. python scripts/probe_input_convention.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.accuracy import top1_and_balanced
from models.wrapper import (
    CONVENTION_IMAGENET,
    CONVENTION_RAW,
    choose_input_convention,
    maybe_wrap,
    write_input_convention,
)
from scripts.train_backbone import (
    REPO as TRAIN_REPO,
    TV_INPUT_SIZE,
    _build_model,
    _dataloader,
    _make_tv_transforms,
    _paper_subset,
)

OUT = REPO / "results" / "paper" / "logs" / "probe_input" / "cifar10_resnet50_seed0.json"
EPOCHS = 15
LR = 1e-3
BATCH = 32
SEED = 0


def _loader(split: str) -> DataLoader:
    transform = _make_tv_transforms(TV_INPUT_SIZE, augment=False, normalize=False)
    dataset = CIFAR10(
        root=str(TRAIN_REPO / "data"), train=True, download=False, transform=transform,
    )
    dataset = _paper_subset(dataset, "cifar10", split)
    return _dataloader(dataset, BATCH, shuffle=False, seed=SEED)


def cache_features(convention: str, loader: DataLoader, device: torch.device):
    """Frozen ResNet-50 in eval. The head is identity so the cache is the penultimate vector."""
    torch.manual_seed(SEED)
    model = _build_model("resnet50", num_classes=10, pretrained=True)
    model.fc = nn.Identity()
    model = maybe_wrap(model, convention)
    model.to(device)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            features.append(model(images.to(device)).cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def fit_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
) -> dict:
    """Adam, cosine stepped once per epoch, cross-entropy. Best val balanced accuracy is restored."""
    torch.manual_seed(SEED)
    head = nn.Linear(features.shape[1], 10).to(device)
    features = features.to(device)
    labels = labels.to(device)
    val_features = val_features.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    best_score = -1.0
    best_top1 = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    count = features.shape[0]
    for _epoch in range(EPOCHS):
        head.train()
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, BATCH):
            index = order[start:start + BATCH]
            logits = head(features[index])
            loss = loss_fn(logits, labels[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        schedule.step()
        head.eval()
        with torch.no_grad():
            pred = head(val_features).argmax(dim=1).cpu()
        top1, balanced = top1_and_balanced(pred, val_labels, 10)
        if balanced > best_score:
            best_score = float(balanced)
            best_top1 = float(top1)
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return {"val_top1": best_top1, "val_balanced_acc": best_score}


def main() -> None:
    from core.device import get_device

    device = get_device()
    print(f"device: {device}", flush=True)
    train_loader = _loader("train")
    val_loader = _loader("val")
    scores = {}
    for convention in (CONVENTION_RAW, CONVENTION_IMAGENET):
        print(f"caching {convention}", flush=True)
        train_f, train_y = cache_features(convention, train_loader, device)
        val_f, val_y = cache_features(convention, val_loader, device)
        print(f"fitting {convention}  features={tuple(train_f.shape)}", flush=True)
        scores[convention] = fit_head(train_f, train_y, val_f, val_y, device)
        row = scores[convention]
        print(
            f"{convention}  val_top1={row['val_top1']:.4f}"
            f"  val_balanced_acc={row['val_balanced_acc']:.4f}",
            flush=True,
        )
    winner = choose_input_convention(
        scores[CONVENTION_RAW]["val_balanced_acc"],
        scores[CONVENTION_IMAGENET]["val_balanced_acc"],
    )
    write_input_convention(winner)
    payload = {
        "dataset": "cifar10",
        "model": "resnet50",
        "seed": SEED,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH,
        "cached_features": True,
        "augmentation": False,
        "split": "paper val",
        "raw": scores[CONVENTION_RAW],
        "imagenet": scores[CONVENTION_IMAGENET],
        "winner": winner,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"input convention: {winner}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
