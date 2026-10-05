"""The model takes raw ``[0, 1]`` input. ImageNet normalisation is a layer inside it.

Library baselines that expect normalised input get it from this wrapper. An
assertion rejects an already-normalised tensor, which is how the earlier
train/eval normalisation mismatch gets caught instead of scoring a broken model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CONVENTION_RAW = "raw"
CONVENTION_IMAGENET = "imagenet"
CONVENTIONS = (CONVENTION_RAW, CONVENTION_IMAGENET)


def choose_input_convention(raw_score: float, imagenet_score: float) -> str:
    """Keep raw on a tie. ImageNet normalisation wins only when its val score is higher."""
    if imagenet_score > raw_score:
        return CONVENTION_IMAGENET
    return CONVENTION_RAW


def default_convention_path() -> "Path":
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "results" / "paper" / "input_convention.txt"


def read_input_convention(path=None) -> str:
    """``raw`` when the file is absent. The probe writes the winner before a restart."""
    from pathlib import Path

    path = Path(path) if path is not None else default_convention_path()
    if not path.is_file():
        return CONVENTION_RAW
    text = path.read_text(encoding="utf-8").strip()
    if text not in CONVENTIONS:
        raise ValueError(f"input convention must be one of {CONVENTIONS}, got {text!r}")
    return text


def write_input_convention(convention: str, path=None) -> None:
    from pathlib import Path

    if convention not in CONVENTIONS:
        raise ValueError(f"input convention must be one of {CONVENTIONS}, got {convention!r}")
    path = Path(path) if path is not None else default_convention_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(convention + "\n", encoding="utf-8")


def maybe_wrap(model: nn.Module, convention: str) -> nn.Module:
    """Raw leaves the module unchanged. ImageNet normalisation is a layer in front of it."""
    if convention == CONVENTION_RAW:
        return model
    if convention == CONVENTION_IMAGENET:
        return NormalizedModel(model)
    raise ValueError(f"input convention must be one of {CONVENTIONS}, got {convention!r}")


def checkpoint_input_convention(blob) -> str:
    """The input convention a checkpoint was trained under.

    ``get_or_train`` saves the unwrapped state dict with an ``input_convention``
    key. A bare state dict, or metadata without the key, predates the switch and
    was trained on raw input.
    """
    if isinstance(blob, dict) and "state_dict" in blob:
        convention = blob.get("input_convention", CONVENTION_RAW)
        if convention not in CONVENTIONS:
            raise ValueError(f"input convention must be one of {CONVENTIONS}, got {convention!r}")
        return convention
    return CONVENTION_RAW


def load_checkpoint_into(model: nn.Module, blob) -> nn.Module:
    """Load ``blob`` into the bare ``model`` and restore its training-time input layer.

    Every checkpoint load goes through here. Loading the state dict alone would
    feed raw input to a model trained on normalised input, which is the
    train/eval mismatch this wrapper exists to prevent.
    """
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state)
    return maybe_wrap(model, checkpoint_input_convention(blob))


def unwrap(model: nn.Module) -> nn.Module:
    """The classifier inside a ``NormalizedModel``, for code that inspects layer types."""
    while isinstance(model, NormalizedModel):
        model = model.model
    return model


class NormalizedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
        tol: float = 1e-4,
    ):
        super().__init__()
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length")
        self.model = model
        self.tol = float(tol)
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = float(x.detach().min())
        hi = float(x.detach().max())
        if lo < -self.tol or hi > 1.0 + self.tol:
            raise ValueError(
                f"NormalizedModel expects raw [0, 1] input, got min={lo:.4f} max={hi:.4f}"
            )
        channels = x.shape[1]
        if channels != self.mean.shape[1]:
            raise ValueError(
                f"NormalizedModel mean has {self.mean.shape[1]} channels, input has {channels}"
            )
        normalised = (x - self.mean) / self.std
        return self.model(normalised)

    def __getattr__(self, name: str):
        """Forward layer lookups (``model.layer4``, ``model.conv1``) to the inner module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("model")
            if inner is None:
                raise
            return getattr(inner, name)
