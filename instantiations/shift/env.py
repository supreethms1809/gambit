"""
EnvBatch generator for robust vs shortcut: env.xs = [x_id, aug1(x), aug2(x)].
Classifier is used only for forward (caller keeps it frozen).
Pass: model predictions or logits differ across envs for at least some examples.
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple
import torch
from core.types import EnvBatch


def env_batch_from_augs(
    x: torch.Tensor,
    aug1: Callable[[torch.Tensor], torch.Tensor],
    aug2: Callable[[torch.Tensor], torch.Tensor],
    env_ids: Optional[List[str]] = None,
) -> EnvBatch:
    """
    Build EnvBatch with xs = [x_id, aug1(x), aug2(x)].
    x_id is the identity (in-distribution); aug1/aug2 produce OOD-style views (e.g. background/style change).
    """
    x_id = x
    x_aug1 = aug1(x)
    x_aug2 = aug2(x)
    xs = [x_id, x_aug1, x_aug2]
    if env_ids is None:
        env_ids = ["id", "aug1", "aug2"]
    return EnvBatch(xs=xs, env_ids=env_ids)


def default_shift_augs(
    color_jitter: float = 0.4,
    greyscale_prob: float = 0.2,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """
    Two different augmentations that change appearance (e.g. color/style) so logits can differ.
    Keeps spatial content roughly so "object" stays; changes cue that can be shortcut.
    """
    def aug1(im: torch.Tensor) -> torch.Tensor:
        out = im.clone()
        if im.shape[1] >= 3 and color_jitter > 0:
            c = torch.rand(3, device=im.device) * (2 * color_jitter) + (1 - color_jitter)
            out = out * c.view(1, 3, 1, 1)
        if greyscale_prob > 0 and torch.rand(1).item() < greyscale_prob:
            g = out.mean(dim=1, keepdim=True).expand_as(out)
            out = (1 - greyscale_prob) * out + greyscale_prob * g
        return out.clamp(0, 1)

    def aug2(im: torch.Tensor) -> torch.Tensor:
        out = im.clone()
        if im.shape[1] >= 3 and color_jitter > 0:
            c = torch.rand(3, device=im.device) * (2 * color_jitter) + (1 - color_jitter)
            out = out * c.view(1, 3, 1, 1)
        if greyscale_prob > 0 and torch.rand(1).item() < greyscale_prob:
            g = out.mean(dim=1, keepdim=True).expand_as(out)
            out = (1 - greyscale_prob) * out + greyscale_prob * g
        return out.clamp(0, 1)

    return aug1, aug2


def logits_differ_across_envs(model: Any, env: EnvBatch) -> bool:
    """Return True if model logits differ across env.xs for at least one example (so shortcut signal exists)."""
    with torch.no_grad():
        logits_list = [model(xe) for xe in env.xs]
    # (B, C) each; check that not all identical
    for b in range(logits_list[0].shape[0]):
        l0 = logits_list[0][b]
        for j in range(1, len(logits_list)):
            if not torch.allclose(l0, logits_list[j][b], atol=1e-5, rtol=1e-5):
                return True
    return False
