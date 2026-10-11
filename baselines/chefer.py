"""Chefer LRP bridge (Chefer et al., CVPR 2021) over the vendored closure.

``from_torchvision`` converts a torchvision ``vit_b_16`` into the vendored
``VisionTransformer`` so ``LRP.generate_LRP`` runs on our checkpoints. The
state-dict map (corrected against torchvision 0.24: its ``in_proj`` is one
combined ``(3E, E)`` matrix in q/k/v order, copied whole):

| torchvision key | vendored key |
|---|---|
| ``conv_proj`` | ``patch_embed.proj`` |
| ``class_token`` | ``cls_token`` |
| ``encoder.pos_embedding`` | ``pos_embed`` |
| ``ln_1`` | ``norm1`` |
| ``self_attention.in_proj_*`` | ``attn.qkv`` |
| ``self_attention.out_proj`` | ``attn.proj`` |
| ``ln_2`` | ``norm2`` |
| ``mlp.0`` | ``mlp.fc1`` |
| ``mlp.3`` | ``mlp.fc2`` |
| ``encoder.ln`` | ``norm`` |
| ``heads.head`` | ``head`` |

Pass the UNWRAPPED torchvision model (unwrap ``NormalizedModel`` first);
normalisation follows that wrapper's convention at call time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_VENDOR = Path(__file__).resolve().parents[1] / "third_party" / "chefer"


def _import_chefer():
    """Import the vendored names without shadowing our own packages.

    Upstream has no ``__init__.py`` under ``baselines/``; the vendored copies
    carry empty markers so the closure imports as regular packages while its
    directory leads ``sys.path``. Our top-level ``baselines`` would otherwise
    win (it is already in ``sys.modules``), so ours is evicted first and
    restored after: later ``baselines.*`` imports resolve to this repo again.
    """
    saved = {
        k: v
        for k, v in sys.modules.items()
        if k == "baselines" or k.startswith(("baselines.", "modules"))
    }
    for k in saved:
        del sys.modules[k]
    added = str(_VENDOR) not in sys.path
    if added:
        sys.path.insert(0, str(_VENDOR))
    try:
        from baselines.ViT.ViT_LRP import VisionTransformer
        from baselines.ViT.ViT_explanation_generator import LRP
    finally:
        if added and str(_VENDOR) in sys.path:
            sys.path.remove(str(_VENDOR))
        for k in [
            k
            for k in sys.modules
            if k == "baselines" or k.startswith(("baselines.", "modules"))
        ]:
            del sys.modules[k]
        sys.modules.update(saved)
    return VisionTransformer, LRP


def _mapped_state(src: dict[str, torch.Tensor], depth: int) -> dict[str, torch.Tensor]:
    """Remap a torchvision ViT-B/16 state dict onto vendored key names."""
    dst: dict[str, torch.Tensor] = {
        "patch_embed.proj.weight": src["conv_proj.weight"],
        "patch_embed.proj.bias": src["conv_proj.bias"],
        "cls_token": src["class_token"],
        "pos_embed": src["encoder.pos_embedding"],
        "norm.weight": src["encoder.ln.weight"],
        "norm.bias": src["encoder.ln.bias"],
        "head.weight": src["heads.head.weight"],
        "head.bias": src["heads.head.bias"],
    }
    for i in range(depth):
        t = f"encoder.layers.encoder_layer_{i}"
        p = f"blocks.{i}"
        dst[f"{p}.norm1.weight"] = src[f"{t}.ln_1.weight"]
        dst[f"{p}.norm1.bias"] = src[f"{t}.ln_1.bias"]
        # Combined (3E, E) q/k/v matrix, same order on both sides.
        dst[f"{p}.attn.qkv.weight"] = src[f"{t}.self_attention.in_proj_weight"]
        dst[f"{p}.attn.qkv.bias"] = src[f"{t}.self_attention.in_proj_bias"]
        dst[f"{p}.attn.proj.weight"] = src[f"{t}.self_attention.out_proj.weight"]
        dst[f"{p}.attn.proj.bias"] = src[f"{t}.self_attention.out_proj.bias"]
        dst[f"{p}.norm2.weight"] = src[f"{t}.ln_2.weight"]
        dst[f"{p}.norm2.bias"] = src[f"{t}.ln_2.bias"]
        dst[f"{p}.mlp.fc1.weight"] = src[f"{t}.mlp.0.weight"]
        dst[f"{p}.mlp.fc1.bias"] = src[f"{t}.mlp.0.bias"]
        dst[f"{p}.mlp.fc2.weight"] = src[f"{t}.mlp.3.weight"]
        dst[f"{p}.mlp.fc2.bias"] = src[f"{t}.mlp.3.bias"]
    return dst


def from_torchvision(model: nn.Module):
    """Convert a torchvision ``vit_b_16`` to the vendored LRP model.

    ViT-B/16 geometry is fixed; the class count is read from the head. The
    converted model is in eval mode on the source device. Logit equality with
    the source (max abs diff) is the test in ``tests/test_chefer.py``.
    """
    src = model.state_dict()
    if "encoder.layers.encoder_layer_0.self_attention.in_proj_weight" not in src:
        raise ValueError("from_torchvision needs a torchvision vit_b_16 state dict")
    depth = 0
    while f"encoder.layers.encoder_layer_{depth}.ln_1.weight" in src:
        depth += 1
    num_classes = int(src["heads.head.weight"].shape[0])
    device = next(model.parameters()).device
    VisionTransformer, _ = _import_chefer()
    converted = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        qkv_bias=True,
    )
    converted.load_state_dict(_mapped_state(src, depth), strict=True)
    # LayerNorm eps follows the source, not the vendored default: every block
    # norm is already 1e-6 on both sides, but the vendored final norm defaults
    # to 1e-5 while torchvision uses 1e-6 there (a 1.8e-4 activation gap with
    # identical weights). The baseline must explain our model, so it matches
    # our model's arithmetic. Named in the dossier.
    converted.norm.eps = float(model.encoder.ln.eps)
    for i in range(depth):
        tm = model.encoder.layers[i]
        converted.blocks[i].norm1.eps = float(tm.ln_1.eps)
        converted.blocks[i].norm2.eps = float(tm.ln_2.eps)
    converted.eval()
    return converted.to(device)


def class_relprop(converted, image: torch.Tensor, class_idx: int) -> torch.Tensor:
    """Author-default class map: ``transformer_attribution`` relprop (14x14).

    Upstream fixes ``alpha=1`` inside ``generate_LRP``. ``image`` is
    ``(1, 3, 224, 224)`` on the model's device. Returns the ``(14, 14)``
    patch relevance for ``class_idx``.
    """
    _, LRP = _import_chefer()
    cam = LRP(converted).generate_LRP(image, index=int(class_idx))
    side = int(round(float(cam.shape[-1]) ** 0.5))
    return cam.reshape(side, side).detach()
