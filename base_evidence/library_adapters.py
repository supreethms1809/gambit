"""
base_evidence/library_adapters.py

`BaseEvidenceProvider` adapters over pytorch-grad-cam and Captum.

CDEA's central claim is that it decomposes *whatever* attribution you give it. With two
hand-rolled backends (Grad-CAM, Integrated Gradients) that claim is asserted rather than
demonstrated. `BaseEvidenceProvider` is a one-method protocol --
``explain(x, model, hypotheses) -> (B, K, R)`` -- so a library method becomes a provider by
running it per hypothesis and pooling to the grid.

Available once `pip install grad-cam captum` has run:

    scorecam, layercam, xgradcam, ablationcam, hirescam, eigencam, gradcam++   (pytorch-grad-cam)
    deeplift, gradientshap, inputxgradient, guidedbackprop                     (Captum)

Every provider returns a non-negative, per-hypothesis region field, matching what
`GradCAMRegionsProvider` emits, so they are drop-in for any script that takes `--evidence`.

Cost, measured on 4 images x 5 hypotheses, ResNet-18, CPU:

    layercam / xgradcam / hirescam / eigencam / gradcam++   0.3 - 0.7 s
    inputxgradient / guidedbackprop                         0.5 s
    gradientshap / ig                                       12 s
    occlusion                                               3 s
    ablationcam                                             199 s
    scorecam                                                239 s

Score-CAM and Ablation-CAM run a forward pass per channel of the target layer, so they are
~60 s per image and impractical past a small subset. Use them on a matched sub-sample and
say so, rather than quietly dropping them.

Note on pooling: CAM methods produce a spatial map that is pooled by adaptive average to
(grid_h, grid_w). Captum's gradient methods attribute per pixel and per channel; those are
summed over channels, clamped at zero, then pooled the same way. Clamping matters -- the
allocator's evidence is a non-negative field, and signed attributions would otherwise let
"evidence against" masquerade as absent evidence.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from core.types import HypothesisSet

CAM_METHODS = ("gradcam", "gradcam++", "scorecam", "layercam", "xgradcam",
               "ablationcam", "hirescam", "eigencam")
CAPTUM_METHODS = ("gradientshap", "inputxgradient", "guidedbackprop", "deeplift")
# DeepLift needs every module used exactly once in forward(). torchvision's BasicBlock
# calls one `self.relu` twice per block, so DeepLift raises on ResNet regardless of the
# in-place fix below. Kept constructible for architectures that do not reuse modules, but
# out of the default sweep so a run does not die halfway through.
DEFAULT_METHODS = ("gradcam", "gradcam++", "layercam", "xgradcam", "hirescam", "eigencam",
                   "inputxgradient", "guidedbackprop", "gradientshap", "ig", "occlusion")
SLOW_METHODS = ("scorecam", "ablationcam")


def _find_target_layer(model: nn.Module) -> nn.Module:
    """The layer GradCAMRegionsProvider uses: last Conv2d, or a ViT's last encoder block.

    A bare last-Conv2d search picks a ViT's patch embedding, so this delegates.
    """
    from base_evidence.gradcam_regions import _find_target_layer as provider_layer

    return provider_layer(model)


def _deinplace(model: nn.Module) -> nn.Module:
    """A copy of `model` with every in-place ReLU made out-of-place.

    Captum's DeepLift attaches hooks that read a module's input, which `ReLU(inplace=True)`
    has already overwritten -- torchvision's ResNet and EfficientNet both use in-place
    ReLUs, so DeepLift raises on them out of the box.
    """
    import copy as _copy
    m = _copy.deepcopy(model)
    for mod in m.modules():
        for name, child in mod.named_children():
            if isinstance(child, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.ELU)) \
                    and getattr(child, "inplace", False):
                setattr(mod, name, type(child)(inplace=False))
    return m


def _pool(field: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """(B, H, W) or (B, 1, H, W) -> (B, gh*gw), non-negative."""
    if field.dim() == 3:
        field = field.unsqueeze(1)
    field = field.clamp_min(0)
    return F.adaptive_avg_pool2d(field, (gh, gw)).flatten(1)


class CamLibraryProvider:
    """pytorch-grad-cam methods as a BaseEvidenceProvider."""

    def __init__(self, method: str, grid_h: int, grid_w: int,
                 target_layer: Optional[nn.Module] = None) -> None:
        if method not in CAM_METHODS:
            raise ValueError(f"method must be one of {CAM_METHODS}")
        self.method = method
        self.grid_h, self.grid_w = grid_h, grid_w
        self.target_layer = target_layer

    def _cls(self):
        import pytorch_grad_cam as pgc
        return {
            "gradcam": pgc.GradCAM, "gradcam++": pgc.GradCAMPlusPlus,
            "scorecam": pgc.ScoreCAM, "layercam": pgc.LayerCAM,
            "xgradcam": pgc.XGradCAM, "ablationcam": pgc.AblationCAM,
            "hirescam": pgc.HiResCAM, "eigencam": pgc.EigenCAM,
        }[self.method]

    def explain(self, x: Any, model: nn.Module, hypotheses: HypothesisSet) -> torch.Tensor:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        layer = self.target_layer or _find_target_layer(model)
        B, K = hypotheses.ids.shape
        # pytorch-grad-cam runs on CPU-or-CUDA tensors and does its own backward; MPS
        # tensors are moved to CPU because several of its methods index with numpy.
        xin = x.detach()
        from base_evidence.gradcam_regions import cam_reshape_transform

        cam = self._cls()(model=model, target_layers=[layer],
                          reshape_transform=cam_reshape_transform(model))
        out = torch.zeros(B, K, self.grid_h * self.grid_w)
        for k in range(K):
            ids = hypotheses.ids[:, k].clamp_min(0).tolist()
            targets = [ClassifierOutputTarget(int(i)) for i in ids]
            g = cam(input_tensor=xin, targets=targets)          # (B, H, W) numpy
            out[:, k] = _pool(torch.from_numpy(g).float(), self.grid_h, self.grid_w)
        return out.to(x.device)


class CaptumRegionsProvider:
    """Captum attribution methods as a BaseEvidenceProvider."""

    def __init__(self, method: str, grid_h: int, grid_w: int,
                 n_samples: int = 16, baseline: str = "zero") -> None:
        if method not in CAPTUM_METHODS:
            raise ValueError(f"method must be one of {CAPTUM_METHODS}")
        self.method = method
        self.grid_h, self.grid_w = grid_h, grid_w
        self.n_samples = n_samples
        self.baseline = baseline

    def _attr(self, model: nn.Module):
        import captum.attr as ca
        if self.method in ("deeplift", "guidedbackprop"):
            model = _deinplace(model)
        if self.method == "gradientshap":
            # GradientShap adds Gaussian noise (stdevs=0.09) to the input, which leaves
            # [0, 1]. NormalizedModel rejects that range, so the path points are
            # clamped back to the image domain the model is defined on.
            inner = model

            def model(inp, _inner=inner):  # noqa: E306
                return _inner(inp.clamp(0.0, 1.0))
        return {
            "deeplift": ca.DeepLift, "gradientshap": ca.GradientShap,
            "inputxgradient": ca.InputXGradient, "guidedbackprop": ca.GuidedBackprop,
        }[self.method](model)

    def explain(self, x: Any, model: nn.Module, hypotheses: HypothesisSet) -> torch.Tensor:
        attr = self._attr(model)
        B, K = hypotheses.ids.shape
        base = torch.zeros_like(x) if self.baseline == "zero" else x.mean(
            dim=(2, 3), keepdim=True).expand_as(x)
        out = torch.zeros(B, K, self.grid_h * self.grid_w, device=x.device)
        for k in range(K):
            ids = hypotheses.ids[:, k].clamp_min(0)
            xin = x.detach().clone().requires_grad_(True)
            if self.method == "gradientshap":
                a = attr.attribute(xin, baselines=base, target=ids,
                                   n_samples=self.n_samples, stdevs=0.09)
            elif self.method == "deeplift":
                try:
                    a = attr.attribute(xin, baselines=base, target=ids)
                except RuntimeError as e:
                    raise RuntimeError(
                        "DeepLift requires every module to be used exactly once in "
                        "forward(); torchvision's BasicBlock reuses a single ReLU per "
                        "block, so ResNet/EfficientNet are unsupported. Use "
                        "'gradientshap' or 'inputxgradient' instead."
                    ) from e
            else:
                a = attr.attribute(xin, target=ids)
            # Signed per-channel attributions -> a non-negative region field, which is
            # what the allocator's evidence contract requires.
            out[:, k] = _pool(a.detach().sum(dim=1), self.grid_h, self.grid_w)
        return out


def build_provider(name: str, grid_h: int, grid_w: int, **kw):
    """Factory over the built-in providers and both libraries."""
    if name in CAM_METHODS:
        return CamLibraryProvider(name, grid_h, grid_w)
    if name in CAPTUM_METHODS:
        return CaptumRegionsProvider(name, grid_h, grid_w, **kw)
    if name == "ig":
        from .integrated_gradients_regions import IntegratedGradientsRegionsProvider
        return IntegratedGradientsRegionsProvider(grid_h=grid_h, grid_w=grid_w,
                                                  steps=kw.get("steps", 16),
                                                  baseline=kw.get("baseline", "zero"))
    if name == "occlusion":
        from .occlusion_regions import OcclusionRegionsProvider
        return OcclusionRegionsProvider(grid_h=grid_h, grid_w=grid_w, **kw)
    raise ValueError(f"unknown provider '{name}'")


ALL_METHODS = CAM_METHODS + CAPTUM_METHODS + ("ig", "occlusion")
__all__ = ["CamLibraryProvider", "CaptumRegionsProvider", "build_provider",
           "ALL_METHODS", "DEFAULT_METHODS", "SLOW_METHODS",
           "CAM_METHODS", "CAPTUM_METHODS"]
