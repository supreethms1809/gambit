"""
core/visualization.py
~~~~~~~~~~~~~~~~~~~~~
Reusable visualization utilities for CDEA explanations.
Uses Plotly for interactive, notebook-friendly figures.

Public API::

    from core.visualization import (
        mask_to_image,
        overlay_rgba,
        extract_layer_activations,
        show_evidence_heatmap,
        show_evidence_region_bars,
        show_feature_activations,
        show_explanation_gallery,
        show_gradcam_vs_ig,
        show_probability_split,
        show_pairwise_margin,
    )

All ``show_*`` functions return a ``plotly.graph_objects.Figure``.
Call ``fig.show()`` in a notebook or ``fig.write_html(path)`` to save.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level helpers (no Plotly/matplotlib dependency)
# ---------------------------------------------------------------------------

def mask_to_image(
    m: torch.Tensor,
    grid_h: int,
    grid_w: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Upsample a region mask to pixel space.

    Args:
        m:              Tensor of shape (R,) or (K, R).
        grid_h, grid_w: Grid dimensions.
        h, w:           Target pixel height and width.

    Returns:
        np.ndarray of shape (H, W) or (K, H, W), float32.
    """
    squeezed = m.dim() == 1
    if squeezed:
        m = m.unsqueeze(0)
    pm = m.detach().cpu().float().view(-1, 1, grid_h, grid_w)
    pm = F.interpolate(pm, size=(h, w), mode="bilinear", align_corners=False)
    out = pm.squeeze(1).numpy()
    if out.shape[0] == 1:
        out = out[0]
    return out


def overlay_rgba(
    unique: np.ndarray,
    shared: np.ndarray,
    alpha: float = 0.55,
    gamma: float = 0.5,
) -> np.ndarray:
    """Build an RGBA overlay from unique and shared mask arrays.

    Color encoding:
        Green  = unique evidence (class-specific)
        Red    = shared evidence (common across top classes)
        Yellow = both

    Args:
        unique: (H, W) float array, unique mask for one class.
        shared: (H, W) float array, shared mask.
        alpha:  Maximum overlay opacity.
        gamma:  Gamma correction for contrast boost (< 1 lifts weak activations).

    Returns:
        (H, W, 4) float32 RGBA array.
    """
    u = np.clip(unique, 0, 1).astype(np.float32)
    s = np.clip(shared,  0, 1).astype(np.float32)
    u = (u / (u.max() + 1e-8)) ** gamma
    s = (s / (s.max() + 1e-8)) ** gamma
    rgba = np.zeros((*u.shape, 4), dtype=np.float32)
    rgba[..., 0] = s           # red   = shared
    rgba[..., 1] = u           # green = unique
    rgba[..., 3] = alpha * np.clip(u + s, 0, 1)
    return rgba


def _to_hwc(x: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) or (H, W) tensor to (H, W, 3) float32 in [0, 1]."""
    img = x.detach().cpu().float()
    if img.dim() == 2:
        img = img.unsqueeze(0)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def _unique_shared_from_masks(
    m_unique: np.ndarray,
    k: int,
    m_shared: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_k_hw, shared_hw) pixel arrays for a given class index k.

    Falls back to approximating shared as min(unique_k, max(other_uniques))
    when no explicit shared mask exists.
    """
    mk = m_unique[k] if m_unique.ndim == 3 else m_unique
    if m_shared is not None:
        return mk, m_shared
    K = m_unique.shape[0]
    others = np.stack([m_unique[j] for j in range(K) if j != k], axis=0)
    if others.shape[0] == 0:
        return mk, np.zeros_like(mk)
    return mk, np.minimum(mk, others.max(axis=0))


def _apply_hot_cmap(arr_hw: np.ndarray) -> np.ndarray:
    """Apply a 'hot' colormap to a [0,1] array, returns uint8 (H,W,3)."""
    t = np.clip(arr_hw, 0, 1).astype(np.float32)
    r = np.clip(t * 3.0, 0, 1)
    g = np.clip(t * 3.0 - 1.0, 0, 1)
    b = np.clip(t * 3.0 - 2.0, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _composite_heatmap(img_hwc: np.ndarray, heat_hw: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
    """Blend a heatmap (hot colormap) onto an HWC float image -> uint8 HWC."""
    img_u8 = (img_hwc * 255).clip(0, 255).astype(np.float32)
    norm = heat_hw / (heat_hw.max() + 1e-8)
    colored = _apply_hot_cmap(norm).astype(np.float32)
    blended = ((1 - alpha) * img_u8 + alpha * colored).clip(0, 255).astype(np.uint8)
    return blended


def _composite_overlay(img_hwc: np.ndarray,
                        unique_hw: np.ndarray,
                        shared_hw: np.ndarray) -> np.ndarray:
    """Blend the unique/shared RGBA overlay onto an HWC float image -> uint8 HWC."""
    img_f = (img_hwc * 255).clip(0, 255).astype(np.float32)
    rgba = overlay_rgba(unique_hw, shared_hw)
    a = rgba[..., 3:4]
    color = rgba[..., :3] * 255
    blended = (img_f * (1 - a) + color * a).clip(0, 255).astype(np.uint8)
    return blended


def _to_uint8(img_hwc: np.ndarray) -> np.ndarray:
    return (img_hwc * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plotly figure-level visualization functions
# ---------------------------------------------------------------------------

def show_evidence_heatmap(
    x: torch.Tensor,
    evidence: torch.Tensor,
    hypothesis_ids: torch.Tensor,
    unit_space,
    class_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    alpha: float = 0.5,
    title: str = "Evidence Heatmaps",
    out_path: Optional[Path] = None,
):
    """Evidence heatmaps (hot colormap) overlaid on the input image.

    Args:
        x:              (B, C, H, W) input tensor.
        evidence:       (B, K, R) evidence tensor from base evidence provider.
        hypothesis_ids: (B, K) integer class IDs.
        unit_space:     VisionGridUnitSpace instance.
        class_names:    Optional list of string class labels.
        sample_idx:     Which sample in the batch to visualize.
        alpha:          Heatmap overlay opacity.
        title:          Figure title.
        out_path:       If given, save as HTML to this path.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    B, C, H, W = x.shape
    gh, gw = unit_space.grid_h, unit_space.grid_w
    img = _to_hwc(x[sample_idx])
    ev  = evidence[sample_idx].detach().cpu()
    ids = hypothesis_ids[sample_idx].detach().cpu()
    K   = ev.shape[0]

    n_cols = K + 1
    labels = ["Input"] + [
        (class_names[int(ids[k])] if class_names and int(ids[k]) < len(class_names)
         else str(int(ids[k])))
        for k in range(K)
    ]

    fig = make_subplots(rows=1, cols=n_cols,
                        subplot_titles=labels,
                        horizontal_spacing=0.02)

    fig.add_trace(go.Image(z=_to_uint8(img), name="Input"), row=1, col=1)

    for k in range(K):
        ev_k = mask_to_image(ev[k], gh, gw, H, W)
        composited = _composite_heatmap(img, ev_k, alpha=alpha)
        fig.add_trace(go.Image(z=composited, name=labels[k + 1]), row=1, col=k + 2)

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(title_text=title, title_x=0.5,
                      height=320, width=300 * n_cols,
                      margin=dict(l=10, r=10, t=60, b=10))

    if out_path is not None:
        fig.write_html(str(out_path))
    return fig


def show_explanation_gallery(
    x: torch.Tensor,
    explanation,
    unit_space,
    class_names: Optional[List[str]] = None,
    probs: Optional[torch.Tensor] = None,
    n_samples: Optional[int] = None,
    max_classes: int = 3,
    title: str = "CDEA Contrastive Explanation Gallery",
    out_path: Optional[Path] = None,
):
    """Multi-sample gallery: Input | Evidence heatmaps | Unique/Shared overlays.

    Each row = one image from the batch.
    Columns = [Input] [Evidence × K] [Overlay × K]

    Color code: green = unique evidence, red = shared, yellow = both.

    Args:
        x:           (B, C, H, W) input tensor.
        explanation: Explanation object from CDEAExplainer.explain().
        unit_space:  VisionGridUnitSpace instance.
        class_names: Optional string labels.
        probs:       (B, num_classes) softmax probabilities.
        n_samples:   How many samples to show (default: all).
        max_classes: Max top classes per image.
        title:       Figure title.
        out_path:    If given, save as HTML to this path.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_cpu = x.detach().cpu()
    B, C, H, W = x_cpu.shape
    gh, gw = unit_space.grid_h, unit_space.grid_w
    n = min(B, n_samples or B)
    K = min(explanation.masks["unique"].shape[1], max_classes)

    m_unique_all = explanation.masks["unique"].detach().cpu()
    m_shared_all = explanation.masks.get("shared")
    if m_shared_all is not None:
        m_shared_all = m_shared_all.detach().cpu()
    evidence_all = explanation.extras.get("evidence")
    if evidence_all is not None:
        evidence_all = evidence_all.detach().cpu()
    hyp_ids_all = explanation.hypotheses.ids.detach().cpu()
    if probs is not None:
        probs = probs.detach().cpu()

    n_cols = 1 + K + K
    col_titles_row0 = (
        ["Input"]
        + [f"Evidence #{k+1}" for k in range(K)]
        + [f"Overlay #{k+1}" for k in range(K)]
    )
    # subplot_titles is a flat list: row by row
    subplot_titles = []
    for b in range(n):
        for c_idx in range(n_cols):
            if b == 0:
                subplot_titles.append(col_titles_row0[c_idx])
            else:
                subplot_titles.append("")

    fig = make_subplots(rows=n, cols=n_cols,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.01,
                        vertical_spacing=0.06)

    for b in range(n):
        img = _to_hwc(x_cpu[b])
        m_unique_b = mask_to_image(m_unique_all[b], gh, gw, H, W)  # (K, H, W)
        m_shared_b = (mask_to_image(m_shared_all[b], gh, gw, H, W)
                      if m_shared_all is not None else None)

        # Col 0: input image with prediction label
        label_input = "Input"
        if probs is not None:
            pred_id = int(probs[b].argmax().item())
            lbl = (class_names[pred_id] if class_names and pred_id < len(class_names)
                   else str(pred_id))
            label_input = f"{lbl} ({float(probs[b, pred_id]):.2f})"
        fig.add_trace(
            go.Image(z=_to_uint8(img), name=label_input,
                     hovertemplate=f"<b>{label_input}</b><extra></extra>"),
            row=b + 1, col=1,
        )

        for k in range(K):
            cls_id  = int(hyp_ids_all[b, k].item())
            cls_lbl = (class_names[cls_id] if class_names and cls_id < len(class_names)
                       else str(cls_id))
            cls_p   = float(probs[b, cls_id].item()) if probs is not None else None
            label_k = f"{cls_lbl} ({cls_p:.2f})" if cls_p is not None else cls_lbl

            # Evidence heatmap
            if evidence_all is not None:
                ev_k = mask_to_image(evidence_all[b, k], gh, gw, H, W)
                composited_ev = _composite_heatmap(img, ev_k)
            else:
                composited_ev = _to_uint8(img)
            fig.add_trace(
                go.Image(z=composited_ev, name=label_k,
                         hovertemplate=f"<b>Evidence: {label_k}</b><extra></extra>"),
                row=b + 1, col=2 + k,
            )

            # Unique/shared overlay
            mk, sh = _unique_shared_from_masks(m_unique_b, k, m_shared_b)
            composited_ov = _composite_overlay(img, mk, sh)
            fig.add_trace(
                go.Image(z=composited_ov, name=f"Overlay {label_k}",
                         hovertemplate=f"<b>Overlay: {label_k}</b><extra></extra>"),
                row=b + 1, col=2 + K + k,
            )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Legend annotation
    legend_html = (
        "<span style='color:green'>■ Unique</span>  "
        "<span style='color:red'>■ Shared</span>  "
        "<span style='color:#cccc00'>■ Both</span>"
    )
    fig.update_layout(
        title_text=f"{title}<br><sup>{legend_html}</sup>",
        title_x=0.5,
        height=max(280 * n, 320),
        width=280 * n_cols,
        showlegend=False,
        margin=dict(l=10, r=10, t=80, b=20),
    )

    if out_path is not None:
        fig.write_html(str(out_path))
    return fig


def show_gradcam_vs_ig(
    x: torch.Tensor,
    explanations: Dict[str, object],
    unit_space,
    class_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    max_classes: int = 3,
    title: str = "Grad-CAM vs Integrated Gradients",
    out_path: Optional[Path] = None,
):
    """Side-by-side comparison of two (or more) evidence provider explanations.

    Rows = evidence providers, Columns = [Input] [Evidence × K] [Overlay × K]

    Args:
        x:            (B, C, H, W) input tensor.
        explanations: Dict mapping provider name → Explanation object.
        unit_space:   VisionGridUnitSpace instance.
        class_names:  Optional string labels.
        sample_idx:   Which sample in the batch to visualize.
        max_classes:  Max number of top classes to show.
        title:        Figure title.
        out_path:     If given, save as HTML to this path.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_cpu = x.detach().cpu()
    B, C, H, W = x_cpu.shape
    gh, gw = unit_space.grid_h, unit_space.grid_w
    img = _to_hwc(x_cpu[sample_idx])

    provider_names = list(explanations.keys())
    n_providers = len(provider_names)
    first_expl = next(iter(explanations.values()))
    K = min(first_expl.masks["unique"].shape[1], max_classes)
    hyp_ids = first_expl.hypotheses.ids[sample_idx].detach().cpu()

    n_cols = 1 + K + K
    col_titles = (
        ["Input"]
        + [f"Evidence: {class_names[int(hyp_ids[k])] if class_names and int(hyp_ids[k]) < len(class_names) else str(int(hyp_ids[k]))}" for k in range(K)]
        + [f"Overlay: {class_names[int(hyp_ids[k])] if class_names and int(hyp_ids[k]) < len(class_names) else str(int(hyp_ids[k]))}" for k in range(K)]
    )

    subplot_titles = []
    for p_idx in range(n_providers):
        for c_idx in range(n_cols):
            if p_idx == 0:
                subplot_titles.append(col_titles[c_idx])
            else:
                subplot_titles.append("")

    fig = make_subplots(rows=n_providers, cols=n_cols,
                        subplot_titles=subplot_titles,
                        row_titles=provider_names,
                        horizontal_spacing=0.01,
                        vertical_spacing=0.08)

    for row, pname in enumerate(provider_names):
        expl = explanations[pname]
        m_unique = expl.masks["unique"][sample_idx].detach().cpu()
        m_shared_t = expl.masks.get("shared")
        m_shared_b = (mask_to_image(m_shared_t[sample_idx].detach().cpu(), gh, gw, H, W)
                      if m_shared_t is not None else None)
        m_unique_b = mask_to_image(m_unique, gh, gw, H, W)
        ev_t = expl.extras.get("evidence")
        ev_b = ev_t[sample_idx].detach().cpu() if ev_t is not None else None

        fig.add_trace(
            go.Image(z=_to_uint8(img), name="Input",
                     hovertemplate=f"<b>{pname}: Input</b><extra></extra>"),
            row=row + 1, col=1,
        )

        for k in range(K):
            cls_id  = int(hyp_ids[k].item())
            cls_lbl = (class_names[cls_id] if class_names and cls_id < len(class_names)
                       else str(cls_id))

            # Evidence
            if ev_b is not None:
                ev_k = mask_to_image(ev_b[k], gh, gw, H, W)
                composited_ev = _composite_heatmap(img, ev_k)
            else:
                composited_ev = _to_uint8(img)
            fig.add_trace(
                go.Image(z=composited_ev,
                         hovertemplate=f"<b>{pname}: {cls_lbl}</b><extra></extra>"),
                row=row + 1, col=2 + k,
            )

            # Overlay
            mk, sh = _unique_shared_from_masks(m_unique_b, k, m_shared_b)
            composited_ov = _composite_overlay(img, mk, sh)
            fig.add_trace(
                go.Image(z=composited_ov,
                         hovertemplate=f"<b>{pname}: overlay {cls_lbl}</b><extra></extra>"),
                row=row + 1, col=2 + K + k,
            )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    legend_html = (
        "<span style='color:green'>■ Unique</span>  "
        "<span style='color:red'>■ Shared</span>  "
        "<span style='color:#cccc00'>■ Both</span>"
    )
    fig.update_layout(
        title_text=f"{title}<br><sup>{legend_html}</sup>",
        title_x=0.5,
        height=max(300 * n_providers, 320),
        width=280 * n_cols,
        showlegend=False,
        margin=dict(l=60, r=10, t=80, b=20),
    )

    if out_path is not None:
        fig.write_html(str(out_path))
    return fig


def show_probability_split(
    explanation,
    class_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    title: str = "Probability Split: Shared vs Shared+Unique",
    out_path: Optional[Path] = None,
):
    """Grouped bar chart of shared-only vs shared+unique softmax probabilities.

    Left panel:  Grouped bars per class (shared vs shared+unique).
    Right panel: Δ probability (unique contribution) bar chart.

    Args:
        explanation: Explanation object containing split probability metrics.
        class_names: Optional string labels.
        sample_idx:  Which sample to visualize.
        title:       Figure title.
        out_path:    If given, save as HTML to this path.

    Returns:
        plotly.graph_objects.Figure, or None if metrics are absent.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    split_shared = explanation.metrics.get("split_shared_only_probs_topm")
    split_plus   = explanation.metrics.get("split_shared_plus_unique_probs_topm")
    if split_shared is None or split_plus is None:
        print("Probability split metrics not available. "
              "Enable use_shared=True in the allocator.")
        return None

    hyp_ids  = explanation.hypotheses.ids[sample_idx].detach().cpu()
    hyp_mask = explanation.hypotheses.mask[sample_idx].detach().cpu()
    valid_k  = int(hyp_mask.sum().item())
    sh0 = split_shared[sample_idx].detach().cpu()
    pl0 = split_plus[sample_idx].detach().cpu()

    labels_k, p_shared_vals, p_plus_vals, delta_vals = [], [], [], []
    for k in range(valid_k):
        cls_id = int(hyp_ids[k].item())
        lbl = (class_names[cls_id] if class_names and 0 <= cls_id < len(class_names)
               else str(cls_id))
        labels_k.append(lbl)
        ps = float(sh0[k].item())
        pp = float(pl0[k].item())
        p_shared_vals.append(ps)
        p_plus_vals.append(pp)
        delta_vals.append(pp - ps)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Shared vs Shared+Unique", "Unique Evidence Contribution (Δ)"),
        horizontal_spacing=0.12,
    )

    # Left: grouped bars
    fig.add_trace(go.Bar(
        name="Shared only",
        x=labels_k,
        y=p_shared_vals,
        marker_color="#d73027",
        opacity=0.85,
        text=[f"{v:.3f}" for v in p_shared_vals],
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name="Shared + Unique",
        x=labels_k,
        y=p_plus_vals,
        marker_color="#1a9850",
        opacity=0.85,
        text=[f"{v:.3f}" for v in p_plus_vals],
        textposition="outside",
    ), row=1, col=1)

    # Right: delta bars
    delta_colors = ["#1a9850" if d >= 0 else "#d73027" for d in delta_vals]
    fig.add_trace(go.Bar(
        name="Δ (unique contribution)",
        x=labels_k,
        y=delta_vals,
        marker_color=delta_colors,
        opacity=0.85,
        text=[f"{v:+.4f}" for v in delta_vals],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=0, line_color="black", line_width=0.8, row=1, col=2)

    fig.update_yaxes(title_text="Softmax probability", row=1, col=1,
                     gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(title_text="Δ probability", row=1, col=2,
                     zeroline=True, zerolinecolor="black", zerolinewidth=1,
                     gridcolor="rgba(0,0,0,0.1)")
    fig.update_xaxes(tickangle=-20)
    fig.update_layout(
        title_text=title, title_x=0.5,
        barmode="group",
        height=420, width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.25),
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=60),
    )

    if out_path is not None:
        fig.write_html(str(out_path))
    return fig


# ---------------------------------------------------------------------------
# Feature extraction utility
# ---------------------------------------------------------------------------

def extract_layer_activations(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: Optional[Any] = None,
) -> torch.Tensor:
    """Run a forward pass and return activations at ``target_layer``.

    Auto-detects the target layer using the same logic as GradCAMRegionsProvider:
      - ViT (torchvision VisionTransformer): ``model.encoder.layers[-1]``
      - CNN: last ``Conv2d`` layer

    Args:
        model:        Eval-mode ``nn.Module``.
        x:            ``(B, C, H, W)`` input tensor on the correct device.
        target_layer: Explicit layer to hook. If None, auto-detected.

    Returns:
        Activation tensor detached to CPU.
        CNN shape:  ``(B, C, h, w)``
        ViT shape:  ``(B, seq_len, dim)``  (index 0 is the CLS token)
    """
    if target_layer is None:
        try:
            from base_evidence.gradcam_regions import _find_target_layer
            target_layer = _find_target_layer(model)
        except Exception as e:
            raise ValueError(
                f"Could not auto-detect target layer: {e}. "
                "Pass target_layer explicitly."
            )

    captured: List[torch.Tensor] = []

    def _hook(_m: Any, _inp: Any, out: Any) -> None:
        captured.append(out.detach().cpu())

    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Forward hook did not capture any activations.")
    return captured[0]


# ---------------------------------------------------------------------------
# Evidence region bar chart
# ---------------------------------------------------------------------------

def show_evidence_region_bars(
    evidence: torch.Tensor,
    hypothesis_ids: torch.Tensor,
    class_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    grid_h: Optional[int] = None,
    grid_w: Optional[int] = None,
    title: str = "Evidence Distribution per Region",
    out_path: Optional[Path] = None,
):
    """Grouped bar chart of per-region evidence for each class hypothesis.

    The spatial heatmap smooths over region boundaries; this chart shows the
    exact numeric evidence mass at each grid cell so inter-class overlap is
    immediately visible.  Bars that heavily overlap across classes indicate
    ambiguous (shared) evidence; non-overlapping bars indicate well-separated
    (unique) evidence — what CDEA aims to maximise.

    Args:
        evidence:       ``(B, K, R)`` evidence tensor from the base provider.
        hypothesis_ids: ``(B, K)`` integer class IDs per sample.
        class_names:    Optional list of string class labels.
        sample_idx:     Which sample in the batch to visualize.
        grid_h, grid_w: Grid dimensions — used to label regions as (row,col)
                        instead of a flat index. Optional.
        title:          Figure title.
        out_path:       If given, save as HTML.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ev  = evidence[sample_idx].detach().cpu().numpy()        # (K, R)
    ids = hypothesis_ids[sample_idx].detach().cpu()
    K, R = ev.shape

    if grid_h and grid_w and grid_h * grid_w == R:
        region_labels = [f"({r // grid_w},{r % grid_w})" for r in range(R)]
    else:
        region_labels = [str(r) for r in range(R)]

    # Panel 1: per-class bar chart.  Panel 2: overlap heatmap.
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Per-class evidence per region",
                        "Class × Region evidence heatmap"),
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.10,
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f"]
    for k in range(K):
        cls_id = int(ids[k].item())
        lbl = (class_names[cls_id]
               if class_names and 0 <= cls_id < len(class_names)
               else str(cls_id))
        fig.add_trace(go.Bar(
            name=lbl,
            x=region_labels,
            y=ev[k].tolist(),
            marker_color=palette[k % len(palette)],
            opacity=0.80,
        ), row=1, col=1)

    # Panel 2: K × R heatmap for a compact view of all classes at once
    class_labels = []
    for k in range(K):
        cls_id = int(ids[k].item())
        class_labels.append(
            class_names[cls_id]
            if class_names and 0 <= cls_id < len(class_names)
            else str(cls_id)
        )
    fig.add_trace(go.Heatmap(
        z=ev,
        x=region_labels,
        y=class_labels,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(len=0.6, thickness=12, x=1.01),
        hovertemplate="Class: %{y}<br>Region: %{x}<br>Evidence: %{z:.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9), row=1, col=1)
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9), row=1, col=2)
    fig.update_yaxes(title_text="Evidence", row=1, col=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(
        title_text=title, title_x=0.5,
        barmode="group",
        height=420, width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0.1),
        plot_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=80),
    )

    if out_path is not None:
        fig.write_html(str(out_path))
    return fig


# ---------------------------------------------------------------------------
# Feature activation channel heatmaps
# ---------------------------------------------------------------------------

def show_feature_activations(
    x: torch.Tensor,
    activations: torch.Tensor,
    unit_space: Any,
    sample_idx: int = 0,
    n_channels: int = 6,
    alpha: float = 0.55,
    title: str = "Feature Activations at Target Layer",
    out_path: Optional[Path] = None,
):
    """Top-N feature channel heatmaps composited on the input image.

    Shows the raw activation map at the GradCAM target layer **before** any
    gradient weighting.  This reveals what the model has encoded at the
    classification-critical layer, independent of any class signal — useful
    for diagnosing whether the backbone is encoding the right content.

    For ViT, the CLS token is dropped and the 2D patch grid is reconstructed
    from the sequence dimension before selecting top channels.

    Args:
        x:           ``(B, C, H, W)`` input tensor (CPU or GPU).
        activations: Output of ``extract_layer_activations``.
                     CNN: ``(B, C, h, w)``  |  ViT: ``(B, seq_len, dim)``
        unit_space:  ``VisionGridUnitSpace`` instance (provides grid_h, grid_w).
        sample_idx:  Which sample in the batch to visualize.
        n_channels:  How many top-activation channels to display.
        alpha:       Heatmap overlay opacity.
        title:       Figure title.
        out_path:    If given, save as HTML.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    gh, gw = unit_space.grid_h, unit_space.grid_w
    x_cpu = x.detach().cpu()
    B, C, H, W = x_cpu.shape
    img = _to_hwc(x_cpu[sample_idx])

    A = activations[sample_idx]  # CNN: (C, h, w) | ViT: (seq_len, dim)

    if A.dim() == 2:
        # ViT: drop CLS token (index 0), reshape to 2D spatial
        A = A[1:, :]  # (num_patches, dim)
        num_patches = A.shape[0]
        ph = pw = int(math.isqrt(num_patches))
        if ph * pw != num_patches:
            ph, pw = 1, num_patches
        A = A.T.reshape(-1, ph, pw)  # (dim, ph, pw)

    # Select top-n channels by mean absolute activation magnitude
    n_top = min(n_channels, A.shape[0])
    channel_scores = A.abs().mean(dim=(1, 2))   # (C,)
    top_indices = channel_scores.topk(n_top).indices.tolist()

    n_cols = n_top + 1  # +1 for original image
    col_titles = ["Input"] + [f"Ch {idx}\n(score={channel_scores[idx]:.3f})"
                               for idx in top_indices]

    fig = make_subplots(rows=1, cols=n_cols,
                        subplot_titles=col_titles,
                        horizontal_spacing=0.02)

    fig.add_trace(go.Image(z=_to_uint8(img), name="Input"), row=1, col=1)

    for i, ch_idx in enumerate(top_indices):
        ch_map = A[ch_idx].float().numpy()  # (h, w) — may be low-res
        # Upsample to input resolution
        ch_up = mask_to_image(torch.from_numpy(ch_map), A.shape[1], A.shape[2], H, W)
        # Shift to [0, 1] (activations can be negative after residual adds)
        ch_up = ch_up - ch_up.min()
        ch_up = ch_up / (ch_up.max() + 1e-8)
        composited = _composite_heatmap(img, ch_up, alpha=alpha)
        fig.add_trace(
            go.Image(z=composited,
                     hovertemplate=f"<b>Channel {ch_idx}</b><extra></extra>"),
            row=1, col=i + 2,
        )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(
        title_text=title, title_x=0.5,
        height=320, width=300 * n_cols,
        margin=dict(l=10, r=10, t=80, b=10),
        showlegend=False,
    )

    if out_path is not None:
        fig.write_html(str(out_path))
    return fig


def show_pairwise_margin(
    explanation,
    class_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    title: str = "Pairwise Margin Heatmap",
    out_path: Optional[Path] = None,
):
    """Heatmap of 'why class k rather than l?' contrastive margins.

    Left panel:  Shared+unique margins  (z_k − z_l  under  keep(m_shared + m_unique[k]))
    Right panel: Margin delta           (unique contribution = shared+unique − shared_only)

    Positive (green) = class k wins over l under its own mask.
    Negative (red)   = mask fails to discriminate k from l.

    Args:
        explanation: Explanation object with pairwise margin metrics.
        class_names: Optional string labels.
        sample_idx:  Which sample to visualize.
        title:       Figure title.
        out_path:    If given, save as HTML to this path.

    Returns:
        plotly.graph_objects.Figure, or None if metrics are absent.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pair_plus  = explanation.metrics.get("pairwise_margin_shared_plus_unique_logits_topm")
    pair_delta = explanation.metrics.get("pairwise_margin_delta_logits_topm")
    if pair_plus is None:
        print("Pairwise margin metrics not available.")
        return None

    hyp_ids  = explanation.hypotheses.ids[sample_idx].detach().cpu()
    hyp_mask = explanation.hypotheses.mask[sample_idx].detach().cpu()
    valid_m  = int(hyp_mask.sum().item())
    pp = pair_plus[sample_idx, :valid_m, :valid_m].detach().cpu().numpy()
    pd_arr = (pair_delta[sample_idx, :valid_m, :valid_m].detach().cpu().numpy()
              if pair_delta is not None else None)

    labels_m = [
        (class_names[int(hyp_ids[k].item())]
         if class_names and int(hyp_ids[k].item()) < len(class_names)
         else str(int(hyp_ids[k].item())))
        for k in range(valid_m)
    ]

    n_panels = 2 if pd_arr is not None else 1
    fig = make_subplots(
        rows=1, cols=n_panels,
        subplot_titles=(
            ["Shared+unique margin<br>(z_k − z_l  under  keep(m_shared + m_unique[k]))",
             "Unique contribution<br>(shared+unique) − (shared only)"][:n_panels]
        ),
        horizontal_spacing=0.12,
    )

    def _add_heatmap(col_idx: int, data: np.ndarray):
        vmax = max(float(np.abs(data).max()), 1e-6)
        text_vals = [[
            "" if i == j else f"{data[i, j]:.3f}"
            for j in range(valid_m)
        ] for i in range(valid_m)]
        fig.add_trace(go.Heatmap(
            z=data,
            x=labels_m,
            y=labels_m,
            colorscale="RdYlGn",
            zmin=-vmax, zmax=vmax,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(len=0.8, thickness=15,
                          x=0.46 if col_idx == 1 else 1.0),
            hovertemplate="Class k: %{y}<br>Class l: %{x}<br>Margin: %{z:.4f}<extra></extra>",
        ), row=1, col=col_idx)

    _add_heatmap(1, pp)
    if pd_arr is not None:
        _add_heatmap(2, pd_arr)

    fig.update_yaxes(title_text="Class k (explained)", row=1, col=1,
                     autorange="reversed")
    if n_panels == 2:
        fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Class l (foil)", tickangle=-30)
    fig.update_layout(
        title_text=title, title_x=0.5,
        height=420,
        width=480 * n_panels,
        margin=dict(l=80, r=20, t=80, b=80),
        plot_bgcolor="white",
    )

    if out_path is not None:
        fig.write_html(str(out_path))
    return fig
