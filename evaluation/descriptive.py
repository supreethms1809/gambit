"""Descriptive measurements outside the method. FORMULATION_DECOMPOSITION.md v2 section 10, open decision 10.

**Co-location share.** At ResNet-50's last layer the head is a global average
pool and a linear layer, so unit (r, c) contributes ``w_kc A_rc / R`` to z_k.
Shared evidence computed per position, ``P = sum_r min(c_k(r)+, c_l(r)+)``
with ``c_j(r) = sum_c w_jc A_rc / R``, is compared with the same computed per
position and channel, ``Q = sum_{r,c} min((w_kc A_rc)+, (w_lc A_rc)+) / R``.
``(P - Q) / P`` is the share of positionally shared evidence that is unique
evidence of the two classes in the same place. As written in the formulation,
``P`` nets each position's channels before the min, so the share can be
negative when channels of opposite sign cancel inside a position. ``P+`` takes
the positive channels of each class instead, ``sum_r min(sum_c (w_kc A_rc)+,
sum_c (w_lc A_rc)+) / R``; then ``Q <= P+`` and ``(P+ - Q) / P+`` is in [0, 1].

**Pairwise interaction.** ``I(a, b) = d({a, b}) - d(a) - d(b)`` for pairs of
the top Omega cells, under CDEA's blur. The head adds contributions exactly,
so any interaction comes from the trunk. ``I > 0`` is redundancy (each cell
backs the other up); ``I < 0`` is synergy.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from core.grid import delete


def head_contributions(model: nn.Module, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``(B, J, C, R)``: ``w_jc A_rc / R`` at a torchvision ResNet's last layer, for the classes ``ids``.

    The contributions and the bias add up to the logits; this is checked.
    """
    from models.wrapper import unwrap

    net = unwrap(model)
    if not (hasattr(net, "layer4") and isinstance(getattr(net, "fc", None), nn.Linear)):
        raise ValueError("the co-location share needs a ResNet with a linear head")
    store = {}
    handle = net.layer4.register_forward_hook(lambda _m, _i, out: store.__setitem__("a", out.detach()))
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        handle.remove()
    a = store["a"].flatten(2)                                  # (B, C, R)
    weight = net.fc.weight.detach()[ids.long()]                # (B, J, C)
    contrib = weight.unsqueeze(-1) * a.unsqueeze(1) / a.shape[-1]
    total = contrib.sum(dim=(-1, -2)) + net.fc.bias.detach()[ids.long()]
    if not torch.allclose(total, logits.gather(1, ids.long()), atol=1e-3, rtol=1e-4):
        raise RuntimeError("head contributions do not add up to the logits")
    return contrib


def colocation(contrib: torch.Tensor, cells: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """``P``, ``P+``, ``Q`` and both shares from ``(B, 2, C, R)`` contributions of k and l.

    Over ``cells`` (``(B, R)`` bool), or every cell.
    """
    positive = contrib.clamp_min(0)
    per_position = torch.minimum(contrib[:, 0].sum(dim=1).clamp_min(0), contrib[:, 1].sum(dim=1).clamp_min(0))
    per_position_pos = torch.minimum(positive[:, 0].sum(dim=1), positive[:, 1].sum(dim=1))
    per_channel = torch.minimum(positive[:, 0], positive[:, 1]).sum(dim=1)
    if cells is not None:
        per_position = per_position * cells
        per_position_pos = per_position_pos * cells
        per_channel = per_channel * cells
    p = per_position.sum(dim=-1)
    p_pos = per_position_pos.sum(dim=-1)
    q = per_channel.sum(dim=-1)
    nan = torch.full_like(p, float("nan"))
    return {"p": p, "p_pos": p_pos, "q": q,
            "share": torch.where(p > 0, (p - q) / p.clamp_min(1e-12), nan),
            "share_pos": torch.where(p_pos > 0, (p_pos - q) / p_pos.clamp_min(1e-12), nan)}


def pairwise_interaction(model: nn.Module, x: torch.Tensor, ids: torch.Tensor, ranking: torch.Tensor,
                         omega: torch.Tensor, grid_h: int, grid_w: int, top: int = 6) -> dict[str, torch.Tensor]:
    """Interactions of the top Omega cells, ranked by ``ranking`` (``(B, R)``), for k and l.

    Returns ``(B, pairs, 2)`` tensors ``i`` (interaction), ``single`` (``|d(a)| + |d(b)|``) and
    ``joint`` (``d({a, b})``), NaN for pairs an image does not have.
    """
    batch, units = ranking.shape
    pairs = list(itertools.combinations(range(top), 2))
    score = torch.where(omega, ranking, torch.full_like(ranking, float("-inf")))
    order = score.argsort(dim=-1, descending=True)[:, :top]
    count = omega.sum(dim=-1)
    ids = ids[:, :2].long()

    def drops(masks):
        with torch.no_grad():
            return full - model(delete(x, masks, grid_h, grid_w)).gather(1, ids)

    with torch.no_grad():
        full = model(x).gather(1, ids)
    single = []
    for t in range(top):
        mask = torch.zeros(batch, units, device=x.device, dtype=x.dtype)
        mask.scatter_(1, order[:, t:t + 1], 1.0)
        single.append(drops(mask))
    shape = (batch, len(pairs), 2)
    out = {name: torch.full(shape, float("nan"), device=x.device) for name in ("i", "single", "joint")}
    for p, (a, b) in enumerate(pairs):
        mask = torch.zeros(batch, units, device=x.device, dtype=x.dtype)
        mask.scatter_(1, order[:, a:a + 1], 1.0)
        mask.scatter_(1, order[:, b:b + 1], 1.0)
        joint = drops(mask)
        valid = (count > b).unsqueeze(-1)
        nan = torch.full_like(joint, float("nan"))
        out["joint"][:, p] = torch.where(valid, joint, nan)
        out["i"][:, p] = torch.where(valid, joint - single[a] - single[b], nan)
        out["single"][:, p] = torch.where(valid, single[a].abs() + single[b].abs(), nan)
    return out
