"""Counterfactual visual explanations by greedy cell replacement.

This is Algorithm 1 of Goyal et al., ICML 2019. A spatial cell of the query
feature map is replaced by a cell of the distractor feature map. The pair is
the one that most raises the distractor's log-probability. That step repeats
until the decision network predicts the distractor class. Query cells that
have already been replaced are left out of later steps. A distractor cell may
be copied again.

The search does not change the classifier. ``decision`` is the network that
sits on top of the spatial feature map and returns logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

Decision = Callable[[torch.Tensor], torch.Tensor]
LogProb = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CVEEdits:
    """Edits in the order they were applied.

    ``rank`` is ``(H, W)``. Entry 1 is the first query cell replaced. A zero
    means that cell was not replaced. ``flipped`` is true when the decision
    network's argmax is the distractor class after these edits.
    """

    query_index: torch.Tensor
    source_index: torch.Tensor
    rank: torch.Tensor
    flipped: bool


def greedy_edits(
    query: torch.Tensor,
    distractor: torch.Tensor,
    decision: Decision,
    distractor_class: int,
    log_probs: LogProb,
    max_edits: int | None = None,
) -> CVEEdits:
    """Run the greedy search on one pair of feature maps.

    ``query`` and ``distractor`` are ``(C, H, W)``. ``log_probs(current, free)``
    returns the distractor log-probability for every free query cell and every
    distractor cell, shaped ``(len(free), H * W)``. Ties keep the earliest
    query index, then the earliest source index.
    """
    _check_maps(query, distractor)
    _, height, width = query.shape
    cells = height * width
    if max_edits is None:
        limit = cells
    else:
        limit = int(max_edits)
        if limit < 1:
            raise ValueError("max_edits must be >= 1")
        limit = min(limit, cells)
    target = int(distractor_class)
    current = query.detach().clone()
    free = torch.arange(cells, device=current.device)
    chosen_query: list[int] = []
    chosen_source: list[int] = []
    with torch.no_grad():
        if _argmax(decision, current) == target:
            return _pack(chosen_query, chosen_source, height, width, True, current.device)
        for _step in range(limit):
            scores = log_probs(current, free)
            if scores.shape != (free.numel(), cells):
                raise ValueError(
                    f"log_probs returned {tuple(scores.shape)}, expected {(free.numel(), cells)}"
                )
            if not torch.isfinite(scores).all():
                raise ValueError("log_probs returned a non-finite score")
            flat = int(torch.argmax(scores).item())
            row, source = divmod(flat, cells)
            query_index = int(free[row].item())
            y, x = divmod(query_index, width)
            sy, sx = divmod(source, width)
            current[:, y, x] = distractor[:, sy, sx]
            chosen_query.append(query_index)
            chosen_source.append(source)
            free = torch.cat((free[:row], free[row + 1 :]))
            if _argmax(decision, current) == target:
                return _pack(chosen_query, chosen_source, height, width, True, current.device)
    return _pack(chosen_query, chosen_source, height, width, False, current.device)


def earliest_edits(rank: torch.Tensor, max_cells: int) -> torch.Tensor:
    """Keep the earliest query edits. Cells the search never touched stay off."""
    if int(max_cells) < 1:
        raise ValueError("max_cells must be >= 1")
    return ((rank > 0) & (rank <= int(max_cells))).to(dtype=torch.float32)


def replacement_log_probs(
    distractor: torch.Tensor,
    decision: Decision,
    distractor_class: int,
) -> LogProb:
    """Score every candidate by editing the map and calling ``decision``.

    This is the search in Algorithm 1. It materialises one map per candidate,
    so it is the reference for a feature grid the size of the paper's (7×7)
    and for the agreement check. A 32×32 grid should use
    ``gap_linear_log_probs`` when the head is global average pooling plus a
    linear layer; that scorer is the same replacement, evaluated in closed form.
    """
    _check_map(distractor)
    channels, height, width = distractor.shape
    cells = height * width
    sources = distractor.detach().reshape(channels, cells).T.contiguous()
    target = int(distractor_class)

    def log_probs(current: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
        count = int(free.numel())
        if count == 0:
            return current.new_empty((0, cells))
        maps = current.detach().reshape(1, 1, channels, height, width).expand(
            count, cells, channels, height, width
        ).clone()
        ys = torch.div(free, width, rounding_mode="floor")
        xs = free.remainder(width)
        for row in range(count):
            maps[row, :, :, int(ys[row].item()), int(xs[row].item())] = sources
        logits = decision(maps.reshape(count * cells, channels, height, width))
        if logits.ndim != 2 or logits.shape[0] != count * cells:
            raise ValueError("decision must return logits shaped (N, K)")
        return torch.log_softmax(logits, dim=-1)[:, target].reshape(count, cells)

    return log_probs


def gap_linear_log_probs(
    distractor: torch.Tensor,
    linear: nn.Linear,
    distractor_class: int,
) -> LogProb:
    """Same scores as ``replacement_log_probs`` when the head is pool-then-linear.

    Replacing cell q with source s changes the pooled vector by
    ``(source - current_q) / (H * W)``. The linear layer is applied to that
    vector. No map is built.
    """
    _check_map(distractor)
    if linear.bias is None:
        raise ValueError("the decision linear layer needs a bias to match the toy head")
    channels, height, width = distractor.shape
    cells = height * width
    sources = distractor.detach().reshape(channels, cells)
    weight = linear.weight.detach()
    bias = linear.bias.detach()
    target = int(distractor_class)

    def log_probs(current: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
        count = int(free.numel())
        if count == 0:
            return current.new_empty((0, cells))
        flat = current.detach().reshape(channels, cells)
        total = flat.sum(dim=1)
        kept = flat[:, free]
        pooled = (total[:, None, None] - kept[:, :, None] + sources[:, None, :]) / cells
        logits = torch.einsum("cqs,kc->qsk", pooled, weight) + bias
        return torch.log_softmax(logits, dim=-1)[..., target]

    return log_probs


def _argmax(decision: Decision, features: torch.Tensor) -> int:
    logits = decision(features.unsqueeze(0))
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("decision must return logits shaped (N, K)")
    return int(torch.argmax(logits, dim=-1).item())


def _pack(
    query_index: list[int],
    source_index: list[int],
    height: int,
    width: int,
    flipped: bool,
    device: torch.device,
) -> CVEEdits:
    rank = torch.zeros(height, width, dtype=torch.long, device=device)
    for step, index in enumerate(query_index, start=1):
        y, x = divmod(index, width)
        rank[y, x] = step
    return CVEEdits(
        query_index=torch.tensor(query_index, dtype=torch.long, device=device),
        source_index=torch.tensor(source_index, dtype=torch.long, device=device),
        rank=rank,
        flipped=flipped,
    )


def _check_maps(query: torch.Tensor, distractor: torch.Tensor) -> None:
    _check_map(query)
    _check_map(distractor)
    if query.shape != distractor.shape:
        raise ValueError("query and distractor feature maps must have the same shape")


def _check_map(features: torch.Tensor) -> None:
    if features.ndim != 3 or features.shape[1] < 1 or features.shape[2] < 1:
        raise ValueError("a feature map is (C, H, W)")
    if not torch.isfinite(features).all():
        raise ValueError("a feature map has a non-finite value")
