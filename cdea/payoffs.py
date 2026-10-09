"""Log-odds payoffs. FORMULATION.md sections 3 and 4.

``s_H`` is the log-odds that the answer lies in H.
``c_k`` is the log-odds of k within H.
A player's payoff is the log-odds it loses when its units are deleted.
When H is every class, ``s_H`` is undefined and there is no shared player.
"""

from __future__ import annotations

import torch

from core.types import HypothesisSet


def unique_log_odds(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``c_k = z_k - LSE_{j in H without k} z_j``. ``ids`` is ``(B, K)``. Returns ``(B, K)``."""
    if ids.ndim != 2:
        raise ValueError("ids must be (batch, hypotheses)")
    if ids.shape[1] < 2:
        raise ValueError("unique log-odds need at least two hypotheses")
    gathered = logits.gather(1, ids.long())
    others = gathered.unsqueeze(1).expand(-1, ids.shape[1], -1).clone()
    eye = torch.eye(ids.shape[1], device=logits.device, dtype=torch.bool)
    others = others.masked_fill(eye.unsqueeze(0), float("-inf"))
    return gathered - torch.logsumexp(others, dim=-1)


def shared_defined(logits: torch.Tensor, ids: torch.Tensor) -> bool:
    """False when H covers every class, so there is nothing outside H."""
    return int(ids.shape[1]) < int(logits.shape[1])


def shared_log_odds(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``s_H = LSE_{j in H} z_j - LSE_{j not in H} z_j``. Returns ``(B,)``."""
    if not shared_defined(logits, ids):
        raise ValueError("shared log-odds are undefined when H covers every class")
    inside = torch.full_like(logits, float("-inf"))
    inside.scatter_(1, ids.long(), logits.gather(1, ids.long()))
    outside = logits.scatter(1, ids.long(), float("-inf"))
    return torch.logsumexp(inside, dim=-1) - torch.logsumexp(outside, dim=-1)


def deletion_payoffs(
    logits_full: torch.Tensor,
    logits_deleted: torch.Tensor,
    hypotheses: HypothesisSet,
    *,
    shared: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``u_k = c_k(x) - c_k(x deleted)``, and the same for ``s_H`` when ``shared``.

    ``logits_deleted`` is ``(B, P, C)`` with one deleted input per player, players
    ordered as the unique hypotheses and then the shared player.
    """
    ids = hypotheses.ids.long()
    unique_full = unique_log_odds(logits_full, ids)
    players = logits_deleted.shape[1]
    unique_count = ids.shape[1]
    deleted_unique = []
    for k in range(unique_count):
        deleted_unique.append(unique_log_odds(logits_deleted[:, k], ids)[:, k])
    unique_payoff = unique_full - torch.stack(deleted_unique, dim=1)
    shared_payoff = None
    if shared:
        if players != unique_count + 1:
            raise ValueError("shared payoff expects one extra deleted input")
        full_s = shared_log_odds(logits_full, ids)
        deleted_s = shared_log_odds(logits_deleted[:, -1], ids)
        shared_payoff = full_s - deleted_s
    return unique_payoff, shared_payoff
