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


# ---------------------------------------------------------------------------
# Decomposition game (FORMULATION_DECOMPOSITION.md section 3). Logit units.
# Players are ordered U_k, U_l, S. The pair is k = rank 0, l = rank 1.
# ---------------------------------------------------------------------------

SMOOTH_EPS = 1e-3


def smooth_abs(t: torch.Tensor, eps: float = SMOOTH_EPS) -> torch.Tensor:
    """``|t|_s = sqrt(t^2 + eps^2)``. At most ``eps`` above ``|t|``."""
    return torch.sqrt(t * t + float(eps) ** 2)


def smooth_min(a: torch.Tensor, b: torch.Tensor, eps: float = SMOOTH_EPS) -> torch.Tensor:
    """``(a + b) / 2 - |a - b|_s / 2``. At most ``eps / 2`` below ``min(a, b)``."""
    return 0.5 * (a + b) - 0.5 * smooth_abs(a - b, eps)


def class_drops(logits_full: torch.Tensor, logits_deleted: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``d_j(A_p) = z_j(x) - z_j(x deleted by A_p)``.

    ``logits_full`` is ``(B, C)``, ``logits_deleted`` is ``(B, P, C)`` and
    ``ids`` is ``(B, J)``. Returns ``(B, P, J)``.
    """
    ids = ids.long()
    full = logits_full.gather(1, ids).unsqueeze(1)
    deleted = logits_deleted.gather(2, ids.unsqueeze(1).expand(-1, logits_deleted.shape[1], -1))
    return full - deleted


def decomposition_payoffs(drops: torch.Tensor, null: torch.Tensor | None = None) -> torch.Tensor:
    """``(u_k, u_l, u_S)`` from the ``(B, 3, 2)`` drops of players U_k, U_l, S on classes k, l.

    With ``null``, each drop becomes the excess drop ``d - nu``. A ``(B, 2)``
    null is one area's (version 1); a ``(B, 3, 2)`` null is each player's own,
    at its size (version 2).
      u_k = d~_k(A_k) - |d~_l(A_k)|_s
      u_l = d~_l(A_l) - |d~_k(A_l)|_s
      u_S = min_s(d~_k(A_S), d~_l(A_S))
    Returns ``(B, 3)``.
    """
    if drops.ndim != 3 or drops.shape[1:] != (3, 2):
        raise ValueError("drops must be (batch, 3 players, 2 classes)")
    if null is None:
        excess = drops
    else:
        excess = drops - (null if null.ndim == 3 else null.unsqueeze(1))
    u_k = excess[:, 0, 0] - smooth_abs(excess[:, 0, 1])
    u_l = excess[:, 1, 1] - smooth_abs(excess[:, 1, 0])
    u_s = smooth_min(excess[:, 2, 0], excess[:, 2, 1])
    return torch.stack([u_k, u_l, u_s], dim=1)


# ---------------------------------------------------------------------------
# The cell-level cross penalty (FORMULATION_DECOMPOSITION.md section 4.1,
# drafted 2026-10-22). Version 2 charged U_k for the set-level |d~_l(U_k)|,
# so a cell that lowers z_l could cancel one that raises it, and a shared cell
# could pay its way into U_k. Here each cell of U_k is charged for its own
# effect on z_l, g_l(r): the excess drop of z_l when cell r alone is deleted.
# The gains stay set-level, so redundant cells still pay together.
# ---------------------------------------------------------------------------

PAYOFFS = ("cell_positive", "cell_abs", "set")


def _cross(t: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "cell_positive":
        return t.clamp_min(0)
    if kind == "cell_abs":
        return t.abs()
    raise ValueError(f"no cell-level cross penalty for payoff {kind!r}")


def cell_charges(cell_excess: torch.Tensor, kind: str) -> torch.Tensor:
    """``(B, 2, R)``: what each cell costs U_k (``phi(g_l)``) and U_l (``phi(g_k)``)."""
    return torch.stack([_cross(cell_excess[:, 1], kind), _cross(cell_excess[:, 0], kind)], dim=1)


def cross_penalty(masks: torch.Tensor, cell_excess: torch.Tensor, kind: str) -> torch.Tensor:
    """``(B, 3)``: U_k pays ``sum_r m_r phi(g_l(r))``, U_l pays ``sum_r m_r phi(g_k(r))``, S pays nothing.

    ``masks`` is ``(B, 3, R)`` (U_k, U_l, S), ``cell_excess`` is ``(B, 2, R)``:
    ``g_k`` and ``g_l``. ``phi`` is ``max(0, .)`` for ``cell_positive``
    (lowering z_l is charged, raising it is not) and ``|.|`` for ``cell_abs``.
    Linear in the masks, so its gradient is exact.
    """
    charge = cell_charges(cell_excess, kind)
    pen = (masks[:, :2] * charge).sum(dim=-1)
    return torch.cat([pen, torch.zeros_like(pen[:, :1])], dim=1)


def partition_payoffs(drops: torch.Tensor, null: torch.Tensor | None, kind: str,
                      penalty: torch.Tensor | None = None) -> torch.Tensor:
    """``(u_k, u_l, u_S)`` of a partition, ``(B, 3)``.

    ``set`` is version 2's ``decomposition_payoffs``. The cell kinds are
      u_k = d~_k(U_k) - P_k,  u_l = d~_l(U_l) - P_l,  u_S = min_s(d~_k(S), d~_l(S)),
    with ``P`` from ``cross_penalty``. An empty unique role pays exactly 0.
    """
    if kind == "set":
        return decomposition_payoffs(drops, null)
    if kind not in PAYOFFS:
        raise ValueError(f"unknown payoff {kind!r}")
    if penalty is None:
        raise ValueError("the cell payoffs need the cross penalty")
    excess = drops if null is None else drops - null
    u_k = excess[:, 0, 0] - penalty[:, 0]
    u_l = excess[:, 1, 1] - penalty[:, 1]
    u_s = smooth_min(excess[:, 2, 0], excess[:, 2, 1])
    return torch.stack([u_k, u_l, u_s], dim=1)


# ---------------------------------------------------------------------------
# The shared rule (FORMULATION_DECOMPOSITION.md sections 3.5 and 4.1, drafted
# 2026-10-22). S is "lowers both", read per cell. Under the cell penalty alone
# a cell that lowers both classes still goes to U_k when its k drop is more
# than twice its l drop, and on planted pairs the polish moved C there. So a
# cell whose single-cell excess drops of k and l are both above the margin may
# not join U_k or U_l, and any other cell may not join S: closing only the
# first half lets a unique cell join S to lift the set-level min's weaker side.
# The welfare still picks between the open roles and the empty one.
# ---------------------------------------------------------------------------

def shared_cells(cell_excess: torch.Tensor, omega: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """``(B, R)`` bool: the Omega cells whose deletion alone lowers both z_k and z_l by more than ``margin``.

    ``cell_excess`` is ``(B, 2, R)``: ``g_k`` and ``g_l``, excess over the null of one cell.
    """
    return omega & (cell_excess[:, 0] > float(margin)) & (cell_excess[:, 1] > float(margin))
