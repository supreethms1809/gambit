"""OptimizationAllocator for contrastive: masks as parameters, sigmoid, disjointness penalty, gradient steps."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from core.types import Tensor, HypothesisSet, EnvBatch


def _disjoint_penalty(m_unique: Tensor) -> Tensor:
    """Penalty for overlap among unique masks: sum over k<l of (m_k * m_l).sum(dim=-1) (each pair once)."""
    # m_unique (B, K, R); dots[b,k,l] = m_k · m_l; symmetric so off-diag total = 2 * sum_{k<l}
    dots = torch.einsum("bkr,blr->bkl", m_unique, m_unique)
    total = dots.sum()
    diag = dots.diagonal(dim1=1, dim2=2).sum()
    off_diag = total - diag
    return off_diag * 0.5


def _partition_penalty(m_unique: Tensor, m_shared: Optional[Tensor] = None) -> Tensor:
    """Soft penalty if sum of masks exceeds 1 per region: relu(sum_k m_k + m_shared - 1).sum()."""
    total = m_unique.sum(dim=1)  # (B, R)
    if m_shared is not None:
        total = total + m_shared
    return F.relu(total - 1.0).sum()


class OptimizationAllocator:
    """
    Allocator that optimizes masks as parameters:
    - m_unique_logits (B,K,R), optionally m_shared_logits (B,R)
    - map to [0,1] via sigmoid
    - enforce disjointness/partition via penalty
    - 30-60 gradient steps per batch
    """

    def __init__(
        self,
        objective: Any,
        num_steps: int = 50,
        lr: float = 0.5,
        use_shared: bool = False,
        lambda_disjoint: float = 0.2,
        lambda_partition: float = 0.0,
        init_from_evidence: bool = True,
        attn_mix: float = 0.35,
        logit_clip: float = 12.0,
        logit_eps: float = 1e-6,
    ):
        self.objective = objective
        self.num_steps = num_steps
        self.lr = lr
        self.use_shared = use_shared
        self.lambda_disjoint = lambda_disjoint
        self.lambda_partition = lambda_partition
        self.init_from_evidence = init_from_evidence
        if not (0.0 <= attn_mix <= 1.0):
            raise ValueError("attn_mix must be in [0, 1]")
        self.attn_mix = attn_mix
        if logit_clip <= 0:
            raise ValueError("logit_clip must be > 0")
        if not (0 < logit_eps < 0.5):
            raise ValueError("logit_eps must be in (0, 0.5)")
        self.logit_clip = logit_clip
        self.logit_eps = logit_eps

    def _attention_mixed_evidence(
        self,
        evidence: Tensor,
        attn: Optional[Tensor],
        valid_mask: Tensor,
    ) -> Tensor:
        """Condition evidence initialization using hypothesis interaction attention."""
        if attn is None or self.attn_mix <= 0.0:
            return evidence
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        attn = attn.to(dtype=evidence.dtype).clamp_min(0.0)

        valid = valid_mask.float()
        pair_mask = valid.unsqueeze(1) * valid.unsqueeze(2)  # (B, K, K)
        attn = attn * pair_mask
        row_sum = attn.sum(dim=-1, keepdim=True)
        uniform = pair_mask / pair_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        attn_norm = torch.where(row_sum > 1e-8, attn / row_sum.clamp_min(1e-8), uniform)
        mixed = torch.einsum("bkl,blr->bkr", attn_norm, evidence)
        out = (1.0 - self.attn_mix) * evidence + self.attn_mix * mixed
        out = out * valid.unsqueeze(-1)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def allocate(
        self,
        x: Any,
        model: Any,
        unit_space: Any,
        hypotheses: HypothesisSet,
        evidence: Tensor,
        tokens: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
        env: Optional[EnvBatch] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        B, K, R = evidence.shape
        device = evidence.device
        dtype = evidence.dtype
        if tokens is not None:
            tokens = tokens.detach()
        if attn is not None:
            attn = attn.detach()

        E_init = self._attention_mixed_evidence(evidence, attn, hypotheses.mask).detach()
        # Init logits from evidence or zeros (inverse-sigmoid: logit(p) = log(p/(1-p)); for p in (0,1) use clamp)
        if self.init_from_evidence:
            E = E_init.clamp(self.logit_eps, 1 - self.logit_eps)
            m_unique_logits = torch.logit(E.clone().to(torch.float32), eps=self.logit_eps).to(device=device)
            m_unique_logits = m_unique_logits.clamp(-self.logit_clip, self.logit_clip)
        else:
            m_unique_logits = torch.zeros(B, K, R, device=device, dtype=torch.float32)

        m_unique_logits = m_unique_logits.detach().clone().requires_grad_(True)
        params = [m_unique_logits]

        if self.use_shared:
            m_shared_logits = torch.zeros(B, R, device=device, dtype=torch.float32, requires_grad=True)
            params.append(m_shared_logits)
        else:
            m_shared_logits = None

        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Freeze model so only mask params get gradients
        model_requires_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        try:
            for _ in range(self.num_steps):
                optimizer.zero_grad()
                # Clamp logits so masks stay in (0.05, 0.95) for non-trivial allocation
                m_unique = torch.sigmoid(m_unique_logits)
                m_shared = torch.sigmoid(m_shared_logits) if m_shared_logits is not None else None
                masks = {"unique": m_unique}
                if m_shared is not None:
                    masks["shared"] = m_shared

                out = self.objective.compute(
                    x=x,
                    model=model,
                    unit_space=unit_space,
                    hypotheses=hypotheses,
                    masks=masks,
                    evidence=evidence,
                    tokens=tokens,
                    attn=attn,
                    env=env,
                    **kwargs,
                )
                loss = out["loss"]
                penalty = self.lambda_disjoint * _disjoint_penalty(m_unique)
                if self.lambda_partition > 0:
                    penalty = penalty + self.lambda_partition * _partition_penalty(m_unique, m_shared)
                (loss + penalty).backward()
                optimizer.step()
                with torch.no_grad():
                    m_unique_logits.clamp_(-self.logit_clip, self.logit_clip)
                    if m_shared_logits is not None:
                        m_shared_logits.clamp_(-self.logit_clip, self.logit_clip)
        finally:
            for p, req in zip(model.parameters(), model_requires_grad):
                p.requires_grad_(req)

        with torch.no_grad():
            m_unique_final = torch.sigmoid(m_unique_logits)
            result = {"unique": m_unique_final}
            if self.use_shared and m_shared_logits is not None:
                result["shared"] = torch.sigmoid(m_shared_logits)
        return result
