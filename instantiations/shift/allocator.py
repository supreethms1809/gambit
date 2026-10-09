"""
RobustShortcutOptimizationAllocator: two masks m_rob (B,R), m_sho (B,R).
Same optimization loop as contrastive: logits -> sigmoid, gradient steps.
Disjointness should be handled inside the objective to avoid double-counting.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from core.eval_mode import eval_mode
from core.types import Tensor, HypothesisSet, EnvBatch


class RobustShortcutOptimizationAllocator:
    """
    Allocator for robust vs shortcut:
    - m_rob_logits (B,R), m_sho_logits (B,R)
    - sigmoid to [0,1]. Overlap is penalised only inside the objective.
    - Requires env: EnvBatch in allocate()
    """

    def __init__(
        self,
        objective: Any,
        num_steps: int = 50,
        lr: float = 0.5,
        lambda_disjoint: float = 0.2,
        init_from_evidence: bool = True,
        init_seed: int = 0,
        loss_scale: float = 1.0,
        init_offset: int = 0,
    ):
        self.objective = objective
        self.num_steps = num_steps
        self.lr = lr
        # Not applied here: the objective already holds the disjointness term.
        # This must equal the objective's lambda_disjoint and is checked in
        # allocate(), so a caller cannot silently set two different values.
        self.lambda_disjoint = lambda_disjoint
        self.init_from_evidence = init_from_evidence
        self.init_seed = int(init_seed)
        if loss_scale <= 0:
            raise ValueError("loss_scale must be > 0")
        if init_offset < 0:
            raise ValueError("init_offset must be >= 0")
        # Same reason as OptimizationAllocator.loss_scale: a chunk's mean is
        # larger than the full-sample mean by N/C, and Adam's epsilon makes
        # that visible. init_offset skips the seeded shortcut jitter so image i
        # gets the same start it would have had in the full sample.
        self.loss_scale = float(loss_scale)
        self.init_offset = int(init_offset)

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
        if env is None:
            raise ValueError("RobustShortcutOptimizationAllocator requires env (EnvBatch)")
        if hasattr(self.objective, "ld"):
            obj_ld = float(getattr(self.objective, "ld"))
            if abs(obj_ld - float(self.lambda_disjoint)) > 1e-8:
                raise ValueError(
                    "lambda_disjoint mismatch between objective (%.6f) and allocator (%.6f)"
                    % (obj_ld, float(self.lambda_disjoint))
                )
        # evidence can be (B,K,R); pool to (B,R) for init
        if evidence.dim() == 3:
            E = evidence.mean(dim=1)
        else:
            E = evidence
        B, R = E.shape
        device = E.device

        if self.init_from_evidence:
            E_clamp = E.clamp(1e-4, 1 - 1e-4).to(torch.float32)
            m_rob_logits = torch.logit(E_clamp * 0.5, eps=1e-4).clamp(-3.0, 3.0)
            m_sho_logits = torch.logit(E_clamp * 0.5, eps=1e-4).clamp(-3.0, 3.0)
            # Break the rob/sho symmetry. Identical starts leave separation
            # entirely to the gradients while the disjoint penalty starts near
            # maximal. A small deterministic jitter (±0.25 logit units) on the
            # shortcut side gives the two masks different descent directions
            # without biasing either toward any region.
            jitter_gen = torch.Generator(device="cpu")
            jitter_gen.manual_seed(self.init_seed)
            left = self.init_offset
            while left:
                take = min(left, 8)
                torch.rand((take, R), generator=jitter_gen)
                left -= take
            noise = torch.rand(m_sho_logits.shape, generator=jitter_gen) - 0.5
            m_sho_logits = (m_sho_logits + 0.5 * noise.to(m_sho_logits.device)).clamp(-3.0, 3.0)
        else:
            m_rob_logits = torch.zeros(B, R, device=device, dtype=torch.float32)
            m_sho_logits = torch.zeros(B, R, device=device, dtype=torch.float32)

        m_rob_logits = m_rob_logits.clone().requires_grad_(True)
        m_sho_logits = m_sho_logits.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([m_rob_logits, m_sho_logits], lr=self.lr)

        model_requires_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        try:
            # Eval mode: train-mode forwards would update BatchNorm stats and
            # sample dropout, so the explained function would drift mid-run.
            with eval_mode(model):
                for _ in range(self.num_steps):
                    optimizer.zero_grad()
                    m_rob = torch.sigmoid(m_rob_logits)
                    m_sho = torch.sigmoid(m_sho_logits)
                    masks = {"robust": m_rob, "shortcut": m_sho}
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
                    # Objective already includes disjointness term; do not add it again here.
                    if self.loss_scale != 1.0:
                        loss = loss * self.loss_scale
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        m_rob_logits.clamp_(-3.0, 3.0)
                        m_sho_logits.clamp_(-3.0, 3.0)
        finally:
            for p, req in zip(model.parameters(), model_requires_grad):
                p.requires_grad_(req)

        with torch.no_grad():
            return {
                "robust": torch.sigmoid(m_rob_logits),
                "shortcut": torch.sigmoid(m_sho_logits),
            }
