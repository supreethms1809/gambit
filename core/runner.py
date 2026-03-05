from __future__ import annotations
from typing import Any, Optional
import torch
from .types import Tensor, HypothesisSet, EnvBatch, Explanation
from .unit_space import EvidenceUnitSpace
from .interaction import InteractionModel
from .base_evidence import BaseEvidenceProvider
from .hypotheses import HypothesisSelector
from .allocator import Allocator
from .objective import AllocationObjective
from .device import get_device


def _env_to_device(env: EnvBatch, device: torch.device) -> EnvBatch:
    xs = [xe.to(device) if isinstance(xe, torch.Tensor) else xe for xe in env.xs]
    return EnvBatch(xs=xs, env_ids=env.env_ids)


class CDEAExplainer:
    
    def __init__(
        self,
        model: Any,
        unit_space: EvidenceUnitSpace,
        selector: HypothesisSelector,
        base_evidence: BaseEvidenceProvider,
        allocator: Allocator,
        objective: AllocationObjective,
        interaction: Optional[InteractionModel] = None,
        normalize_evidence: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.unit_space = unit_space
        self.selector = selector
        self.base_evidence = base_evidence
        self.allocator = allocator
        self.objective = objective
        self.interaction = interaction
        self.normalize_evidence = normalize_evidence
        self.device = get_device(device)

    @torch.no_grad()
    def _forward_model(self, x: Any) -> Tensor:
        logits = self.model(x)
        return logits

    def _normalize(self, E: Tensor, eps: float = 1e-8) -> Tensor:
        denom = E.sum(dim=-1, keepdim=True).clamp_min(eps)
        return E / denom

    def _build_tokens(self,
        x: Any,
        evidence: Tensor,
        hypotheses: HypothesisSet,
    ) -> Optional[Tensor]:
        phi = self.unit_space.embed_units(x)
        if phi is None:
            return None
        tokens = torch.einsum("bkr,brd -> bkd", evidence, phi)
        return tokens

    def explain(self, x: Any, env: Optional[EnvBatch] = None) -> Explanation:
        if isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
        try:
            if next(self.model.parameters()).device != self.device:
                self.model.to(self.device)
        except StopIteration:
            pass
        if self.interaction is not None and isinstance(self.interaction, torch.nn.Module):
            try:
                if next(self.interaction.parameters()).device != self.device:
                    self.interaction.to(self.device)
            except StopIteration:
                pass
        if env is not None and env.xs:
            env = _env_to_device(env, self.device)
        logits = self._forward_model(x)
        probs = torch.softmax(logits, dim=-1)

        hypotheses = self.selector.select(logits, probs)
        evidence = self.base_evidence.explain(x, self.model, hypotheses)
        if self.normalize_evidence:
            evidence = self._normalize(evidence)

        tokens = self._build_tokens(x, evidence, hypotheses)
        attn = None
        if self.interaction is not None and tokens is not None:
            tokens, attn = self.interaction(tokens, hypotheses.mask)

        masks = self.allocator.allocate(
            x=x,
            model=self.model,
            unit_space=self.unit_space,
            hypotheses=hypotheses,
            evidence=evidence,
            tokens=tokens,
            attn=attn,
            env=env,
        )
        metrics = self.objective.compute(
            x=x,
            model=self.model,
            unit_space=self.unit_space,
            hypotheses=hypotheses,
            masks=masks,
            evidence=evidence,
            tokens=tokens,
            attn=attn,
            env=env,
        )
        return Explanation(hypotheses=hypotheses, masks=masks, metrics=metrics, extras={"tokens": tokens, "attn": attn, "evidence": evidence, "probs": probs})