from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from core.types import Tensor, HypothesisSet, EnvBatch

class RobustShortcutObjective:
    def __init__(self, lambda_mean=1.0, lambda_var=0.5, lambda_gap=1.0,
                 lambda_disjoint=0.2, lambda_sparse=0.05, target="pred",
                 lambda_shortcut=0.0):
        self.lambda_mean = lambda_mean
        self.lambda_var = lambda_var
        self.lambda_gap = lambda_gap
        self.lambda_shortcut = lambda_shortcut
        self.lambda_disjoint = lambda_disjoint
        self.lambda_sparse = lambda_sparse
        self.target = target

    def compute(self, x: Any, model: Any, unit_space: Any, hypotheses: HypothesisSet,
                masks: Dict[str, Tensor], evidence: Tensor,
                tokens: Optional[Tensor] = None, attn: Optional[Tensor] = None,
                env: Optional[EnvBatch] = None, **kwargs: Any) -> Dict[str, Tensor]:

        if env is None or not env.xs:
            raise ValueError("RobustShortcutObjective requires a non-empty env batch")

        m_rob = masks["robust"]
        m_sho = masks["shortcut"]

        disjoint = (m_rob * m_sho).sum(dim=-1)
        sparse = (m_rob.abs().sum(dim=-1) + m_sho.abs().sum(dim=-1)) * 0.5

        logits_id = model(env.xs[0])
        if self.target == "pred":
            y = logits_id.argmax(dim=-1)
        elif self.target == "top_hypothesis":
            y = hypotheses.ids[:, 0].clamp_min(0).to(logits_id.device)
        elif self.target == "label":
            y_arg = kwargs.get("y", None)
            if y_arg is None:
                raise ValueError("target='label' requires y in objective kwargs")
            if not isinstance(y_arg, torch.Tensor):
                raise TypeError("y must be a torch.Tensor when target='label'")
            y = y_arg.to(logits_id.device).long()
        else:
            raise ValueError("target must be one of: pred, top_hypothesis, label")

        def suff(m: Tensor, x_env: Any) -> Tensor:
            x_keep = unit_space.keep(x_env, m)
            z = model(x_keep).gather(1, y[:, None]).squeeze(1)
            z0 = model(unit_space.keep(x_env, torch.zeros_like(m))).gather(1, y[:, None]).squeeze(1)
            return z - z0

        suff_rob = []
        suff_sho = []
        for xe in env.xs:
            suff_rob.append(suff(m_rob, xe))
            suff_sho.append(suff(m_sho, xe))
        suff_rob = torch.stack(suff_rob, dim=1)
        suff_sho = torch.stack(suff_sho, dim=1)

        rob_mean = suff_rob.mean(dim=1)
        rob_var = suff_rob.var(dim=1, unbiased=False)

        sho_id = suff_sho[:, 0]
        sho_ood_mean = suff_sho[:, 1:].mean(dim=1)
        gap = sho_id - sho_ood_mean
        sho_mean = suff_sho.mean(dim=1)

        loss = (
            -(self.lambda_mean * rob_mean - self.lambda_var * rob_var
              + self.lambda_gap * gap + self.lambda_shortcut * sho_mean)
            + self.lambda_disjoint * disjoint + self.lambda_sparse * sparse
        )

        return {
            "loss": loss.mean(),
            "rob_mean": rob_mean.mean(),
            "rob_var": rob_var.mean(),
            "sho_gap": gap.mean(),
            "sho_mean": sho_mean.mean(),
            "disjoint": disjoint.mean(),
            "sparse": sparse.mean(),
        }
