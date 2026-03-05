"""
Clean ContrastiveObjective: intervention-based tests.
- Sufficiency: logit of class k on keep(x, m_tot_k)
- Contrastive margin: z_k - max(z_foil)
- Overlap penalty among unique masks
- Sparsity penalty

Optimization minimizes loss => we want to maximize suff and margin, minimize overlap and sparsity.
So loss = - (lambda_suff * suff + lambda_margin * margin) + lambda_overlap * overlap + lambda_sparse * sparse.
Metrics move in the right direction: suff increases, margin increases, overlap decreases, sparse decreases.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from core.types import Tensor, HypothesisSet, EnvBatch


class ContrastiveObjective:
    def __init__(
        self,
        lambda_suff: float = 1.0,
        lambda_margin: float = 1.0,
        lambda_sparse: float = 0.05,
        lambda_overlap: float = 0.2,
        lambda_mass: float = 0.1,
        attn_weight_blend: float = 0.5,
    ):
        self.lambda_suff = lambda_suff
        self.lambda_margin = lambda_margin
        self.lambda_sparse = lambda_sparse
        self.lambda_overlap = lambda_overlap
        self.lambda_mass = lambda_mass
        if not (0.0 <= attn_weight_blend <= 1.0):
            raise ValueError("attn_weight_blend must be in [0, 1]")
        self.attn_weight_blend = attn_weight_blend

    def _hypothesis_weights(self, valid: Tensor, attn: Optional[Tensor]) -> Tensor:
        """Per-sample top-m weights, optionally modulated by interaction attention."""
        valid_f = valid.float()
        uniform = valid_f / valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        if attn is None or self.attn_weight_blend <= 0.0:
            return uniform

        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        attn = attn.to(dtype=valid_f.dtype)

        pair_mask = valid_f.unsqueeze(1) * valid_f.unsqueeze(2)  # (B, K, K)
        attn = attn.clamp_min(0.0) * pair_mask
        eye = torch.eye(attn.shape[-1], device=attn.device, dtype=attn.dtype).unsqueeze(0)
        attn = attn * (1.0 - eye)

        strength = 0.5 * (attn.sum(dim=-1) + attn.sum(dim=-2))
        strength = strength * valid_f

        strength_norm = uniform.clone()
        denom = strength.sum(dim=1, keepdim=True)
        has_strength = denom.squeeze(1) > 1e-8
        if has_strength.any():
            strength_norm[has_strength] = strength[has_strength] / denom[has_strength].clamp_min(1e-8)

        blend = self.attn_weight_blend
        return (1.0 - blend) * uniform + blend * strength_norm

    def compute(
        self,
        x: Any,
        model: Any,
        unit_space: Any,
        hypotheses: HypothesisSet,
        masks: Dict[str, Tensor],
        evidence: Tensor,
        tokens: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
        env: Optional[EnvBatch] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        m_unique = masks["unique"]
        m_shared = masks.get("shared", None)
        B, K, R = m_unique.shape
        valid = hypotheses.mask  # (B, K)
        h_ids = hypotheses.ids   # (B, K)

        if m_shared is None:
            m_tot = m_unique
            m_shared_eff = torch.zeros(B, R, device=m_unique.device, dtype=m_unique.dtype)
        else:
            m_tot = m_unique + m_shared[:, None, :]
            m_shared_eff = m_shared

        # Batch all K keep-interventions into a single model forward pass
        # m_tot is (B, K, R); we need to apply keep for each of the K masks
        x_kept_list = [unit_space.keep(x, m_tot[:, k, :]) for k in range(K)]
        x_stacked = torch.cat(x_kept_list, dim=0)  # (B*K, C, H, W)
        logits_stacked = model(x_stacked)           # (B*K, num_classes)
        probs_stacked = torch.softmax(logits_stacked, dim=-1)

        # Reshape back to (B, K, num_classes)
        num_classes = logits_stacked.shape[-1]
        logits_all_k = logits_stacked.view(K, B, num_classes).permute(1, 0, 2)  # (B, K, num_classes)
        probs_all_k = probs_stacked.view(K, B, num_classes).permute(1, 0, 2)    # (B, K, num_classes)

        # Extract per-hypothesis sufficiency and margin using gathered indices
        h_ids_clamped = h_ids.clamp_min(0)  # (B, K)

        # Sufficiency: logit of class k when keeping mask k
        # For each k, we need logits_all_k[b, k, h_ids[b, k]]
        suff = logits_all_k.gather(2, h_ids_clamped.unsqueeze(2)).squeeze(2)  # (B, K)
        split_plus_logits = suff.clone()
        split_plus_probs = probs_all_k.gather(2, h_ids_clamped.unsqueeze(2)).squeeze(2)  # (B, K)

        # For margin and pairwise reports: logits of ALL hypotheses under each keep(k)
        # keep_logits_topm[b, k, l] = logit of class h_ids[b, l] under keep(x, m_tot[b, k])
        h_ids_expanded = h_ids_clamped.unsqueeze(1).expand(B, K, K)  # (B, K, K)
        keep_logits_topm = logits_all_k.gather(2, h_ids_expanded)    # (B, K, K)
        keep_probs_topm = probs_all_k.gather(2, h_ids_expanded)      # (B, K, K)

        # Contrastive margin: z_k - max(z_foil), foil = other valid hypotheses
        z_all_for_margin = keep_logits_topm.diagonal(dim1=1, dim2=2).unsqueeze(2).expand_as(keep_logits_topm)
        # Actually we need: for each k, gather logits of all hypothesis classes under keep(k)
        # Then margin[b, k] = z_k_under_keep_k - max_{l!=k, valid} z_l_under_keep_k
        z_foil = keep_logits_topm.clone()
        z_foil = z_foil.masked_fill(~valid.unsqueeze(1).expand_as(z_foil), float("-inf"))
        # Zero out the diagonal (self vs self)
        diag_mask = torch.eye(K, device=z_foil.device, dtype=torch.bool).unsqueeze(0).expand(B, K, K)
        z_foil = z_foil.masked_fill(diag_mask, float("-inf"))
        # margin[b, k] = suff[b, k] - max_l z_foil[b, k, l]
        margin = suff - z_foil.max(dim=2).values  # (B, K)

        # Mask keep_logits/probs for reporting
        keep_logits_topm = keep_logits_topm.masked_fill(~valid.unsqueeze(1).expand_as(keep_logits_topm), 0.0)
        keep_probs_topm = keep_probs_topm.masked_fill(~valid.unsqueeze(1).expand_as(keep_probs_topm), 0.0)

        # Overlap among unique masks: sum over k<l of (m_k . m_l) per batch
        dots = torch.einsum("bkr,blr->bkl", m_unique, m_unique)
        off_diag_sum = dots.sum(dim=(1, 2)) - dots.diagonal(dim1=1, dim2=2).sum(dim=1)
        overlap_per_batch = off_diag_sum * 0.5

        # Sparsity: L1 of masks (averaged over K then batch)
        sparse_per_batch = m_unique.abs().sum(dim=-1).mean(dim=1)
        target_mass = evidence.sum(dim=-1).detach()  # (B, K)
        mass_dev_per_batch = (m_unique.sum(dim=-1) - target_mass).abs().mean(dim=1)

        # Mask invalid positions for suff and margin
        h_weights = self._hypothesis_weights(valid, attn)
        suff_mean = (suff.masked_fill(~valid, 0.0) * h_weights).sum(dim=1)
        margin_mean = (margin.masked_fill(~valid, 0.0) * h_weights).sum(dim=1)

        # Probability-split report: shared-only via keep(x, m_shared)
        x_shared = unit_space.keep(x, m_shared_eff)
        logits_shared = model(x_shared)
        probs_shared = torch.softmax(logits_shared, dim=-1)
        split_shared_logits = logits_shared.gather(1, h_ids_clamped).masked_fill(~valid, 0.0)
        split_shared_probs = probs_shared.gather(1, h_ids_clamped).masked_fill(~valid, 0.0)
        split_plus_logits = split_plus_logits.masked_fill(~valid, 0.0)
        split_plus_probs = split_plus_probs.masked_fill(~valid, 0.0)

        # Pairwise "why k rather than l" report
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, K, K)
        z_keep_k = keep_logits_topm.diagonal(dim1=1, dim2=2)  # (B, K)
        p_keep_k = keep_probs_topm.diagonal(dim1=1, dim2=2)   # (B, K)
        pair_margin_plus_logits = z_keep_k.unsqueeze(2) - keep_logits_topm
        pair_margin_plus_probs = p_keep_k.unsqueeze(2) - keep_probs_topm
        pair_margin_shared_logits = split_shared_logits.unsqueeze(2) - split_shared_logits.unsqueeze(1)
        pair_margin_shared_probs = split_shared_probs.unsqueeze(2) - split_shared_probs.unsqueeze(1)
        pair_margin_delta_logits = pair_margin_plus_logits - pair_margin_shared_logits
        pair_margin_delta_probs = pair_margin_plus_probs - pair_margin_shared_probs

        pair_margin_plus_logits = pair_margin_plus_logits.masked_fill(~pair_valid, 0.0)
        pair_margin_plus_probs = pair_margin_plus_probs.masked_fill(~pair_valid, 0.0)
        pair_margin_shared_logits = pair_margin_shared_logits.masked_fill(~pair_valid, 0.0)
        pair_margin_shared_probs = pair_margin_shared_probs.masked_fill(~pair_valid, 0.0)
        pair_margin_delta_logits = pair_margin_delta_logits.masked_fill(~pair_valid, 0.0)
        pair_margin_delta_probs = pair_margin_delta_probs.masked_fill(~pair_valid, 0.0)

        loss = (
            - (self.lambda_suff * suff_mean.mean() + self.lambda_margin * margin_mean.mean())
            + self.lambda_overlap * overlap_per_batch.mean()
            + self.lambda_sparse * sparse_per_batch.mean()
            + self.lambda_mass * mass_dev_per_batch.mean()
        )

        return {
            "loss": loss,
            "suff": suff_mean.mean(),
            "margin": margin_mean.mean(),
            "overlap": overlap_per_batch.mean(),
            "sparse": sparse_per_batch.mean(),
            "mass_dev": mass_dev_per_batch.mean(),
            "split_shared_only_logits_topm": split_shared_logits,
            "split_shared_only_probs_topm": split_shared_probs,
            "split_shared_plus_unique_logits_topm": split_plus_logits,
            "split_shared_plus_unique_probs_topm": split_plus_probs,
            "hypothesis_weights_topm": h_weights.masked_fill(~valid, 0.0),
            "pairwise_margin_shared_only_logits_topm": pair_margin_shared_logits,
            "pairwise_margin_shared_only_probs_topm": pair_margin_shared_probs,
            "pairwise_margin_shared_plus_unique_logits_topm": pair_margin_plus_logits,
            "pairwise_margin_shared_plus_unique_probs_topm": pair_margin_plus_probs,
            "pairwise_margin_delta_logits_topm": pair_margin_delta_logits,
            "pairwise_margin_delta_probs_topm": pair_margin_delta_probs,
        }
