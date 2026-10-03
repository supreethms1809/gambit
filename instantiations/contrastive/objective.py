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
        mass_ref_regions: int = 49,
        lambda_shared_sparse: float = 0.0,
    ):
        self.lambda_suff = lambda_suff
        self.lambda_margin = lambda_margin
        self.lambda_sparse = lambda_sparse
        self.lambda_overlap = lambda_overlap
        self.lambda_mass = lambda_mass
        # Normalized evidence sums to ~1.0 whatever the grid, so an unscaled mass
        # target means the mask covers a constant *absolute* amount — 1/49 of a 7x7
        # grid but only 1/784 of a 28x28 one, i.e. 16x sparser in relative terms.
        # Measured consequences of that dilution: HAM10000 localization fell 0.585 ->
        # 0.446 going 7x7 -> 28x28, and the decomposition's recovery dropped from 84%
        # to 38% of full spread because keep(x, m) became mostly baseline.
        #
        # Scaling the target by R / mass_ref_regions makes the budget a constant
        # *fraction* of the grid instead. The default reference is 49 (7x7), so every
        # existing 7x7 result is bit-for-bit unchanged.
        #
        # The scale is clamped at >= 1.0, i.e. it only ever grows the budget. Dilution
        # at fine grids is the measured failure; coarse grids (the 4x4 smallcnn path)
        # show no such problem, and scaling them *down* would silently change working
        # results to fix nothing.
        if mass_ref_regions <= 0:
            raise ValueError("mass_ref_regions must be > 0")
        self.mass_ref_regions = mass_ref_regions
        if lambda_shared_sparse < 0:
            raise ValueError("lambda_shared_sparse must be >= 0")
        self.lambda_shared_sparse = lambda_shared_sparse
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

        suff = torch.zeros(B, K, device=m_unique.device, dtype=m_unique.dtype)
        margin = torch.zeros(B, K, device=m_unique.device, dtype=m_unique.dtype)
        split_plus_logits = torch.zeros(B, K, device=m_unique.device, dtype=m_unique.dtype)
        split_plus_probs = torch.zeros(B, K, device=m_unique.device, dtype=m_unique.dtype)
        keep_logits_topm = torch.zeros(B, K, K, device=m_unique.device, dtype=m_unique.dtype)
        keep_probs_topm = torch.zeros(B, K, K, device=m_unique.device, dtype=m_unique.dtype)

        for k in range(K):
            mk = m_tot[:, k, :]  # (B, R)
            x_keep = unit_space.keep(x, mk)
            logits_keep = model(x_keep)  # (B, num_classes)
            probs_keep = torch.softmax(logits_keep, dim=-1)

            cls_k = h_ids[:, k].clamp_min(0)
            z_k = logits_keep.gather(1, cls_k.unsqueeze(1)).squeeze(1)
            p_k = probs_keep.gather(1, cls_k.unsqueeze(1)).squeeze(1)
            suff[:, k] = z_k
            split_plus_logits[:, k] = z_k
            split_plus_probs[:, k] = p_k

            # Contrastive margin: z_k - max(z_foil), foil = other valid hypotheses
            z_all = logits_keep.gather(1, h_ids.clamp_min(0))
            p_all = probs_keep.gather(1, h_ids.clamp_min(0))
            keep_logits_topm[:, k, :] = z_all.masked_fill(~valid, 0.0)
            keep_probs_topm[:, k, :] = p_all.masked_fill(~valid, 0.0)
            z_all = z_all.masked_fill(~valid, float("-inf"))
            z_foil = z_all.clone()
            z_foil[:, k] = float("-inf")
            z_foil_max = z_foil.max(dim=1).values
            margin[:, k] = z_k - z_foil_max

        # Overlap among unique masks: sum over k<l of (m_k · m_l) per batch (each pair counted once)
        dots = torch.einsum("bkr,blr->bkl", m_unique, m_unique)
        off_diag_sum = dots.sum(dim=(1, 2)) - dots.diagonal(dim1=1, dim2=2).sum(dim=1)
        overlap_per_batch = off_diag_sum * 0.5  # dots is symmetric so off-diag = 2 * sum_{k<l}

        # Sparsity: L1 of masks (averaged over K then batch)
        sparse_per_batch = m_unique.abs().sum(dim=-1).mean(dim=1)

        # The shared mask appears in m_tot, so growing it always raises sufficiency and
        # margin — more of the image is kept — but it was in no penalty term at all. Its
        # only brake was the allocator's partition cap, which never binds on average
        # (mean region occupancy 0.56 against a cap of 1.0). Left unpenalized it inflates
        # into content-free background. Measured over 96 HAM10000 val images, 7x7 grid:
        #
        #                        lambda_shared_sparse=0.0   =0.25
        #   shared mass                      22.56 / 49      2.76 / 49
        #   regions above 0.5                     47.2%           3.5%
        #   base evidence captured / area    0.99x chance   1.48x chance
        #
        # That last row is the artifact itself: at 0.0 the shared mask captures base
        # evidence at *exactly* its own area fraction, i.e. it is uncorrelated with the
        # evidence field it is supposed to be allocating. It is a blanket, not a mask.
        # This is the "evidence appearing where there is nothing" seen in the figures,
        # and it also made `cdea_shared` scoring at chance on the lesion metric
        # near-tautological: a blanket covering half the frame scores its area fraction
        # by construction. The unique masks were never the problem — they sit at 3.4x
        # chance on both settings.
        #
        # Defaults to 0.0 so existing runs reproduce exactly; set it to constrain shared.
        #
        # The L1 is divided by mass_scale for the same reason target_mass is multiplied by
        # it: so the coefficient means a constant *fraction* of the frame rather than a
        # constant absolute mass. Without this division a value tuned at 7x7 crushes the
        # shared mask at fine grids — measured on brain tumor at lambda 0.25, shared held
        # 4.66% of a 7x7 grid but only 0.77% of a 28x28 one, a 6x harsher constraint, while
        # unique (which is scaled) held 2.0% at both. That made a resolution sweep at fixed
        # lambda a comparison between two different experiments, and it collapsed
        # cdea_unique at 28x28 in a way that looked like a finding about the method.
        mass_scale = max(1.0, m_unique.shape[-1] / float(self.mass_ref_regions))
        shared_sparse_per_batch = (m_shared_eff.abs().sum(dim=-1) / mass_scale
                                   if m_shared is not None
                                   else torch.zeros_like(sparse_per_batch))
        # (B, K), ~1.0 after normalization, then scaled so the budget is a constant
        # fraction of the grid rather than a constant absolute mass (see __init__).
        target_mass = evidence.sum(dim=-1).detach() * mass_scale
        mass_dev_per_batch = (m_unique.sum(dim=-1) - target_mass).abs().mean(dim=1)

        # Mask invalid positions for suff and margin
        h_weights = self._hypothesis_weights(valid, attn)
        suff_mean = (suff.masked_fill(~valid, 0.0) * h_weights).sum(dim=1)
        margin_mean = (margin.masked_fill(~valid, 0.0) * h_weights).sum(dim=1)

        # Probability-split report:
        #  - shared-only: keep(x, m_shared)
        #  - shared+unique(k): keep(x, m_shared + m_unique[k]) (already in split_plus_* above)
        x_shared = unit_space.keep(x, m_shared_eff)
        logits_shared = model(x_shared)
        probs_shared = torch.softmax(logits_shared, dim=-1)
        split_shared_logits = logits_shared.gather(1, h_ids.clamp_min(0)).masked_fill(~valid, 0.0)
        split_shared_probs = probs_shared.gather(1, h_ids.clamp_min(0)).masked_fill(~valid, 0.0)
        split_plus_logits = split_plus_logits.masked_fill(~valid, 0.0)
        split_plus_probs = split_plus_probs.masked_fill(~valid, 0.0)

        # Pairwise "why k rather than l" report:
        #  - shared-only margins
        #  - shared+unique(k) margins
        #  - delta = unique contribution beyond shared baseline
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, K, K)
        z_keep_k = keep_logits_topm.diagonal(dim1=1, dim2=2)  # (B, K), z_k under keep(k)
        p_keep_k = keep_probs_topm.diagonal(dim1=1, dim2=2)   # (B, K), p_k under keep(k)
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
            + self.lambda_shared_sparse * shared_sparse_per_batch.mean()
            + self.lambda_mass * mass_dev_per_batch.mean()
        )

        return {
            "loss": loss,
            "suff": suff_mean.mean(),
            "margin": margin_mean.mean(),
            "overlap": overlap_per_batch.mean(),
            "sparse": sparse_per_batch.mean(),
            "shared_sparse": shared_sparse_per_batch.mean(),
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
