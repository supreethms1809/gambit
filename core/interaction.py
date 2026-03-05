"""
InteractionModel options: None, attention-only, 1-layer transformer.
Only change how tokens are conditioned (or pair weights); rest of pipeline unchanged.
Swap via flag: interaction=get_interaction("none" | "attention" | "transformer", d_model=...).
"""
from __future__ import annotations
from typing import Optional, Protocol, Tuple, Union
import torch
import torch.nn as nn
import math
from .types import Tensor


class InteractionModel(Protocol):
    def __call__(self, tokens: Tensor, token_mask: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        ...


def _apply_attn_mask(attn_scores: Tensor, token_mask: Tensor) -> Tensor:
    """Mask out invalid positions: (B, K, K), token_mask (B, K). Invalid positions get -inf."""
    # attn_scores (B, H, K, K) or (B, K, K); token_mask (B, K) True = valid
    if attn_scores.dim() == 4:
        # (B, H, K, K): mask out key dim (dim -1) and query dim (dim -2) where not valid
        m = token_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
        mq = token_mask.unsqueeze(1).unsqueeze(3)  # (B, 1, K, 1)
        mask = (m.float() * mq.float()).clamp(min=0.0)
        mask = (1 - mask) * (-1e9)
        return attn_scores + mask
    else:
        m = token_mask.unsqueeze(1).float() * token_mask.unsqueeze(2).float()
        return attn_scores + (1 - m).clamp(min=0) * (-1e9)


class NoOpInteraction:
    """No interaction: return tokens and None (same as passing interaction=None)."""

    def __call__(self, tokens: Tensor, token_mask: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return tokens, None


class AttentionOnlyInteraction(nn.Module):
    """Condition tokens with self-attention over the K dimension; return updated tokens and attn weights."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.scale = math.sqrt(d_model // num_heads)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tuple[Tensor, Tensor]:
        B, K, D = tokens.shape
        qkv = self.qkv(tokens).reshape(B, K, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = _apply_attn_mask(attn, token_mask)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, K, D)
        out = self.proj(out)
        # Return residual + output so we don't change scale; optional: just out
        tokens_out = tokens + self.dropout(out)
        attn_out = attn.mean(dim=1)
        return tokens_out, attn_out


class Transformer1LayerInteraction(nn.Module):
    """One transformer layer (self-attention + FFN) over K; return updated tokens and attn weights."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def _key_padding_mask(self, token_mask: Tensor) -> Optional[Tensor]:
        """True = ignore (invalid position). (B, K)."""
        return ~token_mask

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # (B, K, D); key_padding_mask: (B, K) True = ignore
        key_padding = self._key_padding_mask(token_mask)
        attn_out, attn_weights = self.self_attn(
            tokens, tokens, tokens,
            key_padding_mask=key_padding,
            need_weights=True,
            average_attn_weights=True,
        )
        tokens = self.norm1(tokens + self.dropout(attn_out))
        ffn = self.linear2(self.dropout(torch.relu(self.linear1(tokens))))
        tokens = self.norm2(tokens + self.dropout(ffn))
        return tokens, attn_weights


def get_interaction(
    flag: str,
    d_model: Optional[int] = None,
    num_heads: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.0,
) -> Optional[Union[InteractionModel, nn.Module]]:
    """
    Factory: swap a flag to choose interaction. Rest of pipeline unchanged.
    - "none" -> None (or NoOpInteraction() for explicit no-op)
    - "attention" -> AttentionOnlyInteraction(d_model, num_heads)
    - "transformer" -> Transformer1LayerInteraction(d_model, num_heads, dim_feedforward)
    """
    if flag is None or flag.lower() == "none":
        return None
    flag = flag.lower()
    if d_model is None and flag != "none":
        raise ValueError("d_model required for interaction=%s" % flag)
    if flag == "attention":
        return AttentionOnlyInteraction(d_model, num_heads=num_heads, dropout=dropout)
    if flag == "transformer":
        return Transformer1LayerInteraction(
            d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
    raise ValueError("interaction must be 'none' | 'attention' | 'transformer', got %s" % flag)
