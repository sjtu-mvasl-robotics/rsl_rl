"""Explicit, stateless text-to-control fusion; frozen text features are inputs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PelvisCondition(nn.Module):
    """Optional commanded pelvis trajectory, never measured global root XY."""

    def __init__(self, hidden_dim: int, pelvis_dim: int):
        super().__init__()
        self.pelvis_dim = pelvis_dim
        self.project = nn.Sequential(nn.Linear(pelvis_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim, bias=False))
        nn.init.zeros_(self.project[-1].weight)

    def forward(self, hidden, pelvis=None, available=None):
        if pelvis is None:
            if available is not None:
                raise ValueError("pelvis mask without command")
            return hidden
        if pelvis.shape != (*hidden.shape[:-1], self.pelvis_dim):
            raise ValueError("pelvis command shape mismatch")
        if available is None or available.shape != hidden.shape[:-1] or available.dtype != torch.bool:
            raise ValueError("explicit boolean pelvis availability required")
        clean = torch.where(available.unsqueeze(-1), pelvis, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("visible pelvis command must be finite")
        delta = self.project(clean)
        return hidden + torch.where(available.unsqueeze(-1), delta, 0.0)


class LanguageCrossAttention(nn.Module):
    """Fuse each physical timestep with only its currently available instruction.

    Hidden: [..., H], tokens: [..., L, E], token_valid: [..., L],
    available: [...]. No implicit broadcasting, persistent text cache, or learned
    instruction IDs. Callers must explicitly replay historical instructions.
    """

    def __init__(self, hidden_dim: int, text_dim: int = 768,
                 attention_dim: int = 128, heads: int = 4):
        super().__init__()
        if attention_dim % heads:
            raise ValueError("attention_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.text_dim = text_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.query = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key = nn.Linear(text_dim, attention_dim, bias=False)
        self.value = nn.Linear(text_dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, tokens=None, token_valid=None, available=None):
        if tokens is None:
            if token_valid is not None or available is not None:
                raise ValueError("text masks without text tokens")
            return hidden
        prefix = hidden.shape[:-1]
        if tokens.shape[:-2] != prefix or tokens.shape[-1] != self.text_dim:
            raise ValueError("text tokens must match every hidden timestep/batch axis")
        if token_valid is None or token_valid.shape != tokens.shape[:-1]:
            raise ValueError("explicit token_valid required")
        if available is None or available.shape != prefix:
            raise ValueError("explicit per-step text availability required")
        if token_valid.dtype != torch.bool or available.dtype != torch.bool:
            raise ValueError("text visibility masks must be boolean")
        length = tokens.shape[-2]
        if length < 1:
            raise ValueError("text token axis must be nonempty")
        visible = token_valid & available.unsqueeze(-1)
        active = visible.any(-1)
        # Mask payload BEFORE linear/normalization: unavailable/padded NaNs do
        # not leak, and fully absent rows never enter all-masked attention.
        clean = torch.where(visible.unsqueeze(-1), tokens, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("visible text must be finite")
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_active = active.reshape(-1)
        if not flat_active.any():
            return hidden
        text = self.text_norm(clean.reshape(-1, length, self.text_dim)[flat_active])
        count = text.shape[0]
        q = self.query(self.norm(flat_hidden[flat_active])).view(count, self.heads, 1, self.head_dim)
        k = self.key(text).view(count, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(text).view(count, length, self.heads, self.head_dim).transpose(1, 2)
        mask = visible.reshape(-1, length)[flat_active, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        delta = self.output(attended.reshape(count, -1))
        residual = torch.zeros_like(flat_hidden).index_copy(0, flat_active.nonzero().flatten(), delta)
        return (flat_hidden + residual).reshape_as(hidden)
