"""Explicit, stateless text-to-control fusion; frozen text features are inputs."""
from __future__ import annotations

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F


class TextConditionBank:
    """One frozen encoder feature table, shared across environments and time.

    IDs are lookup metadata only. All valid bank features must be finite; masked
    padding may be NaN. Dense per-step tensors remain a supported reference API.
    """
    def __init__(self, tokens, valid):
        if tokens.ndim != 3 or min(tokens.shape) < 1:
            raise ValueError('condition bank must be nonempty [N,L,E]')
        if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('condition bank needs boolean [N,L] token validity')
        if tokens.requires_grad or valid.device != tokens.device:
            raise ValueError('condition bank must contain frozen, co-located encoder features')
        if not torch.isfinite(torch.where(valid[..., None], tokens, 0.0)).all():
            raise ValueError('valid condition bank features must be finite')
        self.tokens, self.valid = tokens, valid

    def select(self, indices):
        return IndexedTextCondition(self, indices)


@dataclass(frozen=True)
class IndexedTextCondition:
    bank: TextConditionBank
    indices: torch.Tensor

    def materialize(self):
        """Diagnostic/legacy bridge; production forwards do not expand features."""
        return self.bank.tokens[self.indices], self.bank.valid[self.indices]


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
    available: [...]. Alternatively pass IndexedTextCondition with [...]
    lookup indices into a frozen bank, and no separate token_valid tensor.
    Indices never become learned embeddings. Historical indices remain explicit.
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
        self._condition_cache = None  # derived no-grad data, never a state_dict buffer
        self.condition_cache_hits = 0
        self.condition_cache_misses = 0
        self.condition_projected_texts = 0

    def _condition_cache_key(self, bank):
        try:
            tensors = (bank.tokens, bank.valid, *self.text_norm.parameters(),
                       *self.key.parameters(), *self.value.parameters())
            stamp = tuple((id(t), t._version, t.dtype, t.device) for t in tensors)
        except RuntimeError:
            # Inference tensors have no mutation version: do not trust a cache.
            return None
        device_type = bank.tokens.device.type
        return (id(bank), stamp, torch.is_autocast_enabled(device_type),
                torch.get_autocast_dtype(device_type))

    def _project_condition(self, tokens, valid):
        self.condition_projected_texts += tokens.shape[0]
        clean = torch.where(valid[..., None], tokens, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError('visible text must be finite')
        text = self.text_norm(clean)
        count, length = text.shape[:2]
        key = self.key(text).view(count, length, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(text).view(count, length, self.heads, self.head_dim).transpose(1, 2)
        return key, value

    def _forward_indexed(self, hidden, condition, token_valid, available):
        bank, ids = condition.bank, condition.indices
        if token_valid is not None:
            raise ValueError('indexed condition carries its own token validity')
        if ids.shape != hidden.shape[:-1] or ids.dtype != torch.long:
            raise ValueError('condition indices must match every hidden batch/time axis')
        if (available is None or available.shape != ids.shape or available.dtype != torch.bool
                or ids.device != hidden.device or bank.tokens.device != hidden.device):
            raise ValueError('explicit co-located boolean condition availability required')
        if bank.tokens.shape[-1] != self.text_dim:
            raise ValueError('condition encoder dimension mismatch')
        if ((ids < 0) | (ids >= bank.tokens.shape[0])).any():
            raise ValueError('condition lookup index out of range')
        if not available.any():
            return hidden
        flat_ids = ids.reshape(-1)
        valid = bank.valid[flat_ids]
        active = available.reshape(-1) & valid.any(-1)
        if not active.any():
            return hidden
        chosen = flat_ids[active]
        if torch.is_grad_enabled():
            # One differentiable projection per DISTINCT text in this forward.
            # index_select backward sums contributions from repeated timesteps.
            unique, inverse = torch.unique(chosen, sorted=True, return_inverse=True)
            key, value = self._project_condition(bank.tokens[unique], bank.valid[unique])
            key, value = key[inverse], value[inverse]
        else:
            stamp = self._condition_cache_key(bank)
            if stamp is not None and self._condition_cache is not None and self._condition_cache[0] == stamp:
                _, _, key, value = self._condition_cache
                self.condition_cache_hits += 1
            else:
                key, value = self._project_condition(bank.tokens, bank.valid)
                self.condition_cache_misses += 1
                # Strong reference prevents object-ID reuse; mutations, load,
                # dtype/device moves and AdamW updates invalidate via the stamp.
                self._condition_cache = (stamp, bank, key, value) if stamp is not None else None
            key, value = key[chosen], value[chosen]
        h = hidden.reshape(-1, hidden.shape[-1])
        q = self.query(self.norm(h[active])).view(-1, self.heads, 1, self.head_dim)
        mask = valid[active, None, None, :]
        attended = F.scaled_dot_product_attention(q, key, value, attn_mask=mask)
        delta = self.output(attended.reshape(q.shape[0], -1))
        residual = torch.zeros_like(h).index_copy(0, active.nonzero().flatten(), delta)
        return (h + residual).reshape_as(hidden)

    def forward(self, hidden, tokens=None, token_valid=None, available=None):
        if isinstance(tokens, IndexedTextCondition):
            return self._forward_indexed(hidden, tokens, token_valid, available)
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
