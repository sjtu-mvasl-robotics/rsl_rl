# Rotary Transformer
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal, List, Dict, Any
from rsl_rl.networks.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction)."""
    def __init__(self, normalized_shape, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
        return x_norm.to(input_dtype)


class RoPEMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Rotary Position Embedding"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_theta: int = 10000,
        rope_freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        bias: bool = False,
        use_sdpa: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV projection for better memory efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Rotary Position Embedding
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            theta=rope_theta,
            freqs_for=rope_freqs_for,
            cache_if_possible=True,
            cache_max_seq_len=64
        )

        self.dropout = nn.Dropout(dropout)
        self.use_sdpa = use_sdpa

    @staticmethod
    def _expand_attention_mask(
        mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        query_len: int,
        key_len: int,
    ) -> torch.Tensor:
        """Expand a MultiheadAttention-style mask for SDPA broadcasting.

        Boolean masks use the ``nn.MultiheadAttention`` convention throughout
        this module: ``True`` means that the query/key pair is blocked.  SDPA
        uses the opposite convention for boolean masks, so callers must first
        convert the returned mask to an additive attention bias.
        """

        if tuple(mask.shape[-2:]) != (query_len, key_len):
            raise ValueError(
                "attention mask must end in "
                f"({query_len}, {key_len}), got {tuple(mask.shape)}"
            )
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3:
            if mask.shape[0] == batch_size:
                return mask.unsqueeze(1)
            if mask.shape[0] == batch_size * num_heads:
                return mask.reshape(
                    batch_size, num_heads, query_len, key_len
                )
            raise ValueError(
                "3-D attention mask leading dimension must be batch_size or "
                f"batch_size * num_heads, got {tuple(mask.shape)}"
            )
        if mask.dim() == 4:
            return mask
        raise ValueError(
            f"attention mask must be 2-D, 3-D, or 4-D, got {mask.dim()}-D"
        )

    def _build_attention_bias(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        batch_size: int,
        query_len: int,
        key_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build one additive mask shared by the SDPA and manual paths.

        ``attn_mask`` and ``key_padding_mask`` follow PyTorch
        MultiheadAttention semantics (``True`` is blocked).  Returning a float
        bias avoids relying on SDPA's inverted boolean-mask convention.
        Fully blocked rows are temporarily made safe and reported separately;
        their attention outputs are zeroed after the kernel call.
        """

        attention_bias = None

        def add_component(component: torch.Tensor) -> None:
            nonlocal attention_bias
            if component.dtype == torch.bool:
                component_bias = torch.zeros(
                    component.shape, dtype=dtype, device=device
                ).masked_fill(component.to(device=device), float("-inf"))
            else:
                component_bias = component.to(device=device, dtype=dtype)
            attention_bias = (
                component_bias
                if attention_bias is None
                else attention_bias + component_bias
            )

        if attn_mask is not None:
            add_component(
                self._expand_attention_mask(
                    attn_mask,
                    batch_size,
                    self.num_heads,
                    query_len,
                    key_len,
                )
            )

        if is_causal:
            causal_blocked = torch.ones(
                query_len, key_len, dtype=torch.bool, device=device
            ).triu(diagonal=1)
            add_component(causal_blocked.unsqueeze(0).unsqueeze(0))

        if key_padding_mask is not None:
            if tuple(key_padding_mask.shape) != (batch_size, key_len):
                raise ValueError(
                    "key padding mask must have shape "
                    f"({batch_size}, {key_len}), got {tuple(key_padding_mask.shape)}"
                )
            add_component(
                key_padding_mask.to(device=device, dtype=torch.bool)
                .unsqueeze(1)
                .unsqueeze(2)
            )

        fully_blocked = None
        if attention_bias is not None:
            fully_blocked = torch.isneginf(attention_bias).all(
                dim=-1, keepdim=True
            )
            # Both the manual softmax and some SDPA kernels can produce NaNs
            # for an all -inf row. Its result is semantically unused, so make
            # the kernel row safe and explicitly zero it afterwards.
            attention_bias = attention_bias.masked_fill(fully_blocked, 0.0)

        return attention_bias, fully_blocked

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        interleave: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            attn_mask: [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            key_padding_mask: [batch_size, seq_len], True for padding positions
            is_causal: bool, whether to apply causal masking

        Returns:
            output: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape

        # Fused QKV projection and split
        qkv = self.qkv_proj(query)  # [batch_size, seq_len, 3*d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys
        if position_ids is not None:
            if tuple(position_ids.shape) != (batch_size, seq_len):
                raise ValueError(
                    "position_ids must match attention [B, T], got "
                    f"{tuple(position_ids.shape)} and {tuple(query.shape)}"
                )
            q = self.rope.rotate_queries_or_keys_at_positions(q, position_ids)
            k = self.rope.rotate_queries_or_keys_at_positions(k, position_ids)
        elif hasattr(self.rope, 'rotate_queries_and_keys') and getattr(self.rope, 'use_xpos', False):
            q, k = self.rope.rotate_queries_and_keys(q, k, interleave=interleave)
        else:
            q = self.rope.rotate_queries_or_keys(q, interleave=interleave)
            k = self.rope.rotate_queries_or_keys(k, interleave=interleave)

        # Use one mask contract for both implementations. Internally True means
        # blocked, matching nn.MultiheadAttention and TransformerEncoder.
        # Convert it to additive bias before entering SDPA, whose bool mask has
        # the opposite meaning (True means allowed).
        native_causal = (
            is_causal and attn_mask is None and key_padding_mask is None
        )
        attention_bias, fully_blocked = self._build_attention_bias(
            attn_mask=None if native_causal else attn_mask,
            key_padding_mask=None if native_causal else key_padding_mask,
            is_causal=False if native_causal else is_causal,
            batch_size=batch_size,
            query_len=seq_len,
            key_len=seq_len,
            dtype=q.dtype,
            device=q.device,
        )

        # Use PyTorch's optimized scaled_dot_product_attention when possible.
        use_sdpa = self.use_sdpa and hasattr(F, 'scaled_dot_product_attention')

        if use_sdpa:
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attention_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=native_causal,
            )
            attn_weights = None

        else:
            # Fallback to manual attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if native_causal:
                causal_blocked = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=query.device
                ).triu(diagonal=1)
                attn_weights = attn_weights.masked_fill(
                    causal_blocked, float("-inf")
                )
            elif attention_bias is not None:
                attn_weights = attn_weights + attention_bias

            attn_weights = F.softmax(attn_weights, dim=-1)
            if fully_blocked is not None:
                attn_weights = attn_weights.masked_fill(fully_blocked, 0.0)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        if fully_blocked is not None:
            attn_output = attn_output.masked_fill(fully_blocked, 0.0)

        query_padding_mask = None
        if key_padding_mask is not None:
            query_padding_mask = key_padding_mask.to(
                device=query.device, dtype=torch.bool
            ).unsqueeze(-1)

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.out_proj(attn_output)
        if query_padding_mask is not None:
            output = output.masked_fill(query_padding_mask, 0.0)

        return output, attn_weights

    def forward_with_kv_cache(
        self,
        query: torch.Tensor,
        position_ids: torch.Tensor,
        token_padding_mask: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
        past_key_padding_mask: Optional[torch.Tensor] = None,
        max_cache_tokens: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Attend new tokens to a bounded append-only KV memory.

        Cached keys are stored *after* RoPE at their episode-local absolute
        positions.  The cache implements the same semantics as a full
        sequence with local causal attention; eviction removes keys from
        direct attention but intentionally does not rewrite contextual hidden
        states already stored by upper layers.
        """

        batch_size, query_len, _ = query.shape
        if tuple(position_ids.shape) != (batch_size, query_len):
            raise ValueError(
                "position_ids must match query [B, T], got "
                f"{tuple(position_ids.shape)} and {tuple(query.shape)}"
            )
        if tuple(token_padding_mask.shape) != (batch_size, query_len):
            raise ValueError(
                "token_padding_mask must match query [B, T], got "
                f"{tuple(token_padding_mask.shape)} and {tuple(query.shape)}"
            )

        qkv = self.qkv_proj(query)
        qkv = qkv.reshape(
            batch_size,
            query_len,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, new_key, new_value = qkv[0], qkv[1], qkv[2]
        q = self.rope.rotate_queries_or_keys_at_positions(q, position_ids)
        new_key = self.rope.rotate_queries_or_keys_at_positions(
            new_key, position_ids
        )

        if past_key is None:
            if past_value is not None or past_key_padding_mask is not None:
                raise ValueError("partial KV cache state is not allowed")
            key = new_key
            value = new_value
            key_padding_mask = token_padding_mask.bool()
            past_len = 0
        else:
            if past_value is None or past_key_padding_mask is None:
                raise ValueError("past key/value/padding must be supplied together")
            if past_key.shape != past_value.shape:
                raise ValueError("past key and value must have identical shapes")
            if past_key.shape[:2] != (batch_size, self.num_heads):
                raise ValueError(
                    "past cache batch/head dimensions do not match attention"
                )
            past_len = past_key.shape[-2]
            if tuple(past_key_padding_mask.shape) != (batch_size, past_len):
                raise ValueError("past padding mask does not match cached keys")
            key = torch.cat((past_key, new_key), dim=-2)
            value = torch.cat((past_value, new_value), dim=-2)
            key_padding_mask = torch.cat(
                (past_key_padding_mask.bool(), token_padding_mask.bool()),
                dim=-1,
            )

        key_len = key.shape[-2]
        # Every cached token is in the past.  Within the new block retain the
        # ordinary triangular order: with [ref_t, obs_t], ref_t cannot see
        # obs_t while obs_t can see ref_t.
        local_causal_block = torch.zeros(
            query_len, key_len, dtype=torch.bool, device=query.device
        )
        local_causal_block[:, past_len:] = torch.ones(
            query_len,
            query_len,
            dtype=torch.bool,
            device=query.device,
        ).triu(diagonal=1)

        attention_bias, fully_blocked = self._build_attention_bias(
            attn_mask=local_causal_block,
            key_padding_mask=key_padding_mask,
            is_causal=False,
            batch_size=batch_size,
            query_len=query_len,
            key_len=key_len,
            dtype=q.dtype,
            device=q.device,
        )
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=key,
                value=value,
                attn_mask=attention_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_weights = torch.matmul(q, key.transpose(-2, -1)) * self.scale
            if attention_bias is not None:
                attn_weights = attn_weights + attention_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            if fully_blocked is not None:
                attn_weights = attn_weights.masked_fill(fully_blocked, 0.0)
            attn_output = torch.matmul(self.dropout(attn_weights), value)

        if fully_blocked is not None:
            attn_output = attn_output.masked_fill(fully_blocked, 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        output = self.out_proj(attn_output)
        output = output.masked_fill(token_padding_mask.bool().unsqueeze(-1), 0.0)

        if max_cache_tokens <= 0:
            cached_key = key[..., :0, :]
            cached_value = value[..., :0, :]
            cached_padding = key_padding_mask[:, :0]
        else:
            cached_key = key[..., -max_cache_tokens:, :].detach()
            cached_value = value[..., -max_cache_tokens:, :].detach()
            cached_padding = key_padding_mask[:, -max_cache_tokens:].detach()
        return output, cached_key, cached_value, cached_padding


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: w2(silu(w1(x)) * w3(x))"""
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with RoPE Multi-Head Attention.

    Supports modern options: pre-norm (norm_first), RMSNorm, SwiGLU FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        use_rmsnorm: bool = False,
        use_swiglu: bool = False,
        rope_theta: int = 10000,
        rope_freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        bias: bool = False,
        use_sdpa: bool = True
    ):
        super().__init__()
        self.norm_first = norm_first
        self.use_swiglu = use_swiglu
        self.batch_first = batch_first

        self.self_attn = RoPEMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            rope_freqs_for=rope_freqs_for,
            bias=bias,
            use_sdpa=use_sdpa
        )

        # Feed-forward network
        if use_swiglu:
            self.ffn = SwiGLUFFN(d_model, dim_feedforward)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
            # Activation function
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'gelu':
                self.activation = F.gelu
            elif activation == 'swish':
                self.activation = F.silu
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        # Normalization
        if use_rmsnorm:
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps, bias=False)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps, bias=False)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # FFN internal dropout (only used in standard FFN path)
        self.dropout = nn.Dropout(dropout)

    def _ff_block(self, src: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block."""
        if self.use_swiglu:
            return self.ffn(src)
        else:
            return self.linear2(self.dropout(self.activation(self.linear1(src))))

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        interleave: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model] if batch_first=True
                 [seq_len, batch_size, d_model] if batch_first=False
            src_mask: attention mask
            src_key_padding_mask: key padding mask
            is_causal: whether to apply causal attention

        Returns:
            output: same shape as src
        """
        if not self.batch_first:
            src = src.transpose(0, 1)

        if self.norm_first:
            # Pre-norm: norm -> sublayer -> residual add
            residual = src
            src = self.norm1(src)
            attn_output, _ = self.self_attn(
                query=src, key=src, value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                interleave=interleave,
                position_ids=position_ids,
            )
            src = residual + self.dropout1(attn_output)

            residual = src
            src = self.norm2(src)
            src = residual + self.dropout2(self._ff_block(src))
        else:
            # Post-norm: sublayer -> residual add -> norm (legacy)
            attn_output, _ = self.self_attn(
                query=src, key=src, value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                interleave=interleave,
                position_ids=position_ids,
            )
            src = self.norm1(src + self.dropout1(attn_output))
            src = self.norm2(src + self.dropout2(self._ff_block(src)))

        if not self.batch_first:
            src = src.transpose(0, 1)

        return src

    def forward_with_kv_cache(
        self,
        src: torch.Tensor,
        position_ids: torch.Tensor,
        token_padding_mask: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_padding_mask: Optional[torch.Tensor] = None,
        max_cache_tokens: int = 0,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Increment one encoder layer and return its bounded KV memory."""

        if not self.batch_first:
            src = src.transpose(0, 1)
        past_key, past_value = (
            past_key_value if past_key_value is not None else (None, None)
        )

        if self.norm_first:
            residual = src
            normalized = self.norm1(src)
            attn_output, key, value, padding = (
                self.self_attn.forward_with_kv_cache(
                    normalized,
                    position_ids=position_ids,
                    token_padding_mask=token_padding_mask,
                    past_key=past_key,
                    past_value=past_value,
                    past_key_padding_mask=past_key_padding_mask,
                    max_cache_tokens=max_cache_tokens,
                )
            )
            src = residual + self.dropout1(attn_output)
            residual = src
            src = residual + self.dropout2(self._ff_block(self.norm2(src)))
        else:
            attn_output, key, value, padding = (
                self.self_attn.forward_with_kv_cache(
                    src,
                    position_ids=position_ids,
                    token_padding_mask=token_padding_mask,
                    past_key=past_key,
                    past_value=past_value,
                    past_key_padding_mask=past_key_padding_mask,
                    max_cache_tokens=max_cache_tokens,
                )
            )
            src = self.norm1(src + self.dropout1(attn_output))
            src = self.norm2(src + self.dropout2(self._ff_block(src)))

        if not self.batch_first:
            src = src.transpose(0, 1)
        return src, (key, value), padding


class RoPETransformerEncoder(nn.Module):
    """Transformer Encoder with RoPE"""

    def __init__(
        self,
        encoder_layer: RoPETransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        interleave: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                interleave=interleave,
                position_ids=position_ids,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class RoPETransformer(nn.Module):
    """Complete Transformer with RoPE for sequence modeling.

    Supports modern architecture options:
    - norm_first: pre-norm (default True) vs post-norm
    - use_rmsnorm: RMSNorm (default False) vs LayerNorm
    - use_swiglu: SwiGLU FFN (default False) vs standard 2-layer FFN
    - init_style: "xavier" (default) or "gpt2" (scaled residual init)
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        use_rmsnorm: bool = False,
        use_swiglu: bool = False,
        bias: bool = False,
        init_style: str = "xavier",
        rope_theta: int = 10000,
        rope_freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        max_seq_len: int = 8192,
        use_sdpa: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.batch_first = batch_first
        self.num_encoder_layers = num_encoder_layers
        self.init_style = init_style

        # Token embedding (optional)
        self.embedding = nn.Embedding(vocab_size, d_model) if vocab_size is not None else None

        # Input projection (for continuous inputs)
        self.input_proj = nn.Linear(d_model, d_model) if vocab_size is None else None

        # Encoder layers
        encoder_layer = RoPETransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            use_rmsnorm=use_rmsnorm,
            use_swiglu=use_swiglu,
            rope_theta=rope_theta,
            rope_freqs_for=rope_freqs_for,
            bias=bias,
            use_sdpa=use_sdpa
        )

        # Pre-norm needs a final norm after last layer; post-norm does not
        if norm_first:
            if use_rmsnorm:
                encoder_norm = RMSNorm(d_model, eps=layer_norm_eps, bias=False)
            else:
                encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        else:
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.encoder = RoPETransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size) if vocab_size is not None else None

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        if self.init_style == "gpt2":
            self._gpt2_init()
        else:
            # Xavier uniform (legacy default)
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _gpt2_init(self):
        """GPT-2 style initialization with residual scaling.

        - Most weights: N(0, 0.02)
        - Output projections (attn out_proj, FFN w2/linear2): N(0, 0.02 / sqrt(2 * num_layers))
        """
        base_std = 0.02
        residual_std = base_std / math.sqrt(2 * self.num_encoder_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)

        # Scale output projections that feed into residual stream
        for layer in self.encoder.layers:
            # Attention output projection
            nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=residual_std)
            # FFN output projection
            if layer.use_swiglu:
                nn.init.normal_(layer.ffn.w2.weight, mean=0.0, std=residual_std)
            else:
                nn.init.normal_(layer.linear2.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        interleave: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: input tensor
                 [batch_size, seq_len] if using token embedding
                 [batch_size, seq_len, d_model] if using continuous input
            src_mask: attention mask
            src_key_padding_mask: key padding mask
            is_causal: whether to apply causal attention

        Returns:
            output: [batch_size, seq_len, vocab_size] or [batch_size, seq_len, d_model]
        """
        if self.embedding is not None:
            src = self.embedding(src) * math.sqrt(self.d_model)
        elif self.input_proj is not None:
            src = self.input_proj(src)

        output = self.encoder(
            src=src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            interleave=interleave,
            position_ids=position_ids,
        )

        if self.output_proj is not None:
            output = self.output_proj(output)

        return output

    def forward_with_kv_cache(
        self,
        src: torch.Tensor,
        position_ids: torch.Tensor,
        token_padding_mask: torch.Tensor,
        cache: Optional[Dict[str, Any]] = None,
        max_cache_tokens: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Incrementally encode new tokens with one KV cache per layer."""

        if self.embedding is not None:
            src = self.embedding(src) * math.sqrt(self.d_model)
        elif self.input_proj is not None:
            src = self.input_proj(src)

        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
        past_padding = None
        if cache is None:
            past_key_values = None
        else:
            past_key_values = cache.get("key_values")
            past_padding = cache.get("key_padding_mask")
            if past_key_values is None or len(past_key_values) != len(
                self.encoder.layers
            ):
                raise ValueError("KV cache must contain one entry per layer")

        output = src
        new_key_values = []
        new_padding = None
        for layer_idx, layer in enumerate(self.encoder.layers):
            output, key_value, layer_padding = layer.forward_with_kv_cache(
                output,
                position_ids=position_ids,
                token_padding_mask=token_padding_mask,
                past_key_value=(
                    None
                    if past_key_values is None
                    else past_key_values[layer_idx]
                ),
                past_key_padding_mask=past_padding,
                max_cache_tokens=max_cache_tokens,
            )
            new_key_values.append(key_value)
            if new_padding is None:
                new_padding = layer_padding
            # Every layer appends and truncates the same token positions, so
            # the padding layout is shared. Avoid a per-layer GPU sync here.

        if self.encoder.norm is not None:
            output = self.encoder.norm(output)
        if self.output_proj is not None:
            output = self.output_proj(output)
        return output, {
            "key_values": new_key_values,
            "key_padding_mask": new_padding,
        }


# Convenience functions

def create_rope_transformer(
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    vocab_size: Optional[int] = None,
    **kwargs
) -> RoPETransformer:
    """Create a RoPE Transformer with standard configuration"""
    return RoPETransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        **kwargs
    )


def create_rope_encoder_layer(
    d_model: int = 512,
    num_heads: int = 8,
    dim_feedforward: Optional[int] = None,
    use_swiglu: bool = False,
    **kwargs
) -> RoPETransformerEncoderLayer:
    """Create a single RoPE Transformer encoder layer"""
    if dim_feedforward is None:
        if use_swiglu:
            # SwiGLU recommended: 8/3 * d_model, rounded to multiple of 64
            dim_feedforward = int(math.ceil(d_model * 8 / 3 / 64) * 64)
        else:
            dim_feedforward = d_model * 4

    return RoPETransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        use_swiglu=use_swiglu,
        **kwargs
    )
