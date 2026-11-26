# Rotary Transformer
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        interleave: bool = False
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
        if hasattr(self.rope, 'rotate_queries_and_keys') and getattr(self.rope, 'use_xpos', False):
            q, k = self.rope.rotate_queries_and_keys(q, k, interleave=interleave)
        else:
            q = self.rope.rotate_queries_or_keys(q, interleave=interleave)
            k = self.rope.rotate_queries_or_keys(k, interleave=interleave)

        # Use PyTorch's optimized scaled_dot_product_attention when possible
        use_sdpa = self.use_sdpa and hasattr(F, 'scaled_dot_product_attention') and (
            attn_mask is None or attn_mask.dtype == torch.bool
        )

        if use_sdpa:
            # Prepare attention mask for SDPA
            sdpa_attn_mask = None
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    sdpa_attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    sdpa_attn_mask = attn_mask.unsqueeze(1)
                else:
                    sdpa_attn_mask = attn_mask

            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
            attn_weights = None

        else:
            # Fallback to manual attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

            if is_causal:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.out_proj(attn_output)

        return output, attn_weights


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
        interleave: bool = False
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
                interleave=interleave
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
                interleave=interleave
            )
            src = self.norm1(src + self.dropout1(attn_output))
            src = self.norm2(src + self.dropout2(self._ff_block(src)))

        if not self.batch_first:
            src = src.transpose(0, 1)

        return src


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
        interleave: bool = False
    ) -> torch.Tensor:
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                interleave=interleave
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
        interleave: bool = False
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
            interleave=interleave
        )

        if self.output_proj is not None:
            output = self.output_proj(output)

        return output


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
