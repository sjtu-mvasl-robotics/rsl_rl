# Rotary Transformer
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal
from rsl_rl.networks.rope import RotaryEmbedding


class RoPEMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Rotary Position Embedding"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_theta: int = 10000,
        rope_freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        bias: bool = True,
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
        
        # Rotary Position Embedding - 针对RL短序列优化
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            theta=rope_theta,
            freqs_for=rope_freqs_for,
            cache_if_possible=True,
            cache_max_seq_len=64  # RL场景短序列优化
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
        
        # Fused QKV projection and split - 用户建议的简单高效方法
        qkv = self.qkv_proj(query)  # [batch_size, seq_len, 3*d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 拆分得到Q, K, V
        
        # Apply RoPE to queries and keys
        # RoPE expects [batch_size, num_heads, seq_len, head_dim] with seq_dim=-2
        # Use simultaneous Q&K rotation when available and use_xpos is enabled
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
                    # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                    sdpa_attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                    sdpa_attn_mask = attn_mask.unsqueeze(1)
                else:
                    sdpa_attn_mask = attn_mask
            
            # Use optimized SDPA
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
            
            # For compatibility, we still need to return attention weights
            # but they will be None when using SDPA for efficiency
            attn_weights = None
            
        else:
            # Fallback to manual attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # attn_weights: [batch_size, num_heads, seq_len, seq_len]
            
            # Apply attention mask
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                    attn_mask = attn_mask.unsqueeze(1)
                
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            
            # Apply causal mask
            if is_causal:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            # Apply key padding mask
            if key_padding_mask is not None:
                # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
            
            # Apply softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            # attn_output: [batch_size, num_heads, seq_len, head_dim]
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with RoPE Multi-Head Attention"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        rope_theta: int = 10000,
        rope_freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        use_sdpa: bool = True
    ):
        super().__init__()
        
        self.self_attn = RoPEMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            rope_freqs_for=rope_freqs_for,
            use_sdpa=use_sdpa
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.batch_first = batch_first
    
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
            # Convert [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
            src = src.transpose(0, 1)
        
        # Self-attention block
        attn_output, _ = self.self_attn(
            query=src,
            key=src, 
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            interleave=interleave
        )
        
        # First residual connection and layer norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # Feed-forward block
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # Second residual connection and layer norm
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        if not self.batch_first:
            # Convert back to [seq_len, batch_size, d_model]
            src = src.transpose(0, 1)
        
        return src


class RoPETransformerEncoder(nn.Module):
    """Transformer Encoder with RoPE"""
    
    def __init__(
        self,
        encoder_layer: RoPETransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            self._get_clones(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        
    def _get_clones(self, module):
        """Create a clone of the module with same parameters but separate instances"""
        import copy
        return copy.deepcopy(module)
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        interleave: bool = False
    ) -> torch.Tensor:
        """
        Args:
            src: input tensor
            mask: attention mask
            src_key_padding_mask: key padding mask  
            is_causal: whether to apply causal attention
            
        Returns:
            output: transformed tensor
        """
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
    """Complete Transformer with RoPE for sequence modeling"""
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        rope_theta: int = 10000,
        rope_freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        max_seq_len: int = 8192,
        use_sdpa: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.batch_first = batch_first
        
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
            rope_theta=rope_theta,
            rope_freqs_for=rope_freqs_for,
            use_sdpa=use_sdpa
        )
        
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if not norm_first else None
        
        self.encoder = RoPETransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size) if vocab_size is not None else None
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
        # Handle input embedding/projection
        if self.embedding is not None:
            # Token input
            src = self.embedding(src) * math.sqrt(self.d_model)
        elif self.input_proj is not None:
            # Continuous input
            src = self.input_proj(src)
        
        # Pass through encoder
        output = self.encoder(
            src=src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            interleave=interleave
        )
        
        # Output projection
        if self.output_proj is not None:
            output = self.output_proj(output)
        
        return output


# Convenience functions for creating standard configurations

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
    **kwargs
) -> RoPETransformerEncoderLayer:
    """Create a single RoPE Transformer encoder layer"""
    if dim_feedforward is None:
        dim_feedforward = d_model * 4
        
    return RoPETransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        **kwargs
    )

