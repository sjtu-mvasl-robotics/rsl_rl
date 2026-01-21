# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Created by Yifei Yao, 2024-12-21

from __future__ import annotations
from calendar import c
from sympy import fu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
# from timm.models.vision_transformer import LayerScale
import time
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import Linear as LoRALinear
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.actor_critic_mlp_v2 import MultiModalMLPV2, FusedMultiModalMLP

def group_by_concat_list(orig_dict: dict, concat_list: list | None = None):
    key_to_idx = {k: i for i, k in enumerate(orig_dict)}

    if concat_list is None:
        grouped_keys = [[k] for k in orig_dict]
        grouped_values = [[v] for v in orig_dict.values()]
        group_idx = [[i] for i in range(len(orig_dict))]
        return grouped_keys, grouped_values, group_idx

    elements_in_groups = {element for sublist in concat_list for element in sublist}

    grouped_keys = list(concat_list)
    grouped_values = [[orig_dict[item] for item in sublist] for sublist in grouped_keys]
    group_idx = [[key_to_idx[item] for item in sublist] for sublist in grouped_keys]

    for key in orig_dict:
        if key not in elements_in_groups:
            grouped_keys.append([key])
            grouped_values.append([orig_dict[key]])
            group_idx.append([key_to_idx[key]])

    return grouped_keys, grouped_values, group_idx

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    y = x / sqrt(mean(x**2) + eps) * weight + bias
    No mean subtraction.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # Scale (gamma)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        # Shift (beta)
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float() # ensure x is float32
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
        return x_norm.to(input_dtype)
    
class SwiGLUEmbedding(nn.Module):
    """A SwiGLU block for embedding a single observation group."""
    def __init__(self, input_dim: int, d_model: int, expansion_factor: int = 2, steps: int = 1):
        super().__init__()
        hidden_dim = int(expansion_factor * d_model)
        
        # The SwiGLU magic: two linear layers for the gate and value
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # The final projection layer
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # x is a single group's observation, e.g., (B, group_input_dim)
        gate = F.silu(self.w1(x)) # Swish activation for the gate
        value = self.w3(x)
        
        # Element-wise multiplication, followed by the final projection
        return self.w2(gate * value)
    
class MLPEmbedding(nn.Module):
    """A simple MLP block for embedding a single observation group."""
    def __init__(self, input_dim: int, d_model: int, expansion_factor: int = 2, steps: int = 1):
        super().__init__()
        hidden_dim = int(expansion_factor * d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, d_model)
        )
        
    def forward(self, x):
        return self.mlp(x)
    
    
class HistoryEncoder(nn.Module):
    """
    HistoryEncoder is a temporal encoder for history observations.
    It uses 1D convolutions for temporal feature extraction and a powerful SwiGLU layer for the final projection.
    """
    def __init__(self, history_length: int, group_per_step_dim: int, d_model: int, use_swiglu: bool = False, swiglu_expansion_factor: int = 2):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=group_per_step_dim, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        )
        flattened_size = 64 * history_length
        current_embedding = SwiGLUEmbedding if use_swiglu else MLPEmbedding
        self.projection = current_embedding(flattened_size, d_model, int(flattened_size / d_model)+1)
    def forward(self, x_seq: torch.Tensor):
        # x_seq has shape (B, history_length, group_per_step_dim)
        x_conv_in = x_seq.permute(0, 2, 1)
        conv_out = self.conv_net(x_conv_in)
        conv_out_flat = torch.flatten(conv_out, 1)
            
            # Use the powerful projection layer
        token = self.projection(conv_out_flat)
        
        return token
    
class HistoryEncoderSimple(nn.Module):
    '''
    A history encoder with fewer parameters for smaller models.
    '''
    def __init__(self, history_length: int, group_per_step_dim: int, d_model: int):
        '''
        This version computes history status from (B, history_length, group_per_step_dim) to (B, d_model) with a simpler architecture.
        We define hidden_dim = 64, and computes kernel size and stride based on history_length:
        kernel_size = max(2, history_length // 4)
        stride = max(1, history_length // 8)
        computation steps:
        1. Linear (B, history_length, group_per_step_dim) -> (B, history_length, 64)
        2. Permute to (B, hidden_dim, history_length)
        3. Conv1d, out_channels=32, kernel_size=kernel_size, stride=stride -> (B, 32, L_out)
        4. ReLU
        5. Conv1d, out_channels=16, kernel_size=kernel_size, stride=stride -> (B, 16, L_out2)
        6. ReLU
        7. Flatten to (B, 16 * L_out2)
        8. Linear to (B, d_model)
        '''
        super().__init__()
        kernel_size = max(2, history_length // 4)
        stride = max(1, history_length // 8)
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=group_per_step_dim, out_channels=32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        self.projection = nn.Linear(64 * self._compute_conv_output_length(history_length, kernel_size, stride, 2), d_model)
    
    def _compute_conv_output_length(self, input_length: int, kernel_size: int, stride: int, num_convs: int) -> int:
        length = input_length
        for _ in range(num_convs):
            length = (length - kernel_size) // stride + 1
        return length
    
    def forward(self, x_seq: torch.Tensor):
        # x_seq has shape (B, history_length, group_per_step_dim)
        x_conv_in = x_seq.permute(0, 2, 1)  # (B, group_per_step_dim, history_length)
        conv_out = self.conv_net(x_conv_in)    # (B, 64, L_out2)
        conv_out_flat = torch.flatten(conv_out, 1)  # (B, 64 * L_out2)
        token = self.projection(conv_out_flat)      # (B, d_model)
        return token
    
class HistoryEmbedding(nn.Module):
    '''
    HistoryEncoder Wrapper for form consistency as MLP Embedding ans SwiGLU Embedding
    '''
    def __init__(self, input_dim: int, d_model: int, expansion_factor: int = 2, steps: int = 1):
        super().__init__()
        self.encoder = HistoryEncoderSimple(history_length=steps, group_per_step_dim=input_dim//steps, d_model=d_model)
        self.steps = steps
    def forward(self, x):
        x = x.reshape(-1, self.steps, x.size(-1)//self.steps)
        return self.encoder(x)

############################################################################################################
#
# Observation Embedding
#
# ObservationEmbedding is the simplest approach for embedding observations.
# It projects observations to sequence length * dim_model, and then reshapes it to (B, seq_len, dim_model).
# This implementation is simple but may amplify the disturbance caused by the observation noise.
#
############################################################################################################

class ObservationEmbedding(nn.Module):
    def __init__(self, num_obs, d_model, max_len=16, apply_norm = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Linear(num_obs, d_model * max_len)
        self.norm = RMSNorm(d_model) if apply_norm else nn.Identity()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.size(0), -1, self.d_model)  # (B, seq_len, dim_model) 
        x = self.norm(x)  # Apply normalization       
        pos = torch.arange(self.max_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return x + self.pos_embedding(pos)
    
    
############################################################################################################
#
# Observation Embedding With Observation Length
#
# ObservationEmbeddingWithObsLen is another simple approach for embedding observations.
# It treats num_obs as seq_len, given each observation a unique id for representation.
# This implementation is simple but requires a lot of memory.
#
############################################################################################################

class ObservationEmbeddingWithObsLen(nn.Module):
    def __init__(self, num_obs, d_model, apply_norm = False): # max len won't be used
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Embedding(num_obs, d_model)
        self.norm = RMSNorm(d_model) if apply_norm else nn.Identity()
        
    def forward(self, x):
        x = x.unsqueeze(-1) # (B, seq_len, 1)
        x = self.embedding(x)  # (B, seq_len, dim_model)
        x = self.norm(x)  # Apply normalization
        idx = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.pos_embedding(idx)  # (B, seq_len, dim_model)
        return x
    
############################################################################################################
#
# Observation Embedding V2
#
# ObservationEmbeddingV2 treats different observation terms as different tokens
# It projects observation terms to dim_model, and then concatenates them to form a sequence.
# This implementation is better than the previous two, but still not perfect.
#
############################################################################################################

class ObservationEmbeddingV2(nn.Module):
    def __init__(self, d_model: int, term_dict: dict[str, int], apply_norm: bool = False, concatenate_term_names: list[list[str]] | None = None, history_length: int = 1):
        super().__init__()
        self.term_names     = list(term_dict.keys())
        self.term_dims      = list(term_dict.values())
        self.group_term_names, self.group_term_dims, self.group_term_idx = group_by_concat_list(term_dict, concatenate_term_names)
        self.seq_len        = len(self.group_term_names)
        self.d_model        = d_model
        self.history_length = history_length
        self.embeddings     = nn.ModuleList([
            SwiGLUEmbedding(sum(dims), d_model) if self.history_length <= 1 else HistoryEncoder(self.history_length, sum(dims)//self.history_length, d_model) for dims in self.group_term_dims
        ])
        self.term_embed     = nn.Embedding(self.seq_len, d_model)
        self.norm           = RMSNorm(d_model) if apply_norm else nn.Identity()
        start               = 0
        self.term_slices = []
        for dim in self.term_dims: # reserve original term slices for later slicing & concatenation
            self.term_slices.append(slice(start, start + dim))
            start += dim

    def forward(self, obs: torch.Tensor):
        # obs shape: (B, num_obs)
        token_list = []

        for i, group_term_dims, group_term_idx in zip(range(self.seq_len), self.group_term_dims, self.group_term_idx):
            if self.history_length > 1:
                group_obs = torch.cat([
                    obs[:, self.term_slices[j]].view(
                        -1,
                        self.history_length,
                        self.term_dims[j] // self.history_length
                    ) for j in group_term_idx], dim=-1) # (B, history_length, sum(group_term_dims) // history_length)
            else:
                group_obs = torch.cat([obs[:, self.term_slices[j]] for j in group_term_idx], dim=-1) # (B, sum(group_term_dims))
            token_i = self.embeddings[i](group_obs) # (B, d_model)
            token_list.append(token_i)
        x = torch.stack(token_list, dim=1) # (B, seq_len, d_model)
        x = self.norm(x)
        idx = torch.arange(self.seq_len, device=obs.device).unsqueeze(0) # (1, seq_len)
        x = x + self.term_embed(idx) # (B, seq_len, d_model)
        return x


    
class MMTransformer(nn.Module):
    def __init__(
            self,
            obs_size,
            ref_obs_size,
            dim_out,
            dim_model,
            max_len = 128,
            num_heads = 8,
            num_layers = 4,
            ffn_ratio = 4,
            dropout = 0.0,
            name = "",
            ls_init_values = 1e-3,
            apply_pooling = False,
            apply_mlp_residual = False,
            **kwargs
    ):
        super().__init__()
        self.name = name
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")
        self.obs_embedding = ObservationEmbedding(obs_size, dim_model, max_len, apply_norm=False)
        if ref_obs_size == 0:
            self.ref_obs_embedding = None
        else:
            self.ref_obs_embedding = ObservationEmbedding(ref_obs_size, dim_model, max_len, apply_norm=False)
        
        self.in_size = obs_size + ref_obs_size
            
        self.mlp_residual = nn.Sequential(
            nn.Linear(obs_size + ref_obs_size, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, dim_out),
        ) if apply_mlp_residual else None
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_model * ffn_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(dim_model))
        self.fc = nn.Sequential(
            # nn.Linear(dim_model, dim_model * 2),
            # nn.GELU(),
            # nn.Linear(dim_model * 2, dim_out),
            nn.Linear(dim_model, dim_out),
        )
        self.apply_pooling = apply_pooling
        # self.out_ls = LayerScale(dim_out, init_values=ls_init_values)

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for the multi-modality transformer.

        Args:
            obs (torch.Tensor): Observation tensor of shape (B, seq_len_obs, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (B, seq_len_ref_obs, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (B,) indicating the presence of ref_obs.

        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        embeddings = []
        padding_masks = []

        assert (self.ref_obs_embedding is not None) or (ref_obs is None), "Cannot run multi-modality mode with ref_obs_size=0"

        # -------------------
        # Process obs embeddings
        # -------------------
        obs_emb = self.obs_embedding(obs)  # Shape: (B, seq_len_obs, dim_model)
        cls_emb = self.cls_token.expand(obs.size(0), -1, -1)  # Shape: (B, 1, dim_model)
        sep_emb = self.sep_token.expand(obs.size(0), -1, -1)  # Shape: (B, 1, dim_model)
        

        # Create padding mask for obs: 1 for non-padding, 0 for padding
        obs_padding_mask = torch.ones(obs_emb.size(0), obs_emb.size(1), dtype=torch.bool, device=obs.device) # Shape: (B, seq_len_obs)
        # obs_seq_pad_mask shape: (seq_len_obs,) -> (B, seq_len_obs)
        # obs_seq_pad_mask is False for positions which should not be considered for calculating attention
        # obs_padding_mask = obs_padding_mask & obs_seq_pad_mask.unsqueeze(0)
        obs_padding_mask = F.pad(obs_padding_mask, (1, 1), value=1.0)  # Account for CLS and SEP tokens
        obs_emb = torch.cat([cls_emb, obs_emb, sep_emb], dim=1)  # Shape: (B, seq_len_obs + 2, dim_model)
        embeddings.append(obs_emb)
        padding_masks.append(obs_padding_mask)  # Shape: (B, seq_len_obs + 2)

        # -------------------
        # Process ref_obs embeddings (if provided)
        # -------------------
        if ref_obs is not None and self.ref_obs_embedding is not None:
            ref_obs_tensor, ref_obs_mask = ref_obs  # Unpack the tuple

            # Preprocess reference observations
            # ref_obs_tensor = self.ref_obs_deque(ref_obs_tensor)
            ref_obs_emb = self.ref_obs_embedding(ref_obs_tensor)  # Shape: (B, seq_len_ref_obs, dim_model)
            # cls_ref_emb = self.cls_token.expand(ref_obs_emb.size(0), -1, -1)  # Shape: (B, 1, dim_model)
            sep_ref_emb = self.sep_token.expand(ref_obs_emb.size(0), -1, -1)  # Shape: (B, 1, dim_model)
            
            # Create padding mask for ref_obs: 1 for non-padding, 0 for padding
            ref_obs_padding_mask = torch.ones(ref_obs_emb.size(0), ref_obs_emb.size(1), dtype=torch.bool, device=ref_obs_emb.device) # Shape: (B, seq_len_ref_obs)
            ref_obs_emb = torch.cat([ref_obs_emb, sep_ref_emb], dim=1)  # Shape: (B, seq_len_ref_obs + 2, dim_model)

            # Apply the ref_obs_mask to zero out embeddings where ref_obs is not present
            # ref_obs_mask shape: (B,) -> (B, 1, 1) for broadcasting
            ref_obs_mask = ref_obs_mask.view(-1, 1, 1)
            ref_obs_emb = ref_obs_emb * ref_obs_mask  # Zero out embeddings where ref_obs_mask is False

            embeddings.append(ref_obs_emb)  # Append to embeddings list

            # ref_obs_seq_pad_mask shape: (seq_len_ref_obs,) -> (B, seq_len_ref_obs)
            # ref_obs_seq_pad_mask is False for positions which should not be considered for calculating attention
            # ref_obs_padding_mask = ref_obs_padding_mask & ref_obs_seq_pad_mask.unsqueeze(0)
            ref_obs_padding_mask = F.pad(ref_obs_padding_mask, (0, 1), value=1.0)  # Account for CLS and SEP tokens

            # Update padding mask: set to 0 where ref_obs_mask is False
            # ref_obs_mask shape: (B, 1, 1) -> (B, 1)
            ref_obs_padding_mask = ref_obs_padding_mask & ref_obs_mask.squeeze(-1).squeeze(-1).unsqueeze(1)
            padding_masks.append(ref_obs_padding_mask)  # Shape: (B, seq_len_ref_obs + 2)

            

        # -------------------
        # Concatenate embeddings and padding masks
        # -------------------
        x = torch.cat(embeddings, dim=1)  # Shape: (B, total_seq_len, dim_model)

        if len(padding_masks) > 1:
            padding_mask = torch.cat(padding_masks, dim=1)  # Shape: (B, total_seq_len)
        else:
            padding_mask = padding_masks[0]  # Shape: (B, total_seq_len)

        padding_mask = ~padding_mask  # Invert mask: True for padding tokens

        # -------------------
        # Transformer processing
        # -------------------
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # Transformer expects shape (S, B, E)
        if not self.apply_pooling:
            x = x[:, 0, :]  # Take the first token's output: Shape (B, dim_model)
        else:
            padding_mask = ~padding_mask # Inverted mask: True for non-padding tokens
            # Apply average pooling over non-padding tokens
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)  # Set padding tokens to zero
            x = x.sum(dim=1)  # Sum over the sequence length dimension, x shape: (B, dim_model)
            x = x / padding_mask.sum(dim=1, keepdim=True)  # Normalize by the number of non-padding tokens

        # -------------------
        # Final fully connected layer
        # -------------------
        # Cache hidden state before projection (for DAgger training)
        self.last_hidden_state = x  # [B, dim_model]
        
        x = self.fc(x)  # Shape: (B, output_dim)
        if self.mlp_residual is not None:
            obs_in = obs
            if ref_obs is not None:
                obs_in = torch.cat([obs, ref_obs_tensor], dim=1)
            
            if obs_in.size(1) != self.in_size: # use zero padding
                obs_in = F.pad(obs_in, (0, self.in_size - obs_in.size(1)), value=0.0)
                
            x = x + self.mlp_residual(obs_in)
        # x = self.out_ls(x)
        return x
    
class MMTransformerWithSeqLen(nn.Module):
    def __init__(
            self,
            obs_size,
            ref_obs_size,
            dim_out,
            dim_model,
            num_heads = 4,
            num_layers = 4,
            ffn_ratio = 4,
            dropout = 0.0,
            name = "",
            ls_init_values = 1e-3,
            apply_mlp_residual = True,
            **kwargs
    ):
        super().__init__()
        self.name = name
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")
        self.obs_embedding = ObservationEmbeddingWithObsLen(obs_size, dim_model)
        if ref_obs_size == 0:
            self.ref_obs_embedding = None
        else:
            self.ref_obs_embedding = ObservationEmbedding(ref_obs_size, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_model * ffn_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(dim_model))
        self.fc = nn.Sequential(
            nn.Linear(dim_model, dim_out),
        )
        
        

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks=None,
        **kwargs
    ) -> torch.Tensor:

        embeddings = []
        padding_masks = []

        assert (self.ref_obs_embedding is not None) or (ref_obs is None), "Cannot run multi-modality mode with ref_obs_size=0"
        obs_emb = self.obs_embedding(obs)  # Shape: (B, seq_len_obs, dim_model)

        # Create padding mask for obs: 1 for non-padding, 0 for padding
        obs_padding_mask = torch.ones(obs_emb.size(0), obs_emb.size(1), dtype=torch.bool, device=obs.device) # Shape: (B, seq_len_obs)
        embeddings.append(obs_emb)
        padding_masks.append(obs_padding_mask)  # Shape: (B, seq_len_obs + 2)
        if ref_obs is not None and self.ref_obs_embedding is not None:
            ref_obs_tensor, ref_obs_mask = ref_obs  # Unpack the tuple
            ref_obs_emb = self.ref_obs_embedding(ref_obs_tensor)  # Shape: (B, seq_len_ref_obs, dim_model)
            ref_obs_padding_mask = torch.ones(ref_obs_emb.size(0), ref_obs_emb.size(1), dtype=torch.bool, device=ref_obs_emb.device) # Shape: (B, seq_len_ref_obs)
            ref_obs_mask = ref_obs_mask.view(-1, 1, 1)
            ref_obs_emb = ref_obs_emb * ref_obs_mask  # Zero out embeddings where ref_obs_mask is False
            embeddings.append(ref_obs_emb)  # Append to embeddings list
            ref_obs_padding_mask = ref_obs_padding_mask & ref_obs_mask.squeeze(-1).squeeze(-1).unsqueeze(1)
            padding_masks.append(ref_obs_padding_mask)  # Shape: (B, seq_len_ref_obs + 2)

        x = torch.cat(embeddings, dim=1)  # Shape: (B, total_seq_len, dim_model)

        if len(padding_masks) > 1:
            padding_mask = torch.cat(padding_masks, dim=1)  # Shape: (B, total_seq_len)
        else:
            padding_mask = padding_masks[0]  # Shape: (B, total_seq_len)

        padding_mask = ~padding_mask  # Invert mask: True for padding tokens
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # Transformer expects shape (S, B, E)
        padding_mask = ~padding_mask # Inverted mask: True for non-padding tokens
        # Apply average pooling over non-padding tokens
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)  # Set padding tokens to zero
        x = x.sum(dim=1)  # Sum over the sequence length dimension, x shape: (B, dim_model)
        x = x / padding_mask.sum(dim=1, keepdim=True)  # Normalize by the number of non-padding tokens
        x = self.fc(x)  # Shape: (B, output_dim)
        return x

class MMTransformerV2(nn.Module):
    def __init__(
            self,
            dim_out,
            dim_model,
            term_dict: dict,
            ref_term_dict: dict | None = None,
            concatenate_term_names: list[list[str]] | None = None,
            concatenate_ref_term_names: list[list[str]] | None = None,
            history_length: int = 1,
            num_heads = 8,
            num_layers = 4,
            ffn_ratio = 4,
            dropout = 0.0,
            name = "",
            ls_init_values = 1e-3,
            apply_pooling = False,
            apply_mlp_residual = True, # Warning: Do not set this to True if you are using a ref & without ref switching environment!
            mlp_weight = 0.5,
            **kwargs
    ):
        super().__init__()
        self.name = name
        self.mlp_weight = mlp_weight
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")
        self.obs_embedding = ObservationEmbeddingV2(dim_model, term_dict, concatenate_term_names=concatenate_term_names, history_length=history_length)
        if ref_term_dict:
            self.ref_obs_embedding = ObservationEmbeddingV2(dim_model, ref_term_dict, concatenate_term_names=concatenate_ref_term_names, history_length=history_length)
        else:
            self.ref_obs_embedding = None
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_model * ffn_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(dim_model))
        self.fc = nn.Sequential(
            nn.Linear(dim_model, dim_out),
        )
       
        
        self.apply_pooling = apply_pooling
        obs_size = sum(term_dict.values())
        ref_obs_size = sum(ref_term_dict.values()) if ref_term_dict else 0
        
        # New MLP residual design: separate primary and reference paths
        if apply_mlp_residual:
            # Primary MLP path (always active, processes obs)
            # self.mlp_primary = nn.Sequential(
            #     nn.Linear(obs_size, 512),
            #     nn.GELU(),
            #     nn.Linear(512, 256),
            #     nn.GELU(),
            #     nn.Linear(256, dim_out),
            # )
            # self.mlp_primary = MultiModalMLPV2(
            #     term_dict={name: term_dict},
            #     output_size=dim_out,
            #     hidden_dims=[512, 256, 128],
            #     history_length=history_length,
            # )
            
            # # Reference MLP path (only active when ref_obs is provided)
            # if ref_obs_size > 0:
            #     self.mlp_ref = MultiModalMLPV2(
            #         term_dict={name: ref_term_dict},
            #         output_size=dim_out,
            #         hidden_dims=[512, 256, 128],
            #         history_length=history_length,
            #     )
                
            #     # Gated fusion for MLP outputs
            #     self.mlp_gate = nn.Sequential(
            #         nn.Linear(dim_out * 3, dim_out),
            #         nn.Sigmoid()
            #     )
            # else:
            #     self.mlp_ref = None
            #     self.mlp_gate = None
            
            self.mlp_bypass = FusedMultiModalMLP(
                term_dict={'name': term_dict},
                ref_term_dict={'ref_name': ref_term_dict} if ref_term_dict else None,
                output_size=dim_model,
                hidden_dims=[512, 256],
                history_length=history_length,
                activation='elu',
                encoder_latent_dim=dim_model,
                encoder_compress_threshold=32,
                fusion_mode='gated',
                use_layer_norm=False,
                fuse_activation=True,
            )
            self.mlp_gate = self.mlp_gate = nn.Sequential(
                nn.Linear(dim_model * 2, dim_model * 4),
                nn.GELU(),
                nn.Linear(dim_model * 4, dim_model), 
                nn.Sigmoid()
            )
            nn.init.constant_(self.mlp_gate[-2].bias, -2.0)  # -3.0 in bias -> sigmoid ~ 0.5 initial gating towards transformer path
            
        else:
            self.mlp_bypass = None
            self.mlp_gate = None
            # self.mlp_primary = None
            # self.mlp_ref = None
            # self.mlp_gate = None
        # self.out_ls = LayerScale(dim_out, init_values=ls_init_values)

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for the multi-modality transformer.

        Args:
            obs (torch.Tensor): Observation tensor of shape (B, seq_len_obs, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (B, seq_len_ref_obs, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (B,) indicating the presence of ref_obs.

        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        embeddings = []
        padding_masks = []

        assert (self.ref_obs_embedding is not None) or (ref_obs is None), "Cannot run multi-modality mode with ref_obs_size=0"        
        # -------------------
        # Process obs embeddings
        # -------------------
        obs_emb = self.obs_embedding(obs)  # Shape: (B, seq_len_obs, dim_model)
        cls_emb = self.cls_token.expand(obs.size(0), -1, -1)  # Shape: (B, 1, dim_model)
        sep_emb = self.sep_token.expand(obs.size(0), -1, -1)  # Shape: (B, 1, dim_model)
        

        # Create padding mask for obs: 1 for non-padding, 0 for padding
        obs_padding_mask = torch.ones(obs_emb.size(0), obs_emb.size(1), dtype=torch.bool, device=obs.device) # Shape: (B, seq_len_obs)
        # obs_seq_pad_mask shape: (seq_len_obs,) -> (B, seq_len_obs)
        # obs_seq_pad_mask is False for positions which should not be considered for calculating attention
        # obs_padding_mask = obs_padding_mask & obs_seq_pad_mask.unsqueeze(0)
        obs_padding_mask = F.pad(obs_padding_mask, (1, 1), value=1.0)  # Account for CLS and SEP tokens
        obs_emb = torch.cat([cls_emb, obs_emb, sep_emb], dim=1)  # Shape: (B, seq_len_obs + 2, dim_model)
        embeddings.append(obs_emb)
        padding_masks.append(obs_padding_mask)  # Shape: (B, seq_len_obs + 2)

        # -------------------
        # Process ref_obs embeddings (if provided)
        # -------------------
        if ref_obs is not None and self.ref_obs_embedding is not None:
            ref_obs_tensor, ref_obs_mask = ref_obs  # Unpack the tuple

            # Preprocess reference observations
            # ref_obs_tensor = self.ref_obs_deque(ref_obs_tensor)
            ref_obs_emb = self.ref_obs_embedding(ref_obs_tensor)  # Shape: (B, seq_len_ref_obs, dim_model)
            # cls_ref_emb = self.cls_token.expand(ref_obs_emb.size(0), -1, -1)  # Shape: (B, 1, dim_model)
            sep_ref_emb = self.sep_token.expand(ref_obs_emb.size(0), -1, -1)  # Shape: (B, 1, dim_model)
            
            # Create padding mask for ref_obs: 1 for non-padding, 0 for padding
            ref_obs_padding_mask = torch.ones(ref_obs_emb.size(0), ref_obs_emb.size(1), dtype=torch.bool, device=ref_obs_emb.device) # Shape: (B, seq_len_ref_obs)
            ref_obs_emb = torch.cat([ref_obs_emb, sep_ref_emb], dim=1)  # Shape: (B, seq_len_ref_obs + 2, dim_model)

            # Apply the ref_obs_mask to zero out embeddings where ref_obs is not present
            # ref_obs_mask shape: (B,) -> (B, 1, 1) for broadcasting
            ref_obs_mask = ref_obs_mask.view(-1, 1, 1)
            ref_obs_emb = ref_obs_emb * ref_obs_mask  # Zero out embeddings where ref_obs_mask is False

            embeddings.append(ref_obs_emb)  # Append to embeddings list

            # ref_obs_seq_pad_mask shape: (seq_len_ref_obs,) -> (B, seq_len_ref_obs)
            # ref_obs_seq_pad_mask is False for positions which should not be considered for calculating attention
            # ref_obs_padding_mask = ref_obs_padding_mask & ref_obs_seq_pad_mask.unsqueeze(0)
            ref_obs_padding_mask = F.pad(ref_obs_padding_mask, (0, 1), value=1.0)  # Account for CLS and SEP tokens

            # Update padding mask: set to 0 where ref_obs_mask is False
            # ref_obs_mask shape: (B, 1, 1) -> (B, 1)
            ref_obs_padding_mask = ref_obs_padding_mask & ref_obs_mask.squeeze(-1).squeeze(-1).unsqueeze(1)
            padding_masks.append(ref_obs_padding_mask)  # Shape: (B, seq_len_ref_obs + 2)

            

        # -------------------
        # Concatenate embeddings and padding masks
        # -------------------
        x = torch.cat(embeddings, dim=1)  # Shape: (B, total_seq_len, dim_model)

        if len(padding_masks) > 1:
            padding_mask = torch.cat(padding_masks, dim=1)  # Shape: (B, total_seq_len)
        else:
            padding_mask = padding_masks[0]  # Shape: (B, total_seq_len)

        padding_mask = ~padding_mask  # Invert mask: True for padding tokens

        # -------------------
        # Transformer processing
        # -------------------
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # Transformer expects shape (S, B, E)
        if not self.apply_pooling:
            x = x[:, 0, :]  # Take the first token's output: Shape (B, dim_model)
        else:
            padding_mask = ~padding_mask # Inverted mask: True for non-padding tokens
            # Apply average pooling over non-padding tokens
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)  # Set padding tokens to zero
            x = x.sum(dim=1)  # Sum over the sequence length dimension, x shape: (B, dim_model)
            x = x / padding_mask.sum(dim=1, keepdim=True)  # Normalize by the number of non-padding tokens

        # -------------------
        # Final fully connected layer
        # -------------------
        # y = self.fc(x)  # Shape: (B, output_dim)
        
        # New MLP residual: separate primary and reference paths with gated fusion
        if self.mlp_bypass is not None:
            x_mlp = self.mlp_bypass(obs, ref_obs)
            gate_input = torch.cat([x, x_mlp], dim=-1)
            gate = self.mlp_gate(gate_input)
            x_fused = (1 - gate) * x_mlp  + gate * x
        else:
            x_fused = x
            
        y = self.fc(x_fused)
        return y

class Transformer(nn.Module):
    '''
        This class is for debugging only.
        Do not use it for real PPO training.
    '''
    def __init__(
            self,
            obs_size,
            ref_obs_size,
            dim_out,
            dim_model,
            max_len = 128,
            num_heads = 8,
            num_layers = 4,
            ffn_ratio = 4,
            dropout = 0.0,
            name = "",
            ls_init_values = 1e-3,
            apply_pooling = True,
            **kwargs
    ):
        super().__init__()
        self.name = name
        self.obs_size = obs_size
        self.ref_obs_size = ref_obs_size
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")
        self.obs_embedding = ObservationEmbedding(obs_size + ref_obs_size, dim_model, max_len)
        # self.obs_embedding = ObservationEmbedding(obs_size + ref_obs_size, dim_model, max_len)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_model * ffn_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(dim_model))
        self.fc = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_out),
        )
        self.apply_pooling = apply_pooling

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for the multi-modality transformer.

        Args:
            obs (torch.Tensor): Observation tensor of shape (B, seq_len_obs, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (B, seq_len_ref_obs, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (B,) indicating the presence of ref_obs.

        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        embeddings = []

        if ref_obs is not None:
            ref_obs_tensor, _ = ref_obs  # Unpack the tuple
        else:
            ref_obs_tensor = torch.zeros(self.ref_obs_size, device=obs.device).unsqueeze(0).expand(obs.size(0), -1)  # Shape: (B, seq_len_ref_obs)
        obs = torch.cat([obs, ref_obs_tensor], dim=1)  # Concatenate obs and ref_obs along the last dimension
        obs_emb = self.obs_embedding(obs)  # Shape: (B, seq_len_obs, dim_model)
        cls_emb = self.cls_token.expand(obs.size(0), -1, -1)  # Shape: (B, 1, dim_model)
        
        obs_emb = torch.cat([cls_emb, obs_emb], dim=1)  # Shape: (B, seq_len_obs + 2, dim_model)
        embeddings.append(obs_emb)
            
        x = torch.cat(embeddings, dim=1)  # Shape: (B, total_seq_len, dim_model)
        # -------------------
        # Transformer processing
        # -------------------
        x = self.transformer(x)  # Transformer expects shape (S, B, E)
        x = x[:, 0, :]  # Take the first token's output: Shape (B, dim_model)

        # -------------------
        # Final fully connected layer
        # -------------------
        x = self.fc(x)  # Shape: (B, output_dim)
        # x = self.out_ls(x)
        return x
    
class DebugMLP(nn.Module):
    '''
        This class is for debugging only.
        Do not use it for real PPO training.
    '''
    def __init__(
            self,
            obs_size,
            ref_obs_size,
            dim_out,
            dim_model = 0,
            max_len = 128,
            num_heads = 8,
            num_layers = 4,
            ffn_ratio = 4,
            dropout = 0.0,
            name = "",
            ls_init_values = 1e-3,
            layer_dims = [512, 256, 128],
            **kwargs
    ):
        super().__init__()
        self.name = name
        self.obs_size = obs_size
        self.ref_obs_size = ref_obs_size
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")
        self.layers_dim = layer_dims
        self.layers = nn.Sequential(
            nn.Linear(obs_size + ref_obs_size, self.layers_dim[0]),
            nn.ReLU(),
            nn.Linear(self.layers_dim[0], self.layers_dim[1]),
            nn.ReLU(),
            nn.Linear(self.layers_dim[1], self.layers_dim[2]),
            nn.ReLU(),
            nn.Linear(self.layers_dim[2], dim_out),
        )

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for the multi-modality transformer.

        Args:
            obs (torch.Tensor): Observation tensor of shape (B, seq_len_obs, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (B, seq_len_ref_obs, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (B,) indicating the presence of ref_obs.

        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        embeddings = []

        if ref_obs is not None:
            ref_obs_tensor, _ = ref_obs  # Unpack the tuple
        else:
            ref_obs_tensor = torch.zeros(self.ref_obs_size, device=obs.device).unsqueeze(0).expand(obs.size(0), -1)  # Shape: (B, seq_len_ref_obs)
        obs = torch.cat([obs, ref_obs_tensor], dim=1)  # Concatenate obs and ref_obs along the last dimension
        x = self.layers(obs)  # Shape: (B, dim_out)
        return x


class ActorCriticMMTransformer(nn.Module):
    is_recurrent = False
    def __init__(
            self,
            num_actor_obs,
            num_actor_ref_obs,
            num_critic_obs,
            num_critic_ref_obs,
            num_actions,
            max_len=16,
            dim_model=128,
            num_layers=4,
            num_heads=8,
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            load_dagger=False,
            load_dagger_path=None,
            load_actor_path=None,
            enable_lora=True,
            dropout=0.05,
            **kwargs
    ):
        super().__init__()
        assert not load_dagger or load_dagger_path, "load_dagger and load_dagger_path must be provided if load_dagger is True"
        self.actor = MMTransformer(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="actor", dropout=dropout, **kwargs)
        self.actor_dagger = MMTransformer(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="actor_dagger", dropout=dropout, **kwargs) if load_dagger else None
        

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        if load_dagger:
            self.load_dagger_weights(load_dagger_path)
            lora_r = kwargs.get('lora_r', 8)
            lora_alpha = kwargs.get('lora_alpha', 16)
            lora_dropout = kwargs.get('lora_dropout', 0.05)
            if enable_lora:
                self.apply_dagger_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            
        self.critic = MMTransformer(num_critic_obs, num_critic_ref_obs, 1, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="critic", dropout=dropout, **kwargs) # 1 for value function
        
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
            
            
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")
        print(f"Dagger Model: {self.actor_dagger}")
        
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)


    def load_actor_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        # check for 'actor' in the state_dict keys
        assert 'model_state_dict' in state_dict.keys(), f"Key 'model_state_dict' not found in state_dict keys: {state_dict.keys()}, check if your model is the correct one created by rsl_rl"
        
        model_state_dict = state_dict['model_state_dict']
        # load the actor_dagger weights through layer name matching (starting with 'actor')
        actor_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor')}
        critic_weights = {k[len('critic.'):]: v for k, v in model_state_dict.items() if k.startswith('critic')}
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        new_actor_weights = {}
        new_critic_weights = {}
        # perform weights checking
        for k, v in actor_weights.items():
            if k not in actor_state_dict:
                print(f"Warning: Key {k} not found in actor state_dict, removing...")
                continue
            if actor_state_dict[k].shape != v.shape:
                print(f"Warning: Shape mismatch for key {k}: {actor_state_dict[k].shape} vs {v.shape}, removing...")
                continue
            new_actor_weights[k] = v
                
        for k, v in critic_weights.items():
            if k not in critic_state_dict:
                print(f"Warning: Key {k} not found in critic state_dict, removing...")
                continue
            if critic_state_dict[k].shape != v.shape:
                print(f"Warning: Shape mismatch for key {k}: {critic_state_dict[k].shape} vs {v.shape}, removing...")
                continue
            new_critic_weights[k] = v   
            
        actor_weights = new_actor_weights
        critic_weights = new_critic_weights
        # perform actor state dict fullfilling
        for k, v in actor_state_dict.items():
            if k not in actor_weights:
                actor_weights[k] = v
                
        for k, v in critic_state_dict.items():
            if k not in critic_weights:
                critic_weights[k] = v
        # load the weights
        # self.actor.load_state_dict(dagger_weights)
        self.actor.load_state_dict(actor_weights)
        # self.critic.load_state_dict(critic_weights)
        print(f"Loaded actor weights from {path} to actor and critic")
        
        # load std
        if self.noise_std_type == "scalar":
            assert "std" in model_state_dict.keys(), f"Key 'std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.std.data = model_state_dict["std"]
        elif self.noise_std_type == "log":
            assert "log_std" in model_state_dict.keys(), f"Key 'log_std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.log_std.data = model_state_dict["log_std"]
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        print(f"Loaded actor noise std from {path}")
        
        
    def load_critic_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        # check for 'actor' in the state_dict keys
        assert 'model_state_dict' in state_dict.keys(), f"Key 'model_state_dict' not found in state_dict keys: {state_dict.keys()}, check if your model is the correct one created by rsl_rl"
        
        model_state_dict = state_dict['model_state_dict']
        # load the actor_dagger weights through layer name matching (starting with 'actor')
        # actor_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor')}
        critic_weights = {k[len('critic.'):]: v for k, v in model_state_dict.items() if k.startswith('critic')}
        # actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        new_actor_weights = {}
        new_critic_weights = {}
        # perform weights checking
        # for k, v in actor_weights.items():
        #     if k not in actor_state_dict:
        #         print(f"Warning: Key {k} not found in actor state_dict, removing...")
        #         continue
        #     if actor_state_dict[k].shape != v.shape:
        #         print(f"Warning: Shape mismatch for key {k}: {actor_state_dict[k].shape} vs {v.shape}, removing...")
        #         continue
        #     new_actor_weights[k] = v
                
        for k, v in critic_weights.items():
            if k not in critic_state_dict:
                print(f"Warning: Key {k} not found in critic state_dict, removing...")
                continue
            if critic_state_dict[k].shape != v.shape:
                print(f"Warning: Shape mismatch for key {k}: {critic_state_dict[k].shape} vs {v.shape}, removing...")
                continue
            new_critic_weights[k] = v   
            
        # actor_weights = new_actor_weights
        critic_weights = new_critic_weights
        # perform actor state dict fullfilling
        # for k, v in actor_state_dict.items():
        #     if k not in actor_weights:
        #         actor_weights[k] = v
                
        for k, v in critic_state_dict.items():
            if k not in critic_weights:
                critic_weights[k] = v
        # load the weights
        # self.actor.load_state_dict(dagger_weights)
        # self.actor.load_state_dict(actor_weights)
        self.critic.load_state_dict(critic_weights)
        print(f"Loaded actor weights from {path} to actor and critic")
        
        # load std
        if self.noise_std_type == "scalar":
            assert "std" in model_state_dict.keys(), f"Key 'std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.std.data = model_state_dict["std"]
        elif self.noise_std_type == "log":
            assert "log_std" in model_state_dict.keys(), f"Key 'log_std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.log_std.data = model_state_dict["log_std"]
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        print(f"Loaded actor noise std from {path}")
        
    def load_dagger_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        # check for 'actor' in the state_dict keys
        assert 'model_state_dict' in state_dict.keys(), f"Key 'model_state_dict' not found in state_dict keys: {state_dict.keys()}, check if your model is the correct one created by rsl_rl"
        
        model_state_dict = state_dict['model_state_dict']
        # load the actor_dagger weights through layer name matching (starting with 'actor')
        dagger_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor')}
        dagger_state_dict = self.actor_dagger.state_dict()
        # perform weights checking
        for k, v in dagger_weights.items():
            if k not in dagger_state_dict:
                raise KeyError(f"Key {k} not found in actor_dagger state_dict")
            if dagger_state_dict[k].shape != v.shape:
                raise ValueError(f"Shape mismatch for key {k}: {dagger_state_dict[k].shape} vs {v.shape}")
        # load the weights
        # self.actor.load_state_dict(dagger_weights)
        self.actor_dagger.load_state_dict(dagger_weights)
        # self.actor.load_state_dict(dagger_weights)
        if self.noise_std_type == "scalar":
            assert "std" in model_state_dict.keys(), f"Key 'std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            # self.std.data = model_state_dict["std"]
            self.std_dagger = nn.Parameter(model_state_dict["std"])
        elif self.noise_std_type == "log":
            assert "log_std" in model_state_dict.keys(), f"Key 'log_std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            # self.log_std.data = model_state_dict["log_std"]
            self.log_std_dagger = nn.Parameter(model_state_dict["log_std"])
            self.log_std_dagger.requires_grad = False
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        self.distribution_dagger = None
            
        print(f"Loaded dagger weights from {path} to actor_dagger")
        
    def apply_dagger_lora(self, r=8, alpha=16, dropout=0.05):
        for param in self.actor_dagger.transformer.parameters():
            param.requires_grad = False
        
        for name, module in self.actor_dagger.transformer.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Replace the linear projections with LoRA layers
                d_model = module.embed_dim
                module.out_proj = LoRALinear(
                    in_features=d_model,
                    out_features=d_model,
                    base_layer=module.out_proj,
                    adapter_name="default",
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    bias=False
                )

        # Unfreeze LoRA parameters explicitly
        for module in self.actor_dagger.transformer.modules():
            if isinstance(module, LoRALinear):
                for param in module.parameters():
                    param.requires_grad = True

        print("LoRA manually applied, other parameters frozen.")

        
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    @property
    def action_mean_dagger(self):
        return self.distribution_dagger.mean
    
    @property
    def action_std_dagger(self):
        return self.distribution_dagger.stddev
    
    @property
    def entropy_dagger(self):
        return self.distribution_dagger.entropy().sum(dim=-1)

    def update_distribution(self, observations, ref_observations=None, **kwargs):
        mean = self.actor(observations, ref_observations, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, ref_observations=None, **kwargs):
        self.update_distribution(observations, ref_observations, **kwargs)
        sample = self.distribution.sample()
        return sample
    
    def update_distribution_dagger(self, observations, ref_observations=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        mean = self.actor_dagger(observations, ref_observations, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std_dagger.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std_dagger).expand_as(mean)
        self.distribution_dagger = Normal(mean, std)
   
    def act_dagger(self, observations, ref_observations=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        self.update_distribution_dagger(observations, ref_observations, **kwargs)
        sample = self.distribution_dagger.sample()
        return sample
    
    def act_dagger_inference(self, observations, ref_observations=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        actions_mean = self.actor_dagger(observations, ref_observations, **kwargs)
        return actions_mean

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_actions_log_prob_dagger(self, actions):
        return self.distribution_dagger.log_prob(actions).sum(dim=-1)
    
    def get_hidden_alignment_loss(self):
        """
        Compute alignment loss between student (actor) and teacher (actor_dagger) 
        hidden representations (before the final projection layer).
        
        This enables ASAP-style training in hidden space rather than action space.
        
        Returns:
            torch.Tensor: L2 distance between hidden states, shape []
        """
        if self.actor_dagger is None:
            raise RuntimeError("actor_dagger is not initialized. Cannot compute hidden alignment loss.")
        
        if not hasattr(self.actor, 'last_hidden_state') or not hasattr(self.actor_dagger, 'last_hidden_state'):
            raise RuntimeError(
                "Hidden states not cached. Make sure to call forward pass on both actor and actor_dagger "
                "before computing alignment loss."
            )
        
        student_hidden = self.actor.last_hidden_state  # [B, dim_model]
        teacher_hidden = self.actor_dagger.last_hidden_state  # [B, dim_model]
        
        # L2 distance (ASAP style)
        loss = (student_hidden - teacher_hidden.detach()).norm(p=2, dim=1).mean()
        
        return loss

    def act_inference(self, observations, ref_observations=None, **kwargs):
        actions_mean = self.actor(observations, ref_observations, **kwargs)
        return actions_mean

    def evaluate(self, critic_observations, ref_critic_observations=None, **kwargs):
        value = self.critic(critic_observations, ref_critic_observations, **kwargs)
        return value

class ActorCriticDebugMLP(nn.Module):
    is_recurrent = False
    def __init__(
            self,
            num_actor_obs,
            num_actor_ref_obs,
            num_critic_obs,
            num_critic_ref_obs,
            num_actions,
            max_len=16,
            dim_model=128,
            num_layers=4,
            num_heads=8,
            init_noise_std=1.0,
            load_dagger=False,
            load_dagger_path=None,
            load_actor_path=None,
            noise_std_type: str = "scalar",
            **kwargs
    ):
        super().__init__()
        self.actor = DebugMLP(num_actor_obs, num_actor_ref_obs, num_actions)
        self.critic = DebugMLP(num_critic_obs, num_critic_ref_obs, 1, layer_dims=[768, 512, 128]) # 1 for value function
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        self.noise_std_type = noise_std_type
        # Action noise
        if noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif noise_std_type == "scalar":    
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        if load_dagger:
            self.actor_dagger = DebugMLP(num_actor_obs, 0, num_actions)
            print(f"Dagger Model: {self.actor_dagger}")
            self.load_dagger_weights(load_dagger_path)
        else:
            self.actor_dagger = None
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
    
    
    def load_actor_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        # check for 'actor' in the state_dict keys
        assert 'model_state_dict' in state_dict.keys(), f"Key 'model_state_dict' not found in state_dict keys: {state_dict.keys()}, check if your model is the correct one created by rsl_rl"
        
        model_state_dict = state_dict['model_state_dict']
        # load the actor_dagger weights through layer name matching (starting with 'actor')
        actor_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor')}
        critic_weights = {k[len('critic.'):]: v for k, v in model_state_dict.items() if k.startswith('critic')}
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        new_actor_weights = {}
        new_critic_weights = {}
        # perform weights checking
        for k, v in actor_weights.items():
            if k not in actor_state_dict:
                print(f"Warning: Key {k} not found in actor state_dict, removing...")
                continue
            if actor_state_dict[k].shape != v.shape:
                print(f"Warning: Shape mismatch for key {k}: {actor_state_dict[k].shape} vs {v.shape}, removing...")
                continue
            new_actor_weights[k] = v
                
        for k, v in critic_weights.items():
            if k not in critic_state_dict:
                print(f"Warning: Key {k} not found in critic state_dict, removing...")
                continue
            if critic_state_dict[k].shape != v.shape:
                print(f"Warning: Shape mismatch for key {k}: {critic_state_dict[k].shape} vs {v.shape}, removing...")
                continue
            new_critic_weights[k] = v   
            
        actor_weights = new_actor_weights
        critic_weights = new_critic_weights
        # perform actor state dict fullfilling
        for k, v in actor_state_dict.items():
            if k not in actor_weights:
                actor_weights[k] = v
                
        for k, v in critic_state_dict.items():
            if k not in critic_weights:
                critic_weights[k] = v
        # load the weights
        # self.actor.load_state_dict(dagger_weights)
        self.actor.load_state_dict(actor_weights)
        # self.critic.load_state_dict(critic_weights)
        print(f"Loaded actor weights from {path} to actor and critic")
        
        # load std
        if self.noise_std_type == "scalar":
            assert "std" in model_state_dict.keys(), f"Key 'std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.std.data = model_state_dict["std"]
        elif self.noise_std_type == "log":
            assert "log_std" in model_state_dict.keys(), f"Key 'log_std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.log_std.data = model_state_dict["log_std"]
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        print(f"Loaded actor noise std from {path}")
        
    def load_dagger_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        # check for 'actor' in the state_dict keys
        assert 'model_state_dict' in state_dict.keys(), f"Key 'model_state_dict' not found in state_dict keys: {state_dict.keys()}, check if your model is the correct one created by rsl_rl"
        
        model_state_dict = state_dict['model_state_dict']
        # load the actor_dagger weights through layer name matching (starting with 'actor')
        dagger_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor')}
        dagger_state_dict = self.actor_dagger.state_dict()
        # perform weights checking
        for k, v in dagger_weights.items():
            if k not in dagger_state_dict:
                raise KeyError(f"Key {k} not found in actor_dagger state_dict")
            if dagger_state_dict[k].shape != v.shape:
                raise ValueError(f"Shape mismatch for key {k}: {dagger_state_dict[k].shape} vs {v.shape}")
        # load the weights
        # self.actor.load_state_dict(dagger_weights)
        self.actor_dagger.load_state_dict(dagger_weights)
        if self.noise_std_type == "scalar":
            assert "std" in model_state_dict.keys(), f"Key 'std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.std.data = model_state_dict["std"]
            self.std_dagger = nn.Parameter(model_state_dict["std"])
            self.std_dagger.requires_grad = False
        elif self.noise_std_type == "log":
            assert "log_std" in model_state_dict.keys(), f"Key 'log_std' not found in state_dict keys: {model_state_dict.keys()}, check if your noise_std_type is correct"
            self.log_std.data = model_state_dict["log_std"]
            self.log_std_dagger = nn.Parameter(model_state_dict["log_std"])
            self.log_std_dagger.requires_grad = False
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        self.distribution_dagger = None
            
        print(f"Loaded dagger weights from {path} to actor_dagger")
        

    def reset(self, dones=None):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def action_mean_dagger(self):
        return self.distribution_dagger.mean
    
    @property
    def action_std_dagger(self):
        return self.distribution_dagger.stddev
    
    @property
    def entropy_dagger(self):
        return self.distribution_dagger.entropy().sum(dim=-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, ref_observations=None, **kwargs):
        mean = self.actor(observations, ref_observations, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, ref_observations=None, **kwargs):
        self.update_distribution(observations, ref_observations, **kwargs)
        sample = self.distribution.sample()
        return sample
        
    # def act_inference(self, observations, ref_observations=None, **kwargs):
    #     return self.actor(observations, ref_observations, **kwargs)

    def act_inference(self, observations, ref_observations=None, **kwargs):
        actions_mean = self.actor(observations, ref_observations, **kwargs)
        return actions_mean
    
    
    def update_distribution_dagger(self, observations, ref_observations=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        mean = self.actor_dagger(observations, ref_observations, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std_dagger.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std_dagger).expand_as(mean)
        self.distribution_dagger = Normal(mean, std)
   
    def act_dagger(self, observations, ref_observations=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        self.update_distribution_dagger(observations, ref_observations, **kwargs)
        sample = self.distribution_dagger.sample()
        return sample
    
    def act_dagger_inference(self, observations, ref_observations=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        actions_mean = self.actor_dagger(observations, ref_observations, **kwargs)
        return actions_mean

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_actions_log_prob_dagger(self, actions):
        return self.distribution_dagger.log_prob(actions).sum(dim=-1)


    def evaluate(self, critic_observations, ref_critic_observations=None, **kwargs):
        value = self.critic(critic_observations, ref_critic_observations, **kwargs)
        return value
    

class ActorCriticMMTransformerV2(ActorCriticMMTransformer):
    def __init__(
            self,
            term_dict,
            ref_term_dict,
            num_actions,
            history_length=1,
            concatenate_term_names=None,
            concatenate_ref_term_names=None,
            max_len=16,
            dim_model=128,
            num_layers=4,
            num_heads=8,
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            load_dagger=False,
            load_dagger_path=None,
            load_actor_path=None,
            enable_lora=False,
            dropout=0.05,
            **kwargs
    ):
        nn.Module.__init__(self)
        assert not load_dagger or load_dagger_path, "load_dagger and load_dagger_path must be provided if load_dagger is True"
        self.actor = MMTransformerV2(num_actions, dim_model, term_dict["policy"], ref_term_dict["policy"], max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="actor", dropout=dropout, concatenate_term_names=concatenate_term_names["policy"], concatenate_ref_term_names=concatenate_ref_term_names["policy"], history_length=history_length, **kwargs)
        self.actor_dagger = MMTransformerV2(num_actions, dim_model, term_dict["policy"], None, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="actor_dagger", dropout=dropout, concatenate_term_names=concatenate_term_names["policy"], concatenate_ref_term_names=None, history_length=history_length, **kwargs) if load_dagger else None
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        if load_dagger:
            self.load_dagger_weights(load_dagger_path)
            lora_r = kwargs.get('lora_r', 8)
            lora_alpha = kwargs.get('lora_alpha', 16)
            lora_dropout = kwargs.get('lora_dropout', 0.05)
            if enable_lora:
                self.apply_dagger_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            
        # self.critic = MMTransformerV2(1, dim_model, term_dict["critic"], ref_term_dict["critic"], max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="critic", dropout=dropout, concatenate_term_names=concatenate_term_names["critic"], concatenate_ref_term_names=concatenate_ref_term_names["critic"], history_length=history_length, **kwargs) # 1 for value function
        
        critic_term_dict = {"critic": term_dict.get("critic", term_dict.get("policy", {}))}
        critic_ref_term_dict = {"critic": ref_term_dict.get("critic", ref_term_dict.get("policy", {}))} if ref_term_dict else None
        
        self.critic = FusedMultiModalMLP(
            term_dict=critic_term_dict,
            ref_term_dict=critic_ref_term_dict,
            output_size=1,  # Value function outputs single value
            hidden_dims=[512, 256, 128],
            activation="elu",
            history_length=history_length,
            encoder_latent_dim=dim_model,
            encoder_compress_threshold=32,
            use_layer_norm=False,
            fusion_mode='gated',
            name="critic"
        )
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")
        print(f"Dagger Model: {self.actor_dagger}")
        
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        
