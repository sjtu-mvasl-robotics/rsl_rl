# Copyright (c) 2021-2025, SJTU MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import warnings

from torch.distributed.checkpoint import state_dict
from rsl_rl.modules import ActorCritic, ActorCriticMMTransformer, SwiGLUEmbedding, group_by_concat_list, HistoryEmbedding
from rsl_rl.modules.actor_critic_mlp_v2 import DirectConcatMLP, FusedMultiModalMLP
from rsl_rl.networks import Memory, RoPETransformer
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from torch.distributions import Normal
import math
from typing import Optional, Tuple, List, Union, Dict
from torch.func import functional_call, vmap



def generate_block_causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a block causal mask for a sequence where each time step consists of two tokens:
    (obs_t, ref_t). The mask ensures that within each time step, both tokens can attend to each other.

    Args:
        T (int): The length of the sequence (i.e., the number of time steps).
        device (torch.device): The device where the tensor is located.
        dtype (torch.dtype): The data type of the tensor.
    Returns:
        torch.Tensor: A boolean mask matrix of shape [2*T, 2*T].
                      True indicates attention is allowed. False indicates attention is blocked.
    """
    inter_step_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    intra_step_mask = torch.ones(2, 2, device=device, dtype=torch.bool)
    block_causal_mask = torch.kron(inter_step_mask, intra_step_mask).to(dtype)
    
    return block_causal_mask
# Sequence Compressor (Deprecated)
class SequenceCompressor(nn.Module):
    """
    Compress the input sequence by a given stride using linear projection and max pooling.
    """
    def __init__(self, d_model: int, stride: int):
        super().__init__()
        self.stride = stride
        self.projector = nn.Linear(d_model * stride, d_model)
        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, seq: torch.Tensor, masks: torch.Tensor):
        raise NotImplementedError("This class is deprecated. Please use Conv1dCompressor instead.")
        # seq: [T, B, D], masks: [T, B]
        T, B, D = seq.shape
        
        padding_needed = (self.stride - T % self.stride) % self.stride
        if padding_needed > 0:
            padding_tensor = torch.zeros(padding_needed, B, D, device=seq.device)
            seq = torch.cat([seq, padding_tensor], dim=0)
            padding_mask = torch.zeros(padding_needed, B, dtype=torch.bool, device=seq.device)
            masks = torch.cat([masks, padding_mask], dim=0)
        
        T_padded = seq.shape[0]

        # [T_padded, B, D] -> [B, T_padded, D] -> [B, T_padded/stride, stride*D]
        seq_chunked = seq.permute(1, 0, 2).reshape(B, T_padded // self.stride, self.stride * D)
        # [B, T_padded/stride, stride*D] -> [B, T_padded/stride, D]
        compressed_seq = self.projector(seq_chunked)
        # -> [T_padded/stride, B, D]
        compressed_seq = compressed_seq.permute(1, 0, 2)
        
        # [T_padded, B] -> [B, 1, T_padded]
        mask_for_pool = masks.permute(1, 0).unsqueeze(1).float()
        new_mask_permuted = self.pool(mask_for_pool) # -> [B, 1, T_padded/stride]
        new_masks = new_mask_permuted.squeeze(1).permute(1, 0).bool() # -> [T_padded/stride, B]
        
        return compressed_seq, new_masks
    

class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, dim_model, expansion_factor: int = 2, steps: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.dim_model = dim_model
        self.expansion_factor = expansion_factor
        
        hidden_dim = int(dim_model * expansion_factor)
        if expansion_factor > 1:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, dim_model)
            )
        else:
            self.projector = nn.Linear(input_dim, dim_model)
    def forward(self, x: torch.Tensor):
        return self.projector(x)


class SonicTemporalChunkEncoder(nn.Module):
    """Compress one causal, non-overlapping frame chunk into one token.

    SONIC's released motion encoder flattens a fixed temporal axis before an
    MLP maps it to a smaller token axis.  This module adapts that contract to
    online policy history: observations and references are first fused per
    frame, then up to ``chunk_size`` episode-local frames are flattened into a
    single token.  ``slot_mask`` makes partial chunks explicit so inference can
    still produce a 50 Hz action before a five-frame chunk is complete.

    This is deliberately a continuous token encoder, not SONIC's FSQ action
    tokenizer.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int,
        has_reference: bool,
    ):
        super().__init__()
        if chunk_size < 2:
            raise ValueError("temporal chunk size must be at least 2")
        self.d_model = int(d_model)
        self.chunk_size = int(chunk_size)
        self.has_reference = bool(has_reference)

        frame_input_dim = d_model
        if self.has_reference:
            frame_input_dim += d_model + 1
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_size * d_model + chunk_size, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def encode_frames(
        self,
        obs_embedding: torch.Tensor,
        obs_mask: torch.Tensor,
        ref_embedding: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse modalities independently at every physical control frame."""

        features = [obs_embedding]
        if self.has_reference:
            if ref_embedding is None or ref_mask is None:
                raise ValueError("chunked reference encoding requires tensor and mask")
            ref_mask = ref_mask.bool() & obs_mask.bool()
            features.extend(
                (
                    ref_embedding * ref_mask.unsqueeze(-1).to(ref_embedding.dtype),
                    ref_mask.unsqueeze(-1).to(obs_embedding.dtype),
                )
            )
        frame_features = self.frame_encoder(torch.cat(features, dim=-1))
        return frame_features * obs_mask.unsqueeze(-1).to(frame_features.dtype)

    def encode_windows(
        self,
        frame_windows: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ``[..., K, D]`` causal windows, including partial chunks."""

        if frame_windows.shape[-2:] != (self.chunk_size, self.d_model):
            raise ValueError(
                "frame windows must end in "
                f"({self.chunk_size}, {self.d_model}), got "
                f"{tuple(frame_windows.shape)}"
            )
        if slot_mask.shape != frame_windows.shape[:-1]:
            raise ValueError("slot mask must match frame-window leading axes")
        slot_mask = slot_mask.bool()
        masked_windows = frame_windows * slot_mask.unsqueeze(-1).to(
            frame_windows.dtype
        )
        encoder_input = torch.cat(
            (
                masked_windows.flatten(start_dim=-2),
                slot_mask.to(frame_windows.dtype),
            ),
            dim=-1,
        )
        token = self.chunk_encoder(encoder_input)
        return token * slot_mask.any(dim=-1, keepdim=True).to(token.dtype)

# Compressor using Conv1d
class Conv1dCompressor(nn.Module):
    def __init__(self, d_model: int, stride: int, kernel_size: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - self.stride) // 2
        
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, seq: torch.Tensor, masks: torch.Tensor):
        # seq: [T, B, D], masks: [T, B]
        
        # 1. Pre-padding to ensure correct structure
        T, B, D = seq.shape
        padding_needed = (self.stride - T % self.stride) % self.stride
        if padding_needed > 0:
            seq = F.pad(seq, (0, 0, 0, 0, 0, padding_needed))
            masks = F.pad(masks, (0, 0, 0, padding_needed), value=False)

        # 2. Conv1d
        compressed_seq = self.conv(seq.permute(1, 2, 0)).permute(2, 0, 1)
        
        # 3. MaxPool1d for new masks
        mask_for_pool = masks.permute(1, 0).unsqueeze(1).float()
        new_mask_permuted = self.pool(mask_for_pool)
        new_masks = new_mask_permuted.squeeze(1).permute(1, 0).bool()
        
        return compressed_seq, new_masks


class PerceiverCompressor(nn.Module):
    """Perceiver-style cross-attention compressor as alternative to Conv1dCompressor.

    Uses learnable latent tokens that cross-attend to the input sequence,
    producing a fixed-size compressed representation. Supports causal masking
    where latent i only attends to the first ceil((i+1)*T/L) timesteps.
    """
    def __init__(self, d_model: int, num_latents: int = 32, num_heads: int = 4, stride: int = 1):
        super().__init__()
        self.num_latents = num_latents
        self.stride = stride  # kept for API compatibility
        self.d_model = d_model
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=False)
        self.norm_latent = nn.LayerNorm(d_model)
        self.norm_seq = nn.LayerNorm(d_model)

    def forward(self, seq: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            seq: [T, B, D] input sequence
            masks: [T, B] boolean mask (True = valid)
        Returns:
            compressed: [num_latents, B, D]
            new_masks: [num_latents, B] boolean mask
        """
        T, B, D = seq.shape

        # Expand latents for batch: [num_latents, B, D]
        latents = self.latents.unsqueeze(1).expand(-1, B, -1)
        latents_normed = self.norm_latent(latents)
        seq_normed = self.norm_seq(seq)

        # key_padding_mask: [B, T], float mask (-inf for padding, 0 for valid)
        key_padding_mask = torch.zeros(B, T, device=seq.device)
        key_padding_mask.masked_fill_(~masks.T.bool(), float('-inf'))

        # Causal cross-attention mask: latent i attends to seq[:ceil((i+1)*T/L)]
        attn_mask = self._build_causal_cross_mask(self.num_latents, T, seq.device)

        compressed, _ = self.cross_attn(
            query=latents_normed,   # [num_latents, B, D]
            key=seq_normed,         # [T, B, D]
            value=seq_normed,       # [T, B, D]
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        compressed = compressed + latents  # residual connection

        # All latents are valid if any token in the sequence is valid
        new_masks = masks.any(dim=0).unsqueeze(0).expand(self.num_latents, -1)  # [num_latents, B]
        return compressed, new_masks

    @staticmethod
    def _build_causal_cross_mask(num_latents: int, T: int, device: torch.device) -> torch.Tensor:
        """Build causal cross-attention mask. Latent i can attend to positions [0, cutoff_i).

        Returns:
            mask: [num_latents, T] float mask, -inf where blocked, 0.0 where allowed
        """
        latent_idx = torch.arange(num_latents, device=device, dtype=torch.float32).unsqueeze(1)
        seq_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
        cutoffs = ((latent_idx + 1) * T / num_latents).ceil()  # [L, 1]
        blocked = seq_idx >= cutoffs  # [L, T] True where blocked
        mask = torch.zeros(num_latents, T, device=device)
        mask.masked_fill_(blocked, float('-inf'))
        return mask


class PerceiverUpsampler(nn.Module):
    """Cross-attention upsampler that maps compressed latents back to original sequence length.

    Uses positional queries to cross-attend into compressed latent representations,
    matching the ConvTranspose1dCompressor interface: forward(compressed_seq, original_len).
    """
    def __init__(self, d_model: int, max_seq_len: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.pos_queries = nn.Embedding(max_seq_len, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=False)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, compressed_seq: torch.Tensor, original_len: int):
        """
        Args:
            compressed_seq: [num_latents, B, D]
            original_len: T_in
        Returns:
            upsampled: [T_in, B, D]
        """
        B = compressed_seq.shape[1]
        device = compressed_seq.device

        # Positional queries: [T_in, B, D]
        pos_idx = torch.arange(original_len, device=device)
        queries = self.pos_queries(pos_idx).unsqueeze(1).expand(-1, B, -1)

        upsampled, _ = self.cross_attn(
            query=self.norm_q(queries),
            key=self.norm_kv(compressed_seq),
            value=compressed_seq,
        )
        upsampled = upsampled + queries  # residual
        return upsampled


class ConvTranspose1dCompressor(nn.Module):
    def __init__(self, d_model, stride=2, kernel_size=4):
        super().__init__()
        padding = (kernel_size - stride) // 2
        
        self.unconv = nn.ConvTranspose1d(d_model, d_model, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, compressed_seq: torch.Tensor, original_len: int):

        seq_permuted = compressed_seq.permute(1, 2, 0)
        
        upsampled_permuted = self.unconv(seq_permuted)
        
        upsampled_seq = upsampled_permuted.permute(2, 0, 1)
        
        return upsampled_seq[:original_len, :, :]

class ObservationSeqEmbedding(nn.Module):
    """
    Embed a sequence of observations into a sequence of embeddings using an MLP.
    """
    def __init__(self, obs_dim: int, d_model: int, mlp_hidden_dims=[64, 32], activation="elu"):
        super().__init__()
        layers = []
        current_dim = obs_dim
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(resolve_nn_activation(activation))  # Create a new instance each time
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, d_model))
        self.projector = nn.Sequential(*layers)
        # self.norm = nn.LayerNorm(d_model)

    def forward(self, obs_seq: torch.Tensor):
        # obs_seq shape: [T, B, obs_dim]
        T, B, _ = obs_seq.shape
        obs_flat = obs_seq.reshape(T * B, -1)
        embedding_flat = self.projector(obs_flat)
        # embedding_flat = self.norm(embedding_flat)
        return embedding_flat.reshape(T, B, -1)
    
class ObservationSeqEmbeddingV2(nn.Module):
    def __init__(self, d_model: int, term_dict: Dict[str, int], concatenate_term_names: Optional[List[List[str]]] = None, SwiGLU = False, nheads = 4, term_steps: Optional[Dict[str, int]] = None,  **kwargs):
        super().__init__()
        
        self.term_names = list(term_dict.keys())
        self.term_dims = list(term_dict.values())
        
        self.group_term_names, self.group_term_dims, self.group_term_idx = group_by_concat_list(term_dict, concatenate_term_names)
        
        self.num_groups = len(self.group_term_names)
        self.d_model = d_model
        embedding_mlp = SwiGLUEmbedding if SwiGLU else MLPEmbedding
        if term_steps is None or len(term_steps) == 0:
            term_steps = {name: 1 for name in self.term_names}
        
        # create group term steps
        self.group_term_steps = []
        for group_names in self.group_term_names:
            steps = [term_steps[name] for name in group_names]
            assert all(s == steps[0] for s in steps), "Expected all terms in a group to have the same number of steps, but got {steps} for group {group_names}."
            self.group_term_steps.append(steps[0])
        embedding_mlps = [embedding_mlp if steps == 1 else HistoryEmbedding for steps in self.group_term_steps]
        
        self.embeddings = nn.ModuleList([
            # embedding_mlp(sum(dims), d_model, 1+int(sum(dims)/d_model)) for dims in self.group_term_dims
            embedding_mlps[i](sum(self.group_term_dims[i]), d_model, 1+int(sum(self.group_term_dims[i])/d_model), self.group_term_steps[i]) for i in range(self.num_groups)
        ])
        self.pre_norm = nn.LayerNorm(d_model)
        # self.attn_pool = nn.MultiheadAttention(d_model, nheads, batch_first=True, dropout=0.0)
        # self.pool_q = nn.Parameter(torch.randn(1, 1, d_model))
        self.gating_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        ) # score
        
        self.norm = nn.LayerNorm(d_model)
        
        start = 0
        self.term_slices = []
        for dim in self.term_dims:
            self.term_slices.append(slice(start, start + dim))
            start += dim

    def forward(self, obs_seq: torch.Tensor):
        # [T, B, obs_dim]
        T, B, _ = obs_seq.shape
        
        # [T * B, obs_dim]
        obs_flat = obs_seq.reshape(T * B, -1)
        
        token_list = []
        for i, group_term_idx in enumerate(self.group_term_idx):
            group_obs_flat = torch.cat([obs_flat[:, self.term_slices[j]] for j in group_term_idx], dim=-1)
            
            token_i = self.embeddings[i](group_obs_flat) # [T * B, d_model]
            token_list.append(token_i)
        structural_token_seq = torch.stack(token_list, dim=1) # [T * B, num_groups, d_model]
        structural_token_seq = self.pre_norm(structural_token_seq)
        # query = self.pool_q.expand(T * B, -1, -1)  # [T*B, 1, d_model]
        # pooled_embedding_flat, _ = self.attn_pool(
        #     query,
        #     structural_token_seq,
        #     structural_token_seq,
        # )  # [T*B, 1, d_model]
        
        # # residual
        # pooled_embedding_flat = self.norm(pooled_embedding_flat + query)
        scores = self.gating_mlp(structural_token_seq) # [T*B, num_groups, 1]
        weights = F.softmax(scores, dim=1) # [T*B, num_groups, 1]
        pooled_embedding_flat = torch.sum(weights * structural_token_seq, dim=1, keepdim=True) # [T*B, 1, d_model]
        normalized_embedding_flat = self.norm(pooled_embedding_flat.squeeze(1))  # [T*B, d_model]
        return normalized_embedding_flat.reshape(T, B, self.d_model)
    
    
class ObservationSeqEmbeddingV3(nn.Module): # Simple MLP concatenation version
    def __init__(self, d_model: int, term_dict: Dict[str, int], concatenate_term_names: Optional[List[List[str]]] = None, SwiGLU = False, nheads = 4, term_steps: Optional[Dict[str, int]] = None,  **kwargs):
        super().__init__()
        
        self.term_names = list(term_dict.keys())
        self.term_dims = list(term_dict.values())
        
        self.group_term_names, self.group_term_dims, self.group_term_idx = group_by_concat_list(term_dict, concatenate_term_names)
        
        self.num_groups = len(self.group_term_names)
        self.d_model = d_model
        self.high_expansion = 2
        self.expanded_feature_dim = d_model
        # embedding_mlp = SwiGLUEmbedding if SwiGLU else MLPEmbedding
        if term_steps is None or len(term_steps) == 0:
            term_steps = {name: 1 for name in self.term_names}
        
        # create group term steps
        self.group_term_steps = []
        for group_names in self.group_term_names:
            steps = [term_steps[name] for name in group_names]
            assert all(s == steps[0] for s in steps), "Expected all terms in a group to have the same number of steps, but got {steps} for group {group_names}."
            self.group_term_steps.append(steps[0])
        embedding_mlps = [nn.Identity() if steps == 1 else HistoryEmbedding for steps in self.group_term_steps]
        
        self.embeddings = nn.ModuleList([
            # embedding_mlp(sum(dims), d_model, 1+int(sum(dims)/d_model)) for dims in self.group_term_dims
            embedding_mlps[i](sum(self.group_term_dims[i]), self.expanded_feature_dim, 1+int(sum(self.group_term_dims[i])/self.expanded_feature_dim), self.group_term_steps[i])  if not isinstance(embedding_mlps[i], nn.Identity) else nn.Identity() for i in range(self.num_groups)
        ])
        # compute in features
        in_features = sum([sum(self.group_term_dims[i]) if isinstance(self.embeddings[i], nn.Identity) else self.expanded_feature_dim for i in range(self.num_groups)])
        # self.pre_norm = nn.LayerNorm(d_model)
        # self.attn_pool = nn.MultiheadAttention(d_model, nheads, batch_first=True, dropout=0.0)
        # self.pool_q = nn.Parameter(torch.randn(1, 1, d_model))
        self.out_mlp = nn.Sequential(
            nn.Linear(in_features, d_model * 2),
            nn.ELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        
        start = 0
        self.term_slices = []
        for dim in self.term_dims:
            self.term_slices.append(slice(start, start + dim))
            start += dim

    def forward(self, obs_seq: torch.Tensor):
        # [T, B, obs_dim]
        T, B, _ = obs_seq.shape
        
        # [T * B, obs_dim]
        obs_flat = obs_seq.reshape(T * B, -1)
        
        token_list = []
        for i, group_term_idx in enumerate(self.group_term_idx):
            group_obs_flat = torch.cat([obs_flat[:, self.term_slices[j]] for j in group_term_idx], dim=-1)
            
            token_i = self.embeddings[i](group_obs_flat) # [T * B, d_model]
            token_list.append(token_i)
        # structural_token_seq = torch.stack(token_list, dim=1) # [T * B, num_groups, d_model]
        structural_token_seq = torch.concat(token_list, dim=-1) # should be [T*B, d_in]
        structural_token_seq = structural_token_seq.reshape(T * B, -1) # [T * B, num_groups * d_model]
        structural_token_seq = self.out_mlp(structural_token_seq) # [T * B, d_model]
        structural_token_seq = self.norm(structural_token_seq)
        # structural_token_seq = self.pre_norm(structural_token_seq)
        # query = self.pool_q.expand(T * B, -1, -1)  # [T*B, 1, d_model]
        # pooled_embedding_flat, _ = self.attn_pool(
        #     query,
        #     structural_token_seq,
        #     structural_token_seq,
        # )  # [T*B, 1, d_model]
        
        # # residual
        # pooled_embedding_flat = self.norm(pooled_embedding_flat + query)
        # scores = self.gating_mlp(structural_token_seq) # [T*B, num_groups, 1]
        # weights = F.softmax(scores, dim=1) # [T*B, num_groups, 1]
        # pooled_embedding_flat = torch.sum(weights * structural_token_seq, dim=1, keepdim=True) # [T*B, 1, d_model]
        # normalized_embedding_flat = self.norm(pooled_embedding_flat.squeeze(1))  # [T*B, d_model]
        return structural_token_seq.reshape(T, B, self.d_model)
    

class TTTKVBLinearMemory(nn.Module):
    """Per-environment fast-weight memory with a vectorized KVB update.

    The slow projections and initial fast weights are ordinary model
    parameters.  ``fast_weights`` are recurrent actor state: one independent
    block-diagonal linear map per environment.  A normalized analytic form of
    the gradient update avoids an inner ``autograd.grad`` call for every
    vectorized environment.  With the normalization omitted, this is the exact
    gradient of
    ``0.5 * ||W k - v||^2``; key normalization is retained here as a bounded
    on-policy stability adaptation.
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        *,
        inner_lr: float = 0.1,
        inner_lr_max: float = 1.0,
        gate_init: float = 1.0e-3,
        learnable_inner_lr: bool = True,
    ):
        super().__init__()
        if dim_model % num_heads != 0:
            raise ValueError(
                "TTT dim_model must be divisible by num_heads, got "
                f"{dim_model} and {num_heads}"
            )
        if not 0.0 < inner_lr < inner_lr_max:
            raise ValueError(
                "TTT inner_lr must lie strictly between zero and "
                f"inner_lr_max, got {inner_lr} and {inner_lr_max}"
            )
        if not 0.0 <= gate_init < 1.0:
            raise ValueError(
                "TTT gate_init must lie in [0, 1), got "
                f"{gate_init}"
            )

        self.dim_model = int(dim_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_model // self.num_heads
        self.inner_lr_max = float(inner_lr_max)

        self.input_norm = nn.LayerNorm(self.dim_model)
        self.qkv_proj = nn.Linear(self.dim_model, 3 * self.dim_model, bias=False)
        self.out_proj = nn.Linear(self.dim_model, self.dim_model, bias=False)
        self.initial_fast_weights = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim, self.head_dim)
        )

        lr_fraction = float(inner_lr) / self.inner_lr_max
        lr_logit = math.log(lr_fraction / (1.0 - lr_fraction))
        self.inner_lr_logit = nn.Parameter(
            torch.full((self.num_heads,), lr_logit),
            requires_grad=bool(learnable_inner_lr),
        )
        self.gate_logit = nn.Parameter(
            torch.full(
                (self.dim_model,),
                math.atanh(float(gate_init)),
            )
        )

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (self.num_heads, self.head_dim, self.head_dim)

    @property
    def inner_lr(self) -> torch.Tensor:
        return self.inner_lr_max * torch.sigmoid(self.inner_lr_logit)

    @property
    def gate(self) -> torch.Tensor:
        return torch.tanh(self.gate_logit)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.initial_fast_weights.to(device=device, dtype=dtype).unsqueeze(0).expand(
            int(batch_size), -1, -1, -1
        ).clone()

    def _validate_state(self, state: torch.Tensor, batch_size: int) -> None:
        expected = (batch_size, *self.state_shape)
        if tuple(state.shape) != expected:
            raise ValueError(
                "TTT fast-weight state must have shape "
                f"{expected}, got {tuple(state.shape)}"
            )

    def step(
        self,
        hidden: torch.Tensor,
        fast_weights: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the current fused token, then query the updated memory."""

        if hidden.ndim != 2 or hidden.shape[-1] != self.dim_model:
            raise ValueError(
                "TTT hidden input must have shape (B, dim_model), got "
                f"{tuple(hidden.shape)}"
            )
        batch_size = hidden.shape[0]
        self._validate_state(fast_weights, batch_size)
        if valid is None:
            valid = torch.ones(batch_size, dtype=torch.bool, device=hidden.device)
        if tuple(valid.shape) != (batch_size,):
            raise ValueError(
                f"TTT valid mask must have shape {(batch_size,)}, got "
                f"{tuple(valid.shape)}"
            )

        state = fast_weights.to(device=hidden.device, dtype=hidden.dtype)
        qkv = self.qkv_proj(self.input_norm(hidden)).reshape(
            batch_size, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=1)

        prediction = torch.einsum("bhij,bhj->bhi", state, key)
        error = prediction - value
        key_norm_sq = key.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        update = error.unsqueeze(-1) * key.unsqueeze(-2)
        update = update / key_norm_sq.unsqueeze(-1)
        step_size = self.inner_lr.to(dtype=hidden.dtype).view(1, -1, 1, 1)
        candidate_state = state - step_size * update
        next_state = torch.where(
            valid.view(batch_size, 1, 1, 1), candidate_state, state
        )

        memory_heads = torch.einsum("bhij,bhj->bhi", next_state, query)
        memory = self.out_proj(memory_heads.reshape(batch_size, self.dim_model))
        fused = hidden + self.gate.to(dtype=hidden.dtype).view(1, -1) * memory
        fused = torch.where(valid.unsqueeze(-1), fused, hidden)
        return fused, next_state

    def forward_sequence(
        self,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        *,
        episode_ids: Optional[torch.Tensor] = None,
        state_reset_masks: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_reset_mask: Optional[torch.Tensor] = None,
        update_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay fast-weight writes in chronological per-environment order.

        ``update_start`` lets PPO prepend raw frames for the short KV cache
        without writing them twice into a fast state captured at rollout start.
        """

        if hidden.ndim != 3 or hidden.shape[-1] != self.dim_model:
            raise ValueError(
                "TTT sequence input must have shape (T, B, dim_model), got "
                f"{tuple(hidden.shape)}"
            )
        time_steps, batch_size, _ = hidden.shape
        if tuple(masks.shape) != (time_steps, batch_size):
            raise ValueError(
                "TTT masks must match the sequence leading dimensions, got "
                f"{tuple(masks.shape)} and {(time_steps, batch_size)}"
            )
        if not 0 <= update_start < time_steps:
            raise ValueError(
                f"TTT update_start must lie in [0, {time_steps}), got "
                f"{update_start}"
            )
        if episode_ids is not None and tuple(episode_ids.shape) != tuple(masks.shape):
            raise ValueError(
                "TTT episode_ids must match masks, got "
                f"{tuple(episode_ids.shape)} and {tuple(masks.shape)}"
            )
        if (
            state_reset_masks is not None
            and tuple(state_reset_masks.shape) != tuple(masks.shape)
        ):
            raise ValueError(
                "TTT state_reset_masks must match masks, got "
                f"{tuple(state_reset_masks.shape)} and {tuple(masks.shape)}"
            )

        if initial_state is None:
            state = self.initial_state(
                batch_size, device=hidden.device, dtype=hidden.dtype
            )
        else:
            self._validate_state(initial_state, batch_size)
            state = initial_state.to(device=hidden.device, dtype=hidden.dtype)

        reset_state = self.initial_fast_weights.to(
            device=hidden.device, dtype=hidden.dtype
        ).unsqueeze(0)
        if initial_state_reset_mask is not None:
            if initial_state is None:
                raise ValueError(
                    "TTT initial_state_reset_mask requires initial_state"
                )
            if tuple(initial_state_reset_mask.shape) != (batch_size,):
                raise ValueError(
                    "TTT initial reset mask must have shape "
                    f"{(batch_size,)}, got "
                    f"{tuple(initial_state_reset_mask.shape)}"
                )
            state = torch.where(
                initial_state_reset_mask.bool().view(batch_size, 1, 1, 1),
                reset_state,
                state,
            )
        outputs = []
        for step_index in range(time_steps):
            if step_index < update_start:
                outputs.append(hidden[step_index])
                continue

            valid = masks[step_index].bool()
            starts_segment = torch.zeros_like(valid)
            if step_index > update_start:
                starts_segment = valid & ~masks[step_index - 1].bool()
                if episode_ids is not None:
                    starts_segment = starts_segment | (
                        valid
                        & masks[step_index - 1].bool()
                        & (
                            episode_ids[step_index]
                            != episode_ids[step_index - 1]
                        )
                    )
            if state_reset_masks is not None:
                starts_segment = starts_segment | (
                    valid & state_reset_masks[step_index].bool()
                )
            state = torch.where(
                starts_segment.view(batch_size, 1, 1, 1),
                reset_state,
                state,
            )

            fused, state = self.step(hidden[step_index], state, valid)
            outputs.append(fused)

        return torch.stack(outputs, dim=0), state


class MMGPT(nn.Module):
    def __init__(
        self,
        obs_size, # actually, it should be named by obs_dim, but I maintained the original name for consistency
        ref_obs_size,
        dim_out,
        dim_model = 64,
        num_heads = 2,
        num_layers = 2,
        ffn_ratio = 4, # For SwiGLU, recommended ratio is 8/3 (~2.67)
        dropout = 0.0,
        name = "",
        term_dict = None,
        ref_term_dict = None,
        ref_term_steps: Optional[Dict[str, int]] = None, # hint: only ref terms support step-wise history encoding for now. We don't provide future information in obs terms.
        concatenate_term_names: Optional[List[List[str]]] = None,
        concatenate_ref_term_names: Optional[List[List[str]]] = None,
        num_steps_per_env = 24, # default, remember to parse this!
        max_seq_len = 24,
        mlp_hidden_dims = [],
        apply_rope = False, # default: APE; set True to use RoPE
        apply_res = True,
        ref_first = True,
        # HINT: The following parameters matters **SIGNIFICANTLY** to the performance for allowing temporal prediction features!
        pred_obs_term_names: Optional[List[str]] = None,
        pred_obs_term_weights: Optional[List[float]] = None,
        aux_backbone_grad: bool = True,
        dreamer_gradient_mode: str = "backbone_only",
        dreamer_target_term_name: str = "target_actions",
        dreamer_target_action_offset: int = 0,
        action_is_delta: bool = False,
        action_clip_value: Optional[float] = None,
        latest_reference_only: bool = False,
        streaming_kv_cache: bool = False,
        streaming_sequence_training: bool = False,
        streaming_context_steps: Optional[int] = None,
        temporal_chunk_size: int = 1,
        temporal_fast_slow_enabled: bool = False,
        temporal_fast_residual_scale: float = 1.0,
        ttt_memory_enabled: bool = False,
        ttt_num_heads: Optional[int] = None,
        ttt_inner_lr: float = 0.1,
        ttt_inner_lr_max: float = 1.0,
        ttt_gate_init: float = 1.0e-3,
        ttt_learnable_inner_lr: bool = True,
        ttt_memory_horizon_steps: int = 0,
        score_prediction_enabled: bool = False,
        apply_mlp_residual = False,
        # some not pretty fix
        default_joint_pos: Optional[torch.Tensor] = None,
        # Compressor config
        compressor_type: str = "conv1d",  # "conv1d" or "perceiver"
        perceiver_num_latents: int = 32,
        perceiver_num_heads: int = 4,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.obs_dim = obs_size
        self.ref_obs_dim = ref_obs_size
        self.dim_out = dim_out
        self.dim_model = dim_model
        self.num_steps_per_env = num_steps_per_env
        self.apply_rope = apply_rope
        self.apply_res = apply_res
        self.ref_first = ref_first
        self.default_joint_pos = default_joint_pos
        self.temporal_chunk_size = int(temporal_chunk_size)
        if self.temporal_chunk_size < 1:
            raise ValueError("temporal_chunk_size must be at least 1")
        self.temporal_chunk_enabled = self.temporal_chunk_size > 1
        self.temporal_fast_slow_enabled = bool(temporal_fast_slow_enabled)
        self.temporal_fast_residual_scale = float(temporal_fast_residual_scale)
        self.ttt_memory_enabled = bool(ttt_memory_enabled)
        self.ttt_num_heads = (
            int(num_heads) if ttt_num_heads is None else int(ttt_num_heads)
        )
        self.ttt_memory_horizon_steps = int(ttt_memory_horizon_steps)
        if self.ttt_memory_horizon_steps < 0:
            raise ValueError("ttt_memory_horizon_steps must be non-negative")
        if self.temporal_fast_residual_scale < 0.0:
            raise ValueError("temporal_fast_residual_scale must be non-negative")
        if self.temporal_fast_slow_enabled and not self.temporal_chunk_enabled:
            raise ValueError(
                "temporal_fast_slow_enabled requires temporal_chunk_size > 1"
            )
        if kwargs:
            print(
                f"MMGPT {self.name}.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        # The legacy compressor derives an implicit stride from max_seq_len.
        # Chunked streaming has an explicit physical-frame/token contract and
        # must not silently instantiate that second compressor.
        self.stride = (
            1
            if self.temporal_chunk_enabled
            else (
                math.ceil(num_steps_per_env / max_seq_len)
                if num_steps_per_env > max_seq_len
                else 1
            )
        )
        # self.obs_embedding = ObservationSeqEmbeddingV2(dim_model, term_dict, concatenate_term_names, nheads=num_heads)
        # self.ref_obs_embedding = ObservationSeqEmbeddingV2(dim_model, ref_term_dict, concatenate_ref_term_names,term_steps=ref_term_steps, nheads=num_heads) if ref_obs_size > 0 else None
        self.obs_embedding = ObservationSeqEmbeddingV3(dim_model, term_dict, concatenate_term_names, nheads=num_heads)
        self.ref_obs_embedding = ObservationSeqEmbeddingV3(dim_model, ref_term_dict, concatenate_ref_term_names,term_steps=ref_term_steps, nheads=num_heads) if ref_obs_size > 0 else None
        self.obs_norm = nn.LayerNorm(dim_model)
        self.ref_obs_norm = nn.LayerNorm(dim_model) if ref_obs_size > 0 else None
        
        # dreamer reward head. Reward is R(s, a, s'), which computes by using obs[t], action[t], obs[t+1]. For reward calculation, we focus mainly on obs, ref obs should be added to the dreamer reward.
        self.dreamer_reward_head = nn.Sequential(
            nn.Linear(dim_model * 2 + dim_out, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, 1)
        )
        if ref_obs_size > 0:
            self.dreamer_ref_reward_head = nn.Sequential(
                nn.Linear(dim_model * 2 + dim_out, dim_model),
                nn.SiLU(),
                nn.Linear(dim_model, 1)
            )
        else:
            self.dreamer_ref_reward_head = None
        
        # action indice
        action_name = 'actions'
        if action_name in term_dict:
            start = 0
            for term_name, term_dim in term_dict.items():
                if term_name == action_name:
                    self.action_slice = (start, start + term_dim)
                    break
                start += term_dim
        else:
            raise ValueError("For MMGPT to work normally, please set `last_action` term in observation terms, and name it as `actions`.")        
        self.pred_obs_term_names = pred_obs_term_names
        self.pred_obs_term_weights = pred_obs_term_weights
        self.aux_backbone_grad = bool(aux_backbone_grad)
        if dreamer_gradient_mode not in {
            "backbone_only",
            "backbone_and_action_head",
            "action_path_only",
        }:
            raise ValueError(
                "dreamer_gradient_mode must be 'backbone_only', "
                "'backbone_and_action_head', or 'action_path_only', "
                f"got {dreamer_gradient_mode!r}"
            )
        self.dreamer_gradient_mode = dreamer_gradient_mode
        if not dreamer_target_term_name:
            raise ValueError("dreamer_target_term_name must be non-empty")
        if dreamer_target_action_offset < 0:
            raise ValueError("dreamer_target_action_offset must be non-negative")
        self.dreamer_target_term_name = dreamer_target_term_name
        self.dreamer_target_action_offset = dreamer_target_action_offset
        self.action_is_delta = bool(action_is_delta)
        if action_clip_value is not None and action_clip_value <= 0.0:
            raise ValueError("action_clip_value must be positive or None")
        self.action_clip_value = (
            None if action_clip_value is None else float(action_clip_value)
        )
        self.latest_reference_only = bool(latest_reference_only)
        self.streaming_kv_cache = bool(streaming_kv_cache)
        self.streaming_sequence_training = bool(streaming_sequence_training)
        if self.streaming_sequence_training and not self.streaming_kv_cache:
            raise ValueError(
                "streaming_sequence_training requires streaming_kv_cache=True"
            )
        self.streaming_context_steps = (
            self.num_steps_per_env
            if streaming_context_steps is None
            else int(streaming_context_steps)
        )
        if self.streaming_context_steps < 1:
            raise ValueError("streaming_context_steps must be at least 1")
        if self.streaming_context_steps > self.num_steps_per_env:
            raise ValueError(
                "streaming_context_steps cannot exceed num_steps_per_env "
                "because rollout storage retains only that raw history"
            )
        self.streaming_context_tokens = math.ceil(
            self.streaming_context_steps / self.temporal_chunk_size
        )
        # Rebuilding an episode-aligned K-frame cache from a rolling raw
        # window can start in the middle of a chunk.  K-1 extra frames ensure
        # that, after discarding that first incomplete chunk, the full context
        # can still be reconstructed under updated parameters.
        self.rollout_history_frames = (
            self.streaming_context_steps + self.temporal_chunk_size - 1
            if self.temporal_chunk_enabled
            else self.num_steps_per_env
        )
        self.score_prediction_enabled = bool(score_prediction_enabled)
        if self.pred_obs_term_names is not None and len(self.pred_obs_term_names) > 0:
            assert self.pred_obs_term_weights is not None and len(self.pred_obs_term_weights) == len(self.pred_obs_term_names), "pred_obs_term_weights should be provided with the same length as pred_obs_term_names."

            self.obs_indices = []
            self.obs_dims = []
            self.obs_term_names = []
            start = 0
            self.action_film = nn.Linear(dim_out, 2 * dim_model)  # for FiLM modulation of obs prediction
            
            # Initialize FiLM layer: Gamma=1, Beta=0
            # This ensures that initially, the latent features are passed through without modification (Gamma=1)
            # and the action bias is zero (Beta=0). This allows gradients to flow from Aux Loss to Backbone.
            nn.init.normal_(self.action_film.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.action_film.bias[:dim_model], 0.5)
            nn.init.constant_(self.action_film.bias[dim_model:], 0.5)
            
            term_slices = {}
            for term_name, term_dim in term_dict.items():
                term_slices[term_name] = (start, start + term_dim)
                start += term_dim
            if len(set(self.pred_obs_term_names)) != len(self.pred_obs_term_names):
                raise ValueError(
                    "pred_obs_term_names must not contain duplicates, got "
                    f"{self.pred_obs_term_names}."
                )
            missing_terms = [
                term_name
                for term_name in self.pred_obs_term_names
                if term_name not in term_slices
            ]
            if missing_terms:
                raise ValueError(
                    "Some pred_obs_term_names are not found in term_dict. "
                    f"Missing {missing_terms}; available names: "
                    f"{list(term_dict.keys())}."
                )
            # Prediction heads and their weights are a positional config
            # contract.  Preserve pred_obs_term_names order rather than the
            # observation-dict order, which may be different (as it is for the
            # SONIC actor).
            for term_name in self.pred_obs_term_names:
                term_start, term_end = term_slices[term_name]
                self.obs_indices.append((term_start, term_end))
                self.obs_dims.append(term_end - term_start)
                self.obs_term_names.append(term_name)
            
            self.aux_heads = nn.ModuleList()
            for dim in self.obs_dims:
                self.aux_heads.append(
                    nn.Linear(dim_model, dim)
                )
                
            self.target_action_slice = None
            if (
                ref_term_dict is not None
                and self.dreamer_target_term_name in ref_term_dict
            ):
                start = 0
                for term_name, term_dim in ref_term_dict.items():
                    if term_name == self.dreamer_target_term_name:
                        target_start = start + self.dreamer_target_action_offset
                        target_end = target_start + dim_out
                        if target_end > start + term_dim:
                            raise ValueError(
                                f"Dreamer target slice [{target_start}, "
                                f"{target_end}) exceeds term "
                                f"{term_name!r} with dimension {term_dim}."
                            )
                        self.target_action_slice = (target_start, target_end)
                        break
                    start += term_dim
            
        else:
            self.action_film = None
            self.aux_heads = None
            self.obs_indices = None
            self.obs_dims = None
            self.obs_term_names = None
            self.target_action_slice = None
                
        self.apply_mlp_residual = apply_mlp_residual
        if apply_mlp_residual:
            self.mlp_bypass = FusedMultiModalMLP(
                term_dict={name: term_dict},
                ref_term_dict={name: ref_term_dict} if ref_obs_size > 0 else None,
                output_size=dim_out,
                hidden_dims=mlp_hidden_dims,
                history_length=1,
                activation="elu",
                encoder_latent_dim=dim_model,
                encoder_compress_threshold=32,
                use_layer_norm=True,
                fusion_mode="gated",
            )
            self.mlp_gate = nn.Sequential(
                nn.Linear(dim_out * 2, dim_out),
                nn.Sigmoid()
            )
        else:
            self.mlp_bypass = None
            self.mlp_gate = None

        # self.compressor = SequenceCompressor(dim_model, self.stride) if self.stride > 1 else None
        self.kernel_size = self.stride
        self.padding = (self.kernel_size - self.stride) // 2 if self.kernel_size > self.stride else 0
        padding_needed = (self.stride - self.num_steps_per_env % self.stride) % self.stride if self.stride > 1 else 0
        L_in_padded = self.num_steps_per_env + padding_needed
        self.compressor_type = compressor_type

        if self.stride > 1:
            if compressor_type == "perceiver":
                self.compressed_len = perceiver_num_latents
                self.compressor = PerceiverCompressor(dim_model, num_latents=perceiver_num_latents, num_heads=perceiver_num_heads, stride=self.stride)
                self.ref_compressor = PerceiverCompressor(dim_model, num_latents=perceiver_num_latents, num_heads=perceiver_num_heads, stride=self.stride) if ref_obs_size > 0 else None
                self.upsampler = PerceiverUpsampler(dim_model, max_seq_len=num_steps_per_env, num_heads=perceiver_num_heads)
            else:
                # Default: Conv1dCompressor
                self.compressed_len = math.floor((L_in_padded + 2 * self.padding - self.kernel_size) / self.stride) + 1
                self.compressor = Conv1dCompressor(dim_model, self.stride, self.kernel_size)
                self.ref_compressor = Conv1dCompressor(dim_model, self.stride, self.kernel_size) if ref_obs_size > 0 else None
                self.upsampler = ConvTranspose1dCompressor(dim_model, self.stride, self.kernel_size)
        else:
            self.compressed_len = self.num_steps_per_env
            self.compressor = None
            self.ref_compressor = None
            self.upsampler = None

        if self.streaming_kv_cache:
            if dropout != 0.0:
                raise ValueError(
                    "streaming_kv_cache requires dropout=0 so rollout and PPO "
                    "recomputation share the same policy distribution"
                )
            if not self.apply_rope:
                raise ValueError("streaming_kv_cache requires apply_rope=True")
            if self.compressor is not None:
                raise ValueError(
                    "streaming_kv_cache currently requires an uncompressed "
                    "one-token-per-modality timestep sequence"
                )
            if self.latest_reference_only:
                raise ValueError(
                    "streaming_kv_cache cannot retroactively erase historical "
                    "reference keys; set latest_reference_only=False"
                )
            if not self.ref_first and self.ref_obs_dim > 0:
                raise ValueError(
                    "streaming_kv_cache requires ref_first=True so obs_t may "
                    "attend to ref_t without ref_t seeing obs_t"
                )
            if self.apply_mlp_residual:
                raise ValueError(
                    "streaming_kv_cache does not yet support apply_mlp_residual"
                )

        if self.temporal_chunk_enabled:
            if not self.streaming_kv_cache:
                raise ValueError(
                    "temporal chunking requires streaming_kv_cache=True"
                )
            if not self.streaming_sequence_training:
                raise ValueError(
                    "temporal chunking requires "
                    "streaming_sequence_training=True so PPO receives the "
                    "episode-local chunk phase"
                )
            if self.streaming_context_steps % self.temporal_chunk_size != 0:
                raise ValueError(
                    "streaming_context_steps must be divisible by "
                    "temporal_chunk_size for an exact physical horizon"
                )
            self.temporal_chunk_encoder = SonicTemporalChunkEncoder(
                dim_model,
                self.temporal_chunk_size,
                has_reference=ref_obs_size > 0,
            )
            # The partial chunk is the 50 Hz path; the last completed slow
            # token contributes context through one feature fusion module.
            # There remains only one action decoder (self.fc).
            self.temporal_context_fusion = nn.Sequential(
                nn.Linear(2 * dim_model + 1, 2 * dim_model),
                nn.SiLU(),
                nn.Linear(2 * dim_model, dim_model),
            )
            self.temporal_output_norm = nn.LayerNorm(dim_model)
            if self.temporal_fast_slow_enabled:
                # The slow path proposes one nominal action for every physical
                # frame in the chunk.  A plan is sampled only at phase zero and
                # is held fixed until the next chunk boundary.
                self.temporal_plan_head = nn.Sequential(
                    nn.LayerNorm(dim_model),
                    nn.Linear(dim_model, 2 * dim_model),
                    nn.SiLU(),
                    nn.Linear(
                        2 * dim_model,
                        self.temporal_chunk_size * dim_out,
                    ),
                )
                nn.init.uniform_(
                    self.temporal_plan_head[-1].weight, -0.003, 0.003
                )
                nn.init.constant_(self.temporal_plan_head[-1].bias, 0.0)

                # The 50 Hz correction path has an explicit current-observation
                # bypass.  It does not need to wait for the K-frame chunk token
                # and therefore preserves feedback inside a slow plan.
                fast_input_dim = dim_model + dim_out + self.temporal_chunk_size
                if ref_obs_size > 0:
                    fast_input_dim += dim_model + 1
                self.temporal_fast_controller = nn.Sequential(
                    nn.Linear(fast_input_dim, 2 * dim_model),
                    nn.SiLU(),
                    nn.Linear(2 * dim_model, dim_model),
                    nn.SiLU(),
                    nn.Linear(dim_model, dim_out),
                )
                # Start from the coherent nominal plan.  PPO/BC must learn how
                # much high-frequency correction is useful rather than being
                # forced through a random residual at initialization.
                nn.init.zeros_(self.temporal_fast_controller[-1].weight)
                nn.init.zeros_(self.temporal_fast_controller[-1].bias)
            else:
                self.temporal_plan_head = None
                self.temporal_fast_controller = None
        else:
            self.temporal_chunk_encoder = None
            self.temporal_context_fusion = None
            self.temporal_output_norm = None
            self.temporal_plan_head = None
            self.temporal_fast_controller = None

        if self.ttt_memory_enabled:
            if not self.streaming_kv_cache or not self.streaming_sequence_training:
                raise ValueError(
                    "ttt_memory_enabled requires streaming_kv_cache=True and "
                    "streaming_sequence_training=True"
                )
            if self.temporal_chunk_enabled:
                raise ValueError(
                    "the first TTT adapter requires temporal_chunk_size=1; "
                    "chunk/TTT interaction is a separate architecture ablation"
                )
            self.ttt_memory = TTTKVBLinearMemory(
                dim_model,
                self.ttt_num_heads,
                inner_lr=ttt_inner_lr,
                inner_lr_max=ttt_inner_lr_max,
                gate_init=ttt_gate_init,
                learnable_inner_lr=ttt_learnable_inner_lr,
            )
        else:
            self.ttt_memory = None
        
        self.pos_emb = nn.Embedding(self.compressed_len, dim_model) if not self.apply_rope else None # *2 for obs and ref_obs
        self.modality_emb = nn.Embedding(2, dim_model) # 0 for obs, 1 for ref_obs
        if not self.apply_rope:
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=ffn_ratio*dim_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.decoder = RoPETransformer(
                d_model=dim_model,
                num_heads=num_heads,
                num_encoder_layers=num_layers,
                dim_feedforward=ffn_ratio*dim_model,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                max_seq_len=max_seq_len * 2,
                use_sdpa=True,
            )
        self.fc = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_model * 2),
            nn.SiLU(),
            nn.Linear(dim_model * 2, dim_out)
        )
        nn.init.uniform_(self.fc[-1].weight, -0.003, 0.003)
        nn.init.constant_(self.fc[-1].bias, 0.0)

        self.inference_buffer_obs = None # for storing inference-time past key values
        self.inference_buffer_ref_obs = None
        self.inference_buffer_ref_obs_mask = None
        self.current_timesteps = None  # Track current timesteps for each environment [B,]
        self.num_envs_cache = 0
        self.inference_kv_cache = None
        self.inference_position_ids = None
        self.inference_kv_cache_dirty = False
        self.inference_chunk_frame_buffer = None
        self.inference_chunk_slot_mask = None
        self.inference_chunk_phase = None
        self.inference_slow_hidden = None
        self.inference_slow_valid = None
        self.inference_chunk_plan = None
        self.inference_chunk_plan_valid = None
        self.inference_ttt_state = None
        self.inference_ttt_state_is_reset = None
        self.inference_ttt_steps_since_reset = None
        self.last_inference_ttt_reset_event = None
        self.last_inference_temporal_phase = None
        
        self.aux_loss = 0.0
        self.aux_loss_fn = nn.MSELoss()
        self.dreamer_loss = 0.0
        self.dreamer_reward = 0.0
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dream_loss(self):
        return self.dreamer_loss if self.dreamer_loss is not None and self.dreamer_loss > 0.0 else torch.tensor(0.0, device=self.device)
    
    @property
    def dream_reward(self):
        return self.dreamer_reward if self.dreamer_reward is not None and self.dreamer_reward > 0.0 else torch.tensor(0.0, device=self.device)

    @property
    def auxiliary_loss(self):
        return self.aux_loss

    def _route_future_prediction_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply only the configured future-prediction backbone stop-gradient."""

        if self.aux_backbone_grad:
            return latent
        return latent.detach()

    def _route_dreamer_inputs(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route Dreamer gradients without changing the forward values.

        ``backbone_only`` is the action-detached cell.  The matched
        ``backbone_and_action_head`` cell removes only that detach.  The
        historical ``action_path_only`` control keeps the action path but
        removes the predictor's direct latent shortcut.
        """

        if self.dreamer_gradient_mode == "backbone_only":
            action = action.detach()
        elif self.dreamer_gradient_mode == "action_path_only":
            latent = latent.detach()
        return latent, action

    def _resolve_prediction_action(
        self,
        previous_action: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> torch.Tensor:
        """Match the environment's raw controller-command coordinates.

        The future predictor is supervised with ``last_action`` from the raw
        observation dict.  That term is the action manager's resolved command:
        in absolute mode it equals the finite policy output, while in delta
        mode it is the previous resolved command plus the policy delta.  The
        action-term scale/default offset is deliberately not applied here; it
        is downstream of both the predictor's training action and this query.
        """

        if previous_action.shape != policy_action.shape:
            raise ValueError(
                "previous and policy actions must have identical shapes, got "
                f"{tuple(previous_action.shape)} and {tuple(policy_action.shape)}"
            )
        finite_action = torch.where(
            torch.isfinite(policy_action),
            policy_action,
            torch.zeros_like(policy_action),
        )
        resolved = (
            previous_action + finite_action
            if self.action_is_delta
            else finite_action
        )
        if self.action_clip_value is not None:
            resolved = resolved.clamp(
                min=-self.action_clip_value,
                max=self.action_clip_value,
            )
        return resolved

    def _masked_prediction_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean prediction error over valid transitions/environments only."""

        valid_mask = valid_mask.bool()
        if valid_mask.any():
            return self.aux_loss_fn(
                prediction[valid_mask],
                target[valid_mask],
            )
        # Preserve a differentiable scalar when a batch contains no valid
        # transition (for example, immediately after every environment reset).
        return prediction.sum() * 0.0
        
    @staticmethod
    def unpad(padded_sequence, masks):
        return padded_sequence.permute(1, 0, 2)[masks.permute(1, 0)] # shape [valid_length, dim]

    def _apply_reference_history_contract(
        self,
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Limit reference information to the current command when requested.

        SONIC combines past proprioception with one current multi-future
        command. MMGPT normally retains both modalities over time; this
        adapter prevents historical command tokens from silently expanding
        the actor information set while leaving the current dreamer target
        unchanged.
        """

        if ref_obs is None or not self.latest_reference_only:
            return ref_obs
        ref_tensor, ref_mask = ref_obs
        latest_mask = torch.zeros_like(ref_mask)
        latest_mask[-1] = ref_mask[-1]
        aligned_ref = ref_tensor * latest_mask.unsqueeze(-1).to(ref_tensor.dtype)
        return aligned_ref, latest_mask
    
    def generate_mask(self) -> torch.Tensor:
        """
        Generate a mask of shape [T_in, B] based on current_timesteps.
        For each environment b, if its timestep is t (0 <= t < T_in),
        then positions [0, T_in-t-1] in the sequence are False (masked out), 
        and positions [T_in-t, T_in-1] are True.
        
        Returns:
            torch.Tensor: shape [T_in, B], True indicates valid positions, False indicates padding positions
        """
        if self.current_timesteps is None:
            raise RuntimeError("current_timesteps not initialized. Call _init_inference_buffer first.")
        
        T_in = self.inference_buffer_obs.shape[0]
        B = self.current_timesteps.shape[0]
        device = self.current_timesteps.device
        
        # Create time index matrix [T_in, B], each column represents timesteps 0, 1, ..., T_in-1
        time_indices = torch.arange(T_in, device=device).unsqueeze(1).expand(T_in, B)
        
        # For each environment b, valid positions start from T_in - current_timesteps[b]
        # i.e., when time_indices >= T_in - current_timesteps[b], it's True
        start_positions = T_in - self.current_timesteps.unsqueeze(0)  # [1, B]
        mask = time_indices >= start_positions  # [T_in, B]
        
        return mask

    def _init_inference_buffer(self, num_envs, device):
        self.num_envs_cache = num_envs
        obs_buffer_shape = (self.rollout_history_frames, num_envs, self.obs_dim)
        self.inference_buffer_obs = torch.zeros(obs_buffer_shape, device=device)
        # Initialize timestep tracking array, initially set to 0
        self.current_timesteps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.inference_kv_cache = None
        self.inference_position_ids = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.inference_kv_cache_dirty = False
        if self.ttt_memory_enabled:
            self.inference_ttt_state = self.ttt_memory.initial_state(
                num_envs,
                device=device,
                dtype=self.ttt_memory.initial_fast_weights.dtype,
            ).detach()
            self.inference_ttt_state_is_reset = torch.ones(
                num_envs, dtype=torch.bool, device=device
            )
            self.inference_ttt_steps_since_reset = torch.zeros(
                num_envs, dtype=torch.long, device=device
            )
            self.last_inference_ttt_reset_event = torch.zeros(
                num_envs, dtype=torch.bool, device=device
            )
        else:
            self.inference_ttt_state = None
            self.inference_ttt_state_is_reset = None
            self.inference_ttt_steps_since_reset = None
            self.last_inference_ttt_reset_event = None
        if self.ref_obs_dim > 0:
            ref_obs_buffer_shape = (self.rollout_history_frames, num_envs, self.ref_obs_dim)
            self.inference_buffer_ref_obs = torch.zeros(ref_obs_buffer_shape, device=device)
            self.inference_buffer_ref_obs_mask = torch.zeros((self.rollout_history_frames, num_envs), dtype=torch.bool, device=device)
        if self.temporal_chunk_enabled:
            self.inference_chunk_frame_buffer = torch.zeros(
                self.temporal_chunk_size,
                num_envs,
                self.dim_model,
                device=device,
            )
            self.inference_chunk_slot_mask = torch.zeros(
                self.temporal_chunk_size,
                num_envs,
                dtype=torch.bool,
                device=device,
            )
            self.inference_chunk_phase = torch.zeros(
                num_envs, dtype=torch.long, device=device
            )
            self.inference_slow_hidden = torch.zeros(
                num_envs, self.dim_model, device=device
            )
            self.inference_slow_valid = torch.zeros(
                num_envs, dtype=torch.bool, device=device
            )
            if self.temporal_fast_slow_enabled:
                self.inference_chunk_plan = torch.zeros(
                    num_envs,
                    self.temporal_chunk_size,
                    self.dim_out,
                    device=device,
                )
                self.inference_chunk_plan_valid = torch.zeros(
                    num_envs, dtype=torch.bool, device=device
                )
            self.last_inference_temporal_phase = torch.zeros(
                num_envs, dtype=torch.long, device=device
            )
            
    def reset(self, dones=None):
        if self.inference_buffer_obs is None:
            return
        if dones is None: # reset all
            self.inference_buffer_obs.fill_(0.0)
            # Reset all environments' timesteps to 0
            if self.current_timesteps is not None:
                self.current_timesteps.fill_(0)
            if self.ref_obs_dim > 0:
                if self.inference_buffer_ref_obs is not None:
                    self.inference_buffer_ref_obs.fill_(0.0)
                if self.inference_buffer_ref_obs_mask is not None:
                    self.inference_buffer_ref_obs_mask.fill_(False)
            self.inference_kv_cache = None
            if self.inference_position_ids is not None:
                self.inference_position_ids.zero_()
            self.inference_kv_cache_dirty = False
            if self.inference_ttt_state is not None:
                self.inference_ttt_state.copy_(
                    self.ttt_memory.initial_state(
                        self.num_envs_cache,
                        device=self.inference_ttt_state.device,
                        dtype=self.inference_ttt_state.dtype,
                    ).detach()
                )
                self.inference_ttt_state_is_reset.fill_(True)
                self.inference_ttt_steps_since_reset.zero_()
                self.last_inference_ttt_reset_event.zero_()
            if self.temporal_chunk_enabled:
                self.inference_chunk_frame_buffer.zero_()
                self.inference_chunk_slot_mask.zero_()
                self.inference_chunk_phase.zero_()
                self.inference_slow_hidden.zero_()
                self.inference_slow_valid.zero_()
                if self.inference_chunk_plan is not None:
                    self.inference_chunk_plan.zero_()
                if self.inference_chunk_plan_valid is not None:
                    self.inference_chunk_plan_valid.zero_()
                self.last_inference_temporal_phase.zero_()
        else:
            # dones are at the environment level
            mask = (dones == 1)
            self.inference_buffer_obs[:, mask, :] = 0.0
            # Reset timesteps to 0 for completed environments: timesteps[dones] = 0
            if self.current_timesteps is not None:
                self.current_timesteps[mask] = 0
            if self.ref_obs_dim > 0:
                if self.inference_buffer_ref_obs is not None:
                    self.inference_buffer_ref_obs[:, mask, :] = 0.0
                if self.inference_buffer_ref_obs_mask is not None:
                    self.inference_buffer_ref_obs_mask[:, mask] = False
            if self.inference_position_ids is not None:
                self.inference_position_ids[mask] = 0
            if self.inference_kv_cache is not None:
                cache_padding = self.inference_kv_cache["key_padding_mask"]
                cache_padding[mask] = True
                for key, value in self.inference_kv_cache["key_values"]:
                    key[mask] = 0.0
                    value[mask] = 0.0
            if self.inference_ttt_state is not None and mask.any():
                reset_state = self.ttt_memory.initial_fast_weights.to(
                    device=self.inference_ttt_state.device,
                    dtype=self.inference_ttt_state.dtype,
                )
                self.inference_ttt_state[mask] = reset_state.detach()
                self.inference_ttt_state_is_reset[mask] = True
                self.inference_ttt_steps_since_reset[mask] = 0
                self.last_inference_ttt_reset_event[mask] = False
            if self.temporal_chunk_enabled:
                self.inference_chunk_frame_buffer[:, mask] = 0.0
                self.inference_chunk_slot_mask[:, mask] = False
                self.inference_chunk_phase[mask] = 0
                self.inference_slow_hidden[mask] = 0.0
                self.inference_slow_valid[mask] = False
                if self.inference_chunk_plan is not None:
                    self.inference_chunk_plan[mask] = 0.0
                if self.inference_chunk_plan_valid is not None:
                    self.inference_chunk_plan_valid[mask] = False
                self.last_inference_temporal_phase[mask] = 0

    def get_inference_ttt_state(
        self,
        observations: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return the pre-action fast state captured at a rollout boundary."""

        state = self.get_inference_ttt_rollout_state(observations)
        return None if state is None else state[0]

    def get_inference_ttt_rollout_state(
        self,
        observations: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return fast weights and which rows are currently at learned W0."""

        if not self.ttt_memory_enabled:
            return None
        if observations.ndim != 2 or observations.shape[-1] != self.obs_dim:
            raise ValueError(
                "MMGPT rollout observations must have shape (B, obs_dim), got "
                f"{tuple(observations.shape)}"
            )
        batch_size = observations.shape[0]
        if self.inference_buffer_obs is None or batch_size != self.num_envs_cache:
            self._init_inference_buffer(batch_size, observations.device)
        return (
            self.inference_ttt_state.detach().clone(),
            self.inference_ttt_state_is_reset.detach().clone(),
        )

    def get_inference_ttt_reset_event(self) -> Optional[torch.Tensor]:
        """Return scheduled fast-state resets applied to the latest action."""

        if not self.ttt_memory_enabled:
            return None
        if self.last_inference_ttt_reset_event is None:
            raise RuntimeError("TTT inference state is not initialized")
        return self.last_inference_ttt_reset_event.detach().clone()

    def _apply_ttt_inference(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.ttt_memory_enabled:
            return hidden
        if self.inference_ttt_state is None:
            raise RuntimeError("TTT inference state is not initialized")
        if self.inference_ttt_steps_since_reset is None:
            raise RuntimeError("TTT inference horizon counter is not initialized")
        reset_event = torch.zeros(
            hidden.shape[0], dtype=torch.bool, device=hidden.device
        )
        if self.ttt_memory_horizon_steps > 0:
            reset_event = (
                self.inference_ttt_steps_since_reset
                >= self.ttt_memory_horizon_steps
            )
            if reset_event.any():
                reset_state = self.ttt_memory.initial_fast_weights.to(
                    device=self.inference_ttt_state.device,
                    dtype=self.inference_ttt_state.dtype,
                )
                self.inference_ttt_state[reset_event] = reset_state.detach()
                self.inference_ttt_state_is_reset[reset_event] = True
                self.inference_ttt_steps_since_reset[reset_event] = 0
        self.last_inference_ttt_reset_event.copy_(reset_event)
        valid = torch.ones(
            hidden.shape[0], dtype=torch.bool, device=hidden.device
        )
        fused, next_state = self.ttt_memory.step(
            hidden, self.inference_ttt_state, valid
        )
        # Rollout state is deliberately graph-free across physical control
        # steps. PPO reconstructs the differentiable trajectory from the saved
        # rollout-start state instead of retaining the collection graph.
        self.inference_ttt_state = next_state.detach()
        self.inference_ttt_state_is_reset.fill_(False)
        self.inference_ttt_steps_since_reset.add_(1)
        return fused

    def invalidate_inference_cache(self):
        """Rebuild cached K/V with current parameters on the next rollout step."""

        if self.streaming_kv_cache and self.inference_buffer_obs is not None:
            self.inference_kv_cache_dirty = True

    @staticmethod
    def _select_kv_cache(
        cache: Optional[Dict[str, object]],
        env_indices: torch.Tensor,
    ) -> Optional[Dict[str, object]]:
        if cache is None:
            return None
        return {
            "key_values": [
                (key[env_indices], value[env_indices])
                for key, value in cache["key_values"]
            ],
            "key_padding_mask": cache["key_padding_mask"][env_indices],
        }

    def _merge_kv_cache_subset(
        self,
        subset_cache: Dict[str, object],
        env_indices: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Scatter caches for asynchronously completed chunks back by env."""

        subset_padding = subset_cache["key_padding_mask"]
        new_length = subset_padding.shape[1]
        if self.inference_kv_cache is None:
            full_key_values = []
            for key, value in subset_cache["key_values"]:
                key_shape = (batch_size, *key.shape[1:])
                value_shape = (batch_size, *value.shape[1:])
                full_key_values.append(
                    (torch.zeros(key_shape, dtype=key.dtype, device=key.device),
                     torch.zeros(value_shape, dtype=value.dtype, device=value.device))
                )
            full_padding = torch.ones(
                batch_size,
                new_length,
                dtype=torch.bool,
                device=subset_padding.device,
            )
            self.inference_kv_cache = {
                "key_values": full_key_values,
                "key_padding_mask": full_padding,
            }
        else:
            old_length = self.inference_kv_cache["key_padding_mask"].shape[1]
            if new_length < old_length:
                raise RuntimeError("subset KV cache unexpectedly shrank")
            if new_length > old_length:
                if new_length != old_length + 1:
                    raise RuntimeError("subset KV cache grew by more than one token")
                expanded_key_values = []
                for key, value in self.inference_kv_cache["key_values"]:
                    expanded_key_values.append(
                        (
                            # Keep each asynchronous environment right-aligned;
                            # append-only attention then drops left padding
                            # before it can evict a valid completed token.
                            F.pad(key, (0, 0, 1, 0)),
                            F.pad(value, (0, 0, 1, 0)),
                        )
                    )
                self.inference_kv_cache["key_values"] = expanded_key_values
                self.inference_kv_cache["key_padding_mask"] = F.pad(
                    self.inference_kv_cache["key_padding_mask"],
                    (1, 0),
                    value=True,
                )

        for layer_index, (subset_key, subset_value) in enumerate(
            subset_cache["key_values"]
        ):
            full_key, full_value = self.inference_kv_cache["key_values"][
                layer_index
            ]
            full_key[env_indices] = subset_key
            full_value[env_indices] = subset_value
        self.inference_kv_cache["key_padding_mask"][env_indices] = subset_padding

    def _streaming_chunk_step(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_write_allowed: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Advance the partial chunk and append only newly completed tokens."""

        required_state = (
            self.inference_chunk_frame_buffer,
            self.inference_chunk_slot_mask,
            self.inference_chunk_phase,
            self.inference_slow_hidden,
            self.inference_slow_valid,
            self.inference_position_ids,
        )
        if any(state is None for state in required_state):
            raise RuntimeError("chunked streaming state is not initialized")
        B = obs.shape[0]
        device = obs.device
        obs_mask = obs_mask.bool()
        phase = self.inference_chunk_phase.clone()
        self.last_inference_temporal_phase = phase.clone()

        obs_embedding = self.obs_norm(self.obs_embedding(obs.unsqueeze(0)))[0]
        ref_embedding = None
        ref_mask = None
        if self.ref_obs_dim > 0:
            if ref_obs is None:
                ref_tensor = torch.zeros(
                    B, self.ref_obs_dim, dtype=obs.dtype, device=device
                )
                ref_mask = torch.zeros(B, dtype=torch.bool, device=device)
            else:
                ref_tensor, ref_mask = ref_obs
                ref_mask = ref_mask.bool() & obs_mask
            ref_embedding = self.ref_obs_norm(
                self.ref_obs_embedding(ref_tensor.unsqueeze(0))
            )[0]
        frame_features = self.temporal_chunk_encoder.encode_frames(
            obs_embedding,
            obs_mask,
            ref_embedding,
            ref_mask,
        )

        valid_envs = torch.nonzero(obs_mask, as_tuple=False).flatten()
        if valid_envs.numel() > 0:
            valid_slots = phase[valid_envs]
            self.inference_chunk_frame_buffer[valid_slots, valid_envs] = (
                frame_features[valid_envs]
            )
            self.inference_chunk_slot_mask[valid_slots, valid_envs] = True
        windows = self.inference_chunk_frame_buffer.permute(1, 0, 2)
        slot_mask = self.inference_chunk_slot_mask.T
        partial_token = self.temporal_chunk_encoder.encode_windows(
            windows, slot_mask
        )
        closes_chunk = obs_mask & (phase == self.temporal_chunk_size - 1)
        completed = closes_chunk & slot_mask.all(dim=-1)
        write_mask = completed
        if cache_write_allowed is not None:
            if tuple(cache_write_allowed.shape) != (B,):
                raise ValueError("cache_write_allowed must have shape [B]")
            write_mask = write_mask & cache_write_allowed.bool()
        write_envs = torch.nonzero(write_mask, as_tuple=False).flatten()
        if write_envs.numel() > 0:
            subset_cache = self._select_kv_cache(
                self.inference_kv_cache, write_envs
            )
            slow_hidden, subset_cache = self.decoder.forward_with_kv_cache(
                partial_token[write_envs].unsqueeze(1),
                position_ids=self.inference_position_ids[write_envs].unsqueeze(1),
                token_padding_mask=torch.zeros(
                    write_envs.numel(), 1, dtype=torch.bool, device=device
                ),
                cache=subset_cache,
                max_cache_tokens=self.streaming_context_tokens,
            )
            self._merge_kv_cache_subset(subset_cache, write_envs, B)
            self.inference_slow_hidden[write_envs] = slow_hidden[:, 0]
            self.inference_slow_valid[write_envs] = True

        hidden = self._fuse_partial_and_slow_context(
            partial_token,
            self.inference_slow_hidden,
            self.inference_slow_valid,
            obs_mask,
        )
        closing_envs = torch.nonzero(closes_chunk, as_tuple=False).flatten()
        if closing_envs.numel() > 0:
            self.inference_chunk_frame_buffer[:, closing_envs] = 0.0
            self.inference_chunk_slot_mask[:, closing_envs] = False
        self.inference_chunk_phase = torch.where(
            obs_mask,
            torch.where(closes_chunk, torch.zeros_like(phase), phase + 1),
            phase,
        )
        self.inference_position_ids = self.inference_position_ids + obs_mask.long()
        return hidden, obs_embedding, ref_embedding, ref_mask, phase

    def _rebuild_chunked_streaming_cache(self) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Rebuild completed-token KV and the current partial chunk from raw frames."""

        masks = self.generate_mask()
        valid_counts = masks.long().sum(dim=0)
        # The current observation has already entered the raw rolling buffer,
        # but its position has not yet been consumed by _streaming_chunk_step.
        target_next_position = self.inference_position_ids + 1
        start_position = target_next_position - valid_counts

        self.inference_kv_cache = None
        self.inference_chunk_frame_buffer.zero_()
        self.inference_chunk_slot_mask.zero_()
        self.inference_slow_hidden.zero_()
        self.inference_slow_valid.zero_()
        if self.inference_chunk_plan is not None:
            self.inference_chunk_plan.zero_()
        if self.inference_chunk_plan_valid is not None:
            self.inference_chunk_plan_valid.zero_()
        self.inference_position_ids = start_position.clone()
        self.inference_chunk_phase = torch.remainder(
            start_position, self.temporal_chunk_size
        )
        cache_eligible = self.inference_chunk_phase == 0
        step_features = None
        for step in range(self.inference_buffer_obs.shape[0]):
            step_mask = masks[step]
            completes_visible_chunk = (
                step_mask
                & (self.inference_chunk_phase == self.temporal_chunk_size - 1)
            )
            ref_step = None
            if self.ref_obs_dim > 0:
                ref_step = (
                    self.inference_buffer_ref_obs[step],
                    self.inference_buffer_ref_obs_mask[step],
                )
            step_features = self._streaming_chunk_step(
                self.inference_buffer_obs[step],
                step_mask,
                ref_step,
                cache_write_allowed=cache_eligible,
            )
            if self.temporal_fast_slow_enabled:
                hidden, _, _, _, temporal_phase = step_features
                self._update_inference_chunk_plan(
                    hidden, temporal_phase, step_mask
                )
            cache_eligible = cache_eligible | completes_visible_chunk
        self.inference_kv_cache_dirty = False
        if step_features is None:
            raise RuntimeError("cannot rebuild an empty chunked context")
        return step_features

    def _streaming_token_step(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Append one environment timestep to the bounded transformer cache."""

        if self.inference_position_ids is None:
            raise RuntimeError("streaming position counters are not initialized")

        B = obs.shape[0]
        device = obs.device
        obs_emb = self.obs_norm(self.obs_embedding(obs.unsqueeze(0)))[0]
        obs_emb = obs_emb + self.modality_emb(
            torch.zeros(B, dtype=torch.long, device=device)
        )

        if self.ref_obs_dim > 0:
            if ref_obs is None:
                ref_tensor = torch.zeros(
                    B, self.ref_obs_dim, dtype=obs.dtype, device=device
                )
                ref_mask = torch.zeros(B, dtype=torch.bool, device=device)
            else:
                ref_tensor, ref_mask = ref_obs
                ref_mask = ref_mask.bool() & obs_mask.bool()
            ref_emb = self.ref_obs_norm(
                self.ref_obs_embedding(ref_tensor.unsqueeze(0))
            )[0]
            ref_emb = ref_emb + self.modality_emb(
                torch.ones(B, dtype=torch.long, device=device)
            )
            tokens = torch.stack((ref_emb, obs_emb), dim=1)
            token_padding = ~torch.stack((ref_mask, obs_mask.bool()), dim=1)
            obs_token_index = 1
        else:
            tokens = obs_emb.unsqueeze(1)
            token_padding = ~obs_mask.bool().unsqueeze(1)
            obs_token_index = 0

        tokens_per_step = tokens.shape[1]
        position_ids = self.inference_position_ids.unsqueeze(1).expand(
            -1, tokens_per_step
        )
        hidden, self.inference_kv_cache = self.decoder.forward_with_kv_cache(
            tokens,
            position_ids=position_ids,
            token_padding_mask=token_padding,
            cache=self.inference_kv_cache,
            max_cache_tokens=(
                (self.streaming_context_steps - 1) * tokens_per_step
            ),
        )
        self.inference_position_ids = self.inference_position_ids + obs_mask.long()
        return hidden[:, obs_token_index]

    def _rebuild_streaming_cache(self) -> torch.Tensor:
        """Prefill cache from the reset-safe raw history retained by rollout."""

        self.inference_kv_cache = None
        self.inference_position_ids.zero_()
        masks = self.generate_mask()
        hidden = None
        for step in range(self.num_steps_per_env):
            ref_step = None
            if self.ref_obs_dim > 0:
                ref_step = (
                    self.inference_buffer_ref_obs[step],
                    self.inference_buffer_ref_obs_mask[step],
                )
            hidden = self._streaming_token_step(
                self.inference_buffer_obs[step], masks[step], ref_step
            )
        self.inference_kv_cache_dirty = False
        if hidden is None:
            raise RuntimeError("cannot rebuild an empty streaming context")
        return hidden

    def _infer_temporal_phases(
        self,
        masks: torch.Tensor,
        episode_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Infer phases only when a sequence begins at an episode boundary.

        PPO supplies stored phases for rolling prefixes.  This fallback keeps
        direct unit-level calls convenient, but deliberately treats the first
        valid frame in each visible segment as phase zero.
        """

        T, B = masks.shape
        phases = torch.zeros(T, B, dtype=torch.long, device=masks.device)
        for step in range(1, T):
            continues = (
                masks[step]
                & masks[step - 1]
                & (episode_ids[step] == episode_ids[step - 1])
            )
            phases[step] = torch.where(
                continues,
                (phases[step - 1] + 1) % self.temporal_chunk_size,
                torch.zeros_like(phases[step - 1]),
            )
        return phases

    def _build_partial_chunk_tokens(
        self,
        frame_features: torch.Tensor,
        masks: torch.Tensor,
        episode_ids: torch.Tensor,
        temporal_phases: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one strictly causal partial token for every physical frame."""

        if self.temporal_chunk_encoder is None:
            raise RuntimeError("temporal chunk encoder is not initialized")
        T, B, D = frame_features.shape
        if tuple(temporal_phases.shape) != (T, B):
            raise ValueError(
                "temporal_phases must match masks, got "
                f"{tuple(temporal_phases.shape)} and {(T, B)}"
            )
        temporal_phases = temporal_phases.long()

        chunk_size = self.temporal_chunk_size
        time_index = torch.arange(T, device=frame_features.device).view(T, 1, 1)
        slot_index = torch.arange(chunk_size, device=frame_features.device).view(
            1, 1, chunk_size
        )
        source_index = (
            time_index - temporal_phases.unsqueeze(-1) + slot_index
        )
        in_bounds = (source_index >= 0) & (source_index < T)
        clamped_source = source_index.clamp(0, max(T - 1, 0))
        source_bt = clamped_source.permute(1, 0, 2)

        expanded_features = frame_features.permute(1, 0, 2).unsqueeze(2).expand(
            B, T, chunk_size, D
        )
        gather_index = source_bt.unsqueeze(-1).expand(B, T, chunk_size, D)
        windows = torch.gather(expanded_features, 1, gather_index).permute(
            1, 0, 2, 3
        )

        flat_source = source_bt.reshape(B, -1)
        source_masks = torch.gather(masks.T, 1, flat_source).reshape(
            B, T, chunk_size
        ).permute(1, 0, 2).bool()
        source_episodes = torch.gather(
            episode_ids.T, 1, flat_source
        ).reshape(B, T, chunk_size).permute(1, 0, 2)
        expected_slots = slot_index <= temporal_phases.unsqueeze(-1)
        slot_mask = (
            in_bounds
            & expected_slots
            & masks.bool().unsqueeze(-1)
            & source_masks
            & (source_episodes == episode_ids.unsqueeze(-1))
        )
        partial_tokens = self.temporal_chunk_encoder.encode_windows(
            windows, slot_mask
        )
        full_chunk_mask = (
            masks.bool()
            & (temporal_phases == chunk_size - 1)
            & slot_mask.all(dim=-1)
        )

        # Absolute offsets are unnecessary for RoPE; preserving the physical
        # 50 Hz differences and the episode-local phase is sufficient.
        positions = torch.zeros(T, B, dtype=torch.long, device=masks.device)
        if T > 0:
            positions[0] = torch.where(
                masks[0], temporal_phases[0], torch.zeros_like(temporal_phases[0])
            )
        for step in range(1, T):
            continues = (
                masks[step]
                & masks[step - 1]
                & (episode_ids[step] == episode_ids[step - 1])
            )
            positions[step] = torch.where(
                continues,
                positions[step - 1] + 1,
                torch.where(
                    masks[step],
                    temporal_phases[step],
                    torch.zeros_like(positions[step - 1]),
                ),
            )
        return partial_tokens, full_chunk_mask, positions

    def _completed_chunk_context(
        self,
        partial_tokens: torch.Tensor,
        full_chunk_mask: torch.Tensor,
        masks: torch.Tensor,
        episode_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the slow transformer only on completed episode-local chunks."""

        T, B, D = partial_tokens.shape
        max_chunks = max(1, math.ceil(T / self.temporal_chunk_size))
        chunk_rank = torch.cumsum(full_chunk_mask.long(), dim=0) - 1
        assignment = F.one_hot(
            chunk_rank.clamp(min=0, max=max_chunks - 1),
            num_classes=max_chunks,
        ).bool() & full_chunk_mask.unsqueeze(-1)
        assignment_float = assignment.to(partial_tokens.dtype)
        packed_tokens = torch.einsum(
            "tbc,tbd->cbd", assignment_float, partial_tokens
        )
        packed_mask = assignment.any(dim=0)  # [B, C]
        packed_positions = (
            assignment.long() * positions.unsqueeze(-1)
        ).sum(dim=0)
        packed_episode_ids = (
            assignment.long() * episode_ids.unsqueeze(-1)
        ).sum(dim=0)

        query_index = torch.arange(
            max_chunks, device=partial_tokens.device
        ).view(max_chunks, 1)
        key_index = torch.arange(
            max_chunks, device=partial_tokens.device
        ).view(1, max_chunks)
        blocked = key_index > query_index
        blocked = blocked | (
            key_index
            < query_index - self.streaming_context_tokens + 1
        )
        segment_blocked = (
            packed_episode_ids.unsqueeze(2)
            != packed_episode_ids.unsqueeze(1)
        )
        attention_mask = blocked.unsqueeze(0) | segment_blocked
        slow_output = self.decoder(
            packed_tokens.permute(1, 0, 2),
            src_mask=attention_mask,
            src_key_padding_mask=~packed_mask,
            position_ids=packed_positions,
        )

        latest_rank = torch.full(
            (B,), -1, dtype=torch.long, device=partial_tokens.device
        )
        latest_by_frame = []
        for step in range(T):
            if step == 0:
                starts_segment = masks[step]
            else:
                starts_segment = masks[step] & (
                    ~masks[step - 1]
                    | (episode_ids[step] != episode_ids[step - 1])
                )
            latest_rank = torch.where(
                starts_segment,
                torch.full_like(latest_rank, -1),
                latest_rank,
            )
            latest_rank = torch.where(
                full_chunk_mask[step], chunk_rank[step], latest_rank
            )
            latest_by_frame.append(latest_rank.clone())
        latest_by_frame = torch.stack(latest_by_frame, dim=0)
        has_context = (latest_by_frame >= 0) & masks.bool()
        gather_rank = latest_by_frame.clamp(min=0, max=max_chunks - 1).T
        gathered = torch.gather(
            slow_output,
            1,
            gather_rank.unsqueeze(-1).expand(B, T, D),
        ).permute(1, 0, 2)
        gathered = gathered * has_context.unsqueeze(-1).to(gathered.dtype)
        return gathered, has_context

    def _fuse_partial_and_slow_context(
        self,
        partial_tokens: torch.Tensor,
        slow_context: torch.Tensor,
        has_context: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.temporal_context_fusion is None or self.temporal_output_norm is None:
            raise RuntimeError("temporal context fusion is not initialized")
        fusion_input = torch.cat(
            (
                partial_tokens,
                slow_context,
                has_context.unsqueeze(-1).to(partial_tokens.dtype),
            ),
            dim=-1,
        )
        context_delta = self.temporal_context_fusion(fusion_input)
        fused = self.temporal_output_norm(
            partial_tokens
            + context_delta * has_context.unsqueeze(-1).to(context_delta.dtype)
        )
        return fused * masks.unsqueeze(-1).to(fused.dtype)

    def _temporal_plan_candidates(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode one K-action proposal from every candidate plan state."""

        if self.temporal_plan_head is None:
            raise RuntimeError("fast/slow temporal plan head is not initialized")
        flat_plan = self.temporal_plan_head(hidden)
        return flat_plan.unflatten(
            -1, (self.temporal_chunk_size, self.dim_out)
        )

    def _persistent_chunk_plan_sequence(
        self,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        episode_ids: torch.Tensor,
        temporal_phases: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hold the phase-zero plan fixed for every visible chunk.

        Streaming PPO includes a reset-safe history prefix.  Therefore every
        optimized step has its true phase-zero source in this sequence.  Direct
        diagnostic calls may begin in the middle of a chunk; those unrecoverable
        prefix frames use their local candidate and are marked invalid so they
        cannot be mistaken for a calibrated chunk plan.
        """

        T, B, _ = hidden.shape
        if tuple(masks.shape) != (T, B):
            raise ValueError("plan masks must match the hidden sequence")
        if tuple(episode_ids.shape) != (T, B):
            raise ValueError("plan episode_ids must match the hidden sequence")
        if tuple(temporal_phases.shape) != (T, B):
            raise ValueError("plan phases must match the hidden sequence")
        if ((temporal_phases < 0) | (
            temporal_phases >= self.temporal_chunk_size
        )).any():
            raise ValueError("temporal phases must lie inside the chunk")

        time_index = torch.arange(T, device=hidden.device).view(T, 1)
        source_index = time_index - temporal_phases.long()
        in_bounds = source_index >= 0
        source_index = source_index.clamp(0, max(T - 1, 0))
        source_bt = source_index.T

        source_mask = torch.gather(masks.T, 1, source_bt).T.bool()
        source_episode = torch.gather(episode_ids.T, 1, source_bt).T
        source_phase = torch.gather(temporal_phases.T, 1, source_bt).T
        valid_source = (
            in_bounds
            & masks.bool()
            & source_mask
            & (source_episode == episode_ids)
            & (source_phase == 0)
        )

        # Decode plans only where a new chunk starts.  A local fallback is
        # required solely for diagnostic prefixes whose true phase-zero frame
        # lies outside the supplied sequence.  This keeps the slow plan head
        # off the 50-Hz path during normal rollout/PPO recomputation.
        needs_candidate = masks.bool() & (
            (temporal_phases == 0) | ~valid_source
        )
        candidates = hidden.new_zeros(
            T,
            B,
            self.temporal_chunk_size,
            self.dim_out,
        )
        if needs_candidate.any():
            candidates[needs_candidate] = self._temporal_plan_candidates(
                hidden[needs_candidate]
            )
        candidates_bt = candidates.permute(1, 0, 2, 3)
        gathered_plan = torch.gather(
            candidates_bt,
            1,
            source_bt.unsqueeze(-1).unsqueeze(-1).expand(
                B,
                T,
                self.temporal_chunk_size,
                self.dim_out,
            ),
        ).permute(1, 0, 2, 3)
        plans = torch.where(
            valid_source.unsqueeze(-1).unsqueeze(-1),
            gathered_plan,
            candidates,
        )
        plans = plans * masks.unsqueeze(-1).unsqueeze(-1).to(plans.dtype)
        return plans, valid_source

    def _compose_temporal_fast_action(
        self,
        obs_embedding: torch.Tensor,
        nominal_action: torch.Tensor,
        temporal_phases: torch.Tensor,
        ref_embedding: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 50-Hz feedback around one slot of a persistent slow plan."""

        if self.temporal_fast_controller is None:
            raise RuntimeError("fast temporal controller is not initialized")
        if obs_embedding.shape[:-1] != nominal_action.shape[:-1]:
            raise ValueError("fast observation and nominal action axes must match")
        if temporal_phases.shape != nominal_action.shape[:-1]:
            raise ValueError("fast temporal phases must match action batch axes")

        features = [obs_embedding]
        if self.ref_obs_dim > 0:
            if ref_embedding is None or ref_mask is None:
                ref_embedding = torch.zeros_like(obs_embedding)
                ref_mask = torch.zeros_like(temporal_phases, dtype=torch.bool)
            ref_mask = ref_mask.bool()
            features.extend(
                (
                    ref_embedding
                    * ref_mask.unsqueeze(-1).to(ref_embedding.dtype),
                    ref_mask.unsqueeze(-1).to(obs_embedding.dtype),
                )
            )
        phase_one_hot = F.one_hot(
            temporal_phases.long(), num_classes=self.temporal_chunk_size
        ).to(obs_embedding.dtype)
        features.extend((nominal_action, phase_one_hot))
        correction = torch.tanh(
            self.temporal_fast_controller(torch.cat(features, dim=-1))
        )
        action = (
            nominal_action
            + self.temporal_fast_residual_scale * correction
        )
        return action, correction

    def _decode_temporal_sequence_actions(
        self,
        hidden: torch.Tensor,
        obs_embedding: torch.Tensor,
        masks: torch.Tensor,
        episode_ids: torch.Tensor,
        temporal_phases: torch.Tensor,
        ref_embedding: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        plans, _ = self._persistent_chunk_plan_sequence(
            hidden, masks, episode_ids, temporal_phases
        )
        nominal_action = torch.gather(
            plans,
            2,
            temporal_phases.long().unsqueeze(-1).unsqueeze(-1).expand(
                *temporal_phases.shape, 1, self.dim_out
            ),
        ).squeeze(2)
        actions, _ = self._compose_temporal_fast_action(
            obs_embedding,
            nominal_action,
            temporal_phases,
            ref_embedding,
            ref_mask,
        )
        return actions * masks.unsqueeze(-1).to(actions.dtype)

    def _update_inference_chunk_plan(
        self,
        hidden: torch.Tensor,
        temporal_phase: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if self.inference_chunk_plan is None or self.inference_chunk_plan_valid is None:
            raise RuntimeError("inference chunk plan state is not initialized")
        starts_plan = valid_mask.bool() & (
            (temporal_phase == 0) | ~self.inference_chunk_plan_valid
        )
        if starts_plan.any():
            self.inference_chunk_plan[starts_plan] = (
                self._temporal_plan_candidates(hidden[starts_plan])
            )
            self.inference_chunk_plan_valid[starts_plan] = True

    def _decode_temporal_inference_action(
        self,
        hidden: torch.Tensor,
        obs_embedding: torch.Tensor,
        temporal_phase: torch.Tensor,
        valid_mask: torch.Tensor,
        ref_embedding: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._update_inference_chunk_plan(hidden, temporal_phase, valid_mask)
        env_index = torch.arange(hidden.shape[0], device=hidden.device)
        nominal_action = self.inference_chunk_plan[
            env_index, temporal_phase.long()
        ]
        actions, _ = self._compose_temporal_fast_action(
            obs_embedding,
            nominal_action,
            temporal_phase,
            ref_embedding,
            ref_mask,
        )
        return actions * valid_mask.unsqueeze(-1).to(actions.dtype)

    def _chunked_streaming_sequence_hidden(
        self,
        obs: torch.Tensor,
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]],
        masks: torch.Tensor,
        episode_ids: Optional[torch.Tensor] = None,
        temporal_phases: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Differentiable PPO path for the 5-frame/one-token variant."""

        T, B, _ = obs.shape
        if episode_ids is None:
            episode_ids = torch.zeros_like(masks, dtype=torch.long)
        if temporal_phases is None:
            temporal_phases = self._infer_temporal_phases(masks, episode_ids)

        obs_embedding = self.obs_norm(self.obs_embedding(obs))
        has_ref = ref_obs is not None and self.ref_obs_dim > 0
        ref_embedding = None
        ref_mask = None
        if self.ref_obs_dim > 0:
            if ref_obs is None:
                ref_tensor = torch.zeros(
                    T, B, self.ref_obs_dim, dtype=obs.dtype, device=obs.device
                )
                ref_mask = torch.zeros(T, B, dtype=torch.bool, device=obs.device)
            else:
                ref_tensor, ref_mask = ref_obs
                ref_mask = ref_mask.bool() & masks.bool()
            ref_embedding = self.ref_obs_norm(self.ref_obs_embedding(ref_tensor))

        frame_features = self.temporal_chunk_encoder.encode_frames(
            obs_embedding,
            masks.bool(),
            ref_embedding,
            ref_mask,
        )
        partial_tokens, full_chunk_mask, positions = (
            self._build_partial_chunk_tokens(
                frame_features,
                masks.bool(),
                episode_ids,
                temporal_phases,
            )
        )
        slow_context, has_context = self._completed_chunk_context(
            partial_tokens,
            full_chunk_mask,
            masks.bool(),
            episode_ids,
            positions,
        )
        hidden = self._fuse_partial_and_slow_context(
            partial_tokens, slow_context, has_context, masks.bool()
        )
        return hidden, obs_embedding, ref_embedding if has_ref else None

    def _streaming_sequence_hidden(
        self,
        obs: torch.Tensor,
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]],
        masks: torch.Tensor,
        episode_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode a full sequence with the same local semantics as KV rollout.

        This differentiable path is the training reference for the inference
        cache. ``episode_ids`` prevents attention across resets inside one
        contiguous rollout batch; padding alone cannot express that boundary.
        """

        T, B, _ = obs.shape
        if tuple(masks.shape) != (T, B):
            raise ValueError(
                f"masks must have shape {(T, B)}, got {tuple(masks.shape)}"
            )
        if episode_ids is None:
            episode_ids = torch.zeros_like(masks, dtype=torch.long)
        elif tuple(episode_ids.shape) != (T, B):
            raise ValueError(
                "episode_ids must match masks, got "
                f"{tuple(episode_ids.shape)} and {tuple(masks.shape)}"
            )

        device = obs.device
        obs_emb = self.obs_norm(self.obs_embedding(obs))
        res_obs_emb = obs_emb
        obs_emb = obs_emb + self.modality_emb(
            torch.zeros(T, B, dtype=torch.long, device=device)
        )

        has_ref = ref_obs is not None and self.ref_obs_dim > 0
        res_ref_obs_emb = None
        if has_ref:
            ref_tensor, ref_mask = ref_obs
            if tuple(ref_tensor.shape[:2]) != (T, B):
                raise ValueError("reference sequence must match observation T/B")
            ref_mask = ref_mask.bool() & masks.bool()
            ref_emb = self.ref_obs_norm(self.ref_obs_embedding(ref_tensor))
            res_ref_obs_emb = ref_emb
            ref_emb = ref_emb + self.modality_emb(
                torch.ones(T, B, dtype=torch.long, device=device)
            )
            combined_seq = torch.stack((ref_emb, obs_emb), dim=1).reshape(
                2 * T, B, -1
            )
            combined_masks = torch.stack(
                (ref_mask, masks.bool()), dim=1
            ).reshape(2 * T, B)
            tokens_per_step = 2
            obs_token_index = 1
        else:
            combined_seq = obs_emb
            combined_masks = masks.bool()
            tokens_per_step = 1
            obs_token_index = 0

        # Episode-local positions are computed per environment. Padded tokens
        # keep position zero and are removed by the key padding mask.
        positions = torch.zeros(T, B, dtype=torch.long, device=device)
        for step in range(1, T):
            same_episode = episode_ids[step] == episode_ids[step - 1]
            positions[step] = torch.where(
                same_episode & masks[step] & masks[step - 1],
                positions[step - 1] + 1,
                torch.zeros_like(positions[step - 1]),
            )

        token_steps = torch.arange(T, device=device).repeat_interleave(
            tokens_per_step
        )
        query_step = token_steps.unsqueeze(1)
        key_step = token_steps.unsqueeze(0)
        token_count = T * tokens_per_step
        local_blocked = torch.ones(
            token_count, token_count, dtype=torch.bool, device=device
        ).triu(diagonal=1)
        local_blocked = local_blocked | (
            key_step
            < query_step - self.streaming_context_steps + 1
        )
        token_episode_ids = episode_ids.repeat_interleave(
            tokens_per_step, dim=0
        ).T
        segment_blocked = (
            token_episode_ids.unsqueeze(2)
            != token_episode_ids.unsqueeze(1)
        )
        attention_mask = local_blocked.unsqueeze(0) | segment_blocked
        token_positions = positions.repeat_interleave(
            tokens_per_step, dim=0
        ).T

        hidden = self.decoder(
            combined_seq.permute(1, 0, 2),
            src_mask=attention_mask,
            src_key_padding_mask=~combined_masks.T,
            position_ids=token_positions,
        ).permute(1, 0, 2)
        return (
            hidden[obs_token_index::tokens_per_step],
            res_obs_emb,
            res_ref_obs_emb,
        )

    def _forward_batch(self, 
                obs: torch.Tensor,
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                masks: Optional[torch.Tensor] = None, # fake optional. Must be provided in batch mode
                memory: Optional[Tuple[torch.Tensor, torch.Tensor | None]] = None, # memory token for both modalities
                raw_observations: Optional[torch.Tensor] = None,
                dreamer_target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                unpad_output: bool = True,
                upsample: bool = True,
                compute_aux_loss: bool = False,
                compute_dreamer_loss: bool = False,
                compute_dreamer_reward: bool = False,
                episode_ids: Optional[torch.Tensor] = None,
                temporal_phases: Optional[torch.Tensor] = None,
                sequence_output_start: Optional[int] = None,
                ):
        """
        Forward pass for the multi-modality transformer in batch mode.
        """
        # A PPO batch forward is followed by an optimizer update. Any K/V made
        # with the previous parameters must therefore be prefetched again
        # before the next rollout; the raw reset-safe history is retained for
        # exactly that purpose.
        self.invalidate_inference_cache()
        
        ##########################################################################################
        # T_x meanings:
        # T_in: original input sequence length
        # T_c: compressed sequence length (after Conv1dCompressor)
        # T_combined: combined sequence length (after merging two modalities, if available)
        # if not has_ref: T_combined == T_c, else T_combined == 2 * T_c
        ##########################################################################################
        
        
        T_in, B, _ = obs.shape
        device = obs.device   
        ttt_initial_state = None
        ttt_initial_state_reset_mask = None
        ttt_state_reset_masks = None
        if self.ttt_memory_enabled:
            if memory is not None and not isinstance(memory, dict):
                raise ValueError(
                    "TTT PPO memory must be streaming metadata, got "
                    f"{type(memory).__name__}"
                )
            if isinstance(memory, dict):
                ttt_initial_state = memory.get("ttt_initial_state")
                ttt_initial_state_reset_mask = memory.get(
                    "ttt_initial_state_reset_mask"
                )
                ttt_state_reset_masks = memory.get("ttt_state_reset_masks")
        if sequence_output_start is not None:
            if not self.streaming_kv_cache:
                raise ValueError(
                    "sequence_output_start is available only for streaming training"
                )
            if not 0 <= sequence_output_start < T_in:
                raise ValueError(
                    "sequence_output_start must index the input sequence"
                )
            if self.temporal_chunk_enabled and temporal_phases is None:
                raise ValueError(
                    "chunked streaming PPO requires stored temporal_phases"
                )
            if self.ttt_memory_enabled and ttt_initial_state is None:
                raise ValueError(
                    "streaming TTT PPO requires the rollout-start fast state"
                )
            if self.ttt_memory_enabled and ttt_initial_state_reset_mask is None:
                raise ValueError(
                    "streaming TTT PPO requires the rollout-start reset mask"
                )
            if (
                self.ttt_memory_enabled
                and self.ttt_memory_horizon_steps > 0
                and ttt_state_reset_masks is None
            ):
                raise ValueError(
                    "finite-horizon streaming TTT PPO requires per-action "
                    "state reset masks"
                )
        if raw_observations is None:
            # Backward-compatible fallback for environments/runners that do not
            # yet expose the pre-normalizer observation path.
            raw_observations = obs
        if raw_observations.shape != obs.shape:
            raise ValueError(
                "raw_observations must match normalized observations exactly, "
                f"got {tuple(raw_observations.shape)} and {tuple(obs.shape)}"
            )
        if T_in > self.num_steps_per_env and not self.streaming_kv_cache:
            obs=obs[-self.num_steps_per_env:, :, :]
            raw_observations = raw_observations[-self.num_steps_per_env:, :, :]
            if masks is not None:
                masks = masks[-self.num_steps_per_env:, :]
            if episode_ids is not None:
                episode_ids = episode_ids[-self.num_steps_per_env:, :]
            if ref_obs is not None:
                ref_obs = (ref_obs[0][-self.num_steps_per_env:, :, :], ref_obs[1][-self.num_steps_per_env:, :])   
            T_in = self.num_steps_per_env
        
        if ref_obs is not None:
            ref_mlp_obs = (ref_obs[0][-1], ref_obs[1][-1])
        else:
            ref_mlp_obs = None

        ref_obs = self._apply_reference_history_contract(ref_obs)
        
        if self.apply_mlp_residual:
            mlp_output = self.mlp_bypass(obs[-1], ref_mlp_obs) # [B, dim_out]    
        else:
            mlp_output = None 
        has_ref = ref_obs is not None and self.ref_obs_dim > 0
        resolved_episode_ids = episode_ids
        resolved_temporal_phases = temporal_phases
        if self.streaming_kv_cache:
            if self.temporal_chunk_enabled:
                if resolved_episode_ids is None:
                    resolved_episode_ids = torch.zeros_like(
                        masks, dtype=torch.long
                    )
                if resolved_temporal_phases is None:
                    resolved_temporal_phases = self._infer_temporal_phases(
                        masks, resolved_episode_ids
                    )
                obs_out_seq, res_obs_emb, res_ref_obs_emb = (
                    self._chunked_streaming_sequence_hidden(
                        obs,
                        ref_obs,
                        masks,
                        episode_ids=resolved_episode_ids,
                        temporal_phases=resolved_temporal_phases,
                    )
                )
            else:
                obs_out_seq, res_obs_emb, res_ref_obs_emb = (
                    self._streaming_sequence_hidden(
                        obs, ref_obs, masks, episode_ids=episode_ids
                    )
                )
            T_c = T_in
            if has_ref:
                ref_obs_tensor, ref_obs_mask = ref_obs
                ref_obs_mask = ref_obs_mask.bool() & masks.bool()
        else:
            obs_emb = self.obs_embedding(obs) # [T_in, B, D]
            res_obs_emb = obs_emb
            # Compress OBS
            if self.compressor is not None:
                obs_emb, new_masks = self.compressor(obs_emb, masks)
                T_c = obs_emb.shape[0]
            else:
                new_masks = masks
                T_c = T_in
            obs_emb = self.obs_norm(obs_emb)
            # Modality embedding
            obs_emb = obs_emb + self.modality_emb(torch.zeros(T_c, B, dtype=torch.long, device=device))

            # Process second modality if available
            if has_ref:
                ref_obs_tensor, ref_obs_mask = ref_obs
                # ref_obs_mask shape: [T, B]
                ref_T_in, ref_B, _ = ref_obs_tensor.shape
                assert ref_B == B, "Batch size of obs and ref_obs must match"
                assert ref_T_in == T_in, "Sequence length of obs and ref_obs must match"
                ref_obs_emb = self.ref_obs_embedding(ref_obs_tensor) # [ref_T_in, B, D]
                res_ref_obs_emb = ref_obs_emb
                # Merge masks
                ref_obs_mask = ref_obs_mask * masks
                # Compress ref_obs
                if self.ref_compressor is not None:
                    ref_obs_emb, new_ref_masks = self.ref_compressor(ref_obs_emb, ref_obs_mask)
                else:
                    new_ref_masks = ref_obs_mask

                ref_obs_emb = self.ref_obs_norm(ref_obs_emb)
                # Modality embedding
                ref_obs_emb = ref_obs_emb + self.modality_emb(torch.ones(T_c, B, dtype=torch.long, device=device))

                # Sequence merging. [obs0, ref_obs0, obs1, ref_obs1, ...]
                if not self.ref_first:
                    combined_seq = torch.stack([obs_emb, ref_obs_emb], dim=1)
                else:
                    combined_seq = torch.stack([ref_obs_emb, obs_emb], dim=1)
                combined_seq = combined_seq.reshape(2 * T_c, B, -1) # [2*T_c, B, D]
                T_combined = combined_seq.shape[0]
                # Merge masks
                if not self.ref_first:
                    combined_masks = torch.stack([new_masks, new_ref_masks], dim=1) # [T_c, 2, B]
                else:
                    combined_masks = torch.stack([new_ref_masks, new_masks], dim=1) # [T_c, 2, B]
                combined_masks = combined_masks.reshape(T_combined, B) # [2*T_c, B]
            else:
                res_ref_obs_emb = None
                combined_seq = obs_emb # [T_c, B, D]
                T_combined = combined_seq.shape[0]
                combined_masks = new_masks # [T_c, B]
            combined_masks = combined_masks.bool()

            src_key_padding_mask = ~combined_masks.T # [B, T], True for padding positions
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T_combined, device=device,dtype=torch.bool)

            #with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
            if not self.apply_rope:
                pos = torch.arange(T_c, device=device).unsqueeze(1).expand(T_c, B)
                if has_ref: # T_combined == 2 * T_c, then we need to repeat interleavely the pos (Original [0, 1, 2], new [0,0,1,1,2,2])
                    pos = pos.repeat_interleave(2, dim=0)
                src = combined_seq + self.pos_emb(pos) # [T_combined, B, D]

                gpt_out = self.decoder(
                    src.permute(1, 0, 2), # [B, T, D]
                    mask=causal_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    # is_causal=True,
                )
            else:
                gpt_out = self.decoder(
                    combined_seq.permute(1, 0, 2), # [B, T, D]
                    src_mask=causal_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    # is_causal=True,
                    interleave=has_ref,
                )

            # Sample obs out seq
            if not self.ref_first:
                obs_indices = torch.arange(0, T_combined, 2, device=device) if has_ref else torch.arange(0, T_combined, 1, device=device) # obs are at even positions or all positions
            else: # obs at odd positions
                obs_indices = torch.arange(1, T_combined, 2, device=device) if has_ref else torch.arange(0, T_combined, 1, device=device) # obs are at odd positions or all positions
            obs_out_seq = gpt_out.permute(1, 0, 2)[obs_indices, :, :] # [T_c, B, D]
        

        if self.upsampler is not None and upsample:
            upsampled_out = self.upsampler(obs_out_seq, T_in) # [T_in, B, D]
        else:
            upsampled_out = obs_out_seq # [T_in, B, D]

        if self.ttt_memory_enabled:
            ttt_update_start = (
                0 if sequence_output_start is None else sequence_output_start
            )
            upsampled_out, _ = self.ttt_memory.forward_sequence(
                upsampled_out,
                masks,
                episode_ids=episode_ids,
                state_reset_masks=ttt_state_reset_masks,
                initial_state=ttt_initial_state,
                initial_state_reset_mask=ttt_initial_state_reset_mask,
                update_start=ttt_update_start,
            )


        # if self.apply_res:
        #     # Add residual connection    
        #     upsampled_out = upsampled_out + res_obs_emb * masks.unsqueeze(-1).float()
        #     if has_ref:
        #         upsampled_out = upsampled_out + res_ref_obs_emb * ref_obs_mask.unsqueeze(-1).float()
        
        # Cache hidden state before projection (for DAgger training)
        if sequence_output_start is not None:
            self.last_hidden_state = upsampled_out[
                sequence_output_start:
            ].flatten(0, 1)
        else:
            self.last_hidden_state = upsampled_out[-1, :, :] if not unpad_output else upsampled_out  # [B, D] or [T_in, B, D]
        
        if compute_aux_loss and self.aux_heads is not None:
            self.aux_loss = 0.0
            obs_actions = raw_observations[:, :, self.action_slice[0]:self.action_slice[1]]  # [T_in, B, action_dim]
            obs_actions = obs_actions[1:] # shift 1 step to ensure matching with input
            gamma, beta = self.action_film(obs_actions).chunk(2, dim=-1)  # [T_in-1, B, D] each
            for obs_term_slice, aux_head, weight in zip(self.obs_indices, self.aux_heads, self.pred_obs_term_weights):
                target = raw_observations[:, :, obs_term_slice[0]:obs_term_slice[1]] # [T_in, B, dim]
                pred_in = self._route_future_prediction_latent(
                    upsampled_out[:-1]
                )  # [T_in-1, B, D]
                # FiLM modulation
                pred_in = gamma * pred_in + beta  # [T_in-1, B, D]
                pred = aux_head(pred_in) # [T_in - 1, B, dim]
                # temporal shifting: note that target (obs) is the last timestep, and pred is current timestep. Thus, the computation should be between pred[:-1] and target[1:]
                target_shifted = target[1:, :, :]
                pred_shifted = pred
                # A one-step prediction is valid only when both endpoints are
                # in the same unmasked trajectory segment. In particular, do
                # not train the world-model head across an episode reset.
                valid_mask = masks[:-1] & masks[1:] # shape: [T_in-1, B]
                if episode_ids is not None:
                    valid_mask = valid_mask & (
                        episode_ids[:-1] == episode_ids[1:]
                    )
                if sequence_output_start is not None:
                    loss_steps = torch.arange(
                        T_in - 1, device=device
                    ).unsqueeze(1)
                    loss_start = sequence_output_start - 1
                    if self.ttt_memory_enabled and ttt_initial_state is not None:
                        # The saved state already contains the raw prefix.  Its
                        # final hidden cannot be reconstructed without writing
                        # that prefix twice, so omit only this boundary pair.
                        loss_start = sequence_output_start
                    valid_mask = valid_mask & (
                        loss_steps >= loss_start
                    )
                aux_loss_i = self._masked_prediction_mse(
                    pred_shifted,
                    target_shifted,
                    valid_mask,
                )
                self.aux_loss += weight * aux_loss_i            
                
        else:
            self.aux_loss = 0.0
        
        # PPO must reach both the action decoder and the shared MMGPT
        # backbone.  Auxiliary gradient routing is controlled independently
        # above and must never detach the policy path.
        if self.temporal_fast_slow_enabled:
            pred_actions = self._decode_temporal_sequence_actions(
                upsampled_out,
                res_obs_emb,
                masks,
                resolved_episode_ids,
                resolved_temporal_phases,
                res_ref_obs_emb,
                ref_obs_mask if has_ref else None,
            )
        else:
            pred_actions = self.fc(upsampled_out) # [T_in, B, dim_out]
        
        if compute_dreamer_reward:
            # dreamer reward computation
            cur_obs_latent = res_obs_emb[:-1, :, :]  # [T_in-1, B, dim_model]
            next_obs_latent = res_obs_emb[1:, :, :]  # [T_in-1, B, dim_model]
            # Score prediction is currently diagnostic-only.  Do not let its
            # untrained head alter the action decoder if it is enabled later.
            cur_actions = pred_actions[:-1, :, :].detach()  # [T_in-1, B, dim_out]
            dreamer_input = torch.cat([cur_obs_latent, cur_actions, next_obs_latent], dim=-1)  # [T_in-1, B, dim_model*2 + dim_out]
            dreamer_rewards = self.dreamer_reward_head(dreamer_input).squeeze(-1)  # [T_in-1, B]
            # mask
            valid_mask = masks[:-1] & masks[1:]  # [T_in-1, B]
            valid_dreamer_rewards = dreamer_rewards * valid_mask.float()  # zero out invalid positions
            self.dreamer_reward = valid_dreamer_rewards.sum() / (valid_mask.sum() + 1e-8)  # average reward over valid positions

            if has_ref and self.dreamer_ref_reward_head is not None:
                ref_obs_tensor, ref_obs_mask = ref_obs
                ref_obs_latent = res_ref_obs_emb  # [T_in, B, dim_model]              
                cur_ref_obs_latent = ref_obs_latent[:-1, :, :]  # [T_in-1, B, dim_model]
                next_ref_obs_latent = ref_obs_latent[1:, :, :]  # [T_in-1, B, dim_model]
                dreamer_ref_input = torch.cat([cur_ref_obs_latent, cur_actions, next_ref_obs_latent], dim=-1)  # [T_in-1, B, dim_model]
                dreamer_ref_rewards = self.dreamer_ref_reward_head(dreamer_ref_input).squeeze(-1)  # [T_in-1, B]
                dreamer_ref_rewards = dreamer_ref_rewards * ref_obs_mask[1:].float()  # zero out invalid positions in ref_obs
                self.dreamer_reward += (dreamer_ref_rewards * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)  # accumulate reward over valid positions
                
                
        
        if compute_dreamer_loss and self.aux_heads is not None and (
            dreamer_target is not None or self.target_action_slice is not None
        ):
            self.dreamer_loss = 0.0
            # extract target actions from ref_obs
            if sequence_output_start is None:
                cur_latent = upsampled_out[-1, :, :]  # [B, dim_out]
                cur_action = pred_actions[-1, :, :]
                previous_action = raw_observations[
                    -1, :, self.action_slice[0]:self.action_slice[1]
                ]
            else:
                cur_latent = upsampled_out[
                    sequence_output_start:
                ].flatten(0, 1)
                cur_action = pred_actions[
                    sequence_output_start:
                ].flatten(0, 1)
                previous_action = raw_observations[
                    sequence_output_start:,
                    :,
                    self.action_slice[0]:self.action_slice[1],
                ].flatten(0, 1)
            cur_action = self._resolve_prediction_action(
                previous_action,
                cur_action,
            )
            cur_latent, cur_action = self._route_dreamer_inputs(
                cur_latent,
                cur_action,
            )
            # no parameter update for aux head
            joint_pred_aux_idx = self.obs_term_names.index('joint_pos')
            film_params = {k: v.detach() for k, v in self.action_film.named_parameters()}
            film_buffers = {k: v.detach() for k, v in self.action_film.named_buffers()}
            cur_gamma, cur_beta = functional_call(self.action_film, (film_params, film_buffers), (cur_action,)).chunk(2, dim=-1)  # [B, D] each
            cur_latent = cur_gamma * cur_latent + cur_beta  # [B, D]
            cur_aux_head = self.aux_heads[joint_pred_aux_idx]
            params = {k: v.detach() for k, v in cur_aux_head.named_parameters()}
            buffers = {k: v.detach() for k, v in cur_aux_head.named_buffers()}
            pred_joint_pos = functional_call(cur_aux_head, (params, buffers), (cur_latent,))  # [B, joint_dim]
            pred_joint_pos += self.default_joint_pos.to(device)  # de-normalize
            if dreamer_target is not None:
                target_joint_pos, target_mask = dreamer_target
            else:
                if sequence_output_start is None:
                    target_joint_pos = ref_obs[0][
                        -1,
                        :,
                        self.target_action_slice[0]:self.target_action_slice[1],
                    ]
                    target_mask = ref_obs[1][-1, :]
                else:
                    target_joint_pos = ref_obs[0][
                        sequence_output_start:,
                        :,
                        self.target_action_slice[0]:self.target_action_slice[1],
                    ].flatten(0, 1)
                    target_mask = ref_obs[1][
                        sequence_output_start:
                    ].flatten(0, 1)
            if target_joint_pos.shape != pred_joint_pos.shape:
                raise ValueError(
                    "Dreamer target must match predicted absolute joint pose, "
                    f"got {tuple(target_joint_pos.shape)} and "
                    f"{tuple(pred_joint_pos.shape)}"
                )
            if target_mask.shape != pred_joint_pos.shape[:-1]:
                raise ValueError(
                    "Dreamer target mask must match the prediction batch, got "
                    f"{tuple(target_mask.shape)} and "
                    f"{tuple(pred_joint_pos.shape[:-1])}"
                )
            self.dreamer_loss = self._masked_prediction_mse(
                pred_joint_pos,
                target_joint_pos,
                target_mask,
            )
                  
            
        else:
            self.dreamer_loss = 0.0
        

        if sequence_output_start is not None:
            res = pred_actions[sequence_output_start:].flatten(0, 1)
        elif unpad_output:
            res = self.unpad(pred_actions, masks).view(self.num_steps_per_env, -1, pred_actions.shape[-1])[-1, :, :] # [valid_length, dim_out] -> [num_steps_per_env, num_envs (B), dim_out] -> [B, dim_out]
        else: # return output last timestep
            res = pred_actions[-1, :, :] # [B, dim_out]
            
        # Fusion with MLP bypass
        if self.apply_mlp_residual:
            gate_input = torch.cat([res, mlp_output], dim=-1) # [B, dim_out * 2]
            gate = self.mlp_gate(gate_input) # [B, dim_out]
            final_output = gate * res + (1 - gate) * mlp_output # [B, dim_out]
        else:
            final_output = res
        return final_output  # [B, dim_out]
    
    def _forward_inference(self, 
                obs: torch.Tensor,
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass for the multi-modality transformer in inference mode.
        """
        # inference mode
        assert obs.dim() == 2, "In inference mode, obs should have shape (B, dim_in)"
        B, _ = obs.shape
        device = obs.device
        if self.inference_buffer_obs is None or B != self.num_envs_cache:
            self._init_inference_buffer(B, device)
        
        # stack in
        self.inference_buffer_obs = torch.roll(self.inference_buffer_obs, shifts=-1, dims=0)
        self.inference_buffer_obs[-1, :, :] = obs
        
        # For each new action, increment the corresponding environment's timestep by 1 (but not exceeding T_in)
        if self.current_timesteps is not None:
            self.current_timesteps = torch.clamp(
                self.current_timesteps + 1,
                max=self.inference_buffer_obs.shape[0],
            )
        
        # right here, we always maintain the reference buffer, in case we need it later
        if self.inference_buffer_ref_obs is not None:
            if self.latest_reference_only:
                self.inference_buffer_ref_obs.zero_()
                self.inference_buffer_ref_obs_mask.zero_()
            else:
                self.inference_buffer_ref_obs = torch.roll(self.inference_buffer_ref_obs, shifts=-1, dims=0)
                self.inference_buffer_ref_obs_mask = torch.roll(self.inference_buffer_ref_obs_mask, shifts=-1, dims=0)
            if ref_obs is not None:
                ref_obs_tensor, ref_obs_mask = ref_obs
                self.inference_buffer_ref_obs[-1, :, :] = ref_obs_tensor
                self.inference_buffer_ref_obs_mask[-1, :] = ref_obs_mask
                
            else:
                self.inference_buffer_ref_obs[-1, :, :].fill_(0.0)
                self.inference_buffer_ref_obs_mask[-1, :].fill_(False)
            
        if self.streaming_kv_cache:
            if self.temporal_chunk_enabled:
                if self.inference_kv_cache_dirty:
                    chunk_step = self._rebuild_chunked_streaming_cache()
                else:
                    ref_step = None
                    if self.ref_obs_dim > 0:
                        ref_step = (
                            self.inference_buffer_ref_obs[-1],
                            self.inference_buffer_ref_obs_mask[-1],
                        )
                    chunk_step = self._streaming_chunk_step(
                        obs,
                        torch.ones(B, dtype=torch.bool, device=device),
                        ref_step,
                    )
                (
                    hidden,
                    obs_embedding,
                    ref_embedding,
                    ref_mask,
                    temporal_phase,
                ) = chunk_step
                if self.temporal_fast_slow_enabled:
                    self.last_hidden_state = hidden
                    return self._decode_temporal_inference_action(
                        hidden,
                        obs_embedding,
                        temporal_phase,
                        torch.ones(B, dtype=torch.bool, device=device),
                        ref_embedding,
                        ref_mask,
                    )
            else:
                if self.inference_kv_cache_dirty:
                    hidden = self._rebuild_streaming_cache()
                else:
                    ref_step = None
                    if self.ref_obs_dim > 0:
                        ref_step = (
                            self.inference_buffer_ref_obs[-1],
                            self.inference_buffer_ref_obs_mask[-1],
                        )
                    hidden = self._streaming_token_step(
                        obs,
                        torch.ones(B, dtype=torch.bool, device=device),
                        ref_step,
                    )
            hidden = self._apply_ttt_inference(hidden)
            self.last_hidden_state = hidden
            return self.fc(hidden)

        obs_seq = self.inference_buffer_obs # [T_in, B, dim_in]
        masks = self.generate_mask() # [T_in, B]    
        if self.ref_obs_embedding is not None:
            ref_obs_seq = self.inference_buffer_ref_obs
            ref_obs_mask = self.inference_buffer_ref_obs_mask
            ref_obs = (ref_obs_seq, ref_obs_mask)
        else:
            ref_obs = None
            
        # call _forward_batch
        return self._forward_batch(obs_seq, ref_obs, masks, unpad_output=False) # [B, dim_out]
        

    def forward(self, 
                obs: torch.Tensor, 
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                masks: Optional[torch.Tensor] = None,
                memory: Optional[Tuple[torch.Tensor, torch.Tensor | None]] = None, # memory token for both modalities
                unpad_output: bool = False,
                **kwargs):
        """
        Forward pass for the mm transformer.

        Args:
            obs (torch.Tensor): Observation tensor of shape (T, B, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (T,B, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (T, B) indicating the presence of ref_obs.
            masks (torch.Tensor): Mask tensor of shape (T, B) indicating valid observations among the trajectory.
            memory (Optional[Tuple[torch.Tensor, torch.Tensor | None]]): Memory tokens for both modalities.
            **kwargs: Only added to avoid errors when unexpected arguments are passed.
            
        ** Important **: Unlike mm transformer, here, padded environments, B, tends to be larger than num_envs.

        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        is_batch_mode = masks is not None
        if is_batch_mode:
            return self._forward_batch(
                obs,
                ref_obs,
                raw_observations=kwargs.get("raw_observations"),
                dreamer_target=kwargs.get("dreamer_target"),
                masks=masks,
                memory=memory,
                unpad_output=unpad_output,
                compute_aux_loss=True,
                compute_dreamer_loss=True,
                compute_dreamer_reward=self.score_prediction_enabled,
                episode_ids=kwargs.get("episode_ids"),
                temporal_phases=kwargs.get("temporal_phases"),
                sequence_output_start=kwargs.get("sequence_output_start"),
            )
        else:
            return self._forward_inference(obs, ref_obs)
        
        
class PrivilegedChunkQ(nn.Module):
    """Calibration-only value head for a complete nominal action plan.

    This module is intentionally separate from the PPO state-value critic.  It
    may consume privileged state during training, but no output or input from
    this head is routed into the actor unless a later algorithm phase adds an
    explicit, frozen-Q actor objective.
    """

    def __init__(
        self,
        critic_obs_dim: int,
        critic_ref_obs_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: Optional[List[int]] = None,
        reference_first: bool = True,
    ):
        super().__init__()
        if chunk_size < 2:
            raise ValueError("chunk Q requires a temporally extended action")
        self.critic_obs_dim = int(critic_obs_dim)
        self.critic_ref_obs_dim = int(critic_ref_obs_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.reference_first = bool(reference_first)

        input_dim = (
            self.critic_obs_dim
            + self.critic_ref_obs_dim
            + self.chunk_size * self.action_dim
        )
        hidden_dims = list(hidden_dims or [512, 256, 128])
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.SiLU()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        critic_observations: torch.Tensor,
        action_chunk: torch.Tensor,
        ref_critic_observations: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> torch.Tensor:
        if critic_observations.shape[-1] != self.critic_obs_dim:
            raise ValueError(
                "chunk-Q critic observation dimension mismatch, got "
                f"{critic_observations.shape[-1]} and {self.critic_obs_dim}"
            )
        expected_action_shape = (self.chunk_size, self.action_dim)
        if action_chunk.shape[-2:] != expected_action_shape:
            raise ValueError(
                "chunk-Q action must end in "
                f"{expected_action_shape}, got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[:-2] != critic_observations.shape[:-1]:
            raise ValueError("chunk-Q state and action batch axes must match")

        components = []
        reference = None
        if self.critic_ref_obs_dim > 0:
            if ref_critic_observations is None:
                reference = critic_observations.new_zeros(
                    *critic_observations.shape[:-1],
                    self.critic_ref_obs_dim,
                )
            else:
                reference, reference_mask = ref_critic_observations
                if reference.shape[:-1] != critic_observations.shape[:-1]:
                    raise ValueError(
                        "chunk-Q reference and state batch axes must match"
                    )
                if reference.shape[-1] != self.critic_ref_obs_dim:
                    raise ValueError("chunk-Q reference dimension mismatch")
                if reference_mask.shape != critic_observations.shape[:-1]:
                    raise ValueError("chunk-Q reference mask axes must match")
                reference = reference * reference_mask.unsqueeze(-1).to(
                    reference.dtype
                )
        if self.reference_first and reference is not None:
            components.append(reference)
        components.append(critic_observations)
        if not self.reference_first and reference is not None:
            components.append(reference)
        components.append(action_chunk.flatten(start_dim=-2))
        return self.network(torch.cat(components, dim=-1))


# MMGPT related Critic Network
# Since MMGPT is so hard to train, we provide a simple MLP critic, in order to stabilize the advantage estimation.
class MMGPTCritic(nn.Module):
    def __init__(
        self,
        obs_size, # actually, it should be named by obs_dim, but I maintained the original name for consistency
        ref_obs_size,
        dim_out,
        dim_model = 64,
        num_heads = 2,
        num_layers = 2,
        ffn_ratio = 4,
        dropout = 0.0,
        name = "",
        term_dict = None,
        ref_term_dict = None,
        ref_term_steps: Optional[Dict[str, int]] = None, # hint: only ref terms support step-wise history encoding for now. We don't provide future information in obs terms.
        concatenate_term_names: Optional[List[List[str]]] = None,
        concatenate_ref_term_names: Optional[List[List[str]]] = None,
        num_steps_per_env = 24, # default, remember to parse this!
        max_seq_len = 24, 
        mlp_hidden_dims = [],
        apply_rope = False, # default: APE; set True to use RoPE
        apply_res = True,
        ref_first = True,
        **kwargs
    ): # hint: we keep the same signature as MMGPT.
        super().__init__()
        self.name = name
        self.obs_dim = obs_size
        self.ref_obs_dim = ref_obs_size
        self.num_steps_per_env = num_steps_per_env
        if kwargs:
            print(
                f"MMGPTCritic {self.name}.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        
        self.obs_embedding = ObservationSeqEmbeddingV3(dim_model, term_dict, concatenate_term_names, nheads=num_heads)
        self.ref_obs_embedding = ObservationSeqEmbeddingV3(dim_model, ref_term_dict, concatenate_ref_term_names,term_steps=ref_term_steps, nheads=num_heads) if ref_obs_size > 0 else None # we maintain the same embedding layers, this does not change.
        self.fc = nn.Sequential(
            nn.Linear(dim_model * (1 + (1 if ref_obs_size > 0 else 0)), dim_model * 2), # expansion required
            nn.SiLU(),
            nn.Linear(dim_model * 2, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_out)
        )
    
    
    def forward(self, 
                obs: torch.Tensor, 
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                masks: Optional[torch.Tensor] = None,
                memory: Optional[Tuple[torch.Tensor, torch.Tensor | None]] = None, # memory 
                unpad_output: bool = False,
                **kwargs):
        """
        Forward pass for the mm transformer critic.

        Args:
            obs (torch.Tensor): Observation tensor of shape (T, B, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (T,B, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (T, B) indicating the presence of ref_obs.
            masks (torch.Tensor): Mask tensor of shape (T, B) indicating valid observations among the trajectory.
            **kwargs: Only added to avoid errors when unexpected arguments are passed.
        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        # NOTE: we always work in batch mode for the critic
        # 1. Unpad obs
        if len(obs.shape) == 3:
            obs_unpad = obs[-1]
        else:
            obs_unpad = obs
        obs_emb = self.obs_embedding(obs_unpad.unsqueeze(1)) # [valid_length, 1, D]
        obs_emb = obs_emb.squeeze(1) # [valid_length, D]
        if ref_obs is not None and self.ref_obs_dim > 0:
            ref_obs_tensor, ref_obs_mask = ref_obs
            if len(ref_obs_tensor.shape) == 3:
                ref_obs_unpad = ref_obs_tensor[-1]
            else:
                ref_obs_unpad = ref_obs_tensor
            ref_obs_emb = self.ref_obs_embedding(ref_obs_unpad.unsqueeze(1)) # [B, 1, D]
            ref_obs_emb = ref_obs_emb.squeeze(1) # [B, D]
            combined_emb = torch.cat([obs_emb, ref_obs_emb], dim=-1) # [B, 2*D]
        else:
            combined_emb = obs_emb # [B, D]
        output = self.fc(combined_emb) # [B, dim_out]
        return output
    
    def reset(self, dones: Optional[torch.Tensor] = None):
        # nothing to reset for the critic
        pass
            
        


# M3GPT (Memory-enhanced Motion-mimicking GPT)
class M3GPT(nn.Module):
    def __init__(
        self,
        obs_size, # actually, it should be named by obs_dim, but I maintained the original name for consistency
        ref_obs_size,
        dim_out,
        dim_model = 64,
        num_heads = 2,
        num_layers = 2,
        ffn_ratio = 4,
        dropout = 0.0,
        name = "",
        term_dict = None,
        ref_term_dict = None,
        concatenate_term_names: Optional[List[List[str]]] = None,
        concatenate_ref_term_names: Optional[List[List[str]]] = None,
        num_steps_per_env = 24, # default, remember to parse this!
        max_seq_len = 24, 
        mlp_hidden_dims = [],
        apply_rope = False, # default: APE; set True to use RoPE
        **kwargs
    ):
        
        super().__init__()
        
        if max_seq_len < num_steps_per_env:
            warnings.warn(f"max_seq_len {max_seq_len} is smaller than num_steps_per_env {num_steps_per_env}. This configuration is not recommended for M3GPT. We suggest using uncompressed sequences (i.e., set max_seq_len >= num_steps_per_env) for better memory modeling.", UserWarning)
        self.name = name
        self.obs_dim = obs_size
        self.ref_obs_dim = ref_obs_size
        self.num_steps_per_env = num_steps_per_env
        self.apply_rope = apply_rope
        self.dim_model = dim_model
        if kwargs:
            print(
                f"MMGPT {self.name}.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        self.stride = math.ceil(num_steps_per_env / max_seq_len) if num_steps_per_env > max_seq_len else 1
        self.obs_embedding = ObservationSeqEmbeddingV2(dim_model, term_dict, concatenate_term_names)
        self.ref_obs_embedding = ObservationSeqEmbeddingV2(dim_model, ref_term_dict, concatenate_ref_term_names) if ref_obs_size > 0 else None
        self.obs_norm = nn.LayerNorm(dim_model)
        self.ref_obs_norm = nn.LayerNorm(dim_model) if ref_obs_size > 0 else None

        self.kernel_size = self.stride
        self.padding = (self.kernel_size - self.stride) // 2 if self.kernel_size > self.stride else 0
        padding_needed = (self.stride - self.num_steps_per_env % self.stride) % self.stride if self.stride > 1 else 0
        L_in_padded = self.num_steps_per_env + padding_needed
        
        if self.stride > 1:
            self.compressed_len = math.floor((L_in_padded + 2 * self.padding - self.kernel_size) / self.stride) + 1
        else:
            self.compressed_len = self.num_steps_per_env
        self.compressor = Conv1dCompressor(dim_model, self.stride, self.kernel_size) if self.stride > 1 else None
        self.ref_compressor = Conv1dCompressor(dim_model, self.stride, self.kernel_size) if self.stride > 1 and ref_obs_size > 0 else None
        self.upsampler = ConvTranspose1dCompressor(dim_model, self.stride, self.kernel_size) if self.stride > 1 else None
        
        # Positional embedding size: (compressed_len + 2 memory tokens) * (2 if has ref_obs else 1)
        max_seq_with_memory = (self.compressed_len + 2) * (2 if ref_obs_size > 0 else 1)
        self.pos_emb = nn.Embedding(max_seq_with_memory, dim_model) if not self.apply_rope else None
        self.modality_emb = nn.Embedding(2, dim_model) # 0 for obs, 1 for ref_obs
        if not self.apply_rope:
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=ffn_ratio*dim_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.decoder = RoPETransformer(
                d_model=dim_model,
                num_heads=num_heads,
                num_encoder_layers=num_layers,
                dim_feedforward=ffn_ratio*dim_model,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                max_seq_len=(self.compressed_len + 2) * 2,  # compressed_len + 2 memory tokens, *2 for interleaving with ref
                use_sdpa=True,
            )
        self.fc = nn.Sequential(
            nn.Linear(dim_model, dim_model * ffn_ratio),
            nn.ELU(),
            nn.Linear(dim_model * ffn_ratio, dim_out)
        )

        self.inference_buffer_obs = None # for storing inference-time past key values
        self.inference_buffer_ref_obs = None
        self.inference_buffer_ref_obs_mask = None
        self.current_timesteps = None  # Track current timesteps for each environment [B,]
        self.num_envs_cache = 0
        self.memory = None
        
    @staticmethod
    def unpad(padded_sequence, masks):
        return padded_sequence.permute(1, 0, 2)[masks.permute(1, 0)] # shape [valid_length, dim]
    
    def generate_mask(self) -> torch.Tensor:
        """
        Generate a mask of shape [T_in, B] based on current_timesteps.
        For each environment b, if its timestep is t (0 <= t < T_in),
        then positions [0, T_in-t-1] in the sequence are False (masked out), 
        and positions [T_in-t, T_in-1] are True.
        
        Returns:
            torch.Tensor: shape [T_in, B], True indicates valid positions, False indicates padding positions
        """
        if self.current_timesteps is None:
            raise RuntimeError("current_timesteps not initialized. Call _init_inference_buffer first.")
        
        T_in = self.num_steps_per_env
        B = self.current_timesteps.shape[0]
        device = self.current_timesteps.device
        
        # Create time index matrix [T_in, B], each column represents timesteps 0, 1, ..., T_in-1
        time_indices = torch.arange(T_in, device=device).unsqueeze(1).expand(T_in, B)
        
        # For each environment b, valid positions start from T_in - current_timesteps[b]
        # i.e., when time_indices >= T_in - current_timesteps[b], it's True
        start_positions = T_in - self.current_timesteps.unsqueeze(0)  # [1, B]
        mask = time_indices >= start_positions  # [T_in, B]
        
        return mask

    def _init_inference_buffer(self, num_envs, device):
        self.num_envs_cache = num_envs
        memory_shape = (2, num_envs, self.dim_model)
        self.memory = torch.zeros(memory_shape, device=device) # 2 modalities
        obs_buffer_shape = (self.num_steps_per_env, num_envs, self.obs_dim)
        self.inference_buffer_obs = torch.zeros(obs_buffer_shape, device=device)
        # Initialize timestep tracking array, initially set to 0
        self.current_timesteps = torch.zeros(num_envs, dtype=torch.long, device=device)
        if self.ref_obs_dim > 0:
            ref_obs_buffer_shape = (self.num_steps_per_env , num_envs, self.ref_obs_dim)
            self.inference_buffer_ref_obs = torch.zeros(ref_obs_buffer_shape, device=device)
            self.inference_buffer_ref_obs_mask = torch.zeros((self.num_steps_per_env, num_envs), dtype=torch.bool, device=device)
            
    def reset(self, dones=None):
        if self.inference_buffer_obs is None:
            return
        if dones is None: # reset all
            self.memory.fill_(0.0)
            self.inference_buffer_obs.fill_(0.0)
            # Reset all environments' timesteps to 0
            if self.current_timesteps is not None:
                self.current_timesteps.fill_(0)
            if self.ref_obs_dim > 0:
                if self.inference_buffer_ref_obs is not None:
                    self.inference_buffer_ref_obs.fill_(0.0)
                if self.inference_buffer_ref_obs_mask is not None:
                    self.inference_buffer_ref_obs_mask.fill_(False)
        else:
            # dones are at the environment level
            mask = (dones == 1)
            self.memory[:, mask, :] = 0.0
            self.inference_buffer_obs[:, mask, :] = 0.0
            # Reset timesteps to 0 for completed environments: timesteps[dones] = 0
            if self.current_timesteps is not None:
                self.current_timesteps[mask] = 0
            if self.ref_obs_dim > 0:
                if self.inference_buffer_ref_obs is not None:
                    self.inference_buffer_ref_obs[:, mask, :] = 0.0
                if self.inference_buffer_ref_obs_mask is not None:
                    self.inference_buffer_ref_obs_mask[:, mask] = False

    
    def _forward_memory_idx(self,
                               current_seq: torch.Tensor,
                               current_masks: torch.Tensor,
                               memory_timestep: int,
                               apply_rope: bool = False):
        """
        Compute memory token[idx] based on the current sequence and masks.
        """
        assert memory_timestep * 2 + 2 <= current_seq.shape[0], f"memory_timestep {memory_timestep} is too large for current_seq with length {current_seq.shape[0]}"
        
        mem_0 = current_seq[0:2, :, :] # [2, B, D]
        T_c, B, D = current_seq.shape
        rec_times = torch.arange(4, device=current_seq.device).unsqueeze(1).expand(4, B) if not apply_rope else None
        causal_mask = nn.Transformer.generate_square_subsequent_mask(4, device=current_seq.device, dtype=torch.bool)
        for i in range(memory_timestep):
            start_idx = 2 + i * 2
            end_idx = start_idx + 2
            segment = current_seq[start_idx:end_idx, :, :] # [2, B, D]
            mask_segment = current_masks[start_idx:end_idx, :] # [2, B]
            in_seq = torch.cat([mem_0, segment], dim=0) # [4, B, D]
            in_mask = torch.cat([mask_segment, mask_segment], dim=0) # [4, B]
            src_key_padding_mask = ~in_mask.T # [B, T], True for padding positions
            if not apply_rope:
                pos = rec_times
                src = in_seq + self.pos_emb(pos) # [T_combined, B, D]
                out = self.decoder(
                    src.permute(1, 0, 2), # [B, T, D]
                    mask=causal_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    # is_causal=True,
                )
            else:
                out = self.decoder(
                    in_seq.permute(1, 0, 2), # [B, T, D]
                    src_mask=causal_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    # is_causal=True,
                    interleave=False,
                )
            mem_0 = out.permute(1, 0, 2)[-2:, :, :] # [2, B, D]
            
        # also compute the out seq
        segment_out = current_seq[2 + memory_timestep * 2:, :, :] # [T_c - 2 - memory_timestep*2, B, D]
        mask_out = current_masks[2 + memory_timestep * 2:, :] # [T_c - 2 - memory_timestep*2, B]
        mask_out = torch.cat([mask_out[0:2, :], mask_out], dim=0) # [T_c - memory_timestep*2, B]
        out_seq = torch.cat([mem_0, segment_out], dim=0) # [T_c - memory_timestep*2, B, D]
        src_key_padding_mask_out = ~mask_out.T # [B, T], True for padding positions
        T_out = out_seq.shape[0]
        causal_mask_out = nn.Transformer.generate_square_subsequent_mask(T_out, device=current_seq.device, dtype=torch.bool)
        if not apply_rope:
            pos_out = torch.arange(T_out, device=current_seq.device).unsqueeze(1).expand(T_out, B)
            out_seq_pos = out_seq + self.pos_emb(pos_out) # [T_out, B, D]
            out_seq_final = self.decoder(
                out_seq_pos.permute(1, 0, 2), # [B, T, D]
                mask=causal_mask_out,
                src_key_padding_mask=src_key_padding_mask_out,
                # is_causal=True,
            )
        else:
            out_seq_final = self.decoder(
                out_seq.permute(1, 0, 2), # [B, T, D]
                src_mask=causal_mask_out,
                src_key_padding_mask=src_key_padding_mask_out,
                # is_causal=True,
                interleave=False,
            )
        out_seq_final = out_seq_final # [B, T_c - memory_timestep*2, D]
        return mem_0, out_seq_final
        
    def _forward_batch(self,
                obs: torch.Tensor,
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                masks: Optional[torch.Tensor] = None, # fake optional. Must be provided in batch mode
                memory: Optional[Tuple[torch.Tensor, torch.Tensor | None]] = None, # memory token for both modalities
                unpad_output: bool = True,
                upsample: bool = True,
                detach_memory: bool = True,
                compute_consistency_loss: bool = False,
                ):
        """
        Forward pass for the multi-modality transformer in batch mode.
        """
        assert (memory is not None or self.memory is not None), "Memory must be provided in batch mode, either as input or from the module state."
        if memory is None:
            memory = self.memory
        else:
            if memory[1] is None:
                memory = (memory[0], torch.zeros_like(memory[0]))
            memory = torch.stack(memory, dim=0) # [2, B, D]
        if detach_memory:
            memory = memory.detach()
        T_in, B, _ = obs.shape
        device = obs.device        
        obs_emb = self.obs_embedding(obs) # [T_in, B, D]
        res_obs_emb = obs_emb
        # Compress OBS
        if self.compressor is not None:
            obs_emb, new_masks = self.compressor(obs_emb, masks)
            T_c = obs_emb.shape[0]
        else:
            new_masks = masks
            T_c = T_in
        obs_emb = self.obs_norm(obs_emb)
        # Modality embedding
        obs_emb = obs_emb + self.modality_emb(torch.zeros(T_c, B, dtype=torch.long, device=device))
        
        # Process second modality if available
        has_ref = ref_obs is not None and self.ref_obs_dim > 0
        if has_ref:
            ref_obs_tensor, ref_obs_mask = ref_obs
            # ref_obs_mask shape: [T, B]
            ref_T_in, ref_B, _ = ref_obs_tensor.shape
            assert ref_B == B, "Batch size of obs and ref_obs must match"
            assert ref_T_in == T_in, "Sequence length of obs and ref_obs must match"
            ref_obs_emb = self.ref_obs_embedding(ref_obs_tensor) # [ref_T_in, B, D]
            res_ref_obs_emb = ref_obs_emb
            # Merge masks
            ref_obs_mask = ref_obs_mask * masks
            # Compress ref_obs
            if self.ref_compressor is not None:
                ref_obs_emb, new_ref_masks = self.ref_compressor(ref_obs_emb, ref_obs_mask)
            else:
                new_ref_masks = ref_obs_mask

            ref_obs_emb = self.ref_obs_norm(ref_obs_emb)
            # Modality embedding
            ref_obs_emb = ref_obs_emb + self.modality_emb(torch.ones(T_c, B, dtype=torch.long, device=device))
            
            # Sequence merging. [obs0, ref_obs0, obs1, ref_obs1, ...]
            combined_seq = torch.stack([obs_emb, ref_obs_emb], dim=1)
            combined_seq = combined_seq.reshape(2 * T_c, B, -1) # [2*T_c, B, D]
            T_combined = combined_seq.shape[0]
            # Merge masks
            combined_masks = torch.stack([new_masks, new_ref_masks], dim=1) # [T_c, 2, B]
            combined_masks = combined_masks.reshape(T_combined, B) # [2*T_c, B]
        else:
            combined_seq = obs_emb # [T_c, B, D]
            T_combined = combined_seq.shape[0]
            combined_masks = new_masks # [T_c, B]
        combined_masks = combined_masks.bool()
        
        combined_seq = torch.cat([memory, combined_seq], dim=0) # [2 + T_c, B, D]

        src_key_padding_mask = ~combined_masks.T # [B, T], True for padding positions
        src_key_padding_mask = torch.cat([src_key_padding_mask[:, 0:2], src_key_padding_mask], dim=1) # [B, 2 + T_c]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(combined_seq.shape[0], device=device,dtype=torch.bool)
        
        #with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
        if not self.apply_rope:
            pos = torch.arange(combined_seq.shape[0], device=device).unsqueeze(1).expand(combined_seq.shape[0], B)
            src = combined_seq + self.pos_emb(pos) # [T_combined+2, B, D]
            
            gpt_out = self.decoder(
                src.permute(1, 0, 2), # [B, T_combined+2, D]
                mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
                # is_causal=True,
            )
        else:
            gpt_out = self.decoder(
                combined_seq.permute(1, 0, 2), # [B, T+2, D]
                src_mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
                # is_causal=True,
                interleave=has_ref,
            )
            
        gpt_out = gpt_out[ :, 2:, :] # remove memory tokens, [B, T_c, D]
        self.memory = gpt_out[:, :2, :].permute(1, 0, 2) # update memory, [2, B, D]
        
        
        consistency_loss = None
        if compute_consistency_loss:
            rand_idx = torch.randint(0, T_c // 2 - 1, (1,)).item() # int in [0, T_c // 2 - 1). We assume T_c >= 2, otherwise computing memory will not be necessary.
            orig_seq = gpt_out[:, rand_idx * 2:, :].detach() # shape [B, T_c - rand_idx*2, D]
            mem_at_idx, out_seq = self._forward_memory_idx(combined_seq, combined_masks, rand_idx, apply_rope=self.apply_rope) # [2, B, D], [B, T_c - rand_idx*2, D]
            consistency_loss = F.mse_loss(out_seq, orig_seq) # scalar
            
            
    
        # Sample obs out seq
        obs_indices = torch.arange(0, T_combined, 2, device=device) if has_ref else torch.arange(0, T_combined, 1, device=device) # obs are at even positions or all positions
        obs_out_seq = gpt_out.permute(1, 0, 2)[obs_indices, :, :] # [T_c, B, D]
        

        if self.upsampler is not None and upsample:
            upsampled_out = self.upsampler(obs_out_seq, T_in) # [T_in, B, D]
        else:
            upsampled_out = obs_out_seq # [T_in, B, D]
            

            
        upsampled_out += res_obs_emb
        if has_ref:
            upsampled_out += res_ref_obs_emb * ref_obs_mask.unsqueeze(-1).float()
        
        upsampled_out = self.fc(upsampled_out) # [T_in, B, dim_out]
        

        if unpad_output:
            return self.unpad(upsampled_out, masks).view(self.num_steps_per_env, -1, upsampled_out.shape[-1]), consistency_loss # [valid_length, dim_out] -> [num_steps_per_env, num_envs, dim_out]
        else: # return output last timestep
            return upsampled_out[-1, :, :], consistency_loss # [B, dim_out]
    
    def _forward_inference(self, 
                obs: torch.Tensor,
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass for the multi-modality transformer in inference mode.
        """
        # inference mode
        assert obs.dim() == 2, "In inference mode, obs should have shape (B, dim_in)"
        B, _ = obs.shape
        device = obs.device
        if self.inference_buffer_obs is None or B != self.num_envs_cache:
            self._init_inference_buffer(B, device)
        
        # stack in
        self.inference_buffer_obs = torch.roll(self.inference_buffer_obs, shifts=-1, dims=0)
        self.inference_buffer_obs[-1, :, :] = obs
        
        # For each new action, increment the corresponding environment's timestep by 1 (but not exceeding T_in)
        if self.current_timesteps is not None:
            self.current_timesteps = torch.clamp(self.current_timesteps + 1, max=self.num_steps_per_env)
        
        # right here, we always maintain the reference buffer, in case we need it later
        if self.inference_buffer_ref_obs is not None:
            self.inference_buffer_ref_obs = torch.roll(self.inference_buffer_ref_obs, shifts=-1, dims=0)
            self.inference_buffer_ref_obs_mask = torch.roll(self.inference_buffer_ref_obs_mask, shifts=-1, dims=0)
            if ref_obs is not None:
                ref_obs_tensor, ref_obs_mask = ref_obs
                self.inference_buffer_ref_obs[-1, :, :] = ref_obs_tensor
                self.inference_buffer_ref_obs_mask[-1, :] = ref_obs_mask
                
            else:
                self.inference_buffer_ref_obs[-1, :, :].fill_(0.0)
                self.inference_buffer_ref_obs_mask[-1, :].fill_(False)
            
        obs_seq = self.inference_buffer_obs # [T_in, B, dim_in]
        masks = self.generate_mask() # [T_in, B]    
        if self.ref_obs_embedding is not None:
            ref_obs_seq = self.inference_buffer_ref_obs
            ref_obs_mask = self.inference_buffer_ref_obs_mask
            ref_obs = (ref_obs_seq, ref_obs_mask)
        else:
            ref_obs = None
            
        # call _forward_batch
        return self._forward_batch(obs_seq, ref_obs, masks, unpad_output=False, compute_consistency_loss=False) # [B, dim_out]
        

    def forward(self, 
                obs: torch.Tensor, 
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                masks: Optional[torch.Tensor] = None,
                memory: Optional[Tuple[torch.Tensor, torch.Tensor | None]] = None, # memory token for both modalities
                unpad_output: bool = False,
                compute_consistency_loss: bool = False,
                **kwargs):
        """
        Forward pass for the mm transformer.

        Args:
            obs (torch.Tensor): Observation tensor of shape (T, B, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (T,B, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (T, B) indicating the presence of ref_obs.
            masks (torch.Tensor): Mask tensor of shape (T, B) indicating valid observations among the trajectory.
            memory (Optional[Tuple[torch.Tensor, torch.Tensor | None]]): Memory tokens for both modalities.
            **kwargs: Only added to avoid errors when unexpected arguments are passed.
            
        ** Important **: Unlike mm transformer, here, padded environments, B, tends to be larger than num_envs.

        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        is_batch_mode = masks is not None
        if is_batch_mode:
            return self._forward_batch(obs, ref_obs, masks, memory=memory, unpad_output=unpad_output, compute_consistency_loss=compute_consistency_loss)
        else:
            return self._forward_inference(obs, ref_obs)

    def get_hidden_state(self):
        if self.memory is None:
            raise RuntimeError("Memory is not initialized.")
        return self.memory.clone().detach()
    
    def get_hidden_state_with_grad(self):
        if self.memory is None:
            raise RuntimeError("Memory is not initialized.")
        return self.memory

# Actor-Critic with MM-GPT backbone
class ActorCriticMMGPT(ActorCriticMMTransformer):
    is_recurrent = True
    def __init__(self,
                 num_actor_obs,
                 num_actor_ref_obs,
                 num_critic_obs,
                 num_critic_ref_obs,
                 num_actions,
                 term_dict,
                 ref_term_dict,
                 ref_term_steps: Optional[Dict[str, int]] = None,
                 concatenate_term_names: Optional[List[List[str]]] = None,
                 concatenate_ref_term_names: Optional[List[List[str]]] = None,
                 pred_obs_term_names: Optional[List[str]] = None,
                 pred_obs_term_weights: Optional[List[float]] = None,
                 dim_model=64,
                 num_heads=2,
                 num_layers=2,
                 num_steps_per_env=24,
                 max_seq_len=16,
                 mlp_hidden_dims = [128, 64],
                 init_noise_std=1.0,
                 min_noise_std: Optional[float] = None,
                 max_noise_std: Optional[float] = None,
                 noise_std_type: str = "scalar",
                 load_dagger=False,
                 load_dagger_path=None,
                 load_actor_path=None,
                 load_critic_path=None,
                 load_std_path=None,
                 enable_lora=False,
                 dropout=0.0,
                 apply_rope=False,
                 default_joint_pos: Optional[torch.Tensor] = None,
                 use_mlp_dagger: bool = False,
                 dagger_history_length: int = 8,
                 critic_history_length: int = 8,
                 critic_hidden_dims: Optional[List[int]] = None,
                 activation: str = "elu",
                 use_layer_norm: bool = False,
                 fusion_mode: str = "gated",
                 critic_fusion_mode: Optional[str] = None,
                 chunk_q_enabled: bool = False,
                 chunk_q_hidden_dims: Optional[List[int]] = None,
                 actor_hidden_dims: Optional[List[int]] = None,
                 joint_decoder_hidden_dims: Optional[List[int]] = None,
                 **kwargs
                 ):
        nn.Module.__init__(self)
        del actor_hidden_dims, joint_decoder_hidden_dims
        if ref_term_steps is None:
            ref_term_steps = {"policy": None, "critic": None}
        assert not load_dagger or load_dagger_path, "load_dagger and load_dagger_path must be provided if load_dagger is True"
        self.actor = MMGPT(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, num_heads, num_layers, dropout=dropout, name="actor", num_steps_per_env=num_steps_per_env, max_seq_len=max_seq_len, mlp_hidden_dims=mlp_hidden_dims, apply_rope=apply_rope, term_dict=term_dict["policy"], ref_term_dict=ref_term_dict["policy"], concatenate_term_names=concatenate_term_names["policy"], concatenate_ref_term_names=concatenate_ref_term_names["policy"], ref_term_steps=ref_term_steps["policy"], pred_obs_term_names=pred_obs_term_names, pred_obs_term_weights=pred_obs_term_weights, default_joint_pos=default_joint_pos, **kwargs)
        
        policy_term_dict = {"policy": term_dict.get("policy", {})}
        policy_ref_term_dict = {"policy": ref_term_dict.get("policy", {})} if ref_term_dict else None
        self.use_mlp_dagger = use_mlp_dagger
        if not use_mlp_dagger:
            self.actor_dagger = MMGPT(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, num_heads, num_layers, dropout=dropout, name="actor_dagger", num_steps_per_env=num_steps_per_env, max_seq_len=max_seq_len, mlp_hidden_dims=mlp_hidden_dims, apply_rope=apply_rope, term_dict=term_dict["policy"], ref_term_dict=ref_term_dict["policy"], concatenate_term_names=concatenate_term_names["policy"], concatenate_ref_term_names=concatenate_ref_term_names["policy"], ref_term_steps=ref_term_steps["policy"], pred_obs_term_names=pred_obs_term_names, pred_obs_term_weights=pred_obs_term_weights, default_joint_pos=default_joint_pos, **kwargs) if load_dagger else None
            self.dagger_obs_lambda = lambda obs: obs  # use full sequence
            self.dagger_ref_obs_lambda = (lambda ref_obs: ref_obs) if num_actor_ref_obs > 0 else None
        else:
            self.actor_dagger = FusedMultiModalMLP(
                term_dict=policy_term_dict,
                ref_term_dict=policy_ref_term_dict,
                output_size=num_actions,
                hidden_dims=[512, 256, 128],
                activation="elu",
                history_length=dagger_history_length,
                encoder_latent_dim=128,
                encoder_compress_threshold=32,
                use_layer_norm=False,
                fusion_mode='gated',
                name="actor_dagger",
                reorgnize_obs=True,
            ) if load_dagger else None
            self.dagger_obs_lambda = lambda obs: obs[-dagger_history_length:, :, :]  # take last num_dagger_history_length steps
            self.dagger_ref_obs_lambda = (lambda ref_obs: (ref_obs[0][-dagger_history_length:], ref_obs[1][-dagger_history_length:])) if num_actor_ref_obs > 0 else None
        
        # Action noise
        if min_noise_std is not None and min_noise_std <= 0.0:
            raise ValueError("min_noise_std must be positive when specified")
        if max_noise_std is not None and max_noise_std <= 0.0:
            raise ValueError("max_noise_std must be positive when specified")
        if (
            min_noise_std is not None
            and max_noise_std is not None
            and min_noise_std > max_noise_std
        ):
            raise ValueError("min_noise_std cannot exceed max_noise_std")
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            if load_dagger:
                self.std_dagger = nn.Parameter(
                    init_noise_std * torch.ones(num_actions),
                    requires_grad=False,
                )
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            if load_dagger:
                self.log_std_dagger = nn.Parameter(
                    torch.log(init_noise_std * torch.ones(num_actions)),
                    requires_grad=False,
                )
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        if load_dagger:
            self.load_dagger_weights(load_dagger_path)
            lora_r = kwargs.get('lora_r', 8)
            lora_alpha = kwargs.get('lora_alpha', 16)
            lora_dropout = kwargs.get('lora_dropout', 0.0)
            if enable_lora:
                self.apply_dagger_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                
                


        # self.critic = MMGPT(num_critic_obs, num_critic_ref_obs, 1, dim_model, num_heads, num_layers, dropout=dropout, name="critic", num_steps_per_env=num_steps_per_env, max_seq_len=max_seq_len, mlp_hidden_dims=mlp_hidden_dims, apply_rope=apply_rope, term_dict=term_dict["critic"], ref_term_dict=ref_term_dict["critic"], concatenate_term_names=concatenate_term_names["critic"], concatenate_ref_term_names=concatenate_ref_term_names["critic"], ref_term_steps=ref_term_steps["critic"], **kwargs) # 1 for value function
        
        critic_term_dict = {"critic": term_dict.get("critic", term_dict.get("policy", {}))}
        critic_ref_term_dict = {"critic": ref_term_dict.get("critic", ref_term_dict.get("policy", {}))} if ref_term_dict else None
        
        self.critic_history_length = critic_history_length
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
        resolved_critic_fusion_mode = critic_fusion_mode or fusion_mode
        self._critic_uses_direct_concat = resolved_critic_fusion_mode in {
            "direct_concat",
            "direct_concat_ref_first",
        }
        if self._critic_uses_direct_concat:
            if self.critic_history_length != 1:
                raise ValueError(
                    "DirectConcatMLP consumes one already-composed critic "
                    "sample; set critic_history_length=1."
                )
            self.critic = DirectConcatMLP(
                term_dict=critic_term_dict,
                ref_term_dict=critic_ref_term_dict,
                output_size=1,
                hidden_dims=critic_hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
                reference_first=(
                    resolved_critic_fusion_mode
                    == "direct_concat_ref_first"
                ),
            )
            # Recurrent PPO batches are [T, B, D]. The SONIC privileged
            # history is already flattened inside D, so select one value
            # sample per padded trajectory instead of introducing a second
            # temporal axis in the critic.
            self.critic_obs_lambda = lambda obs: obs[-1, :, :]
            self.critic_ref_obs_lambda = (
                (lambda ref_obs: (ref_obs[0][-1, :, :], ref_obs[1][-1, :]))
                if num_critic_ref_obs > 0
                else None
            )
        else:
            self.critic = FusedMultiModalMLP(
                term_dict=critic_term_dict,
                ref_term_dict=critic_ref_term_dict,
                output_size=1,
                hidden_dims=critic_hidden_dims,
                activation=activation,
                history_length=self.critic_history_length,
                encoder_latent_dim=dim_model,
                encoder_compress_threshold=32,
                use_layer_norm=use_layer_norm,
                fusion_mode=resolved_critic_fusion_mode,
                name="critic",
                reorgnize_obs=True,
            )
            self.critic_obs_lambda = (
                lambda obs: obs[-self.critic_history_length:, :, :]
            )
            self.critic_ref_obs_lambda = (
                (
                    lambda ref_obs: (
                        ref_obs[0][-self.critic_history_length:],
                        ref_obs[1][-self.critic_history_length:],
                    )
                )
                if num_critic_ref_obs > 0
                else None
            )

        self.chunk_q_enabled = bool(chunk_q_enabled)
        if self.chunk_q_enabled:
            chunk_size = int(kwargs.get("temporal_chunk_size", 1))
            if chunk_size < 2:
                raise ValueError(
                    "chunk_q_enabled requires temporal_chunk_size > 1"
                )
            self.chunk_q = PrivilegedChunkQ(
                critic_obs_dim=num_critic_obs,
                critic_ref_obs_dim=num_critic_ref_obs,
                action_dim=num_actions,
                chunk_size=chunk_size,
                hidden_dims=chunk_q_hidden_dims,
                reference_first=(
                    resolved_critic_fusion_mode
                    == "direct_concat_ref_first"
                ),
            )
        else:
            self.chunk_q = None
        
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        if load_critic_path:
            self.load_critic_weights(load_critic_path)
            
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Model: {self.critic}")
        print(f"Chunk-Q Model: {self.chunk_q}")
        print(f"Dagger Model: {self.actor_dagger}")
        
        # disable args validation for speedup
        if load_std_path:
            self.load_std_weights(load_std_path)
        
        
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        
    def load_std_weights(self, path):
        state_dict = torch.load(path, map_location='cpu')
        model_state_dict = state_dict['model_state_dict']
        if self.noise_std_type == "scalar":
            self.std.data = model_state_dict['std']
        elif self.noise_std_type == "log":
            self.log_std.data = model_state_dict['log_std']
        print(f"Loaded std weights from {path}")
        
        
    @property
    def actor_aux_loss(self):
        return self.actor.aux_loss
    

    @property
    def actor_dreamer_loss(self):
        return self.actor.dream_loss
    
    @property
    def actor_dreamer_reward(self):
        return self.actor.dreamer_reward

    def _clamp_action_std(self, std: torch.Tensor) -> torch.Tensor:
        if self.min_noise_std is None and self.max_noise_std is None:
            return std
        return torch.clamp(
            std,
            min=self.min_noise_std,
            max=self.max_noise_std,
        )

    def update_distribution(self, observations, ref_observations=None, masks=None, memory=None, **kwargs):
        mean = self.actor(observations, ref_observations, masks, memory, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, self._clamp_action_std(std))

    def act(self, observations, ref_observations=None, masks=None, memory=None, **kwargs):
        self.update_distribution(observations, ref_observations, masks, memory, **kwargs)
        sample = self.distribution.sample()
        return sample

    def update_distribution_dagger(self, observations, ref_observations=None, masks=None, memory=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        if masks is not None:
            observations = self.dagger_obs_lambda(observations)
            if self.dagger_ref_obs_lambda is not None and ref_observations is not None:
                ref_observations = self.dagger_ref_obs_lambda(ref_observations)
        mean = self.actor_dagger(observations, ref_observations, masks=masks, memory=memory, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std_dagger.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std_dagger).expand_as(mean)
        self.distribution_dagger = Normal(mean, self._clamp_action_std(std))

    def act_dagger(self, observations, ref_observations=None, masks=None, memory=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        self.update_distribution_dagger(observations, ref_observations, masks, memory, **kwargs)
        sample = self.distribution_dagger.sample()
        return sample

    def act_dagger_inference(self, observations, ref_observations=None, masks=None, memory=None, **kwargs):
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        if masks is not None:
            observations = self.dagger_obs_lambda(observations)
            if self.dagger_ref_obs_lambda is not None and ref_observations is not None:
                ref_observations = self.dagger_ref_obs_lambda(ref_observations)
        actions_mean = self.actor_dagger(observations, ref_observations, masks=masks, memory=memory, **kwargs)
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

    def act_inference(self, observations, ref_observations=None, masks=None, memory=None, **kwargs):
        actions_mean = self.actor(observations, ref_observations, masks, memory, **kwargs)
        return actions_mean

    def evaluate(self, critic_observations, ref_critic_observations=None, masks=None, memory=None, **kwargs):
        if masks is not None:
            critic_observations = self.critic_obs_lambda(critic_observations)
            if self.critic_ref_obs_lambda is not None and ref_critic_observations is not None:
                ref_critic_observations = self.critic_ref_obs_lambda(ref_critic_observations)
        value = self.critic(critic_observations, ref_critic_observations, masks=masks, memory=memory, **kwargs)
        return value

    def evaluate_chunk_q(
        self,
        critic_observations: torch.Tensor,
        action_chunk: torch.Tensor,
        ref_critic_observations: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> torch.Tensor:
        if self.chunk_q is None:
            raise RuntimeError("chunk-Q calibration head is disabled")
        return self.chunk_q(
            critic_observations,
            action_chunk,
            ref_critic_observations,
        )
        
    def reset(self, dones=None):
        self.actor.reset(dones)
        self.critic.reset(dones)
        if self.actor_dagger is not None:
            self.actor_dagger.reset(dones)

    def get_actor_recurrent_state(
        self,
        observations: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Capture actor-only state before the first action of a rollout."""

        return self.actor.get_inference_ttt_rollout_state(observations)

    def get_actor_recurrent_reset_event(self) -> Optional[torch.Tensor]:
        """Expose the finite-horizon reset applied to the latest action."""

        return self.actor.get_inference_ttt_reset_event()

    def get_hidden_states(self):
        # TTT uses the optimized rollout-start-only storage path above.  The
        # generic RNN API would replicate this large state at every physical
        # transition and is intentionally left inactive.
        return None, None
