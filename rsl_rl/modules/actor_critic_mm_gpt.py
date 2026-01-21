# Copyright (c) 2021-2025, SJTU MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import warnings

from torch.distributed.checkpoint import state_dict
from rsl_rl.modules import ActorCritic, ActorCriticMMTransformer, SwiGLUEmbedding, group_by_concat_list, HistoryEmbedding
from rsl_rl.modules.actor_critic_mlp_v2 import FusedMultiModalMLP
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
    

class MMGPT(nn.Module):
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
        # HINT: The following parameters matters **SIGNIFICANTLY** to the performance for allowing temporal prediction features!
        pred_obs_term_names: Optional[List[str]] = None,
        pred_obs_term_weights: Optional[List[float]] = None,
        apply_mlp_residual = False,     
        # some not pretty fix
        default_joint_pos: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if dim_model > 128:
            ffn_ratio = 2
        else:
            ffn_ratio = 4
        super().__init__()
        self.name = name
        self.obs_dim = obs_size
        self.ref_obs_dim = ref_obs_size
        self.num_steps_per_env = num_steps_per_env
        self.apply_rope = apply_rope
        self.apply_res = apply_res
        self.ref_first = ref_first
        self.default_joint_pos = default_joint_pos
        if kwargs:
            print(
                f"MMGPT {self.name}.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        self.stride = math.ceil(num_steps_per_env / max_seq_len) if num_steps_per_env > max_seq_len else 1
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
            
            for term_name, term_dim in term_dict.items():
                if term_name in self.pred_obs_term_names:
                    self.obs_indices.append((start, start + term_dim))
                    self.obs_dims.append(term_dim)
                    self.obs_term_names.append(term_name)
                start += term_dim
            if len(self.obs_indices) != len(self.pred_obs_term_names):
                raise ValueError("Some pred_obs_term_names are not found in term_dict. Expected names: {}, but got: {}.".format(list(term_dict.keys()), self.pred_obs_term_names))
            
            self.aux_heads = nn.ModuleList()
            for dim in self.obs_dims:
                self.aux_heads.append(
                    nn.Linear(dim_model, dim)
                )
                
            if ref_term_dict is not None: # extract target_actions
                if 'target_actions' not in ref_term_dict:
                    raise ValueError("If you want to predict observation terms, please provide 'target_actions' term in ref observation terms for MMGPT.")
                start = 0
                for i, (term_name, term_dim) in enumerate(ref_term_dict.items()):
                    if term_name == 'target_actions':
                        self.target_action_slice = (start, start + term_dim)
                        break
                    start += term_dim
            
            else:
                self.target_action_slice = None
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
        
        if self.stride > 1:
            self.compressed_len = math.floor((L_in_padded + 2 * self.padding - self.kernel_size) / self.stride) + 1
        else:
            self.compressed_len = self.num_steps_per_env
        self.compressor = Conv1dCompressor(dim_model, self.stride, self.kernel_size) if self.stride > 1 else None
        self.ref_compressor = Conv1dCompressor(dim_model, self.stride, self.kernel_size) if self.stride > 1 and ref_obs_size > 0 else None
        self.upsampler = ConvTranspose1dCompressor(dim_model, self.stride, self.kernel_size) if self.stride > 1 else None
        
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
            nn.Linear(dim_model * 2, dim_model * 2),
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
            self.inference_buffer_obs[:, mask, :] = 0.0
            # Reset timesteps to 0 for completed environments: timesteps[dones] = 0
            if self.current_timesteps is not None:
                self.current_timesteps[mask] = 0
            if self.ref_obs_dim > 0:
                if self.inference_buffer_ref_obs is not None:
                    self.inference_buffer_ref_obs[:, mask, :] = 0.0
                if self.inference_buffer_ref_obs_mask is not None:
                    self.inference_buffer_ref_obs_mask[:, mask] = False

    def _forward_batch(self, 
                obs: torch.Tensor,
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                masks: Optional[torch.Tensor] = None, # fake optional. Must be provided in batch mode
                memory: Optional[Tuple[torch.Tensor, torch.Tensor | None]] = None, # memory token for both modalities
                unpad_output: bool = True,
                upsample: bool = True,
                compute_aux_loss: bool = False,
                compute_dreamer_loss: bool = False,
                compute_dreamer_reward: bool = False,
                ):
        """
        Forward pass for the multi-modality transformer in batch mode.
        """
        
        ##########################################################################################
        # T_x meanings:
        # T_in: original input sequence length
        # T_c: compressed sequence length (after Conv1dCompressor)
        # T_combined: combined sequence length (after merging two modalities, if available)
        # if not has_ref: T_combined == T_c, else T_combined == 2 * T_c
        ##########################################################################################
        
        
        T_in, B, _ = obs.shape
        device = obs.device   
        if T_in > self.num_steps_per_env:
            obs=obs[-self.num_steps_per_env:, :, :]
            if masks is not None:
                masks = masks[-self.num_steps_per_env:, :]
            if ref_obs is not None:
                ref_obs = (ref_obs[0][-self.num_steps_per_env:, :, :], ref_obs[1][-self.num_steps_per_env:, :])   
            T_in = self.num_steps_per_env
        
        if ref_obs is not None:
            ref_mlp_obs = (ref_obs[0][-1], ref_obs[1][-1])
        else:
            ref_mlp_obs = None
        
        if self.apply_mlp_residual:
            mlp_output = self.mlp_bypass(obs[-1], ref_mlp_obs) # [B, dim_out]    
        else:
            mlp_output = None 
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
            

        # if self.apply_res:
        #     # Add residual connection    
        #     upsampled_out = upsampled_out + res_obs_emb * masks.unsqueeze(-1).float()
        #     if has_ref:
        #         upsampled_out = upsampled_out + res_ref_obs_emb * ref_obs_mask.unsqueeze(-1).float()
        
        # Cache hidden state before projection (for DAgger training)
        self.last_hidden_state = upsampled_out[-1, :, :] if not unpad_output else upsampled_out  # [B, D] or [T_in, B, D]
        
        if compute_aux_loss and self.aux_heads is not None:
            self.aux_loss = 0.0
            obs_actions = obs[:, :, self.action_slice[0]:self.action_slice[1]]  # [T_in, B, action_dim]
            obs_actions = obs_actions[1:] # shift 1 step to ensure matching with input
            gamma, beta = self.action_film(obs_actions).chunk(2, dim=-1)  # [T_in-1, B, D] each
            for obs_term_slice, aux_head, weight in zip(self.obs_indices, self.aux_heads, self.pred_obs_term_weights):
                target = obs[:, :, obs_term_slice[0]:obs_term_slice[1]] # [T_in, B, dim]
                pred_in = upsampled_out[:-1]  # [T_in-1, B, D]
                # FiLM modulation
                pred_in = gamma * pred_in + beta  # [T_in-1, B, D]
                pred = aux_head(pred_in) # [T_in - 1, B, dim]
                # temporal shifting: note that target (obs) is the last timestep, and pred is current timestep. Thus, the computation should be between pred[:-1] and target[1:]
                target_shifted = target[1:, :, :]
                pred_shifted = pred
                # compute mask: only consider valid positions in both target and pred, the mask is aligned with target_shifted
                valid_mask = masks[1:] # shape: [T_in-1, B]
                target_valid = target_shifted[valid_mask]
                pred_valid = pred_shifted[valid_mask]
                aux_loss_i = self.aux_loss_fn(pred_valid, target_valid)
                self.aux_loss += weight * aux_loss_i            
                
        else:
            self.aux_loss = 0.0
        
        pred_actions = self.fc(upsampled_out.detach()) # [T_in, B, dim_out]
        
        if compute_dreamer_reward:
            # dreamer reward computation
            cur_obs_latent = res_obs_emb[:-1, :, :]  # [T_in-1, B, dim_model]
            next_obs_latent = res_obs_emb[1:, :, :]  # [T_in-1, B, dim_model]
            cur_actions = pred_actions[:-1, :, :]  # [T_in-1, B, dim_out]
            dreamer_input = torch.cat([cur_obs_latent, cur_actions, next_obs_latent], dim=-1)  # [T_in-1, B, dim_model*2 + dim_out]
            dreamer_rewards = self.dreamer_reward_head(dreamer_input).squeeze(-1)  # [T_in-1, B]
            # mask
            valid_mask = masks[1:]  # [T_in-1, B]
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
                
                
        
        if compute_dreamer_loss and self.target_action_slice is not None:
            self.dreamer_loss = 0.0
            # extract target actions from ref_obs
            cur_latent = upsampled_out[-1, :, :]  # [B, dim_out]
            cur_action = pred_actions[-1, :, :]
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
            target_joint_pos = ref_obs[0][-1, :, self.target_action_slice[0]:self.target_action_slice[1]]  # [B, action_dim]
            target_mask = ref_obs[1][-1, :]  # [B,]
            pred_joint_pos_valid = pred_joint_pos * target_mask.unsqueeze(-1) # [B, joint_dim]
            target_joint_pos_valid = target_joint_pos * target_mask.unsqueeze(-1) # [B, joint_dim]

            self.dreamer_loss = self.aux_loss_fn(pred_joint_pos_valid, target_joint_pos_valid)
                  
            
        else:
            self.dreamer_loss = 0.0
        

        if unpad_output:
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
            return self._forward_batch(obs, ref_obs, masks, memory=memory, unpad_output=unpad_output, compute_aux_loss=True, compute_dreamer_loss=True, compute_dreamer_reward=True) # compute aux loss only in batch mode
        else:
            return self._forward_inference(obs, ref_obs)
        
        
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
                 noise_std_type: str = "scalar",
                 load_dagger=False,
                 load_dagger_path=None,
                 load_actor_path=None,
                 load_critic_path=None,
                 load_std_path=None,
                 enable_lora=False,
                 dropout=0.1,
                 apply_rope=False,
                 default_joint_pos: Optional[torch.Tensor] = None,
                 use_mlp_dagger: bool = False,
                 dagger_history_length: int = 8,
                 critic_history_length: int = 8,
                 **kwargs
                 ):
        nn.Module.__init__(self)
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
                
                


        # self.critic = MMGPT(num_critic_obs, num_critic_ref_obs, 1, dim_model, num_heads, num_layers, dropout=dropout, name="critic", num_steps_per_env=num_steps_per_env, max_seq_len=max_seq_len, mlp_hidden_dims=mlp_hidden_dims, apply_rope=apply_rope, term_dict=term_dict["critic"], ref_term_dict=ref_term_dict["critic"], concatenate_term_names=concatenate_term_names["critic"], concatenate_ref_term_names=concatenate_ref_term_names["critic"], ref_term_steps=ref_term_steps["critic"], **kwargs) # 1 for value function
        
        critic_term_dict = {"critic": term_dict.get("critic", term_dict.get("policy", {}))}
        critic_ref_term_dict = {"critic": ref_term_dict.get("critic", ref_term_dict.get("policy", {}))} if ref_term_dict else None
        
        self.critic_history_length = critic_history_length
        
        self.critic = FusedMultiModalMLP(
            term_dict=critic_term_dict,
            ref_term_dict=critic_ref_term_dict,
            output_size=1,  # Value function outputs single value
            hidden_dims=[512, 256, 128],
            activation="elu",
            history_length=self.critic_history_length,
            encoder_latent_dim=dim_model,
            encoder_compress_threshold=32,
            use_layer_norm=False,
            fusion_mode='gated',
            name="critic",
            reorgnize_obs=True,
        )
        
        self.critic_obs_lambda = lambda obs: obs[-self.critic_history_length:, :, :]  # take last num_critic_history_length steps
        self.critic_ref_obs_lambda = (lambda ref_obs: (ref_obs[0][-self.critic_history_length:], ref_obs[1][-self.critic_history_length:])) if num_critic_ref_obs > 0 else None
        
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        if load_critic_path:
            self.load_critic_weights(load_critic_path)
            
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Model: {self.critic}")
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

    def update_distribution(self, observations, ref_observations=None, masks=None, memory=None, **kwargs):
        mean = self.actor(observations, ref_observations, masks, memory, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

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
        self.distribution_dagger = Normal(mean, std)

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
        
    def reset(self, dones=None):
        self.actor.reset(dones)
        self.critic.reset(dones)
        if self.actor_dagger is not None:
            self.actor_dagger.reset(dones)
            
    def get_hidden_states(self):
        return None, None