# Copyright (c) 2021-2025, SJTU MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from rsl_rl.modules import ActorCritic, ActorCriticMMTransformer
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from torch.distributions import Normal
import math
from typing import Optional, Tuple, List, Union


# Sequence Compressor
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
    

class ObservationSeqEmbedding(nn.Module):
    """
    Embed a sequence of observations into a sequence of embeddings using an MLP.
    """
    def __init__(self, obs_dim: int, d_model: int, mlp_hidden_dims=[256, 128], activation="elu"):
        super().__init__()
        layers = []
        current_dim = obs_dim
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(resolve_nn_activation(activation))  # Create a new instance each time
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, d_model))
        self.projector = nn.Sequential(*layers)

    def forward(self, obs_seq: torch.Tensor):
        # obs_seq shape: [T, B, obs_dim]
        T, B, _ = obs_seq.shape
        obs_flat = obs_seq.view(T * B, -1)
        embedding_flat = self.projector(obs_flat)
        return embedding_flat.view(T, B, -1)
    

class MMGPT(nn.Module):
    def __init__(
        self,
        obs_size, # actually, it should be named by obs_dim, but I maintained the original name for consistency
        ref_obs_size,
        dim_out,
        dim_model = 256,
        num_heads = 8,
        num_layers = 3,
        ffn_ratio = 4,
        dropout = 0.0,
        name = "",
        num_steps_per_env = 24, # default, remember to parse this!
        max_seq_len = 16, 
        mlp_hidden_dims = [256, 128],
        **kwargs
    ):
        
        super().__init__()
        self.name = name
        self.obs_dim = obs_size
        self.ref_obs_dim = ref_obs_size
        if kwargs:
            print(
                f"MMGPT {self.name}.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        self.stride = math.ceil(num_steps_per_env / max_seq_len) if num_steps_per_env > max_seq_len else 1
        self.compressed_len = (num_steps_per_env + self.stride - 1) // self.stride
        # example calculation: num_steps_per_env=24, max_seq_len=16 -> stride=2, compressed_len=12
        self.obs_embedding = ObservationSeqEmbedding(obs_size, dim_model, mlp_hidden_dims)
        self.ref_obs_embedding = ObservationSeqEmbedding(ref_obs_size, dim_model, mlp_hidden_dims) if ref_obs_size > 0 else None

        self.compressor = SequenceCompressor(dim_model, self.stride) if self.stride > 1 else None
        
        self.pos_emb = nn.Embedding(self.compressed_len * 2, dim_model) # *2 for obs and ref_obs
        self.modality_emb = nn.Embedding(2, dim_model) # 0 for obs, 1 for ref_obs
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=ffn_ratio*dim_model, dropout=dropout, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_model, dim_out)
        
        self.inference_buffer_obs = None # for storing inference-time past key values
        self.inference_buffer_ref_obs = None
        self.inference_buffer_ref_obs_mask = None
        self.num_envs_cache = 0
        
    @staticmethod
    def unpad(padded_sequence, masks):
        return padded_sequence.permute(1, 0, 2)[masks.permute(1, 0)] # shape [valid_length, dim]

    def _init_inference_buffer(self, num_envs, device):
        self.num_envs_cache = num_envs
        obs_buffer_shape = (self.compressed_len, num_envs, self.obs_dim)
        self.inference_buffer_obs = torch.zeros(obs_buffer_shape, device=device)
        if self.ref_obs_dim > 0:
            ref_obs_buffer_shape = (self.compressed_len, num_envs, self.ref_obs_dim)
            self.inference_buffer_ref_obs = torch.zeros(ref_obs_buffer_shape, device=device)
            self.inference_buffer_ref_obs_mask = torch.zeros((self.compressed_len, num_envs), dtype=torch.bool, device=device)
            
    def reset(self, dones=None):
        if self.inference_buffer_obs is None:
            return
        if dones is None: # reset all
            self.inference_buffer_obs.fill_(0.0)
            if self.ref_obs_dim > 0:
                if self.inference_buffer_ref_obs is not None:
                    self.inference_buffer_ref_obs.fill_(0.0)
                if self.inference_buffer_ref_obs_mask is not None:
                    self.inference_buffer_ref_obs_mask.fill_(False)
        else:
            # dones are at the environment level
            mask = (dones == 1)
            self.inference_buffer_obs[:, mask, :] = 0.0
            if self.ref_obs_dim > 0:
                if self.inference_buffer_ref_obs is not None:
                    self.inference_buffer_ref_obs[:, mask, :] = 0.0
                if self.inference_buffer_ref_obs_mask is not None:
                    self.inference_buffer_ref_obs_mask[:, mask] = False

    def _forward_batch(self, 
                obs: torch.Tensor,
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                masks: Optional[torch.Tensor] = None,
                unpad_output: bool = True,
                upsample: bool = True,
                ):
        """
        Forward pass for the multi-modality transformer in batch mode.
        """
        T_in, B, _ = obs.shape
        device = obs.device
        
        obs_emb = self.obs_embedding(obs) # [T_in, B, D]
        # Modality embedding
        obs_emb = obs_emb + self.modality_emb(torch.zeros(T_in, B, dtype=torch.long, device=device))
        
        # Process second modality if available
        has_ref = ref_obs is not None and self.ref_obs_dim > 0
        if has_ref:
            ref_obs_tensor, ref_obs_mask = ref_obs
            # ref_obs_mask shape: [T, B]
            ref_T_in, ref_B, _ = ref_obs_tensor.shape
            assert ref_B == B, "Batch size of obs and ref_obs must match"
            assert ref_T_in == T_in, "Sequence length of obs and ref_obs must match"
            ref_obs_emb = self.ref_obs_embedding(ref_obs_tensor) # [ref_T_in, B, D]
            ref_obs_emb = ref_obs_emb + self.modality_emb(torch.ones(ref_T_in, B, dtype=torch.long, device=device))
            
            # Sequence merging. [obs0, ref_obs0, obs1, ref_obs1, ...]
            combined_seq = torch.stack([obs_emb, ref_obs_emb], dim=1)
            combined_seq = combined_seq.reshape(2 * T_in, B, -1) # [2*T_in, B, D]
            T_combined = combined_seq.shape[0]
            
            obs_attn_mask = masks # [T_in, B]
            ref_attn_mask = masks & ref_obs_mask # [T_in, B]
            stacked_masks = torch.stack([obs_attn_mask, ref_attn_mask], dim=1) # [T_in, 2, B]
            combined_masks = stacked_masks.reshape(T_combined, B) # [2*T_in, B]
        else:
            combined_seq = obs_emb # [T_in, B, D]
            T_combined = combined_seq.shape[0]
            combined_masks = masks # [T_in, B]
            
        # Compression
        if self.compressor is not None and upsample:
            continued_seq, new_masks = self.compressor(combined_seq, combined_masks) 
        else:
            continued_seq, new_masks = combined_seq, combined_masks
        T = continued_seq.shape[0] # compressed length
        pos = torch.arange(T, device=device).unsqueeze(1).expand(T, B)
        src = continued_seq + self.pos_emb(pos) # [T, B, D]
        # For compatibility, use simpler attention without explicit causal mask
        src_key_padding_mask = ~new_masks.T # [B, T], True for padding positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        gpt_out = self.decoder(
            src.permute(1, 0, 2), # [B, T, D]
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        if self.stride > 1 and upsample:
            upsampled_out = gpt_out.permute(1, 0, 2).repeat_interleave(self.stride, dim=0)[:T_combined, :, :] # [T_combined, B, D]
        else:
            upsampled_out = gpt_out.permute(1, 0, 2) # [T_combined, B, D]

        obs_indices = torch.arange(0, T_combined, 2, device=device) if has_ref else torch.arange(0, T_combined, 1, device=device) # obs are at even positions or all positions
        obs_feature_seq_padded = upsampled_out[obs_indices, :, :] # [T_in, B, D] or [T_in, B, D]. We drop ref_obs features since attention has been applied.
        out_seq = self.fc(obs_feature_seq_padded) # [T_in, B, dim_out]
        if unpad_output:
            return self.unpad(out_seq, masks) # [valid_length, dim_out]
        else: # return output last timestep
            return out_seq[-1, :, :] # [B, dim_out]
    
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
        masks = torch.ones_like(obs_seq[..., 0], dtype=torch.bool) # [T_in, B]. During inference, the stacks are controlled to be always valid
        if self.ref_obs_embedding is not None:
            ref_obs_seq = self.inference_buffer_ref_obs
            ref_obs_mask = self.inference_buffer_ref_obs_mask
            ref_obs = (ref_obs_seq, ref_obs_mask)
        else:
            ref_obs = None
            
        # call _forward_batch
        return self._forward_batch(obs_seq, ref_obs, masks, unpad_output=False, upsample=False) # [B, dim_out]
        

    def forward(self, 
                obs: torch.Tensor, 
                ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                masks: Optional[torch.Tensor] = None,
                **kwargs):
        """
        Forward pass for the multi-modality transformer.

        Args:
            obs (torch.Tensor): Observation tensor of shape (T, B, dim_in).
            ref_obs (Optional[Tuple[torch.Tensor, torch.Tensor]]): 
                A tuple containing:
                - ref_obs_tensor (torch.Tensor): Reference observations of shape (T,B, dim_in).
                - ref_obs_mask (torch.Tensor): Mask tensor of shape (T, B) indicating the presence of ref_obs.
            masks (torch.Tensor): Mask tensor of shape (T, B) indicating valid observations among the trajectory.
            **kwargs: Only added to avoid errors when unexpected arguments are passed.
            
        ** Important **: Unlike mm transformer, here, padded environments, B, tends to be larger than num_envs.

        Returns:
            torch.Tensor: Output tensor after transformer and fully connected layer.
        """
        is_batch_mode = masks is not None
        if is_batch_mode:
            return self._forward_batch(obs, ref_obs, masks)
        else:
            return self._forward_inference(obs, ref_obs)


# Actor-Critic with MM-GPT backbone
class ActorCriticMMGPT(ActorCriticMMTransformer):
    is_recurrent = True
    def __init__(self,
                 num_actor_obs,
                 num_actor_ref_obs,
                 num_critic_obs,
                 num_critic_ref_obs,
                 num_actions,
                 dim_model=256,
                 num_heads=8,
                 num_layers=3,
                 num_steps_per_env=24,
                 max_seq_len=16,
                 mlp_hidden_dims = [256, 128],
                 init_noise_std=1.0,
                 noise_std_type: str = "scalar",
                 load_dagger=False,
                 load_dagger_path=None,
                 load_actor_path=None,
                 enable_lora=False,
                 dropout=0.1,
                 **kwargs
                 ):
        nn.Module.__init__(self)
        assert not load_dagger or load_dagger_path, "load_dagger and load_dagger_path must be provided if load_dagger is True"
        self.actor = MMGPT(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, num_heads, num_layers, dropout=dropout, name="actor", num_steps_per_env=num_steps_per_env, max_seq_len=max_seq_len, mlp_hidden_dims=mlp_hidden_dims, **kwargs)
        self.actor_dagger = MMGPT(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, num_heads, num_layers, dropout=dropout, name="actor_dagger", num_steps_per_env=num_steps_per_env, max_seq_len=max_seq_len, mlp_hidden_dims=mlp_hidden_dims, **kwargs) if load_dagger else None
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

        self.critic = MMGPT(num_critic_obs, num_critic_ref_obs, 1, dim_model, num_heads, num_layers, dropout=dropout, name="critic", num_steps_per_env=num_steps_per_env, max_seq_len=max_seq_len, mlp_hidden_dims=mlp_hidden_dims, **kwargs) # 1 for value function
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")
        print(f"Dagger Model: {self.actor_dagger}")
        
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        
    def reset(self, dones=None):
        self.actor.reset(dones)
        self.critic.reset(dones)
        if self.actor_dagger is not None:
            self.actor_dagger.reset(dones)
    