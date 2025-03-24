# Copyright 2024 SJTU MVASL Lab
# Written by Yifei Yao, 2024-12-21
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
# from timm.models.vision_transformer import LayerScale
import time

class ObservationEmbedding(nn.Module):
    def __init__(self, num_obs, d_model, max_len=16, apply_norm = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        if apply_norm:
            self.embedding = nn.Sequential(
                nn.Linear(num_obs, d_model * max_len),
                nn.LayerNorm(d_model * max_len),
            )
        else:
            self.embedding = nn.Linear(num_obs, d_model * max_len)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.size(0), -1, self.d_model)  # (B, seq_len, dim_model)        
        pos = torch.arange(self.max_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return x + self.pos_embedding(pos)
    

    
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
            **kwargs
    ):
        super().__init__()
        self.name = name
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")
        self.obs_embedding = ObservationEmbedding(obs_size, dim_model, max_len)
        if ref_obs_size == 0:
            self.ref_obs_embedding = None
        else:
            self.ref_obs_embedding = ObservationEmbedding(ref_obs_size, dim_model, max_len)
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
            # nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_out),
        )
        # self.out_ls = LayerScale(dim_out, init_values=ls_init_values)

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
        x = x[:, 0, :]  # Take the first token's output: Shape (B, dim_model)

        # -------------------
        # Final fully connected layer
        # -------------------
        x = self.fc(x)  # Shape: (B, output_dim)
        # x = self.out_ls(x)
        return x
    

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

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
            dim_model,
            max_len = 128,
            num_heads = 8,
            num_layers = 4,
            ffn_ratio = 4,
            dropout = 0.0,
            name = "",
            ls_init_values = 1e-3,
            **kwargs
    ):
        super().__init__()
        self.name = name
        self.obs_size = obs_size
        self.ref_obs_size = ref_obs_size
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")
        self.layers = nn.Sequential(
            nn.Linear(obs_size + ref_obs_size, 768),
            nn.ReLU(),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, dim_out),
        )

    def forward(
        self, 
        obs: torch.Tensor, 
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
            **kwargs
    ):
        super().__init__()
        self.actor = MMTransformer(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="actor", **kwargs)
        self.critic = MMTransformer(num_critic_obs, num_critic_ref_obs, 1, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="critic", **kwargs) # 1 for value function
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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

    def update_distribution(self, observations, ref_observations=None):
        mean = self.actor(observations, ref_observations)
        self.distribution = Normal(mean, 0.0 * mean + self.std)

    def act(self, observations, ref_observations=None, **kwargs):
        self.update_distribution(observations, ref_observations)
        sample = self.distribution.sample()
        return sample

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, ref_observations=None):
        actions_mean = self.actor(observations, ref_observations)
        return actions_mean

    def evaluate(self, critic_observations, ref_critic_observations =None, **kwargs):
        value = self.critic(critic_observations, ref_critic_observations)
        return value
    