# Copyright 2024 SJTU MVASL Lab
# Written by Yifei Yao, 2024-12-21
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional
from collections import deque

class ObsDeque(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        self.deque = deque(maxlen=max_len)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, obs_size)
        x = x.unsqueeze(1) # (batch_size, 1, obs_size)
        if len(self.deque) == 0:
            self.deque.extend([torch.zeros_like(x, dtype=x.dtype, device=x.device) for _ in range(self.max_len)])
        self.deque.append(x)
        return torch.cat(list(self.deque), dim=1)
    


class ObservationEmbedding(nn.Module):
    def __init__(self, num_obs, d_model, max_len=16):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(num_obs, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        return self.embedding(x) + self.pos_embedding(pos)
        # x_emb = self.embedding(x)
        # pe = self.pos_embedding(pos)
        # return x_emb + pe
    

    
class Transformer(nn.Module):
    def __init__(
            self,
            obs_size,
            ref_obs_size,
            dim_out,
            dim_model,
            max_len = 16,
            num_heads = 8,
            num_layers = 4,
            ffn_ratio = 4,
            dropout = 0.1,
            **kwargs
    ):
        super().__init__()
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")

        self.obs_deque = ObsDeque(max_len)
        self.obs_embedding = ObservationEmbedding(obs_size, dim_model, max_len)
        self.ref_obs_deque = ObsDeque(max_len)
        self.ref_obs_embedding = ObservationEmbedding(ref_obs_size, dim_model, max_len)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_model * ffn_ratio,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Like BERT, we use the last token to represent the whole sequence
        self.fc = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_out)
        )

    def forward(self, obs: torch.Tensor, ref_obs: Optional[torch.Tensor] = None):
        embeddings = []
        padding_mask = []
        obs = self.obs_deque(obs)

        # obs embedding
        obs_emb = self.obs_embedding(obs)
        cls = self.cls_token.expand(obs.size(0), -1, -1)
        sep = self.sep_token.expand(obs.size(0), -1, -1)
        obs_emb = torch.cat([cls, obs_emb, sep], dim=1)
        embeddings.append(obs_emb)
        # obs padding mask: 0 for padding, 1 for non-padding
        obs_padding_mask = torch.ones(obs.size(0), obs.size(1), dtype=torch.bool, device=obs.device)
        obs_padding_mask = F.pad(obs_padding_mask, (1, 1), value=1)
        padding_mask.append(obs_padding_mask)

        # ref obs embedding
        if ref_obs is not None:
            ref_obs = self.ref_obs_deque(ref_obs)
            ref_obs_emb = self.ref_obs_embedding(ref_obs)
            cls = self.cls_token.expand(ref_obs.size(0), -1, -1)
            sep = self.sep_token.expand(ref_obs.size(0), -1, -1)
            ref_obs_emb = torch.cat([cls, ref_obs_emb, sep], dim=1)
            embeddings.append(ref_obs_emb)
            # ref obs padding mask: 0 for padding, 1 for non-padding
            ref_obs_padding_mask = torch.ones(ref_obs.size(0), ref_obs.size(1), dtype=torch.bool, device=ref_obs.device)
            ref_obs_padding_mask = F.pad(ref_obs_padding_mask, (1, 1), value=1)
            padding_mask.append(ref_obs_padding_mask)
        
        x = torch.cat(embeddings, dim=1)
        if len(padding_mask) > 1:
            padding_mask = torch.cat(padding_mask, dim=1)
        else:
            padding_mask = padding_mask[0]
        padding_mask = ~padding_mask

        # x: (batch_size, seq_len, dim_model)
        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=padding_mask) # (seq_len, batch_size, dim_model)
        x = x[-1] # take the last token
        x = self.fc(x)
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
        self.actor_transformer = Transformer(num_actor_obs, num_actor_ref_obs, num_actions, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, **kwargs)
        self.critic_transformer = Transformer(num_critic_obs, num_critic_ref_obs, 1, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, **kwargs) # 1 for value function
        print(f"Actor Transformer: {self.actor_transformer}")
        print(f"Critic Transformer: {self.critic_transformer}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, ref_observations=None, **kwargs):
        self.update_distribution(observations, ref_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, ref_observations=None):
        actions_mean = self.actor(observations, ref_observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    

