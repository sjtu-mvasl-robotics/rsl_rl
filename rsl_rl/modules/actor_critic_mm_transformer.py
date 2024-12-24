# Copyright 2024 SJTU MVASL Lab
# Written by Yifei Yao, 2024-12-21
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
from collections import deque


class ObsDeque(nn.Module):
    def __init__(self, max_len, obs_size):
        super().__init__()
        self.max_len = max_len
        self.obs_size = obs_size
        # Initialize buffer with zeros
        self.register_buffer('buffer', torch.zeros(1, max_len, obs_size))
        # Initialize current position
        self.register_buffer('current_pos', torch.tensor(0, dtype=torch.long))
        self.padded_len = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Insert new observations into the buffer and return the updated buffer.

        Args:
            x (torch.Tensor): New observations, shape (batch_size, obs_size)

        Returns:
            buffer (torch.Tensor): Updated buffer, shape (batch_size, max_len, obs_size)
            seq_mask (torch.Tensor, dtype=torch.bool): Mask tensor indicating which positions are padded, shape (max_len,)
        """
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch_size, 1, obs_size)

        # Expand buffer to match batch size if necessary
        if self.buffer.size(0) != batch_size:
            # Reinitialize buffer for new batch size
            self.buffer = torch.zeros(batch_size, self.max_len, self.obs_size, device=x.device, dtype=x.dtype)
            self.current_pos = torch.zeros(1, dtype=torch.long, device=x.device)
            self.padded_len = self.max_len

        if self.padded_len and self.padded_len > 0:
            self.padded_len -= 1

        pos = self.current_pos.item()
        # Insert new observation at the current position
        self.buffer[:, pos, :] = x.squeeze(1)
        # Update position
        self.current_pos = (self.current_pos + 1) % self.max_len
        # create mask, 0 to padded_len - 1 is False, others are True. If padded_len is None or 0, all are True
        if not self.padded_len:
            seq_mask = torch.ones(self.max_len, dtype=torch.bool, device=x.device)
        else:
            seq_mask = torch.arange(self.max_len, device=x.device) >= self.padded_len

        return self.buffer, seq_mask

    


class ObservationEmbedding(nn.Module):
    def __init__(self, num_obs, d_model, max_len=16):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(num_obs, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
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
            name = "",
            **kwargs
    ):
        super().__init__()
        self.name = name
        if kwargs:
            print(f"Transformer.__init__ got unexpected arguments, which will be ignored: {kwargs.keys()}")

        self.obs_deque = ObsDeque(max_len, obs_size)
        self.obs_embedding = ObservationEmbedding(obs_size, dim_model, max_len)
        self.ref_obs_size = ref_obs_size
        if self.ref_obs_size == 0:
            self.ref_obs_deque = None
            self.ref_obs_embedding = None
        else:
            self.ref_obs_deque = ObsDeque(max_len, ref_obs_size)
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Like BERT, we use the last token to represent the whole sequence
        self.fc = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_out)
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
        padding_masks = []

        assert (self.ref_obs_deque is not None) or (ref_obs is None), "Cannot run multi-modality mode with ref_obs_size=0"

        # Preprocess observations
        obs, obs_seq_pad_mask = self.obs_deque(obs)

        # -------------------
        # Process obs embeddings
        # -------------------
        obs_emb = self.obs_embedding(obs)  # Shape: (B, seq_len_obs, dim_model)
        cls_emb = self.cls_token.expand(obs.size(0), -1, -1)  # Shape: (B, 1, dim_model)
        sep_emb = self.sep_token.expand(obs.size(0), -1, -1)  # Shape: (B, 1, dim_model)
        obs_emb = torch.cat([cls_emb, obs_emb, sep_emb], dim=1)  # Shape: (B, seq_len_obs + 2, dim_model)
        embeddings.append(obs_emb)

        # Create padding mask for obs: 1 for non-padding, 0 for padding
        obs_padding_mask = torch.ones(obs.size(0), obs.size(1), dtype=torch.bool, device=obs.device) # Shape: (B, seq_len_obs)
        # obs_seq_pad_mask shape: (seq_len_obs,) -> (B, seq_len_obs)
        # obs_seq_pad_mask is False for positions which should not be considered for calculating attention
        obs_padding_mask = obs_padding_mask & obs_seq_pad_mask.unsqueeze(0)
        obs_padding_mask = F.pad(obs_padding_mask, (1, 1), value=1)  # Account for CLS and SEP tokens
        padding_masks.append(obs_padding_mask)  # Shape: (B, seq_len_obs + 2)

        # -------------------
        # Process ref_obs embeddings (if provided)
        # -------------------
        if ref_obs is not None and self.ref_obs_deque is not None:
            ref_obs_tensor, ref_obs_mask = ref_obs  # Unpack the tuple

            # Preprocess reference observations
            ref_obs_tensor, ref_obs_seq_pad_mask = self.ref_obs_deque(ref_obs_tensor)
            ref_obs_emb = self.ref_obs_embedding(ref_obs_tensor)  # Shape: (B, seq_len_ref_obs, dim_model)
            cls_ref_emb = self.cls_token.expand(ref_obs_emb.size(0), -1, -1)  # Shape: (B, 1, dim_model)
            sep_ref_emb = self.sep_token.expand(ref_obs_emb.size(0), -1, -1)  # Shape: (B, 1, dim_model)
            ref_obs_emb = torch.cat([cls_ref_emb, ref_obs_emb, sep_ref_emb], dim=1)  # Shape: (B, seq_len_ref_obs + 2, dim_model)

            # Apply the ref_obs_mask to zero out embeddings where ref_obs is not present
            # ref_obs_mask shape: (B,) -> (B, 1, 1) for broadcasting
            ref_obs_mask = ref_obs_mask.view(-1, 1, 1)
            ref_obs_emb = ref_obs_emb * ref_obs_mask  # Zero out embeddings where ref_obs_mask is False

            embeddings.append(ref_obs_emb)  # Append to embeddings list

            # Create padding mask for ref_obs: 1 for non-padding, 0 for padding
            ref_obs_padding_mask = torch.ones(ref_obs_tensor.size(0), ref_obs_tensor.size(1), dtype=torch.bool, device=ref_obs_tensor.device) # Shape: (B, seq_len_ref_obs)
            # ref_obs_seq_pad_mask shape: (seq_len_ref_obs,) -> (B, seq_len_ref_obs)
            # ref_obs_seq_pad_mask is False for positions which should not be considered for calculating attention
            ref_obs_padding_mask = ref_obs_padding_mask & ref_obs_seq_pad_mask.unsqueeze(0)
            ref_obs_padding_mask = F.pad(ref_obs_padding_mask, (1, 1), value=1)  # Account for CLS and SEP tokens

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
        # x = x.transpose(0, 1)  # Shape: (total_seq_len, B, dim_model)
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # Transformer expects shape (S, B, E)
        x = x[:,-1, :]  # Take the last token's output: Shape (B, dim_model)

        # -------------------
        # Final fully connected layer
        # -------------------
        x = self.fc(x)  # Shape: (B, output_dim)
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
        self.actor = Transformer(obs_size=num_actor_obs, ref_obs_size=num_actor_ref_obs, dim_out=num_actions, dim_model=dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="actor", **kwargs)
        self.critic = Transformer(obs_size=num_critic_obs, ref_obs_size=num_critic_ref_obs, dim_out=1, dim_model=dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="critic", **kwargs) # 1 for value function
        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, ref_observations=None, **kwargs):
        self.update_distribution(observations, ref_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, ref_observations=None):
        actions_mean = self.actor(observations, ref_observations)
        return actions_mean

    def evaluate(self, critic_observations, ref_critic_observations =None, **kwargs):
        value = self.critic(critic_observations, ref_critic_observations)
        return value
    

if __name__ == "__main__":
    # Test ActorCriticMMTransformer
    num_actor_obs = 64
    num_actor_ref_obs = 40
    num_critic_obs = 64
    num_critic_ref_obs = 40
    num_actions = 29
    max_len = 4
    dim_model = 128
    num_layers = 4
    num_heads = 4
    init_noise_std = 1.0
    batch_size = 4096
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    actor_critic = ActorCriticMMTransformer(
        num_actor_obs=num_actor_obs,
        num_actor_ref_obs=num_actor_ref_obs,
        num_critic_obs=num_critic_obs,
        num_critic_ref_obs=num_critic_ref_obs,
        num_actions=num_actions,
        max_len=max_len,
        dim_model=dim_model,
        num_layers=num_layers,
        num_heads=num_heads,
        init_noise_std=init_noise_std
    ).to(device)

    # Test forward pass
    actor_obs = torch.randn(batch_size, num_actor_obs, device=device)
    actor_ref_obs = torch.randn(batch_size, num_actor_ref_obs, device=device)
    actor_ref_obs_mask = torch.randint(0, 2, (batch_size,), device=device).bool()
    critic_obs = torch.randn(batch_size, num_critic_obs, device=device)
    critic_ref_obs = torch.randn(batch_size,num_critic_ref_obs, device=device)
    critic_ref_obs_mask = torch.randint(0, 2, (batch_size,), device=device).bool()

    import time
    start = time.time()
    for _ in range(100):
        actions = actor_critic.act(actor_obs, (actor_ref_obs, actor_ref_obs_mask))
        values = actor_critic.evaluate(critic_obs, (critic_ref_obs, critic_ref_obs_mask))
    total_time = time.time() - start
    print(f"Single inference time: {total_time / 100:.6f} s")
    print(f"Actions: {actions}")
    print(f"Values: {values}")
    print(f"Action mean: {actor_critic.action_mean}")
    print(f"Action std: {actor_critic.action_std}")
    print(f"Entropy: {actor_critic.entropy}")
    print(f"Actions log prob: {actor_critic.get_actions_log_prob(actions)}")
    print(f"Actions inference: {actor_critic.act_inference(actor_obs, actor_ref_obs)}")
