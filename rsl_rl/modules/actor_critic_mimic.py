# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Actor-Critic architecture for PPOMimic.

This module implements a teacher-student architecture with:
- History encoder: Learns from proprioceptive history (student, deployable)
- Privilege encoder: Uses privileged information (teacher, training only)
- Motion encoder: Encodes future motion targets
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Optional

from .encoder_modules import ConvEncoder, MLPEncoder


class ActorMimic(nn.Module):
    """
    Actor network for PPOMimic.
    
    Architecture:
    - Motion encoder: Encodes future motion targets
    - History encoder: Encodes proprioceptive history (student, deployable)
    - Privilege encoder: Encodes privileged observations (teacher, training only)
    - Actor backbone: MLP that takes [obs, motion_encoding, latent] -> actions
    """
    
    def __init__(
        self,
        num_actor_obs: int,
        num_privileged_obs: int,
        num_prop_history: int,
        num_motion_targets: int,
        num_actions: int,
        history_length: int,
        future_num_steps: int,
        # Motion encoder config
        motion_hidden_dim: int = 60,
        motion_output_dim: int = 128,
        # History encoder config
        history_hidden_dim: int = 30,
        history_output_dim: int = 64,
        # Privilege encoder config
        priv_hidden_dims: List[int] = [64],
        # Actor backbone config
        actor_hidden_dims: List[int] = [768, 512, 256],
        activation: str = "SiLU",
        use_layernorm: bool = True,
        # Action distribution config
        init_noise_std: float = 1.0,
        fix_sigma: bool = False,
        min_sigma: float = 0.2,
        max_sigma: float = 1.2,
    ):
        """
        Args:
            num_actor_obs: Dimension of actor observations
            num_privileged_obs: Dimension of privileged observations
            num_prop_history: Dimension of proprioceptive history (per timestep)
            num_motion_targets: Dimension of motion targets (per timestep)
            num_actions: Number of actions
            history_length: Number of timesteps in history
            future_num_steps: Number of future timesteps for motion targets
            motion_hidden_dim: Hidden dimension for motion encoder
            motion_output_dim: Output dimension for motion encoder
            history_hidden_dim: Hidden dimension for history encoder
            history_output_dim: Output dimension for history encoder
            priv_hidden_dims: Hidden dimensions for privilege encoder
            actor_hidden_dims: Hidden dimensions for actor backbone
            activation: Activation function name
            use_layernorm: Whether to use layer normalization in actor backbone
            init_noise_std: Initial standard deviation for action noise
            fix_sigma: Whether to fix action std
            min_sigma: Minimum action std
            max_sigma: Maximum action std
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.history_output_dim = history_output_dim
        
        # Motion encoder: Encodes future motion targets
        self.motion_encoder = ConvEncoder(
            input_dim=num_motion_targets,
            hidden_dim=motion_hidden_dim,
            output_dim=motion_output_dim,
            num_timesteps=future_num_steps,
            activation=activation,
        )
        
        # History encoder: Encodes proprioceptive history (student, deployable)
        self.history_encoder = ConvEncoder(
            input_dim=num_prop_history,
            hidden_dim=history_hidden_dim,
            output_dim=history_output_dim,
            num_timesteps=history_length,
            activation=activation,
        )
        
        # Privilege encoder: Encodes privileged observations (teacher, training only)
        # Output dimension must match history_output_dim for alignment
        self.priv_encoder = MLPEncoder(
            input_dim=num_privileged_obs,
            output_dim=history_output_dim,
            hidden_dims=priv_hidden_dims,
            activation=activation,
            use_layernorm=False,
        )
        
        # Actor backbone: [obs, motion_encoding, latent] -> actions
        actor_input_dim = num_actor_obs + motion_output_dim + history_output_dim
        
        # Build actor MLP
        layers = []
        prev_dim = actor_input_dim
        
        # Get activation function
        activation_upper = activation.upper()
        if activation_upper == "SILU":
            act_fn = nn.SiLU
        elif activation_upper == "RELU":
            act_fn = nn.ReLU
        elif activation_upper == "ELU":
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        for hidden_dim in actor_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.actor_backbone = nn.Sequential(*layers)
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.fix_sigma = fix_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        if self.fix_sigma:
            self.std.requires_grad = False
    
    def forward(
        self,
        obs: torch.Tensor,
        prop_history: torch.Tensor,
        motion_targets: torch.Tensor,
        priv_obs: Optional[torch.Tensor] = None,
        use_privilege: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through actor.
        
        Args:
            obs: Actor observations [batch, num_actor_obs]
            prop_history: Proprioceptive history [batch, history_length * num_prop_history]
            motion_targets: Future motion targets [batch, future_num_steps * num_motion_targets]
            priv_obs: Privileged observations [batch, num_privileged_obs]
            use_privilege: Whether to use privilege encoder (True) or history encoder (False)
            
        Returns:
            Action mean [batch, num_actions]
        """
        # Encode motion targets
        motion_encoding = self.motion_encoder(motion_targets)
        
        # Encode latent: use privilege or history encoder
        if use_privilege:
            if priv_obs is None:
                raise ValueError("priv_obs must be provided when use_privilege=True")
            latent = self.priv_encoder(priv_obs)
        else:
            latent = self.history_encoder(prop_history)
        
        # Concatenate: [obs, motion_encoding, latent]
        actor_input = torch.cat([obs, motion_encoding, latent], dim=-1)
        
        # Forward through actor backbone
        action_mean = self.actor_backbone(actor_input)
        
        return action_mean


class CriticMimic(nn.Module):
    """
    Critic network for PPOMimic.
    
    Architecture:
    Input: [obs, priv_obs, motion_encoding] -> MLP -> value
    """
    
    def __init__(
        self,
        num_critic_obs: int,
        num_privileged_obs: int,
        num_motion_targets: int,
        future_num_steps: int,
        # Motion encoder config (shared with actor)
        motion_hidden_dim: int = 60,
        motion_output_dim: int = 128,
        # Critic backbone config
        critic_hidden_dims: List[int] = [768, 512, 256],
        activation: str = "SiLU",
        use_layernorm: bool = True,
    ):
        """
        Args:
            num_critic_obs: Dimension of critic observations
            num_privileged_obs: Dimension of privileged observations
            num_motion_targets: Dimension of motion targets (per timestep)
            future_num_steps: Number of future timesteps
            motion_hidden_dim: Hidden dimension for motion encoder
            motion_output_dim: Output dimension for motion encoder
            critic_hidden_dims: Hidden dimensions for critic backbone
            activation: Activation function name
            use_layernorm: Whether to use layer normalization
        """
        super().__init__()
        
        # Motion encoder (shared architecture with actor)
        self.motion_encoder = ConvEncoder(
            input_dim=num_motion_targets,
            hidden_dim=motion_hidden_dim,
            output_dim=motion_output_dim,
            num_timesteps=future_num_steps,
            activation=activation,
        )
        
        # Critic backbone: [obs, priv_obs, motion_encoding] -> value
        critic_input_dim = num_critic_obs + num_privileged_obs + motion_output_dim
        
        # Build critic MLP
        layers = []
        prev_dim = critic_input_dim
        
        # Get activation function
        activation_upper = activation.upper()
        if activation_upper == "SILU":
            act_fn = nn.SiLU
        elif activation_upper == "RELU":
            act_fn = nn.ReLU
        elif activation_upper == "ELU":
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        for hidden_dim in critic_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.critic_backbone = nn.Sequential(*layers)
    
    def forward(
        self,
        obs: torch.Tensor,
        priv_obs: torch.Tensor,
        motion_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through critic.
        
        Args:
            obs: Critic observations [batch, num_critic_obs]
            priv_obs: Privileged observations [batch, num_privileged_obs]
            motion_targets: Future motion targets [batch, future_num_steps * num_motion_targets]
            
        Returns:
            Value estimate [batch, 1]
        """
        # Encode motion targets
        motion_encoding = self.motion_encoder(motion_targets)
        
        # Concatenate: [obs, priv_obs, motion_encoding]
        critic_input = torch.cat([obs, priv_obs, motion_encoding], dim=-1)
        
        # Forward through critic backbone
        value = self.critic_backbone(critic_input)
        
        return value


class ActorCriticMimic(nn.Module):
    """
    Actor-Critic network for PPOMimic.
    
    Combines ActorMimic and CriticMimic with action distribution.
    """
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_privileged_obs: int,
        num_prop_history: int,
        num_motion_targets: int,
        num_actions: int,
        history_length: int,
        future_num_steps: int,
        # Actor config
        actor_hidden_dims: List[int] = [768, 512, 256],
        motion_hidden_dim: int = 60,
        motion_output_dim: int = 128,
        history_hidden_dim: int = 30,
        history_output_dim: int = 64,
        priv_hidden_dims: List[int] = [64],
        # Critic config
        critic_hidden_dims: List[int] = [768, 512, 256],
        # Common config
        activation: str = "SiLU",
        use_layernorm: bool = True,
        # Action distribution config
        init_noise_std: float = 1.0,
        fix_sigma: bool = False,
        min_sigma: float = 0.2,
        max_sigma: float = 1.2,
        **kwargs
    ):
        """Initialize Actor-Critic network."""
        super().__init__()
        
        if kwargs:
            print(f"ActorCriticMimic.__init__ got unexpected arguments: {kwargs.keys()}")
        
        # Actor network
        self.actor = ActorMimic(
            num_actor_obs=num_actor_obs,
            num_privileged_obs=num_privileged_obs,
            num_prop_history=num_prop_history,
            num_motion_targets=num_motion_targets,
            num_actions=num_actions,
            history_length=history_length,
            future_num_steps=future_num_steps,
            motion_hidden_dim=motion_hidden_dim,
            motion_output_dim=motion_output_dim,
            history_hidden_dim=history_hidden_dim,
            history_output_dim=history_output_dim,
            priv_hidden_dims=priv_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
            init_noise_std=init_noise_std,
            fix_sigma=fix_sigma,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
        )
        
        # Critic network
        self.critic = CriticMimic(
            num_critic_obs=num_critic_obs,
            num_privileged_obs=num_privileged_obs,
            num_motion_targets=num_motion_targets,
            future_num_steps=future_num_steps,
            motion_hidden_dim=motion_hidden_dim,
            motion_output_dim=motion_output_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
        )
        
        # Action distribution
        self.distribution = None
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
    
    def reset(self, dones=None):
        """Reset hidden states (not used in this architecture)."""
        pass
    
    def forward(self):
        raise NotImplementedError
    
    def update_distribution(self, obs_dict, use_privilege=True):
        """
        Update action distribution.
        
        Args:
            obs_dict: Dictionary containing:
                - 'actor_obs': [batch, num_actor_obs]
                - 'prop_history': [batch, history_length * num_prop_history]
                - 'motion_targets': [batch, future_num_steps * num_motion_targets]
                - 'priv_obs': [batch, num_privileged_obs] (if use_privilege=True)
            use_privilege: Whether to use privilege encoder
        """
        mean = self.actor(
            obs=obs_dict['actor_obs'],
            prop_history=obs_dict['prop_history'],
            motion_targets=obs_dict['motion_targets'],
            priv_obs=obs_dict.get('priv_obs'),
            use_privilege=use_privilege,
        )
        
        # Clamp std
        std = (mean * 0.0 + self.actor.std).clamp(
            min=self.actor.min_sigma,
            max=self.actor.max_sigma
        )
        
        self.distribution = Normal(mean, std)
    
    def act(self, obs_dict, use_privilege=True):
        """
        Sample actions from policy.
        
        Args:
            obs_dict: Dictionary of observations
            use_privilege: Whether to use privilege encoder
            
        Returns:
            Sampled actions
        """
        self.update_distribution(obs_dict, use_privilege)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, obs_dict, use_privilege=False):
        """
        Deterministic action for inference (uses history encoder by default).
        
        Args:
            obs_dict: Dictionary of observations
            use_privilege: Whether to use privilege encoder (default: False for deployment)
            
        Returns:
            Action mean (deterministic)
        """
        return self.actor(
            obs=obs_dict['actor_obs'],
            prop_history=obs_dict['prop_history'],
            motion_targets=obs_dict['motion_targets'],
            priv_obs=obs_dict.get('priv_obs'),
            use_privilege=use_privilege,
        )
    
    def evaluate(self, obs_dict):
        """
        Evaluate state value.
        
        Args:
            obs_dict: Dictionary containing:
                - 'critic_obs': [batch, num_critic_obs]
                - 'priv_obs': [batch, num_privileged_obs]
                - 'motion_targets': [batch, future_num_steps * num_motion_targets]
                
        Returns:
            Value estimate [batch, 1]
        """
        return self.critic(
            obs=obs_dict['critic_obs'],
            priv_obs=obs_dict['priv_obs'],
            motion_targets=obs_dict['motion_targets'],
        )
