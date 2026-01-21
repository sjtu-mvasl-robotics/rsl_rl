# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticMLPV3: Enhanced MLP with ASAP-inspired temporal encoding.

Key features:
1. ConvEncoder for history (similar to ASAP's history_encoder)
2. PrivilegeEncoder as teacher signal (similar to ASAP's priv_encoder)
3. Optional MotionEncoder for future targets
4. Supports privileged training with latent alignment
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional, Tuple

from .conv_encoder import ConvEncoder, PrivilegeEncoder, MotionEncoder


class ActorMLPV3(nn.Module):
    """
    Actor network with ASAP-inspired temporal encoding.
    
    Components:
    - HistoryEncoder: Conv-based encoding of observation history
    - PrivilegeEncoder: MLP encoding of privilege info (training only)
    - MotionEncoder: Optional encoding of future motion targets
    - MainMLP: Final decision network
    """
    
    def __init__(
        self,
        obs_dict: dict[str, int],
        num_actions: int,
        history_length: int = 1,
        # History encoder config
        history_obs_keys: list[str] = None,  # Which obs to use for history
        history_hidden_dim: int = 64,
        history_output_dim: int = 64,
        # Privilege encoder config (optional)
        privilege_obs_keys: list[str] = None,
        privilege_hidden_dims: list[int] = [64],
        # Motion encoder config (optional)
        motion_obs_key: str = None,
        motion_hidden_dim: int = 64,
        motion_output_dim: int = 128,
        motion_time_steps: int = 5,
        # Main MLP config
        hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        use_layer_norm: bool = False,
        # Noise config
        init_noise_std: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.history_length = history_length
        self.obs_dict = obs_dict
        
        # Calculate base observation dimension (non-history terms)
        self.base_obs_dim = 0
        self.history_obs_dim = 0
        
        if history_obs_keys is None:
            # Use all observations for history
            history_obs_keys = list(obs_dict.keys())
        
        self.history_obs_keys = history_obs_keys
        
        for key, dim in obs_dict.items():
            if key in history_obs_keys and history_length > 1 and dim % history_length == 0:
                # This is a history observation
                self.history_obs_dim += dim // history_length
            else:
                # This is a base observation (no history)
                self.base_obs_dim += dim
        
        # Build encoders
        total_encoded_dim = self.base_obs_dim
        
        # History encoder (if history_length > 1)
        if history_length > 1 and self.history_obs_dim > 0:
            self.history_encoder = ConvEncoder(
                input_dim=self.history_obs_dim,
                output_dim=history_output_dim,
                hidden_dim=history_hidden_dim,
                time_steps=history_length,
                activation=activation,
                use_adaptive_config=True,
            )
            total_encoded_dim += history_output_dim
            self.use_history_encoder = True
        else:
            self.history_encoder = None
            self.use_history_encoder = False
        
        # Privilege encoder (optional, for training)
        if privilege_obs_keys is not None:
            privilege_dim = sum(obs_dict.get(key, 0) for key in privilege_obs_keys)
            self.privilege_encoder = PrivilegeEncoder(
                input_dim=privilege_dim,
                output_dim=history_output_dim,  # Same dim as history encoder
                hidden_dims=privilege_hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
            )
            self.privilege_obs_keys = privilege_obs_keys
            self.use_privilege_encoder = True
        else:
            self.privilege_encoder = None
            self.privilege_obs_keys = []
            self.use_privilege_encoder = False
        
        # Motion encoder (optional, for future targets)
        if motion_obs_key is not None and motion_obs_key in obs_dict:
            motion_dim = obs_dict[motion_obs_key] // motion_time_steps
            self.motion_encoder = MotionEncoder(
                input_dim=motion_dim,
                output_dim=motion_output_dim,
                hidden_dim=motion_hidden_dim,
                time_steps=motion_time_steps,
                activation=activation,
            )
            total_encoded_dim += motion_output_dim
            self.motion_obs_key = motion_obs_key
            self.use_motion_encoder = True
        else:
            self.motion_encoder = None
            self.motion_obs_key = None
            self.use_motion_encoder = False
        
        # Main MLP
        self.main_mlp = self._build_mlp(
            total_encoded_dim,
            hidden_dims,
            activation,
            use_layer_norm
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_actions)
        
        # Cache for encodings (used in latent alignment)
        self.last_hist_encoding = None
        self.last_priv_encoding = None
    
    def _build_mlp(self, input_dim, hidden_dims, activation, use_layer_norm):
        """Build MLP layers"""
        layers = []
        prev_dim = input_dim
        
        act_fn = self._get_activation(activation)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        """Get activation function by name"""
        activations = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation.lower(), nn.ELU())
    
    def extract_history_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract history observations from full observation tensor"""
        if not self.use_history_encoder:
            return None
        
        history_features = []
        start = 0
        
        for key in self.obs_dict.keys():
            dim = self.obs_dict[key]
            if key in self.history_obs_keys and self.history_length > 1 and dim % self.history_length == 0:
                history_features.append(obs[:, start:start+dim])
            start += dim
        
        if not history_features:
            return None
        
        return torch.cat(history_features, dim=-1)
    
    def extract_privilege_obs(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract privilege observations"""
        if not self.use_privilege_encoder:
            return None
        
        privilege_features = []
        start = 0
        
        for key in self.obs_dict.keys():
            dim = self.obs_dict[key]
            if key in self.privilege_obs_keys:
                privilege_features.append(obs[:, start:start+dim])
            start += dim
        
        if not privilege_features:
            return None
        
        return torch.cat(privilege_features, dim=-1)
    
    def extract_base_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract non-history observations"""
        base_features = []
        start = 0
        
        for key in self.obs_dict.keys():
            dim = self.obs_dict[key]
            # Include if not in history OR if it's not divisible by history_length
            if key not in self.history_obs_keys or self.history_length <= 1 or dim % self.history_length != 0:
                base_features.append(obs[:, start:start+dim])
            start += dim
        
        if not base_features:
            return torch.zeros(obs.shape[0], 0, device=obs.device)
        
        return torch.cat(base_features, dim=-1)
    
    def forward(self, obs: torch.Tensor, use_privilege: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observations (B, obs_dim)
            use_privilege: If True, use privilege encoder instead of history encoder
        
        Returns:
            action_mean: Mean actions (B, num_actions)
        """
        features = []
        
        # Base observations
        base_obs = self.extract_base_obs(obs)
        if base_obs.shape[-1] > 0:
            features.append(base_obs)
        
        # History or Privilege encoding
        if use_privilege and self.use_privilege_encoder:
            # Use privilege encoder (training mode)
            privilege_obs = self.extract_privilege_obs(obs)
            if privilege_obs is not None:
                priv_encoding = self.privilege_encoder(privilege_obs)
                features.append(priv_encoding)
                self.last_priv_encoding = priv_encoding  # Keep gradient for latent alignment
        elif self.use_history_encoder:
            # Use history encoder (inference mode)
            history_obs = self.extract_history_obs(obs)
            if history_obs is not None:
                hist_encoding = self.history_encoder(history_obs)
                features.append(hist_encoding)
                self.last_hist_encoding = hist_encoding  # Keep gradient for latent alignment
        
        # Motion encoding (if available)
        if self.use_motion_encoder and self.motion_obs_key in self.obs_dict:
            # Motion observations should be provided separately or extracted
            # For now, assume they're part of obs
            # TODO: Extract motion obs properly
            pass
        
        # Concatenate all features
        if not features:
            raise ValueError("No features to process!")
        
        x = torch.cat(features, dim=-1)
        
        # Main MLP
        x = self.main_mlp(x)
        action_mean = self.output_layer(x)
        
        return action_mean
    
    def act(self, obs: torch.Tensor, use_privilege: bool = False) -> torch.Tensor:
        """
        Forward pass and return action mean (no sampling here).
        Distribution creation is handled by ActorCriticMLPV3.
        """
        return self.forward(obs, use_privilege=use_privilege)
    
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference"""
        return self.forward(obs, use_privilege=False)
    
    def get_latent_alignment_loss(self) -> torch.Tensor:
        """
        Compute loss for aligning history encoder with privilege encoder.
        This is the key training signal from ASAP.
        
        The privilege encoder acts as a fixed teacher (detached), while the
        history encoder is the student that learns to mimic its representations.
        """
        if self.last_hist_encoding is None or self.last_priv_encoding is None:
            # Return a tensor with requires_grad for compatibility
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # L2 distance between history (student) and privilege (teacher) encodings
        # Detach privilege encoding to prevent backprop through teacher
        loss = (self.last_hist_encoding - self.last_priv_encoding.detach()).norm(p=2, dim=1).mean()
        
        return loss


class CriticMLPV3(nn.Module):
    """
    Critic network, potentially with privilege observations.
    Simpler than Actor since it doesn't need history encoding during deployment.
    """
    
    def __init__(
        self,
        obs_dict: dict[str, int],
        hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.obs_dict = obs_dict
        input_dim = sum(obs_dict.values())
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # Output layer (value function)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        """Get activation function by name"""
        activations = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(activation.lower(), nn.ELU())
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observations (B, obs_dim), can include privilege info
        
        Returns:
            values: State values (B, 1)
        """
        return self.network(obs)


class ActorCriticMLPV3(nn.Module):
    """
    Complete Actor-Critic with ASAP-inspired design.
    """
    
    is_recurrent = False
    supports_latent_alignment = True  # Enable ASAP-style privilege-history alignment in MMPPO
    
    def __init__(
        self,
        actor_obs_dict: dict[str, int],
        critic_obs_dict: dict[str, int],
        num_actions: int,
        # Actor config
        history_length: int = 1,
        history_obs_keys: list[str] = None,
        privilege_obs_keys: list[str] = None,
        actor_hidden_dims: list[int] = [256, 128],
        # Critic config  
        critic_hidden_dims: list[int] = [256, 128],
        # Common config
        activation: str = "elu",
        use_layer_norm: bool = False,
        init_noise_std: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        self.actor = ActorMLPV3(
            obs_dict=actor_obs_dict,
            num_actions=num_actions,
            history_length=history_length,
            history_obs_keys=history_obs_keys,
            privilege_obs_keys=privilege_obs_keys,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            init_noise_std=init_noise_std,
            **kwargs
        )
        
        self.critic = CriticMLPV3(
            obs_dict=critic_obs_dict,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        
        # Action noise (registered at ActorCritic level, not in Actor)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.noise_std_type = "scalar"  # MLPv3 uses scalar std (compatibility with MMPPO)
        
        # Distribution state
        self.distribution = None
        self.action_mean = None
        self.action_std = None
    
    def reset(self, dones=None):
        """Reset for compatibility"""
        pass
    
    def forward(self):
        raise NotImplementedError("Use act() or evaluate() instead")
    
    def act(self, observations: torch.Tensor, use_privilege: bool = False, **kwargs) -> torch.Tensor:
        """Sample actions from the policy"""
        # Get action mean from actor
        action_mean = self.actor.act(observations, use_privilege=use_privilege)
        
        # Create distribution with std from parent class
        self.action_mean = action_mean
        self.action_std = action_mean * 0.0 + self.std
        self.distribution = Normal(action_mean, self.action_std)
        
        # Sample actions
        return self.distribution.sample()
    
    def act_inference(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Deterministic actions (mean of the policy)"""
        return self.actor.act_inference(observations)
    
    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate state values"""
        return self.critic(critic_observations)
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of current distribution"""
        return self.distribution.entropy().sum(dim=-1)
    
    def get_latent_alignment_loss(self) -> torch.Tensor:
        """Get the privilege-history alignment loss (for ASAP-style training)"""
        return self.actor.get_latent_alignment_loss()
