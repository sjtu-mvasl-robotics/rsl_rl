# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Multi-Modal MLP V2 with term_dict support and HistoryEncoder
# Created by Yifei Yao (with AI assistance), 2025

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple


class TemporalFeatureEncoder(nn.Module):
    """
    Temporal feature encoder for MLP with history observations.
    
    Process flow:
    1. Split obs by terms: [a1,a2,a3,a4, b1,b2,b3,b4, ...] -> separate terms
    2. Reshape each term: term.view(B, T, per_step_dim)
    3. Compress if per_step_dim > threshold: Linear projection to latent_dim
    4. Concatenate all terms: (B, T, total_per_step_dim)
    5. Conv1D to compress temporal dimension: (B, total_per_step_dim, T) -> (B, total_per_step_dim, 1) -> (B, total_per_step_dim)
    
    Note: Conv1D only compresses the temporal dimension (T), not the feature dimension.
          Output dimension = total_per_step_dim (sum of per-step dims after compression)
    """
    def __init__(
        self,
        term_dims: list[int],
        term_names: list[str],
        history_length: int,
        latent_dim: int = 32,
        compress_threshold: int = 32,
        activation: str = "elu"
    ):
        """
        Args:
            term_dims: List of dimensions for each term (e.g., [12, 12, 4])
            term_names: List of term names (for debugging)
            history_length: Length of history sequence
            latent_dim: Latent dimension for large per-step features
            compress_threshold: If per_step_dim > this, project to latent_dim
            activation: Activation function
        """
        super().__init__()
        
        self.term_dims = term_dims
        self.term_names = term_names
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.compress_threshold = compress_threshold
        
        # Build projection list (only register nn.Linear when needed)
        self.projection_indices = []  # Which terms need projection
        projection_layers = []
        self.has_history = []
        self.per_step_dims = []
        
        total_per_step_dim = 0
        
        for i, term_dim in enumerate(term_dims):
            if history_length > 1 and term_dim % history_length == 0:
                # This term has history
                per_step_dim = term_dim // history_length
                self.has_history.append(True)
                
                # Only create projection if needed
                if per_step_dim > compress_threshold:
                    projection_layers.append(nn.Linear(per_step_dim, latent_dim))
                    self.projection_indices.append(i)
                    self.per_step_dims.append(latent_dim)
                    total_per_step_dim += latent_dim
                else:
                    self.per_step_dims.append(per_step_dim)
                    total_per_step_dim += per_step_dim
            else:
                # No history
                self.has_history.append(False)
                self.per_step_dims.append(0)
        
        # Register projections as ModuleList (only actual nn.Linear layers)
        if projection_layers:
            self.projections = nn.ModuleList(projection_layers)
        else:
            self.projections = nn.ModuleList()
        
        # Store output dimension (total per-step dim after compression)
        self.output_dim = total_per_step_dim
        
        # Conv1D for temporal compression
        # Input: (B, total_per_step_dim, T)
        # Output: (B, total_per_step_dim, 1) after pool -> (B, total_per_step_dim)
        # Conv only compresses temporal dimension, keeps feature dimension unchanged
        if total_per_step_dim > 0:
            # Conv1D to process temporal sequence
            # Keep channel size same as input
            self.temporal_conv = nn.Conv1d(
                in_channels=total_per_step_dim,
                out_channels=total_per_step_dim,  # Keep same channel size
                kernel_size=3,
                padding=1
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            # Activation function
            activations = {
                "elu": nn.ELU(),
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
                "silu": nn.SiLU(),
            }
            self.activation = activations.get(activation.lower(), nn.ELU())
        else:
            self.temporal_conv = None
    
    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, total_obs_dim) in IsaacLab format [a1,a2,a3,a4, b1,b2,b3,b4, ...]
        
        Returns:
            encoded: (B, output_dim) - temporally compressed features
        """
        B = obs.shape[0]
        
        if self.temporal_conv is None:
            # No history terms, return empty
            return torch.zeros(B, self.output_dim, device=obs.device, dtype=obs.dtype)
        
        # Split observations by term and process each
        temporal_features = []  # List of (B, T, d) tensors
        start = 0
        proj_idx = 0
        
        for i, term_dim in enumerate(self.term_dims):
            term_obs = obs[:, start:start+term_dim]  # (B, term_dim)
            start += term_dim
            
            if not self.has_history[i]:
                continue  # Skip non-history terms
            
            # Reshape: (B, term_dim) -> (B, T, per_step_dim)
            per_step_dim = term_dim // self.history_length
            feature = term_obs.view(B, self.history_length, per_step_dim)  # (B, T, d)
            
            # Compress if needed
            if i in self.projection_indices:
                feature = self.projections[proj_idx](feature)  # (B, T, latent_dim)
                proj_idx += 1
            
            temporal_features.append(feature)
        
        # Concatenate all temporal features: (B, T, total_per_step_dim)
        if not temporal_features:
            return torch.zeros(B, self.output_dim, device=obs.device, dtype=obs.dtype)
        
        feature_group = torch.cat(temporal_features, dim=-1)  # (B, T, total_per_step_dim)
        
        # Conv1D expects (B, C, T), so transpose
        feature_group = feature_group.transpose(1, 2)  # (B, total_per_step_dim, T)
        
        # Temporal convolution
        conv_out = self.activation(self.temporal_conv(feature_group))  # (B, output_dim, T)
        
        # Pool over time dimension
        pooled = self.pool(conv_out).squeeze(-1)  # (B, output_dim)
        
        return pooled


class MultiModalMLPV2(nn.Module):
    """
    Multi-Modal MLP V2 that processes observations using term_dict structure.
    
    Similar to ObservationEmbeddingV2 in MMTransformer, but outputs to MLP hidden layers
    instead of transformer tokens.
    
    Key features:
    - Processes each term separately
    - Uses HistoryEncoder for terms with history_length > 1
    - Concatenates all processed terms before feeding to main MLP
    """
    
    def __init__(
        self,
        term_dict: dict[str, dict[str, int]],
        output_size: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        history_length: int = 1,
        encoder_latent_dim: int = 128,
        encoder_compress_threshold: int = 32,
        use_layer_norm: bool = False,
        name: str = ""
    ):
        """
        Args:
            term_dict: Dictionary of observation terms, e.g., 
                       {"policy": {"base_lin_vel": 30, "base_ang_vel": 30, ...}}
            output_size: Dimension of output
            hidden_dims: List of hidden layer dimensions for main MLP
            activation: Activation function name
            history_length: Length of history sequence (if terms contain temporal data)
            encoder_latent_dim: Latent dimension for compressing large per-step features
            encoder_compress_threshold: Compress per-step features > this to latent_dim
            use_layer_norm: Whether to use LayerNorm
            name: Name for debugging
        """
        super().__init__()
        self.name = name
        self.history_length = history_length
        self.output_size = output_size
        
        # Parse term_dict
        self.group_key = list(term_dict.keys())[0]
        self.terms = term_dict[self.group_key]
        
        self.term_names = list(self.terms.keys())
        self.term_dims = list(self.terms.values())
        self.total_obs_dim = sum(self.term_dims)
        
        # Separate history terms from non-history terms
        self.non_history_dims = []
        for i, term_dim in enumerate(self.term_dims):
            if history_length <= 1 or term_dim % history_length != 0:
                self.non_history_dims.append((i, term_dim))
        
        total_non_history = sum(dim for _, dim in self.non_history_dims)
        
        # Use unified temporal encoder
        if history_length > 1:
            self.temporal_encoder = TemporalFeatureEncoder(
                term_dims=self.term_dims,
                term_names=self.term_names,
                history_length=history_length,
                latent_dim=encoder_latent_dim,
                compress_threshold=encoder_compress_threshold,
                activation=activation
            )
            total_encoded_dim = self.temporal_encoder.output_dim + total_non_history
        else:
            # No history: direct pass-through
            self.temporal_encoder = None
            total_encoded_dim = self.total_obs_dim
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Main MLP processes encoded features
        if hidden_dims:
            self.main_mlp = self._build_mlp(total_encoded_dim, hidden_dims, use_layer_norm)
            self.output_layer = nn.Linear(hidden_dims[-1], output_size)
        else:
            # No hidden layers, direct projection
            self.main_mlp = nn.Identity()
            self.output_layer = nn.Linear(total_encoded_dim, output_size)
    
    def _get_activation(self, activation: str):
        """Get activation function by name"""
        activations = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        return activations[activation.lower()]
    
    def _build_mlp(self, input_size: int, hidden_dims: list[int], use_layer_norm: bool):
        """Build MLP with given dimensions"""
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Flattened observations (B, total_obs_dim)
                 In IsaacLab format: [a_1, a_2, a_3, b_1, b_2, b_3, ...]
        
        Returns:
            output: (B, output_size)
        """
        if self.temporal_encoder is not None:
            # Extract temporal features
            temporal_features = self.temporal_encoder(obs)  # (B, encoder_output_dim)
            
            # Extract non-history features
            non_history_features = []
            start = 0
            for i, term_dim in enumerate(self.term_dims):
                if (i, term_dim) in [(idx, dim) for idx, dim in self.non_history_dims]:
                    non_history_features.append(obs[:, start:start+term_dim])
                start += term_dim
            
            # Concatenate temporal and non-history
            if non_history_features:
                non_history = torch.cat(non_history_features, dim=-1)
                concatenated = torch.cat([temporal_features, non_history], dim=-1)
            else:
                concatenated = temporal_features
        else:
            # No history: direct pass-through
            concatenated = obs
        
        # Pass through main MLP
        features = self.main_mlp(concatenated)  # (B, hidden_dims[-1])
        output = self.output_layer(features)  # (B, output_size)
        
        return output


class FusedMultiModalMLP(nn.Module):
    """
    Wrapper that combines main and ref MultiModalMLPV2 networks.
    This provides a unified interface similar to MMTransformerV2.
    """
    def __init__(
        self,
        term_dict: dict[str, dict[str, int]],
        ref_term_dict: Optional[dict[str, dict[str, int]]],
        output_size: int,
        hidden_dims: list[int],
        activation: str,
        history_length: int,
        encoder_latent_dim: int,
        encoder_compress_threshold: int,
        use_layer_norm: bool,
        fusion_mode: str,
        name: str = "fused_mlp"
    ):
        super().__init__()
        
        self.fusion_mode = fusion_mode
        self.has_ref = ref_term_dict is not None and bool(list(ref_term_dict.values())[0])
        
        # Main network
        self.main_net = MultiModalMLPV2(
            term_dict=term_dict,
            output_size=output_size,
            hidden_dims=hidden_dims,
            activation=activation,
            history_length=history_length,
            encoder_latent_dim=encoder_latent_dim,
            encoder_compress_threshold=encoder_compress_threshold,
            use_layer_norm=use_layer_norm,
            name=f"{name}_main"
        )
        
        # Reference network (if available)
        if self.has_ref:
            self.ref_net = MultiModalMLPV2(
                term_dict=ref_term_dict,
                output_size=output_size,
                hidden_dims=hidden_dims,
                activation=activation,
                history_length=history_length,
                encoder_latent_dim=encoder_latent_dim,
                encoder_compress_threshold=encoder_compress_threshold,
                use_layer_norm=use_layer_norm,
                name=f"{name}_ref"
            )
            
            # Fusion layers
            if fusion_mode == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(output_size * 2, output_size),
                    nn.Sigmoid()
                )
            elif fusion_mode == "concat":
                self.fusion_proj = nn.Linear(output_size * 2, output_size)
        else:
            self.ref_net = None
    
    def forward(
        self,
        observations: torch.Tensor,
        ref_observations: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            observations: Main observations (B, obs_dim)
            ref_observations: Optional tuple of (ref_obs, ref_mask)
                            ref_obs: (B, ref_obs_dim)
                            ref_mask: (B,) boolean mask
        
        Returns:
            output: (B, output_size)
        """
        main_output = self.main_net(observations)
        
        if not self.has_ref or ref_observations is None:
            return main_output
        
        ref_obs, ref_mask = ref_observations
        ref_output = self.ref_net(ref_obs)
        
        # Apply mask: only use ref_output where mask is True
        if ref_mask is not None:
            ref_mask = ref_mask.unsqueeze(-1).float()  # (B, 1)
            ref_output = ref_output * ref_mask
        
        # Fuse main and ref outputs
        if self.fusion_mode == "gated":
            gate_input = torch.cat([main_output, ref_output], dim=-1)
            gate = self.gate(gate_input)
            output = main_output * (1 - gate) + ref_output * gate
        elif self.fusion_mode == "add":
            output = main_output + ref_output
        elif self.fusion_mode == "concat":
            concat = torch.cat([main_output, ref_output], dim=-1)
            output = self.fusion_proj(concat)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        return output


class ActorCriticMLPV2(nn.Module):
    """
    Actor-Critic with Multi-Modal MLP V2 architecture.
    Uses term_dict structure and HistoryEncoder for temporal information.
    """
    is_recurrent = False
    
    def __init__(
        self,
        term_dict: dict[str, dict[str, int]],
        ref_term_dict: dict[str, dict[str, int]],
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        history_length: int = 1,
        encoder_latent_dim: int = 128,
        encoder_compress_threshold: int = 32,
        use_layer_norm: bool = False,
        fusion_mode: str = "gated",  # How to fuse actor obs and ref_obs
        load_dagger: bool = False,
        load_dagger_path: Optional[str] = None,
        load_actor_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            term_dict: Dictionary of observation terms for actor/critic
                      e.g., {"policy": {"base_lin_vel": 30, ...}, "critic": {...}}
            ref_term_dict: Dictionary of reference observation terms
            num_actions: Number of action dimensions
            actor_hidden_dims: Hidden layer dimensions for actor
            critic_hidden_dims: Hidden layer dimensions for critic
            activation: Activation function
            init_noise_std: Initial standard deviation for action noise
            noise_std_type: "scalar" or "log"
            history_length: Length of history sequence
            encoder_latent_dim: Latent dimension for compressing large per-step features
            encoder_compress_threshold: Compress per-step features > this to latent_dim
            use_layer_norm: Whether to use LayerNorm
            fusion_mode: How to fuse main and reference features ("gated", "add", "concat")
            load_dagger: Whether to load dagger (teacher) model
            load_dagger_path: Path to dagger model weights
            load_actor_path: Path to pretrained actor weights
        """
        super().__init__()
        
        assert not load_dagger or load_dagger_path, \
            "load_dagger and load_dagger_path must be provided if load_dagger is True"
        
        # Extract actor and critic term dicts
        actor_term_dict = {"policy": term_dict.get("policy", {})}
        critic_term_dict = {"critic": term_dict.get("critic", term_dict.get("policy", {}))}
        
        actor_ref_term_dict = {"policy": ref_term_dict.get("policy", {})} if ref_term_dict else None
        critic_ref_term_dict = {"critic": ref_term_dict.get("critic", ref_term_dict.get("policy", {}))} if ref_term_dict else None
        
        # Actor network (fuses main + ref observations)
        self.actor = FusedMultiModalMLP(
            term_dict=actor_term_dict,
            ref_term_dict=actor_ref_term_dict,
            output_size=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            history_length=history_length,
            encoder_latent_dim=encoder_latent_dim,
            encoder_compress_threshold=encoder_compress_threshold,
            use_layer_norm=use_layer_norm,
            fusion_mode=fusion_mode,
            name="actor"
        )
        
        # Dagger (teacher) network - only uses main observations, no ref_obs
        if load_dagger:
            self.actor_dagger = MultiModalMLPV2(
                term_dict=actor_term_dict,
                output_size=num_actions,
                hidden_dims=actor_hidden_dims,
                activation=activation,
                history_length=history_length,
                encoder_latent_dim=encoder_latent_dim,
                encoder_compress_threshold=encoder_compress_threshold,
                use_layer_norm=use_layer_norm,
                name="actor_dagger"
            )
        else:
            self.actor_dagger = None
        
        # Critic network (fuses main + ref observations)
        self.critic = FusedMultiModalMLP(
            term_dict=critic_term_dict,
            ref_term_dict=critic_ref_term_dict,
            output_size=1,  # Value function outputs single value
            hidden_dims=critic_hidden_dims,
            activation=activation,
            history_length=history_length,
            encoder_latent_dim=encoder_latent_dim,
            encoder_compress_threshold=encoder_compress_threshold,
            use_layer_norm=use_layer_norm,
            fusion_mode=fusion_mode,
            name="critic"
        )
        
        # Action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            if load_dagger:
                self.std_dagger = nn.Parameter(init_noise_std * torch.ones(num_actions))
                self.std_dagger.requires_grad = False
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            if load_dagger:
                self.log_std_dagger = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
                self.log_std_dagger.requires_grad = False
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")
        
        self.distribution = None
        self.distribution_dagger = None
        
        # Load weights if specified
        if load_dagger:
            self.load_dagger_weights(load_dagger_path)
        
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)
        
        print(f"Actor MLP V2: {self.actor}")
        print(f"Critic MLP V2: {self.critic}")
        if self.actor_dagger:
            print(f"Dagger MLP V2: {self.actor_dagger}")
    
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
    
    def reset(self, dones=None):
        pass
    
    def update_distribution(self, observations, ref_observations=None, **kwargs):
        """Update action distribution for actor"""
        mean = self.actor(observations, ref_observations)
        
        # Set std
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        self.distribution = Normal(mean, std)
    
    def update_distribution_dagger(self, observations, ref_observations=None, **kwargs):
        """Update action distribution for dagger (teacher)"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        
        mean = self.actor_dagger(observations)
        
        if self.noise_std_type == "scalar":
            std = self.std_dagger.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std_dagger).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        self.distribution_dagger = Normal(mean, std)
    
    def act(self, observations, ref_observations=None, **kwargs):
        """Sample action from distribution"""
        self.update_distribution(observations, ref_observations, **kwargs)
        return self.distribution.sample()
    
    def act_dagger(self, observations, ref_observations=None, **kwargs):
        """Sample action from dagger distribution"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        self.update_distribution_dagger(observations, ref_observations, **kwargs)
        return self.distribution_dagger.sample()
    
    def act_inference(self, observations, ref_observations=None, **kwargs):
        """Get mean action for inference"""
        return self.actor(observations, ref_observations)
    
    def act_dagger_inference(self, observations, ref_observations=None, **kwargs):
        """Get mean action for dagger inference"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        return self.actor_dagger(observations)
    
    def evaluate(self, critic_observations, ref_critic_observations=None, **kwargs):
        """Evaluate value function"""
        return self.critic(critic_observations, ref_critic_observations)
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_actions_log_prob_dagger(self, actions):
        """Get log probability of actions from dagger"""
        return self.distribution_dagger.log_prob(actions).sum(dim=-1)
    
    def load_actor_weights(self, path: str):
        """Load pretrained actor weights"""
        # TODO: Implement weight loading with shape checking
        print(f"Warning: load_actor_weights not fully implemented for ActorCriticMLPV2")
        pass
    
    def load_dagger_weights(self, path: str):
        """Load dagger (teacher) weights"""
        # TODO: Implement weight loading
        print(f"Warning: load_dagger_weights not fully implemented for ActorCriticMLPV2")
        pass
