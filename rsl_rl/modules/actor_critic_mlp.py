# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Multi-Modal MLP with Reference Observation Support
# Created by Yifei Yao (with AI assistance), 2025

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple


class MultiModalMLP(nn.Module):
    """
    Multi-Modal MLP that can optionally incorporate reference observations.
    
    Architecture:
    - Primary path: obs -> hidden_layers -> output
    - Reference path (optional): ref_obs -> hidden_layers -> ref_output
    - Fusion: output + gate(output, ref_output) * ref_output
    
    When ref_obs is None, only the primary path is active.
    When ref_obs is provided, both paths are computed and fused.
    """
    
    def __init__(
        self,
        obs_size: int,
        ref_obs_size: int,
        output_size: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        use_layer_norm: bool = False,
        fusion_mode: str = "gated",  # "gated", "concat", "add", "attention"
        name: str = ""
    ):
        """
        Args:
            obs_size: Dimension of main observation
            ref_obs_size: Dimension of reference observation
            output_size: Dimension of output
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name ("elu", "relu", "gelu", "tanh")
            use_layer_norm: Whether to use LayerNorm after each hidden layer
            fusion_mode: How to fuse main and reference features
                - "gated": Gated fusion with learned gate
                - "concat": Concatenate and project
                - "add": Simple addition
                - "attention": Attention-based fusion
            name: Name for debugging
        """
        super().__init__()
        self.name = name
        self.obs_size = obs_size
        self.ref_obs_size = ref_obs_size
        self.output_size = output_size
        self.fusion_mode = fusion_mode
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Primary MLP path (always active)
        self.primary_mlp = self._build_mlp(obs_size, hidden_dims, use_layer_norm)
        self.primary_output = nn.Linear(hidden_dims[-1], output_size)
        
        # Reference MLP path (only active when ref_obs is provided)
        if ref_obs_size > 0:
            self.ref_mlp = self._build_mlp(ref_obs_size, hidden_dims, use_layer_norm)
            self.ref_output = nn.Linear(hidden_dims[-1], output_size)
            
            # Fusion layer
            if fusion_mode == "gated":
                # Gated fusion: learns when to use reference information
                self.gate = nn.Sequential(
                    nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], output_size),
                    nn.Sigmoid()  # Gate values in [0, 1]
                )
            elif fusion_mode == "concat":
                # Concatenation fusion
                self.fusion_proj = nn.Linear(output_size * 2, output_size)
            elif fusion_mode == "attention":
                # Attention-based fusion
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dims[-1],
                    num_heads=4,
                    batch_first=True
                )
                self.attention_proj = nn.Linear(hidden_dims[-1], output_size)
            # "add" mode doesn't need extra parameters
        else:
            self.ref_mlp = None
            self.ref_output = None
    
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
    
    def forward(
        self,
        obs: torch.Tensor,
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with optional reference observations.
        
        Args:
            obs: Main observations (B, obs_size)
            ref_obs: Optional tuple of (ref_obs_tensor, ref_obs_mask)
                - ref_obs_tensor: (B, ref_obs_size)
                - ref_obs_mask: (B,) boolean mask indicating valid ref_obs
        
        Returns:
            output: (B, output_size)
        """
        # Primary path (always computed)
        primary_features = self.primary_mlp(obs)  # (B, hidden_dims[-1])
        primary_out = self.primary_output(primary_features)  # (B, output_size)
        
        # If no reference observations, return primary output directly
        if ref_obs is None or self.ref_mlp is None:
            return primary_out
        
        # Unpack reference observations
        ref_obs_tensor, ref_obs_mask = ref_obs
        
        # Reference path
        ref_features = self.ref_mlp(ref_obs_tensor)  # (B, hidden_dims[-1])
        ref_out = self.ref_output(ref_features)  # (B, output_size)
        
        # Fusion based on mode
        if self.fusion_mode == "gated":
            # Gated fusion: gate decides how much to use ref information
            combined_features = torch.cat([primary_features, ref_features], dim=-1)
            gate = self.gate(combined_features)  # (B, output_size)
            
            # Apply mask: where ref_obs_mask is False, gate should be 0
            gate = gate * ref_obs_mask.float().unsqueeze(-1)
            
            output = primary_out + gate * ref_out
            
        elif self.fusion_mode == "concat":
            # Concatenation fusion
            concat_out = torch.cat([primary_out, ref_out], dim=-1)
            output = self.fusion_proj(concat_out)
            
            # Apply mask: where ref_obs_mask is False, use only primary
            mask_expanded = ref_obs_mask.float().unsqueeze(-1)
            output = primary_out * (1 - mask_expanded) + output * mask_expanded
            
        elif self.fusion_mode == "add":
            # Simple addition with mask
            mask_expanded = ref_obs_mask.float().unsqueeze(-1)
            output = primary_out + ref_out * mask_expanded
            
        elif self.fusion_mode == "attention":
            # Attention-based fusion
            # Treat primary as query, ref as key/value
            primary_feat_expanded = primary_features.unsqueeze(1)  # (B, 1, hidden)
            ref_feat_expanded = ref_features.unsqueeze(1)  # (B, 1, hidden)
            
            # Attention
            attn_out, _ = self.attention(
                primary_feat_expanded,
                ref_feat_expanded,
                ref_feat_expanded,
                key_padding_mask=~ref_obs_mask.unsqueeze(1)
            )
            attn_out = self.attention_proj(attn_out.squeeze(1))
            
            # Combine
            mask_expanded = ref_obs_mask.float().unsqueeze(-1)
            output = primary_out * (1 - mask_expanded) + attn_out * mask_expanded
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        return output


class ActorCriticMLP(nn.Module):
    """
    Actor-Critic with Multi-Modal MLP architecture.
    Supports optional reference observations with flexible fusion.
    """
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs: int,
        num_actor_ref_obs: int,
        num_critic_obs: int,
        num_critic_ref_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        use_layer_norm: bool = False,
        fusion_mode: str = "gated",
        load_dagger: bool = False,
        load_dagger_path: Optional[str] = None,
        load_actor_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            num_actor_obs: Dimension of actor observation
            num_actor_ref_obs: Dimension of actor reference observation
            num_critic_obs: Dimension of critic observation
            num_critic_ref_obs: Dimension of critic reference observation
            num_actions: Number of action dimensions
            actor_hidden_dims: Hidden layer dimensions for actor
            critic_hidden_dims: Hidden layer dimensions for critic
            activation: Activation function
            init_noise_std: Initial standard deviation for action noise
            noise_std_type: "scalar" or "log"
            use_layer_norm: Whether to use LayerNorm
            fusion_mode: How to fuse main and reference features
            load_dagger: Whether to load dagger (teacher) model
            load_dagger_path: Path to dagger model weights
            load_actor_path: Path to pretrained actor weights
        """
        super().__init__()
        
        assert not load_dagger or load_dagger_path, \
            "load_dagger and load_dagger_path must be provided if load_dagger is True"
        
        # Actor network
        self.actor = MultiModalMLP(
            obs_size=num_actor_obs,
            ref_obs_size=num_actor_ref_obs,
            output_size=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            fusion_mode=fusion_mode,
            name="actor"
        )
        
        # Dagger (teacher) network - no reference observations
        if load_dagger:
            self.actor_dagger = MultiModalMLP(
                obs_size=num_actor_obs,
                ref_obs_size=0,  # Teacher doesn't use ref_obs
                output_size=num_actions,
                hidden_dims=actor_hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
                fusion_mode=fusion_mode,
                name="actor_dagger"
            )
        else:
            self.actor_dagger = None
        
        # Critic network
        self.critic = MultiModalMLP(
            obs_size=num_critic_obs,
            ref_obs_size=num_critic_ref_obs,
            output_size=1,  # Value function outputs single value
            hidden_dims=critic_hidden_dims,
            activation=activation,
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
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            if load_dagger:
                self.log_std_dagger = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")
        
        # Load weights if specified
        if load_dagger:
            self.load_dagger_weights(load_dagger_path)
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        
        self.distribution = None
        self.distribution_dagger = None if not load_dagger else None
        
        # Disable validation for speedup
        Normal.set_default_validate_args(False)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        if load_dagger:
            print(f"Dagger MLP: {self.actor_dagger}")
    
    def load_actor_weights(self, path: str):
        """Load pretrained actor weights"""
        state_dict = torch.load(path, map_location="cpu")
        assert 'model_state_dict' in state_dict.keys(), \
            f"Key 'model_state_dict' not found in {path}"
        
        model_state_dict = state_dict['model_state_dict']
        
        # Extract actor and critic weights
        actor_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor.')}
        critic_weights = {k[len('critic.'):]: v for k, v in model_state_dict.items() if k.startswith('critic.')}
        
        # Check what parameters will be loaded/missing
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        
        # Report missing parameters (in current model but not in checkpoint)
        missing_actor_keys = set(actor_state_dict.keys()) - set(actor_weights.keys())
        missing_critic_keys = set(critic_state_dict.keys()) - set(critic_weights.keys())
        
        # Report unexpected parameters (in checkpoint but not in current model)
        unexpected_actor_keys = set(actor_weights.keys()) - set(actor_state_dict.keys())
        unexpected_critic_keys = set(critic_weights.keys()) - set(critic_state_dict.keys())
        
        if missing_actor_keys:
            print(f"[Actor] Missing keys (will use random initialization):")
            for k in sorted(missing_actor_keys):
                print(f"  - {k}")
        
        if missing_critic_keys:
            print(f"[Critic] Missing keys (will use random initialization):")
            for k in sorted(missing_critic_keys):
                print(f"  - {k}")
        
        if unexpected_actor_keys:
            print(f"[Actor] Unexpected keys (will be ignored):")
            for k in sorted(unexpected_actor_keys):
                print(f"  - {k}")
        
        if unexpected_critic_keys:
            print(f"[Critic] Unexpected keys (will be ignored):")
            for k in sorted(unexpected_critic_keys):
                print(f"  - {k}")
        
        # Load with shape checking
        self.actor.load_state_dict(actor_weights, strict=False)
        self.critic.load_state_dict(critic_weights, strict=False)
        
        # Load std
        if self.noise_std_type == "scalar" and "std" in model_state_dict:
            self.std.data = model_state_dict["std"]
        elif self.noise_std_type == "log" and "log_std" in model_state_dict:
            self.log_std.data = model_state_dict["log_std"]
        
        print(f"Loaded actor weights from {path}")
    
    def load_dagger_weights(self, path: str):
        """Load dagger (teacher) weights"""
        state_dict = torch.load(path, map_location="cpu")
        assert 'model_state_dict' in state_dict.keys(), \
            f"Key 'model_state_dict' not found in {path}"
        
        model_state_dict = state_dict['model_state_dict']
        dagger_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor.')}
        
        # Check what parameters will be loaded/missing
        dagger_state_dict = self.actor_dagger.state_dict()
        
        # Report missing and unexpected parameters
        missing_keys = set(dagger_state_dict.keys()) - set(dagger_weights.keys())
        unexpected_keys = set(dagger_weights.keys()) - set(dagger_state_dict.keys())
        
        if missing_keys:
            print(f"[Actor Dagger] Missing keys (will use random initialization):")
            for k in sorted(missing_keys):
                print(f"  - {k}")
        
        if unexpected_keys:
            print(f"[Actor Dagger] Unexpected keys (will be ignored):")
            for k in sorted(unexpected_keys):
                print(f"  - {k}")
        
        self.actor_dagger.load_state_dict(dagger_weights, strict=False)
        
        # Load std for dagger
        if self.noise_std_type == "scalar" and "std" in model_state_dict:
            self.std_dagger.data = model_state_dict["std"]
            self.std_dagger.requires_grad = False
        elif self.noise_std_type == "log" and "log_std" in model_state_dict:
            self.log_std_dagger.data = model_state_dict["log_std"]
            self.log_std_dagger.requires_grad = False
        
        print(f"Loaded dagger weights from {path}")
    
    def reset(self, dones=None):
        """Reset (for recurrent compatibility, does nothing for MLP)"""
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
        """Update action distribution for actor"""
        mean = self.actor(observations, ref_observations, **kwargs)
        
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)
    
    def act(self, observations, ref_observations=None, **kwargs):
        """Sample actions from actor"""
        self.update_distribution(observations, ref_observations, **kwargs)
        return self.distribution.sample()
    
    def act_inference(self, observations, ref_observations=None, **kwargs):
        """Get mean actions (for inference)"""
        return self.actor(observations, ref_observations, **kwargs)
    
    def update_distribution_dagger(self, observations, ref_observations=None, **kwargs):
        """Update action distribution for dagger (teacher)"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        mean = self.actor_dagger(observations, ref_observations, **kwargs)
        
        if self.noise_std_type == "scalar":
            std = self.std_dagger.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std_dagger).expand_as(mean)
        
        self.distribution_dagger = Normal(mean, std)
    
    def act_dagger(self, observations, ref_observations=None, **kwargs):
        """Sample actions from dagger (teacher)"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        self.update_distribution_dagger(observations, ref_observations, **kwargs)
        return self.distribution_dagger.sample()
    
    def act_dagger_inference(self, observations, ref_observations=None, **kwargs):
        """Get mean actions from dagger (for inference)"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        return self.actor_dagger(observations, ref_observations, **kwargs)
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_actions_log_prob_dagger(self, actions):
        """Get log probability of actions under dagger distribution"""
        return self.distribution_dagger.log_prob(actions).sum(dim=-1)
    
    def evaluate(self, critic_observations, ref_critic_observations=None, **kwargs):
        """Evaluate value function"""
        return self.critic(critic_observations, ref_critic_observations, **kwargs)


# ============================================================================
# V2: Multi-Modal MLP with Temporal History Encoding
# ============================================================================

class HistoryEncoder(nn.Module):
    """
    HistoryEncoder for temporal feature extraction.
    Uses 1D convolutions to extract temporal patterns from history observations.
    
    This is imported from MMTransformer design for consistency.
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
        self.projection = current_embedding(flattened_size, d_model, int(flattened_size / d_model) + 1)
        
    def forward(self, x_seq: torch.Tensor):
        # x_seq has shape (B, history_length, group_per_step_dim)
        x_conv_in = x_seq.permute(0, 2, 1)  # (B, group_per_step_dim, history_length)
        conv_out = self.conv_net(x_conv_in)  # (B, 64, history_length)
        conv_out_flat = torch.flatten(conv_out, 1)  # (B, 64 * history_length)
        
        # Use the powerful projection layer
        token = self.projection(conv_out_flat)  # (B, d_model)
        
        return token


class SwiGLUEmbedding(nn.Module):
    """A SwiGLU block for embedding."""
    def __init__(self, input_dim: int, d_model: int, expansion_factor: int = 2):
        super().__init__()
        hidden_dim = int(expansion_factor * d_model)
        
        # The SwiGLU magic: two linear layers for the gate and value
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # The final projection layer
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x))  # Swish activation for the gate
        value = self.w3(x)
        
        # Element-wise multiplication, followed by the final projection
        return self.w2(gate * value)


class MLPEmbedding(nn.Module):
    """A simple MLP block for embedding."""
    def __init__(self, input_dim: int, d_model: int, expansion_factor: int = 2):
        super().__init__()
        hidden_dim = int(expansion_factor * d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, d_model)
        )
        
    def forward(self, x):
        return self.mlp(x)


class MultiModalMLPV2(nn.Module):
    """
    Multi-Modal MLP V2 with proper temporal history encoding.
    
    Key difference from V1:
    - Understands IsaacLab's stacking format: [a_1, a_2, a_3, b_1, b_2, b_3]
    - Uses term_dict to correctly slice and rearrange temporal data
    - Applies HistoryEncoder per observation term
    - Supports flexible fusion modes
    
    Architecture:
    - Primary path: obs -> MLP -> output
    - Reference path: ref_obs -> reshape by term_dict -> HistoryEncoder per term -> concatenate -> MLP -> ref_output
    - Fusion: gated/concat/add/attention combination
    """
    
    def __init__(
        self,
        obs_size: int,
        ref_term_dict: dict[str, int],  # NEW: term dictionary for ref_obs
        output_size: int,
        history_length: int = 1,  # NEW: history length for temporal data
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        use_layer_norm: bool = False,
        fusion_mode: str = "gated",
        use_history_encoder: bool = True,  # NEW: whether to use HistoryEncoder
        history_encoder_dim: int = 128,  # NEW: output dim per term after HistoryEncoder
        use_swiglu: bool = False,  # NEW: use SwiGLU in HistoryEncoder
        name: str = ""
    ):
        """
        Args:
            obs_size: Dimension of main observation (current step only)
            ref_term_dict: Dictionary mapping term names to their TOTAL dimensions in ref_obs
                          For example: {'lin_vel': 6, 'ang_vel': 6} means lin_vel has 3*2 timesteps
            output_size: Output dimension
            history_length: Number of timesteps in history
            hidden_dims: MLP hidden dimensions
            activation: Activation function
            use_layer_norm: Whether to use LayerNorm
            fusion_mode: "gated", "concat", "add", "attention"
            use_history_encoder: Whether to use HistoryEncoder for temporal encoding
            history_encoder_dim: Output dimension of HistoryEncoder per term
            use_swiglu: Use SwiGLU in HistoryEncoder
            name: Debug name
        """
        super().__init__()
        self.name = name
        self.obs_size = obs_size
        self.ref_term_dict = ref_term_dict
        self.output_size = output_size
        self.history_length = history_length
        self.fusion_mode = fusion_mode
        self.use_history_encoder = use_history_encoder
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Primary MLP path (always active, current observation)
        self.primary_mlp = self._build_mlp(obs_size, hidden_dims, use_layer_norm)
        self.primary_output = nn.Linear(hidden_dims[-1], output_size)
        
        # Reference path with temporal encoding
        if ref_term_dict and sum(ref_term_dict.values()) > 0:
            self.term_names = list(ref_term_dict.keys())
            self.term_dims = list(ref_term_dict.values())
            self.ref_obs_size = sum(self.term_dims)
            
            # Calculate slices for each term in the flattened ref_obs
            # Format: [a_1, a_2, a_3, b_1, b_2, b_3] where a has 3 dims, b has 3 dims
            start = 0
            self.term_slices = []
            for dim in self.term_dims:
                self.term_slices.append(slice(start, start + dim))
                start += dim
            
            if use_history_encoder and history_length > 1:
                # Use HistoryEncoder for each term
                self.history_encoders = nn.ModuleList()
                for term_dim in self.term_dims:
                    per_step_dim = term_dim // history_length
                    assert term_dim == per_step_dim * history_length, \
                        f"term_dim ({term_dim}) must be divisible by history_length ({history_length})"
                    
                    encoder = HistoryEncoder(
                        history_length=history_length,
                        group_per_step_dim=per_step_dim,
                        d_model=history_encoder_dim,
                        use_swiglu=use_swiglu
                    )
                    self.history_encoders.append(encoder)
                
                # After encoding, concatenate all term encodings
                ref_encoded_size = len(self.term_names) * history_encoder_dim
                self.ref_mlp = self._build_mlp(ref_encoded_size, hidden_dims, use_layer_norm)
            else:
                # No history encoding, treat as flat
                self.history_encoders = None
                self.ref_mlp = self._build_mlp(self.ref_obs_size, hidden_dims, use_layer_norm)
            
            self.ref_output = nn.Linear(hidden_dims[-1], output_size)
            
            # Fusion layer
            if fusion_mode == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], output_size),
                    nn.Sigmoid()
                )
            elif fusion_mode == "concat":
                self.fusion_proj = nn.Linear(output_size * 2, output_size)
            elif fusion_mode == "attention":
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dims[-1],
                    num_heads=4,
                    batch_first=True
                )
                self.attention_proj = nn.Linear(hidden_dims[-1], output_size)
        else:
            self.ref_mlp = None
            self.ref_output = None
            self.history_encoders = None
    
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
    
    def forward(
        self,
        obs: torch.Tensor,
        ref_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with optional reference observations.
        
        Args:
            obs: Main observations (B, obs_size) - current step only
            ref_obs: Optional tuple of (ref_obs_tensor, ref_obs_mask)
                - ref_obs_tensor: (B, ref_obs_size) - flattened temporal data
                  Format: [a_1, a_2, ..., a_T, b_1, b_2, ..., b_T]
                - ref_obs_mask: (B,) boolean mask indicating valid ref_obs
        
        Returns:
            output: (B, output_size)
        """
        # Primary path (current observation)
        primary_features = self.primary_mlp(obs)  # (B, hidden_dims[-1])
        primary_out = self.primary_output(primary_features)  # (B, output_size)
        
        # If no reference observations, return primary output
        if ref_obs is None or self.ref_mlp is None:
            return primary_out
        
        # Unpack reference observations
        ref_obs_tensor, ref_obs_mask = ref_obs
        batch_size = ref_obs_tensor.size(0)
        
        # Process reference observations with temporal encoding
        if self.history_encoders is not None:
            # Extract and encode each term separately
            encoded_terms = []
            for i, term_slice in enumerate(self.term_slices):
                # Extract term data: (B, term_dim)
                term_data = ref_obs_tensor[:, term_slice]
                
                # Reshape to temporal format: (B, history_length, per_step_dim)
                # Example: [a_1, a_2, a_3] -> [[a_1], [a_2], [a_3]]
                per_step_dim = self.term_dims[i] // self.history_length
                term_data_reshaped = term_data.view(batch_size, self.history_length, per_step_dim)
                
                # Encode temporal features: (B, history_encoder_dim)
                term_encoded = self.history_encoders[i](term_data_reshaped)
                encoded_terms.append(term_encoded)
            
            # Concatenate all encoded terms: (B, num_terms * history_encoder_dim)
            ref_encoded = torch.cat(encoded_terms, dim=-1)
            ref_features = self.ref_mlp(ref_encoded)  # (B, hidden_dims[-1])
        else:
            # No history encoding, use ref_obs directly
            ref_features = self.ref_mlp(ref_obs_tensor)  # (B, hidden_dims[-1])
        
        ref_out = self.ref_output(ref_features)  # (B, output_size)
        
        # Fusion based on mode
        if self.fusion_mode == "gated":
            combined_features = torch.cat([primary_features, ref_features], dim=-1)
            gate = self.gate(combined_features)  # (B, output_size)
            
            # Apply mask
            gate = gate * ref_obs_mask.float().unsqueeze(-1)
            
            output = primary_out + gate * ref_out
            
        elif self.fusion_mode == "concat":
            concat_out = torch.cat([primary_out, ref_out], dim=-1)
            output = self.fusion_proj(concat_out)
            
            # Apply mask
            mask_expanded = ref_obs_mask.float().unsqueeze(-1)
            output = primary_out * (1 - mask_expanded) + output * mask_expanded
            
        elif self.fusion_mode == "add":
            # Simple addition with mask
            ref_out_masked = ref_out * ref_obs_mask.float().unsqueeze(-1)
            output = primary_out + ref_out_masked
            
        elif self.fusion_mode == "attention":
            # Attention-based fusion
            # Treat primary and ref as sequence
            seq = torch.stack([primary_features, ref_features], dim=1)  # (B, 2, hidden_dims[-1])
            
            # Self-attention
            attn_out, _ = self.attention(seq, seq, seq)  # (B, 2, hidden_dims[-1])
            attn_out = attn_out.mean(dim=1)  # (B, hidden_dims[-1])
            output = self.attention_proj(attn_out)  # (B, output_size)
            
            # Apply mask
            mask_expanded = ref_obs_mask.float().unsqueeze(-1)
            output = primary_out * (1 - mask_expanded) + output * mask_expanded
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        return output


class ActorCriticMLPV2(nn.Module):
    """
    Actor-Critic with Multi-Modal MLP V2 architecture.
    
    V2 Features:
    - Proper temporal history encoding via HistoryEncoder
    - Term-based observation slicing (compatible with IsaacLab's stacking format)
    - Flexible fusion modes for combining current and historical information
    """
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs: int,
        actor_ref_term_dict: dict[str, int],  # NEW: term dict for actor ref_obs
        num_critic_obs: int,
        critic_ref_term_dict: dict[str, int],  # NEW: term dict for critic ref_obs
        num_actions: int,
        history_length: int = 1,  # NEW: history length
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        use_layer_norm: bool = False,
        fusion_mode: str = "gated",
        use_history_encoder: bool = True,  # NEW
        history_encoder_dim: int = 128,  # NEW
        use_swiglu: bool = False,  # NEW
        load_dagger: bool = False,
        load_dagger_path: Optional[str] = None,
        load_actor_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            num_actor_obs: Dimension of actor current observation
            actor_ref_term_dict: Term dictionary for actor reference observations
                Example: {'lin_vel': 6, 'ang_vel': 6} for 2 timesteps of 3D vectors
            num_critic_obs: Dimension of critic current observation
            critic_ref_term_dict: Term dictionary for critic reference observations
            num_actions: Number of action dimensions
            history_length: Number of timesteps in history
            actor_hidden_dims: Hidden layer dimensions for actor
            critic_hidden_dims: Hidden layer dimensions for critic
            activation: Activation function
            init_noise_std: Initial standard deviation for action noise
            noise_std_type: "scalar" or "log"
            use_layer_norm: Whether to use LayerNorm
            fusion_mode: How to fuse current and historical features
            use_history_encoder: Whether to use HistoryEncoder
            history_encoder_dim: Output dimension of HistoryEncoder per term
            use_swiglu: Use SwiGLU in HistoryEncoder
            load_dagger: Whether to load dagger (teacher) model
            load_dagger_path: Path to dagger model weights
            load_actor_path: Path to pretrained actor weights
        """
        super().__init__()
        
        assert not load_dagger or load_dagger_path, \
            "load_dagger and load_dagger_path must be provided if load_dagger is True"
        
        # Actor network
        self.actor = MultiModalMLPV2(
            obs_size=num_actor_obs,
            ref_term_dict=actor_ref_term_dict,
            output_size=num_actions,
            history_length=history_length,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            fusion_mode=fusion_mode,
            use_history_encoder=use_history_encoder,
            history_encoder_dim=history_encoder_dim,
            use_swiglu=use_swiglu,
            name="actor"
        )
        
        # Dagger (teacher) network - no reference observations
        if load_dagger:
            self.actor_dagger = MultiModalMLPV2(
                obs_size=num_actor_obs,
                ref_term_dict={},  # Teacher doesn't use ref_obs
                output_size=num_actions,
                history_length=1,
                hidden_dims=actor_hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
                fusion_mode=fusion_mode,
                use_history_encoder=False,
                name="actor_dagger"
            )
        else:
            self.actor_dagger = None
        
        # Critic network
        self.critic = MultiModalMLPV2(
            obs_size=num_critic_obs,
            ref_term_dict=critic_ref_term_dict,
            output_size=1,  # Value function outputs single value
            history_length=history_length,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            fusion_mode=fusion_mode,
            use_history_encoder=use_history_encoder,
            history_encoder_dim=history_encoder_dim,
            use_swiglu=use_swiglu,
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
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}")
        
        # Load pretrained weights
        if load_dagger:
            self.load_dagger_weights(load_dagger_path)
        
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        
        self.distribution = None
        self.distribution_dagger = None if not load_dagger else None
        
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)
        
        print(f"[ActorCriticMLPV2] Initialized with history_length={history_length}")
        print(f"  Actor ref terms: {list(actor_ref_term_dict.keys())}")
        print(f"  Critic ref terms: {list(critic_ref_term_dict.keys())}")
        print(f"  Use HistoryEncoder: {use_history_encoder}")
    
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
        """Update action distribution"""
        mean = self.actor(observations, ref_observations, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)
    
    def act(self, observations, ref_observations=None, **kwargs):
        """Sample actions from distribution"""
        self.update_distribution(observations, ref_observations, **kwargs)
        return self.distribution.sample()
    
    def act_inference(self, observations, ref_observations=None, **kwargs):
        """Get mean actions (for inference)"""
        return self.actor(observations, ref_observations, **kwargs)
    
    def update_distribution_dagger(self, observations, ref_observations=None, **kwargs):
        """Update dagger distribution"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        mean = self.actor_dagger(observations, ref_observations, **kwargs)
        if self.noise_std_type == "scalar":
            std = self.std_dagger.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std_dagger).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution_dagger = Normal(mean, std)
    
    def act_dagger(self, observations, ref_observations=None, **kwargs):
        """Sample actions from dagger distribution"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        self.update_distribution_dagger(observations, ref_observations, **kwargs)
        return self.distribution_dagger.sample()
    
    def act_dagger_inference(self, observations, ref_observations=None, **kwargs):
        """Get mean actions from dagger (for inference)"""
        assert self.actor_dagger is not None, "actor_dagger is not initialized"
        return self.actor_dagger(observations, ref_observations, **kwargs)
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_actions_log_prob_dagger(self, actions):
        """Get log probability of actions under dagger distribution"""
        return self.distribution_dagger.log_prob(actions).sum(dim=-1)
    
    def evaluate(self, critic_observations, ref_critic_observations=None, **kwargs):
        """Evaluate value function"""
        return self.critic(critic_observations, ref_critic_observations, **kwargs)
    
    def load_actor_weights(self, path: str):
        """Load pretrained actor weights"""
        state_dict = torch.load(path, map_location="cpu")
        assert 'model_state_dict' in state_dict.keys(), \
            f"Key 'model_state_dict' not found in {path}"
        
        model_state_dict = state_dict['model_state_dict']
        
        # Extract actor and critic weights
        actor_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor.')}
        critic_weights = {k[len('critic.'):]: v for k, v in model_state_dict.items() if k.startswith('critic.')}
        
        # Check what parameters will be loaded/missing
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        
        # Report missing parameters
        missing_actor_keys = set(actor_state_dict.keys()) - set(actor_weights.keys())
        missing_critic_keys = set(critic_state_dict.keys()) - set(critic_weights.keys())
        
        if missing_actor_keys:
            print(f"[Actor] Missing keys (will use random initialization):")
            for k in sorted(missing_actor_keys):
                print(f"  - {k}")
        
        if missing_critic_keys:
            print(f"[Critic] Missing keys (will use random initialization):")
            for k in sorted(missing_critic_keys):
                print(f"  - {k}")
        
        # Load with shape checking
        self.actor.load_state_dict(actor_weights, strict=False)
        self.critic.load_state_dict(critic_weights, strict=False)
        
        # Load std
        if self.noise_std_type == "scalar" and "std" in model_state_dict:
            self.std.data = model_state_dict["std"]
        elif self.noise_std_type == "log" and "log_std" in model_state_dict:
            self.log_std.data = model_state_dict["log_std"]
        
        print(f"Loaded actor weights from {path}")
    
    def load_dagger_weights(self, path: str):
        """Load dagger (teacher) weights"""
        state_dict = torch.load(path, map_location="cpu")
        assert 'model_state_dict' in state_dict.keys(), \
            f"Key 'model_state_dict' not found in {path}"
        
        model_state_dict = state_dict['model_state_dict']
        dagger_weights = {k[len('actor.'):]: v for k, v in model_state_dict.items() if k.startswith('actor.')}
        
        # Check what parameters will be loaded/missing
        dagger_state_dict = self.actor_dagger.state_dict()
        
        # Report missing and unexpected parameters
        missing_keys = set(dagger_state_dict.keys()) - set(dagger_weights.keys())
        
        if missing_keys:
            print(f"[Actor Dagger] Missing keys (will use random initialization):")
            for k in sorted(missing_keys):
                print(f"  - {k}")
        
        self.actor_dagger.load_state_dict(dagger_weights, strict=False)
        
        # Load std for dagger
        if self.noise_std_type == "scalar" and "std" in model_state_dict:
            self.std_dagger.data = model_state_dict["std"]
            self.std_dagger.requires_grad = False
        elif self.noise_std_type == "log" and "log_std" in model_state_dict:
            self.log_std_dagger.data = model_state_dict["log_std"]
            self.log_std_dagger.requires_grad = False
        
        print(f"Loaded dagger weights from {path}")
