# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Residual Mixture-of-Experts variant of ActorCriticMLPV2.

Adds residual expert networks + soft gating on top of the base MLP actor.
Inspired by MoRE (TeleHuman/MoRE) and GMT/EGM motion tracking architectures.

Key idea: base actor provides a stable backbone, experts provide motion-specific
residual corrections, gating network routes based on ref obs (motion identity).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .actor_critic_mlp_v2 import (
    ActorCriticMLPV2,
    FusedMultiModalMLP,
    MultiModalMLPV2,
)


class MultiModalMLPV2MoE(MultiModalMLPV2):
    """MultiModalMLPV2 with residual MoE experts.
    
    Architecture:
        encoded_obs -> base main_mlp -> actor_feature
        encoded_obs -> expert_i_mlp -> expert_feature_i  (for i in num_experts)
        residual = sum(gate_weight_i * expert_feature_i)
        output = output_layer(actor_feature + residual)
    
    The gate_net is NOT here — it lives in FusedMultiModalMLPMoE where it can
    access ref_obs to determine motion identity.
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
        reorgnize_obs: bool = False,
        name: str = "",
        no_pooling: bool = True,
        # MoE params
        num_experts: int = 4,
        expert_hidden_dims: list[int] = [256, 256],
    ):
        # Initialize base class (builds main_mlp + output_layer)
        super().__init__(
            term_dict=term_dict,
            output_size=output_size,
            hidden_dims=hidden_dims,
            activation=activation,
            history_length=history_length,
            encoder_latent_dim=encoder_latent_dim,
            encoder_compress_threshold=encoder_compress_threshold,
            use_layer_norm=use_layer_norm,
            reorgnize_obs=reorgnize_obs,
            name=name,
            no_pooling=no_pooling,
        )

        self.num_experts = num_experts

        # Compute the input dim that main_mlp sees
        if self.temporal_encoder is not None:
            total_non_history = sum(dim for _, dim in self.non_history_dims)
            expert_input_dim = self.temporal_encoder.output_dim + total_non_history
        else:
            expert_input_dim = self.total_obs_dim

        # Expert feature dim = last hidden dim of base actor (before output_layer)
        expert_output_dim = hidden_dims[-1] if hidden_dims else expert_input_dim

        # Build expert networks: each maps encoded_obs -> expert_feature
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers = []
            prev_dim = expert_input_dim
            for h_dim in expert_hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h_dim))
                layers.append(self.activation)
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, expert_output_dim))
            self.experts.append(nn.Sequential(*layers))

    def forward_with_experts(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning base feature, expert features, and encoded obs.
        
        Returns:
            actor_feature: (B, hidden_dims[-1]) - base actor backbone output
            expert_features: (B, num_experts, hidden_dims[-1]) - each expert's output
            encoded_obs: (B, encoded_dim) - for gate_net input
        """
        # Same encoding as parent forward()
        if self.temporal_encoder is not None:
            temporal_features = self.temporal_encoder(obs)
            non_history_features = []
            start = 0
            for i, term_dim in enumerate(self.term_dims):
                if (i, term_dim) in [(idx, dim) for idx, dim in self.non_history_dims]:
                    non_history_features.append(obs[:, start:start + term_dim])
                start += term_dim
            if non_history_features:
                non_history = torch.cat(non_history_features, dim=-1)
                encoded_obs = torch.cat([temporal_features, non_history], dim=-1)
            else:
                encoded_obs = temporal_features
        else:
            encoded_obs = obs

        # Base actor backbone
        actor_feature = self.main_mlp(encoded_obs)  # (B, hidden_dims[-1])

        # Expert features
        expert_features = torch.stack(
            [expert(encoded_obs) for expert in self.experts], dim=1
        )  # (B, num_experts, hidden_dims[-1])

        return actor_feature, expert_features, encoded_obs

    def forward(self, obs: torch.Tensor, gate_weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Forward with optional MoE gating.
        
        If gate_weights is None, falls back to base MLPv2 behavior (no experts).
        """
        if gate_weights is None:
            return super().forward(obs, **kwargs)

        actor_feature, expert_features, _ = self.forward_with_experts(obs)

        # Weighted sum of expert features: (B, num_experts, 1) * (B, num_experts, D) -> sum -> (B, D)
        residual = torch.sum(gate_weights.unsqueeze(-1) * expert_features, dim=1)

        # Add residual to base actor feature
        output = self.output_layer(actor_feature + residual)
        return output


class FusedMultiModalMLPMoE(FusedMultiModalMLP):
    """FusedMultiModalMLP with MoE gating conditioned on ref observations.
    
    The gate_net takes encoded main obs + ref obs encoding to determine
    which experts to activate. This allows the gate to route based on
    "what motion is being tracked" (encoded in ref_obs).
    """

    def __init__(
        self,
        term_dict,
        ref_term_dict,
        output_size: int,
        hidden_dims: list[int],
        activation: str,
        history_length: int,
        encoder_latent_dim: int,
        encoder_compress_threshold: int,
        use_layer_norm: bool,
        fusion_mode: str,
        name: str = "fused_mlp_moe",
        fuse_activation: bool = False,
        reorgnize_obs: bool = False,
        # MoE params
        num_experts: int = 4,
        expert_hidden_dims: list[int] = [256, 256],
        gate_hidden_dims: list[int] = [128, 64],
    ):
        # DON'T call super().__init__ — we need to replace main_net with MoE version
        nn.Module.__init__(self)

        self.reorgnize_obs = reorgnize_obs
        self.history_length = history_length
        self.fusion_mode = fusion_mode
        self.fuse_activation = fuse_activation
        self.has_ref = ref_term_dict is not None and bool(list(ref_term_dict.values())[0])
        self.num_experts = num_experts

        # Main network: MoE version
        self.main_net = MultiModalMLPV2MoE(
            term_dict=term_dict,
            output_size=output_size,
            hidden_dims=hidden_dims,
            activation=activation,
            history_length=history_length,
            encoder_latent_dim=encoder_latent_dim,
            encoder_compress_threshold=encoder_compress_threshold,
            use_layer_norm=use_layer_norm,
            name=f"{name}_main",
            reorgnize_obs=reorgnize_obs,
            num_experts=num_experts,
            expert_hidden_dims=expert_hidden_dims,
        )

        self.activation = self._get_activation(activation)

        # Reference network (same as parent — no MoE needed for ref)
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
                name=f"{name}_ref",
                reorgnize_obs=reorgnize_obs,
            )

            # Fusion layers (same as parent)
            if fusion_mode == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(output_size * 2, output_size),
                    nn.Sigmoid(),
                )
            elif fusion_mode == "concat":
                self.fusion_proj = nn.Linear(output_size * 2, output_size)
        else:
            self.ref_net = None

        # Gate network for MoE routing
        # Input: encoded main obs + ref_obs (flattened)
        # Compute encoded_obs dim from main_net
        if self.main_net.temporal_encoder is not None:
            total_non_history = sum(dim for _, dim in self.main_net.non_history_dims)
            encoded_obs_dim = self.main_net.temporal_encoder.output_dim + total_non_history
        else:
            encoded_obs_dim = self.main_net.total_obs_dim

        # Compute ref_obs dim from ref_term_dict
        # In forward, gate sees raw ref_obs (possibly with history if reorgnize_obs)
        ref_obs_dim = 0
        if self.has_ref and ref_term_dict is not None:
            ref_terms = list(ref_term_dict.values())[0]  # e.g. {"term_name": dim, ...}
            single_ref_dim = sum(ref_terms.values())
            # When reorgnize_obs, ref_obs gets stacked into (B, history_length, ref_dim)
            # then flattened to (B, history_length * ref_dim) for gate input
            ref_obs_dim = single_ref_dim * history_length if reorgnize_obs else single_ref_dim

        gate_input_dim = encoded_obs_dim + ref_obs_dim
        act_fn = self._get_activation(activation)
        gate_layers = []
        prev_dim = gate_input_dim
        for h_dim in gate_hidden_dims:
            gate_layers.append(nn.Linear(prev_dim, h_dim))
            gate_layers.append(act_fn)
            prev_dim = h_dim
        gate_layers.append(nn.Linear(prev_dim, num_experts))
        self.gate_net = nn.Sequential(*gate_layers)

        self.obs_buffer = None
        self.ref_obs_buffer = None



    def forward(
        self,
        observations: torch.Tensor,
        ref_observations: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Handle reorgnize_obs (MMGPT compatibility) same as parent
        if self.reorgnize_obs:
            if len(observations.shape) == 2:
                self._update_buffer(observations)
                observations = self.obs_buffer.permute(1, 0, 2)
                if ref_observations is not None:
                    ref_obs, ref_mask = ref_observations
                    self._update_ref_buffer(ref_obs)
                    ref_observations = (self.ref_obs_buffer.permute(1, 0, 2), ref_mask)
            else:
                observations = observations.permute(1, 0, 2)
                ref_observations = (
                    None
                    if ref_observations is None
                    else (ref_observations[0].permute(1, 0, 2), ref_observations[1][-1])
                )

        # Get base actor feature + expert features + encoded obs from MoE main_net
        actor_feature, expert_features, encoded_obs = self.main_net.forward_with_experts(
            observations
        )

        # Build gate input: encoded_obs + ref_obs (if available)
        if self.has_ref and ref_observations is not None:
            ref_obs, ref_mask = ref_observations
            # Flatten ref_obs if needed
            if ref_obs.dim() > 2:
                ref_obs_flat = ref_obs.reshape(ref_obs.shape[0], -1)
            else:
                ref_obs_flat = ref_obs
            gate_input = torch.cat([encoded_obs, ref_obs_flat], dim=-1)
        else:
            gate_input = encoded_obs

        gate_weights = F.softmax(self.gate_net(gate_input), dim=-1)  # (B, num_experts)

        # Combine: actor_feature + weighted expert residual
        residual = torch.sum(
            gate_weights.unsqueeze(-1) * expert_features, dim=1
        )  # (B, hidden_dims[-1])
        main_output = self.main_net.output_layer(actor_feature + residual)

        # Fuse with ref network (same as parent FusedMultiModalMLP)
        if not self.has_ref or ref_observations is None:
            return main_output

        ref_obs, ref_mask = ref_observations
        ref_output = self.ref_net(ref_obs)

        if ref_mask is not None:
            ref_mask = ref_mask.unsqueeze(-1).float()
            ref_output = ref_output * ref_mask

        if self.fusion_mode == "gated":
            gate_input_fuse = torch.cat([main_output, ref_output], dim=-1)
            gate_fuse = self.gate(gate_input_fuse)
            output = main_output * (1 - gate_fuse) + ref_output * gate_fuse
        elif self.fusion_mode == "add":
            output = main_output + ref_output
        elif self.fusion_mode == "concat":
            concat = torch.cat([main_output, ref_output], dim=-1)
            output = self.fusion_proj(concat)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        if self.fuse_activation:
            output = self.activation(output)
        return output


class ActorCriticMLPV2MoE(ActorCriticMLPV2):
    """ActorCriticMLPV2 with Residual MoE on the actor network.
    
    Only the actor uses MoE experts. The critic remains unchanged.
    Config usage: set class_name="ActorCriticMLPV2MoE" and add moe params.
    """

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
        fusion_mode: str = "gated",
        load_dagger: bool = False,
        load_dagger_path: Optional[str] = None,
        load_actor_path: Optional[str] = None,
        load_critic_path: Optional[str] = None,
        # MoE params
        num_experts: int = 4,
        expert_hidden_dims: list[int] = [256, 256],
        gate_hidden_dims: list[int] = [128, 64],
        **kwargs,
    ):
        # Call nn.Module init directly — we'll build everything ourselves
        # to avoid parent creating a non-MoE actor
        nn.Module.__init__(self)

        from torch.distributions import Normal

        assert not load_dagger or load_dagger_path, \
            "load_dagger and load_dagger_path must be provided if load_dagger is True"

        # Extract actor and critic term dicts
        actor_term_dict = {"policy": term_dict.get("policy", {})}
        critic_term_dict = {"critic": term_dict.get("critic", term_dict.get("policy", {}))}

        actor_ref_term_dict = (
            {"policy": ref_term_dict.get("policy", {})} if ref_term_dict else None
        )
        critic_ref_term_dict = (
            {"critic": ref_term_dict.get("critic", ref_term_dict.get("policy", {}))}
            if ref_term_dict
            else None
        )

        # Actor network: MoE version
        self.actor = FusedMultiModalMLPMoE(
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
            name="actor_moe",
            num_experts=num_experts,
            expert_hidden_dims=expert_hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
        )

        # Dagger: not supported for MoE yet
        self.actor_dagger = None
        if load_dagger:
            print("Warning: load_dagger not supported for ActorCriticMLPV2MoE, ignoring.")

        # Critic network: standard (no MoE needed for value function)
        self.critic = FusedMultiModalMLP(
            term_dict=critic_term_dict,
            ref_term_dict=critic_ref_term_dict,
            output_size=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            history_length=history_length,
            encoder_latent_dim=encoder_latent_dim,
            encoder_compress_threshold=encoder_compress_threshold,
            use_layer_norm=use_layer_norm,
            fusion_mode=fusion_mode,
            name="critic",
        )

        # Action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        self.distribution_dagger = None

        # Load weights if specified
        if load_actor_path:
            self.load_actor_weights(load_actor_path)
        if load_critic_path:
            self.load_critic_weights(load_critic_path)

        Normal.set_default_validate_args(False)

        print(f"ActorCriticMLPV2MoE: num_experts={num_experts}, "
              f"expert_hidden_dims={expert_hidden_dims}, "
              f"gate_hidden_dims={gate_hidden_dims}")
        print(f"Actor MoE: {self.actor}")
        print(f"Critic: {self.critic}")
