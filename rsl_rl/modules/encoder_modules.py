# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Encoder modules for temporal and spatial feature extraction.

These modules are used in PPOMimic for encoding:
- History observations (proprioceptive history)
- Motion targets (future motion reference)
- Privilege observations
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class ConvEncoder(nn.Module):
    """
    1D Convolutional encoder for temporal sequences.
    
    Used for encoding:
    - Proprioceptive history: [batch, history_length * obs_dim]
    - Motion targets: [batch, num_future_steps * target_dim]
    
    Architecture:
    Input -> Conv1d layers -> Flatten -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_timesteps: int,
        activation: str = "SiLU",
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim: Dimension of input features per timestep
            hidden_dim: Hidden dimension for conv layers
            output_dim: Output embedding dimension
            num_timesteps: Number of timesteps in the sequence
            activation: Activation function name
            num_layers: Number of conv layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        
        # Get activation function
        activation_upper = activation.upper()
        if activation_upper == "SILU":
            act_fn = nn.SiLU()
        elif activation_upper == "RELU":
            act_fn = nn.ReLU()
        elif activation_upper == "ELU":
            act_fn = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build conv layers
        layers = []
        
        # First conv layer: input_dim -> hidden_dim
        layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1))
        layers.append(act_fn)
        
        # Additional conv layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(act_fn)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output projection: hidden_dim * num_timesteps -> output_dim
        self.output_proj = nn.Linear(hidden_dim * num_timesteps, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, input_dim * num_timesteps]
            
        Returns:
            Encoded tensor of shape [batch, output_dim]
        """
        batch_size = x.shape[0]
        
        # Reshape to [batch, num_timesteps, input_dim]
        x = x.view(batch_size, self.num_timesteps, self.input_dim)
        
        # Transpose to [batch, input_dim, num_timesteps] for Conv1d
        x = x.transpose(1, 2)
        
        # Apply conv layers: [batch, hidden_dim, num_timesteps]
        x = self.conv_layers(x)
        
        # Flatten: [batch, hidden_dim * num_timesteps]
        x = x.flatten(1)
        
        # Project to output: [batch, output_dim]
        x = self.output_proj(x)
        
        return x


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder.
    
    Used for encoding privilege observations or other non-temporal features.
    
    Architecture:
    Input -> MLP layers -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64],
        activation: str = "SiLU",
        use_layernorm: bool = False,
    ):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            use_layernorm: Whether to use layer normalization
        """
        super().__init__()
        
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
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, input_dim]
            
        Returns:
            Encoded tensor of shape [batch, output_dim]
        """
        return self.mlp(x)
