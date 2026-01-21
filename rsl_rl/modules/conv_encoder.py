# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Conv-based temporal encoders inspired by ASAP's design.
Used for encoding history observations into compact latent representations.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for temporal sequences.
    
    Architecture:
    1. Linear per-timestep encoding: (B*T, input_dim) → (B*T, hidden_dim)
    2. Reshape and permute: (B*T, hidden_dim) → (B, hidden_dim, T)
    3. Conv1D for temporal compression: (B, hidden_dim, T) → (B, out_channels[-1], T')
    4. Flatten and output projection: (B, out_channels[-1] * T') → (B, output_dim)
    
    Inspired by ASAP's design for history encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        time_steps: int,
        activation: str = "elu",
        use_adaptive_config: bool = True,
        custom_conv_config: Optional[dict] = None,
    ):
        """
        Args:
            input_dim: Dimension of input features per timestep
            output_dim: Dimension of output latent representation
            hidden_dim: Hidden dimension for per-timestep encoding
            time_steps: Number of timesteps in history
            activation: Activation function name
            use_adaptive_config: Auto-configure conv layers based on time_steps
            custom_conv_config: Manual conv config with keys:
                - out_channels: List[int]
                - kernel_sizes: List[int]
                - strides: List[int]
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        
        # Per-timestep encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation)
        )
        
        # Conv layers for temporal compression
        if use_adaptive_config:
            conv_config = self._get_adaptive_conv_config(time_steps)
        else:
            if custom_conv_config is None:
                raise ValueError("custom_conv_config must be provided when use_adaptive_config=False")
            conv_config = custom_conv_config
        
        self.conv_module = self._build_conv_layers(
            hidden_dim,
            conv_config["out_channels"],
            conv_config["kernel_sizes"],
            conv_config["strides"],
            activation
        )
        
        # Calculate output size after conv
        conv_output_length = self._calculate_conv_output_length(
            time_steps,
            conv_config["kernel_sizes"],
            conv_config["strides"]
        )
        
        # Output projection
        final_features = conv_config["out_channels"][-1] * conv_output_length
        self.output_layer = nn.Linear(final_features, output_dim)
    
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
    
    def _get_adaptive_conv_config(self, tsteps: int) -> dict:
        """
        Auto-configure conv layers based on time_steps.
        Inspired by ASAP's adaptive configuration.
        """
        if tsteps <= 5:
            # Short history: gentle compression
            out_channels = [32, 16]
            kernel_sizes = [2, 2]
            strides = [1, 1]
        elif tsteps <= 10:
            # Medium history
            out_channels = [32, 16]
            kernel_sizes = [3, 2]
            strides = [2, 1]
        elif tsteps <= 20:
            # Long history: more aggressive compression
            out_channels = [64, 32]
            kernel_sizes = [4, 3]
            strides = [2, 2]
        else:
            # Very long history
            out_channels = [64, 32, 16]
            kernel_sizes = [5, 3, 2]
            strides = [2, 2, 1]
        
        return {
            "out_channels": out_channels,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
        }
    
    def _build_conv_layers(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        activation: str
    ) -> nn.Module:
        """Build sequential conv layers"""
        layers = []
        in_ch = in_channels
        
        act_fn = self._get_activation(activation)
        
        for out_ch, kernel_size, stride in zip(out_channels, kernel_sizes, strides):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride))
            layers.append(self._get_activation(activation))  # New instance each time
            in_ch = out_ch
        
        return nn.Sequential(*layers)
    
    def _calculate_conv_output_length(
        self,
        input_length: int,
        kernel_sizes: List[int],
        strides: List[int],
        paddings: Optional[List[int]] = None
    ) -> int:
        """Calculate output sequence length after conv layers"""
        if paddings is None:
            paddings = [0] * len(kernel_sizes)
        
        length = input_length
        for kernel_size, stride, padding in zip(kernel_sizes, strides, paddings):
            length = (length + 2 * padding - kernel_size) // stride + 1
        
        return length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T * input_dim) or (B, T, input_dim)
        
        Returns:
            Encoded tensor of shape (B, output_dim)
        """
        B = x.shape[0]
        
        # Handle both flat and shaped inputs
        if x.dim() == 2:
            # Flat input: (B, T * input_dim)
            x = x.view(B * self.time_steps, self.input_dim)
        else:
            # Shaped input: (B, T, input_dim)
            x = x.view(B * self.time_steps, self.input_dim)
        
        # Per-timestep encoding: (B*T, input_dim) → (B*T, hidden_dim)
        x = self.encoder(x)
        
        # Reshape for Conv1D: (B*T, hidden_dim) → (B, hidden_dim, T)
        x = x.view(B, self.time_steps, self.hidden_dim).permute(0, 2, 1)
        
        # Temporal convolution: (B, hidden_dim, T) → (B, out_channels[-1], T')
        x = self.conv_module(x)
        
        # Flatten and project: (B, out_channels[-1] * T') → (B, output_dim)
        x = x.flatten(start_dim=1)
        x = self.output_layer(x)
        
        return x


class PrivilegeEncoder(nn.Module):
    """
    MLP encoder for privilege observations.
    Used as teacher signal for history encoder training.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64],
        activation: str = "elu",
        use_layer_norm: bool = False,
    ):
        """
        Args:
            input_dim: Dimension of privilege observations
            output_dim: Dimension of output (should match HistoryEncoder output)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            use_layer_norm: Whether to use LayerNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Privilege observations of shape (B, input_dim)
        
        Returns:
            Encoded privilege features of shape (B, output_dim)
        """
        return self.network(x)


class MotionEncoder(nn.Module):
    """
    Encoder for future motion targets.
    Similar to HistoryEncoder but for future predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        time_steps: int,
        activation: str = "elu",
    ):
        """
        Args:
            input_dim: Dimension of motion targets per timestep
            output_dim: Dimension of output latent representation
            hidden_dim: Hidden dimension for per-timestep encoding
            time_steps: Number of future timesteps
            activation: Activation function name
        """
        super().__init__()
        
        # Reuse ConvEncoder implementation
        self.encoder = ConvEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            time_steps=time_steps,
            activation=activation,
            use_adaptive_config=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Future motion targets of shape (B, T * input_dim) or (B, T, input_dim)
        
        Returns:
            Encoded motion features of shape (B, output_dim)
        """
        return self.encoder(x)
