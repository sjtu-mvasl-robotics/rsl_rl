# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import sys
from rsl_rl.utils import resolve_nn_activation
from typing import Callable, Any


def _build_mlp(
        input_dim: int,
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        hidden_dims: list[int] = [512, 256],
        device: str = "cpu",
        **kwargs
) -> nn.Module:
    layers = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers).to(device)


AVAILABLE_BACKBONES = {
    "mlp": _build_mlp,
}

def build_backbone(
        input_dim: int,
        output_dim: int,
        backbone: str = "mlp",
        activation: str = "elu",
        device: str = "cpu",
        **kwargs
) -> nn.Module:
    if backbone not in AVAILABLE_BACKBONES.keys():
        raise ValueError(f"Invalid backbone: {backbone}. Available backbones: {AVAILABLE_BACKBONES.keys()}. To add custom backbones, register them in {__name__}.AVAILABLE_BACKBONES.")
    return AVAILABLE_BACKBONES[backbone](
        input_dim=input_dim,
        output_dim=output_dim,
        activation=resolve_nn_activation(activation),
        device=device,
        **kwargs
    )
