# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural networks."""

from .memory import Memory
from .rope import RotaryEmbedding
from .rope_transformer import RoPETransformer, RoPETransformerEncoder, RoPETransformerEncoderLayer, SwiGLUFFN
from .rope_transformer import RMSNorm as RoPERMSNorm  # Avoid conflict with modules.actor_critic_mm_transformer.RMSNorm

__all__ = ["Memory", "RotaryEmbedding", "RoPETransformer", "RoPETransformerEncoder", "RoPETransformerEncoderLayer", "SwiGLUFFN", "RoPERMSNorm"]
