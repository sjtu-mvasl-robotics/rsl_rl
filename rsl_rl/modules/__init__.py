# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .actor_critic_mm_transformer import ActorCriticMMTransformer, ActorCriticMMTransformerV2, ActorCriticDebugMLP, SwiGLUEmbedding, group_by_concat_list
from .actor_critic_mm_gpt import ActorCriticMMGPT
from .actor_critic_mlp import ActorCriticMLP, MultiModalMLP
from .actor_critic_mlp_v2 import ActorCriticMLPV2, MultiModalMLPV2
from .student_teacher_transformer import StudentTeacherMMTransformer, StudentTeacherMMTransformerV2
from .amp import AMPNet

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "ActorCriticMMTransformer",
    "ActorCriticMMTransformerV2",
    "group_by_concat_list",
    "SwiGLUEmbedding",
    "ActorCriticDebugMLP",
    "ActorCriticMLP",
    "MultiModalMLP",
    "ActorCriticMLPV2",
    "MultiModalMLPV2",
    "StudentTeacherMMTransformer",
    "StudentTeacherMMTransformerV2",
    "AMPNet",
    "ActorCriticMMGPT",
]
