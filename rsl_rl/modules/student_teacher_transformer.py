# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.actor_critic_mm_transformer import MMTransformer, MMTransformerV2

class StudentTeacherMMTransformer(nn.Module):
    is_recurrent = False
    def __init__(
            self,
            num_teacher_obs,
            num_teacher_ref_obs,
            num_student_obs,
            num_student_ref_obs,
            num_actions,
            max_len=16,
            dim_model=128,
            num_layers=4,
            num_heads=8,
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            dropout=0.05,
            **kwargs
    ):
        super().__init__()

        self.load_teacher = False
        
        
        self.student = MMTransformer(num_student_obs, num_student_ref_obs, num_actions, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="student", dropout=dropout, **kwargs)

        self.teacher = MMTransformer(num_teacher_obs, num_teacher_ref_obs, num_actions, dim_model, max_len=max_len, num_heads=num_heads, num_layers=num_layers, name="teacher", dropout=dropout, **kwargs)

        self.teacher.eval()
            
        print(f"Student Transformer: {self.student}")
        print(f"Teacher Transformer: {self.teacher}")
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        

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

    def update_distribution(self, observations, ref_observations=None):
        mean = self.actor(observations, ref_observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, ref_observations=None, **kwargs):
        self.update_distribution(observations, ref_observations)
        sample = self.distribution.sample()
        return sample

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, ref_observations=None):
        actions_mean = self.student(observations, ref_observations)
        return actions_mean

    def evaluate(self, teacher_observations, ref_teacher_observations =None, **kwargs):
        with torch.no_grad():
            actions = self.teacher(teacher_observations, ref_teacher_observations)
        return actions
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
