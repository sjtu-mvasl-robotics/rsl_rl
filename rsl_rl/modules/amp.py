# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from rsl_rl.modules.normalizer import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation, build_backbone

""" Adversial Motion Prior (AMP): https://arxiv.org/pdf/2104.02180

AMPNet takes in [cur_state, next_state] and outputs a score for action classification (-1: not expert, 1: expert). It is basically a GAN discriminator.
"""

class HingeLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum"], "reduction must be either mean or sum"
        self.classifier_activation = nn.Tanh()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = self.classifier_activation(x)
        loss = (1 - x * label).relu()
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss

class AMPNet(nn.Module):
    def __init__(
            self,
            backbone_input_dim: int,
            backbone_output_dim: int,
            backbone: str = "mlp",
            activation: str = "elu",
            out_activation: str = "tanh",
            device: str = "cpu",
            label_smoothing: float = 0.0,
            **kwargs
    ):
        super().__init__()
        assert out_activation in ["tanh", "sigmoid"], "out_activation must be either tanh or sigmoid"

        if "net_kwargs" in kwargs:
            net_kwargs = kwargs.pop("net_kwargs")
        else:
            net_kwargs = {}

        self.backbone = build_backbone(
            input_dim=backbone_input_dim,
            output_dim=backbone_output_dim,
            backbone=backbone,
            activation=activation,
            device=device,
            **net_kwargs
        )

        self.out_layer = nn.Linear(backbone_output_dim, 1)
        self.out_activation = resolve_nn_activation(out_activation)
        self.policy_score = -1 if out_activation == "tanh" else 0 # sigmoid has a range of [0, 1]
        self.label_smoothing = label_smoothing
        self.expert_score = 1
        self.loss_fn = HingeLoss(reduction="none") if out_activation == "tanh" else nn.BCEWithLogitsLoss(reduction="none") # hinge loss for tanh, binary cross entropy for sigmoid
        self.to(device)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.out_layer(x) # predicted score for AMP
        return x
    
    def policy_loss(self, y: torch.Tensor) -> torch.Tensor:
        tgt_score = torch.ones_like(y) * (self.policy_score + self.label_smoothing)
        loss = self.loss_fn(y, tgt_score) # shape: (num_envs, 1)
        return loss.mean()
    
    def policy_acc(self, y: torch.Tensor) -> torch.Tensor:
        pred = self.out_activation(y)
        tgt = torch.ones_like(y) * self.policy_score
        bound = (self.policy_score + self.expert_score) / 2
        acc = ((pred < bound).float() == (tgt < bound).float()).float().mean()
        return acc
    
    def expert_loss(self, y: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the expert's score.

        Parameters:
            y: torch.Tensor, shape: (num_envs, 1)
            tgt_mask: torch.Tensor, shape: (num_envs, )

        Returns:
            loss: torch.Tensor, shape: (1,)
        """
        tgt_score = torch.ones_like(y) * (self.expert_score - self.label_smoothing)
        loss = self.loss_fn(y, tgt_score)
        tgt_mask = tgt_mask.unsqueeze(-1).float()
        loss = (loss * tgt_mask).sum() / (tgt_mask.sum() + 1e-6)
        return loss
    
    def expert_acc(self, y: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        pred = self.out_activation(y)
        tgt = torch.ones_like(y) * (self.expert_score - self.label_smoothing)
        bound = (self.policy_score + self.expert_score) / 2
        acc = ((pred > bound).float() == (tgt > bound).float()).float() # shape: (num_envs, )
        tgt_mask = tgt_mask.unsqueeze(-1).float()
        acc = (acc * tgt_mask).sum() / (tgt_mask.sum() + 1e-6)
        return acc
    
    def expert_grad_penalty(self, expert_cur_state: torch.Tensor, expert_next_state: torch.Tensor, expert_available_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient penalty of the expert's score with respect to the current state.

        AMPNet takes in [cur_state, next_state] and outputs a score for action classification (-1: not expert, 1: expert). This function records the gradient of the score with respect to input states. The mask is used to mask out environments where expert is not available:
        
        Parameters:
            expert_cur_state: torch.Tensor, shape: (num_envs, state_dim)
            expert_next_state: torch.Tensor, shape: (num_envs, state_dim)
            expert_available_mask: torch.Tensor, shape: (num_envs, 1)

        Returns:
            expert_grad_penalty: torch.Tensor, shape: (1,)
        """
        expert_available_mask = expert_available_mask.unsqueeze(-1)
        assert expert_cur_state.shape == expert_next_state.shape
        expert_input = torch.cat([expert_cur_state, expert_next_state], dim=-1)
        expert_input.requires_grad = True

        expert_score = self.forward(expert_input)
        ones = torch.ones_like(expert_score)
        ones = ones * expert_available_mask

        expert_grad = grad(
            outputs=expert_score,
            inputs=expert_input,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_penalty_raw = (expert_grad.norm(p=2, dim=-1) - 0).pow(2) # shape: (num_envs, 1)
        grad_penalty = grad_penalty_raw.sum() / (expert_available_mask.sum() + 1e-6) # shape: (1,)

        return grad_penalty

    def amp_reward(self, cur_state: torch.Tensor, next_state: torch.Tensor, epsilon: float = 1e-4, reward_shift: float = 0.55) -> torch.Tensor:
        """
        Compute the AMP reward for the given current and next states.

        The reward is computed as the saturated cross entropy between the predicted score and the expert's score (1: expert, -1: not expert).

        Parameters:
            cur_state: torch.Tensor, shape: (num_envs, state_dim)
            next_state: torch.Tensor, shape: (num_envs, state_dim)
            epsilon: float, the threshold for the saturated cross entropy
            reward_shift: float, the shift of the reward (we tend to avoid positive reward when p_policy is close to 0 in order to avoid rapid reward growth)
        Returns:
            amp_reward: torch.Tensor, shape: (1,)
        """
        assert cur_state.shape == next_state.shape
        with torch.no_grad():
            expert_score = self.forward(torch.cat([cur_state, next_state], dim=-1)) # shape: (num_envs, 1)
            expert_score = self.out_activation(expert_score) # shape: (num_envs, 1)

            reward = -torch.log(
                (self.expert_score - expert_score - self.label_smoothing).clamp(min=epsilon)
            ) # shape: (num_envs, 1)

            reward -= reward_shift
            reward = reward.clamp(min=0.0)
        
        return reward.squeeze(-1)
    
    def amp_score(self, cur_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Compute the AMP score for the given current and next states.
        """
        assert cur_state.shape == next_state.shape
        with torch.no_grad():
            score = self.forward(torch.cat([cur_state, next_state], dim=-1)) # shape: (num_envs, 1)
            score = self.out_activation(score) # shape: (num_envs, 1)
            return score.squeeze(-1)