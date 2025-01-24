#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticMMTransformer
from rsl_rl.storage import RolloutStorageMM
import time


class MMPPO:
    actor_critic: ActorCriticMMTransformer

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        ref_action_idx=0,
        teacher_coef=None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.ref_action_idx = ref_action_idx
        self.teacher_coef = teacher_coef

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorageMM.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, actor_ref_obs_shape, critic_obs_shape, critic_ref_obs_shape, action_shape):
        self.storage = RolloutStorageMM(
            num_envs,
            num_transitions_per_env,
            obs_shape=actor_obs_shape,
            ref_obs_shape=actor_ref_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            privileged_ref_obs_shape=critic_ref_obs_shape,
            actions_shape=action_shape,
            device=self.device,
        )


    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, ref_obs, critic_obs, ref_critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, ref_observations=ref_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, ref_critic_observations=ref_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.reference_observations = ref_obs[0] if ref_obs is not None else None
        self.transition.reference_observations_mask = ref_obs[1] if ref_obs is not None else None
        self.transition.critic_observations = critic_obs 
        self.transition.critic_reference_observations = ref_critic_obs[0] if ref_critic_obs is not None else None
        self.transition.critic_reference_observations_mask = ref_critic_obs[1] if ref_critic_obs is not None else None
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        assert self.storage is not None, "Storage is not initialized. Please call init_storage before process_env_step."
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        assert self.storage is not None, "Storage is not initialized. Please call init_storage before compute_returns."
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = torch.tensor(0.0).to(self.device)
        mean_surrogate_loss = torch.tensor(0.0).to(self.device)
        mean_imitation_loss = torch.tensor(0.0).to(self.device)
        # if self.actor_critic.is_recurrent:
        #     generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs) #
        # else:
        assert self.actor_critic.is_recurrent == False, "MM-PPO does not support recurrent actor-critic networks."
        assert self.storage is not None, "Storage is not initialized. Please call init_storage before update."
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            ref_obs_batch,
            critic_obs_batch,
            critic_ref_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, ref_observations=ref_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, ref_critic_observations=critic_ref_obs_batch
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Imitation Entropy loss (optional, arxiv: 2409.08904)
            # This loss is calculated only when critic_ref_obs_batch is not None, otherwise 0.0
            # By default, we assume that ref_action_batch = critic_ref_obs_batch[0][:, ref_action_idx: num_actions + ref_action_idx] where ref_action_idx = 0
            # Loss: \sum_i [\sqrt(2 * \pi * \sigma_i^2) + (mu_i - ref_action_i)^2 / (2 * \sigma_i^2)]
            if self.teacher_coef is not None and critic_ref_obs_batch is not None:
                ref_action_batch = critic_ref_obs_batch[0][:, self.ref_action_idx: self.ref_action_idx + actions_batch.shape[-1]]
                ref_action_mask = critic_ref_obs_batch[1].float()
                imitation_loss = torch.sum((
                    (1 / actions_batch.shape[-1]) * (torch.sum((torch.square(mu_batch - ref_action_batch) / (2.0 * torch.square(sigma_batch) + 1e-5)), axis=-1) + torch.sum(torch.log(sigma_batch), axis=-1))
                ) * ref_action_mask) / (ref_action_mask.sum() + 1e-5)
                imitation_loss = self.teacher_coef * imitation_loss
            else:
                imitation_loss = 0.0     

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + imitation_loss
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss
            mean_surrogate_loss += surrogate_loss
            mean_imitation_loss += imitation_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates
        self.storage.clear()

        return mean_value_loss.item(), mean_surrogate_loss.item(), mean_imitation_loss.item()
