# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO with Mimic Learning (PPOMimic)

Implements a three-stage training process:
1. Teacher-only stage: Train with privileged information (teacher)
2. Hybrid stage: Combine teacher and student with DAgger updates
3. Student stage: Fully deploy with history encoder

This implementation is based on PBHC/humanoidverse's ppo_mimic.py
and adapted to rsl_rl's interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from rsl_rl.storage import RolloutStorage


class PPOMimic:
    """
    PPO with Mimic Learning algorithm for teacher-student distillation.
    
    Key features:
    - Three-stage training: teacher-only, hybrid, student
    - DAgger-style latent space alignment
    - Privilege regularization (ASAP-style)
    - Motion encoder + History encoder architecture
    """

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        # Teacher-student specific parameters
        teacher_only_interval=0,  # Epochs for teacher-only training
        hybrid_training_intervals=0,  # Epochs for hybrid training
        dagger_update_freq=20,  # How often to update history encoder
        priv_reg_coef_schedule=None,  # [min_coef, max_coef, warmup_end, ramp_end]
        # Multi-GPU support
        multi_gpu_cfg: dict | None = None,
        **kwargs
    ):
        self.device = device
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        
        # PPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        
        # Teacher-student parameters
        self.teacher_only_interval = teacher_only_interval
        self.hybrid_training_intervals = hybrid_training_intervals
        self.dagger_update_freq = dagger_update_freq
        
        # Privilege regularization schedule
        if priv_reg_coef_schedule is None:
            # Default: [min_coef, max_coef, warmup_end_epoch, ramp_end_epoch]
            self.priv_reg_coef_schedule = [0.0, 0.1, 2000, 3000]
        else:
            self.priv_reg_coef_schedule = priv_reg_coef_schedule
        
        # Training counter
        self.counter = 0
        self.current_stage = 1  # 1: teacher-only, 2: hybrid, 3: student
        
        # Multi-GPU configuration
        self.is_multi_gpu = multi_gpu_cfg is not None
        if self.is_multi_gpu:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        
        # Optimizers
        # Main optimizer for actor-critic
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=self.learning_rate)
        
        # Separate optimizer for history encoder (DAgger updates)
        if hasattr(self.actor_critic, 'actor') and hasattr(self.actor_critic.actor, 'history_encoder'):
            self.hist_encoder_optimizer = optim.AdamW(
                self.actor_critic.actor.history_encoder.parameters(),
                lr=self.learning_rate
            )
        else:
            self.hist_encoder_optimizer = None
        
        # Storage
        self.storage = None
        
    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape
    ):
        """Initialize rollout storage."""
        self.storage = RolloutStorage(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=actions_shape,
            device=self.device
        )
    
    def test_mode(self):
        """Set networks to evaluation mode."""
        self.actor_critic.eval()
    
    def train_mode(self):
        """Set networks to training mode."""
        self.actor_critic.train()
    
    def act(self, obs, critic_obs):
        """
        Sample actions from the policy.
        
        During rollout, we use privilege information (use_privilege=True).
        """
        # Use privilege encoder during data collection
        return self.actor_critic.act(obs, use_privilege=True)
    
    def process_env_step(self, rewards, dones, infos):
        """Process environment step and store transition."""
        # Store transition
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device)
        
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        """Compute returns using GAE."""
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def get_priv_reg_coef(self, epoch):
        """
        Compute privilege regularization coefficient based on training stage.
        
        Schedule: [min_coef, max_coef, warmup_end_epoch, ramp_end_epoch]
        - epoch < warmup_end: coef = min_coef (no constraint, free exploration)
        - warmup_end <= epoch < ramp_end: linear increase from min to max
        - epoch >= ramp_end: coef = max_coef (light constraint)
        """
        min_coef, max_coef, warmup_end, ramp_end = self.priv_reg_coef_schedule
        
        if epoch < warmup_end:
            return min_coef
        elif epoch < ramp_end:
            # Linear ramp
            progress = (epoch - warmup_end) / (ramp_end - warmup_end)
            return min_coef + progress * (max_coef - min_coef)
        else:
            return max_coef
    
    def update(self, epoch=0):
        """
        Update policy using PPO with mimic learning.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            dict: Dictionary of loss values
        """
        # Update current training stage based on epoch
        if epoch < self.teacher_only_interval:
            self.current_stage = 1  # Teacher-only stage
        elif epoch < self.hybrid_training_intervals:
            self.current_stage = 2  # Hybrid stage
        else:
            self.current_stage = 3  # Student stage
        
        # Initialize loss tracking
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_priv_reg_loss = 0.0
        mean_hist_latent_loss = 0.0
        
        # Generate mini-batches
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs
        )
        
        for (
            obs_batch,
            critic_obs_batch,
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
            
            # ===== Forward pass with privilege encoder =====
            self.actor_critic.act(obs_batch, use_privilege=True)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            # ===== Adaptive learning rate (KL divergence) =====
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1e-5))
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    
                    # Multi-GPU: reduce KL across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    
                    # Update learning rate
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    # Broadcast learning rate to all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                
                # Update optimizer learning rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
            
            # ===== PPO loss computation =====
            if epoch < self.teacher_only_interval:
                # Teacher-only stage: skip policy gradient, only train critic
                surrogate_loss = torch.tensor(0.0, device=self.device)
            else:
                # Normal PPO: compute surrogate loss
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
            
            # Total loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            # ===== Privilege regularization loss =====
            # Constrains privilege encoder to stay close to history encoder
            priv_reg_coef = self.get_priv_reg_coef(epoch)
            
            if priv_reg_coef > 0 and hasattr(self.actor_critic.actor, 'priv_encoder') and hasattr(self.actor_critic.actor, 'history_encoder'):
                # Get privilege latent (trainable)
                priv_latent = self.actor_critic.actor.priv_encoder(obs_batch)
                
                # Get history latent (detached, serves as target)
                with torch.no_grad():
                    hist_latent = self.actor_critic.actor.history_encoder(obs_batch)
                
                # L2 alignment loss
                priv_reg_loss = (priv_latent - hist_latent.detach()).norm(p=2, dim=1).mean()
                
                # Add to total loss
                loss = loss + priv_reg_coef * priv_reg_loss
                mean_priv_reg_loss += priv_reg_loss.item()
            
            # ===== Gradient step =====
            self.optimizer.zero_grad()
            loss.backward()
            
            # Multi-GPU: reduce gradients
            if self.is_multi_gpu:
                self.reduce_parameters()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
        
        # ===== DAgger update for history encoder =====
        # Update history encoder to mimic privilege encoder's latent representations
        if self.counter % self.dagger_update_freq == 0 and self.hist_encoder_optimizer is not None and epoch >= self.hybrid_training_intervals:
            # Regenerate mini-batches for DAgger update
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs
            )
            
            for (
                obs_batch,
                critic_obs_batch,
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
                # Get privilege latent (detached, serves as target)
                with torch.no_grad():
                    priv_latent = self.actor_critic.actor.priv_encoder(obs_batch)
                
                # Get history latent (trainable)
                hist_latent = self.actor_critic.actor.history_encoder(obs_batch)
                
                # L2 loss: history encoder learns to mimic privilege encoder
                hist_latent_loss = (priv_latent.detach() - hist_latent).norm(p=2, dim=1).mean()
                
                # Gradient step
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.actor.history_encoder.parameters(),
                    self.max_grad_norm
                )
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        
        # Update counter
        self.counter += 1
        
        # Compute mean losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_priv_reg_loss /= num_updates
        
        if self.counter % self.dagger_update_freq == 0 and mean_hist_latent_loss > 0:
            mean_hist_latent_loss /= num_updates
        
        # Clear storage
        self.storage.clear()
        
        # Return loss dictionary
        loss_dict = {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_entropy": mean_entropy,
            "mean_priv_reg_loss": mean_priv_reg_loss,
            "mean_hist_latent_loss": mean_hist_latent_loss,
        }
        
        return loss_dict
    
    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        if not self.is_multi_gpu:
            return
        
        for param in self.actor_critic.parameters():
            torch.distributed.broadcast(param.data, src=0)
    
    def reduce_parameters(self):
        """Reduce gradients across all GPUs."""
        if not self.is_multi_gpu:
            return
        
        for param in self.actor_critic.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                param.grad.data /= self.gpu_world_size
