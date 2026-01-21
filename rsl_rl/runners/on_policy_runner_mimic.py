# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
On-policy runner for PPOMimic training.

This runner adapts rsl_rl's environment interface to PPOMimic's 
dictionary-based observation format.
"""

from __future__ import annotations

import os
import time
import statistics
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import torch
import rsl_rl
from rsl_rl.algorithms import PPOMimic
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticMimic, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class OnPolicyRunnerMimic:
    """
    On-policy runner for PPOMimic training.
    
    Adapts rsl_rl's environment interface to PPOMimic's dictionary-based format.
    Requires environment to provide:
    - obs: base observations
    - extras["observations"]["priv_obs"]: privileged observations
    - extras["observations"]["prop_history"]: proprioceptive history
    - extras["observations"]["motion_targets"]: future motion targets
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Get observations to determine dimensions
        obs, extras = self.env.get_observations()
        
        # Extract observation components from extras
        num_actor_obs = obs.shape[1]
        
        # Get privileged observations
        if "priv_obs" in extras["observations"]:
            num_privileged_obs = extras["observations"]["priv_obs"].shape[1]
        else:
            raise ValueError("PPOMimic requires 'priv_obs' in extras['observations']")
        
        # Get proprioceptive history
        if "prop_history" in extras["observations"]:
            prop_history = extras["observations"]["prop_history"]
            num_prop_history_total = prop_history.shape[1]
            # Need to determine history_length and num_prop_history from config or data
            # Assuming format: [batch, history_length * num_prop_history]
            history_length = self.policy_cfg.get("history_length", 50)
            num_prop_history = num_prop_history_total // history_length
        else:
            raise ValueError("PPOMimic requires 'prop_history' in extras['observations']")
        
        # Get motion targets
        if "motion_targets" in extras["observations"]:
            motion_targets = extras["observations"]["motion_targets"]
            num_motion_targets_total = motion_targets.shape[1]
            future_num_steps = self.policy_cfg.get("future_num_steps", 10)
            num_motion_targets = num_motion_targets_total // future_num_steps
        else:
            raise ValueError("PPOMimic requires 'motion_targets' in extras['observations']")
        
        # Critic observations (same as actor for now, can use privileged)
        num_critic_obs = num_actor_obs
        
        print(f"Observation dimensions:")
        print(f"  - num_actor_obs: {num_actor_obs}")
        print(f"  - num_critic_obs: {num_critic_obs}")
        print(f"  - num_privileged_obs: {num_privileged_obs}")
        print(f"  - num_prop_history: {num_prop_history} x {history_length} = {num_prop_history_total}")
        print(f"  - num_motion_targets: {num_motion_targets} x {future_num_steps} = {num_motion_targets_total}")
        
        # Filter out 'class_name' from policy config
        actor_critic_kwargs = {k: v for k, v in self.policy_cfg.items() if k != 'class_name'}
        
        # Initialize actor-critic network
        actor_critic = ActorCriticMimic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_privileged_obs=num_privileged_obs,
            num_prop_history=num_prop_history,
            num_motion_targets=num_motion_targets,
            num_actions=self.env.num_actions,
            history_length=history_length,
            future_num_steps=future_num_steps,
            **actor_critic_kwargs
        ).to(self.device)
        
        # Initialize PPOMimic algorithm
        self.alg: PPOMimic = PPOMimic(
            actor_critic=actor_critic,
            device=self.device,
            **self.alg_cfg
        )
        
        # Get runner config
        runner_cfg = self.cfg.get("runner", {})
        self.num_steps_per_env = self.cfg.get("num_steps_per_env") or runner_cfg.get("num_steps_per_env", 24)
        self.save_interval = self.cfg.get("save_interval") or runner_cfg.get("save_interval", 50)
        self.max_iterations = self.cfg.get("max_iterations") or runner_cfg.get("max_iterations", 1000)
        
        # Normalization (optional for PPOMimic)
        self.empirical_normalization = self.cfg.get("empirical_normalization") or runner_cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            self.priv_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.priv_obs_normalizer = torch.nn.Identity().to(self.device)
        
        # Note: PPOMimic uses its own internal storage management
        # We'll collect data in lists and convert to tensors for updates
        
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        
    def _convert_obs_to_dict(self, obs, extras):
        """Convert rsl_rl observations to PPOMimic dictionary format."""
        obs_dict = {
            'actor_obs': obs,
            'critic_obs': obs,  # Can use privileged obs for critic
            'prop_history': extras["observations"]["prop_history"],
            'motion_targets': extras["observations"]["motion_targets"],
            'priv_obs': extras["observations"]["priv_obs"],
        }
        return obs_dict
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True):
        # Initialize writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError(f"Logger type {self.logger_type} not supported")
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        
        obs, extras = self.env.get_observations()
        obs, extras = obs.to(self.device), {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                            for k, v in extras.items()}
        
        # Convert nested dict
        extras["observations"] = {k: v.to(self.device) for k, v in extras["observations"].items()}
        
        self.alg.train_mode()
        
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # === Rollout phase ===
            # Collect trajectories
            rollout_data = {
                'obs': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'values': [],
                'actions_log_prob': [],
                'action_mean': [],
                'action_sigma': [],
            }
            
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Convert to dictionary format
                    obs_dict = self._convert_obs_to_dict(obs, extras)
                    
                    # Normalize observations
                    obs_dict['actor_obs'] = self.obs_normalizer(obs_dict['actor_obs'])
                    obs_dict['critic_obs'] = self.obs_normalizer(obs_dict['critic_obs'])
                    obs_dict['priv_obs'] = self.priv_obs_normalizer(obs_dict['priv_obs'])
                    
                    # Sample actions (using privilege encoder during training)
                    actions = self.alg.actor_critic.act(obs_dict, use_privilege=True)
                    
                    # Get action log prob and value
                    actions_log_prob = self.alg.actor_critic.get_actions_log_prob(actions)
                    values = self.alg.actor_critic.evaluate(obs_dict)
                    
                    # Store data
                    rollout_data['obs'].append(obs_dict)
                    rollout_data['actions'].append(actions)
                    rollout_data['actions_log_prob'].append(actions_log_prob)
                    rollout_data['action_mean'].append(self.alg.actor_critic.action_mean)
                    rollout_data['action_sigma'].append(self.alg.actor_critic.action_std)
                    rollout_data['values'].append(values)
                    
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    extras = infos
                    extras["observations"] = {k: v.to(self.device) for k, v in extras["observations"].items()}
                    
                    rollout_data['rewards'].append(rewards)
                    rollout_data['dones'].append(dones)
                    
                    # Logging
                    if self.log_dir is not None:
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
            
            stop = time.time()
            collection_time = stop - start
            
            # === Learning phase ===
            start = stop
            
            # Compute returns using GAE
            # For PPOMimic, we need to manually compute returns
            with torch.no_grad():
                # Get last value
                obs_dict = self._convert_obs_to_dict(obs, extras)
                obs_dict['actor_obs'] = self.obs_normalizer(obs_dict['actor_obs'])
                obs_dict['critic_obs'] = self.obs_normalizer(obs_dict['critic_obs'])
                obs_dict['priv_obs'] = self.priv_obs_normalizer(obs_dict['priv_obs'])
                last_values = self.alg.actor_critic.evaluate(obs_dict)
                
                # Compute returns and advantages
                returns = []
                advantages = []
                gae = 0
                
                for step in reversed(range(self.num_steps_per_env)):
                    if step == self.num_steps_per_env - 1:
                        next_values = last_values
                        next_non_terminal = 1.0 - rollout_data['dones'][step].float()
                    else:
                        next_values = rollout_data['values'][step + 1]
                        next_non_terminal = 1.0 - rollout_data['dones'][step].float()
                    
                    delta = (rollout_data['rewards'][step] + 
                            self.alg.gamma * next_values * next_non_terminal - 
                            rollout_data['values'][step])
                    gae = delta + self.alg.gamma * self.alg.lam * next_non_terminal * gae
                    
                    returns.insert(0, gae + rollout_data['values'][step])
                    advantages.insert(0, gae)
                
                returns = torch.stack(returns)
                advantages = torch.stack(advantages)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Create a simple storage structure for PPOMimic
            # Stack all data
            stacked_obs_dicts = {
                key: torch.stack([rollout_data['obs'][i][key] for i in range(self.num_steps_per_env)])
                for key in rollout_data['obs'][0].keys()
            }
            stacked_actions = torch.stack(rollout_data['actions'])
            stacked_old_log_probs = torch.stack(rollout_data['actions_log_prob'])
            stacked_old_means = torch.stack(rollout_data['action_mean'])
            stacked_old_stds = torch.stack(rollout_data['action_sigma'])
            stacked_values = torch.stack(rollout_data['values'])
            
            # Flatten batch dimension: [num_steps, num_envs, ...] -> [num_steps * num_envs, ...]
            def flatten_batch(x):
                return x.reshape(-1, *x.shape[2:])
            
            flat_obs_dict = {k: flatten_batch(v) for k, v in stacked_obs_dicts.items()}
            flat_actions = flatten_batch(stacked_actions)
            flat_old_log_probs = flatten_batch(stacked_old_log_probs)
            flat_old_means = flatten_batch(stacked_old_means)
            flat_old_stds = flatten_batch(stacked_old_stds)
            flat_values = flatten_batch(stacked_values)
            flat_returns = flatten_batch(returns)
            flat_advantages = flatten_batch(advantages)
            
            # Mini-batch updates
            total_samples = flat_actions.shape[0]
            batch_size = total_samples // self.alg.num_mini_batches
            
            mean_value_loss = 0.0
            mean_surrogate_loss = 0.0
            mean_entropy = 0.0
            mean_priv_reg_loss = 0.0
            mean_hist_latent_loss = 0.0
            
            for epoch in range(self.alg.num_learning_epochs):
                # Shuffle indices
                indices = torch.randperm(total_samples, device=self.device)
                
                for start_idx in range(0, total_samples, batch_size):
                    end_idx = min(start_idx + batch_size, total_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get mini-batch
                    obs_batch = {k: v[batch_indices] for k, v in flat_obs_dict.items()}
                    actions_batch = flat_actions[batch_indices]
                    old_log_probs_batch = flat_old_log_probs[batch_indices]
                    old_means_batch = flat_old_means[batch_indices]
                    old_stds_batch = flat_old_stds[batch_indices]
                    values_batch = flat_values[batch_indices]
                    returns_batch = flat_returns[batch_indices]
                    advantages_batch = flat_advantages[batch_indices]
                    
                    # Forward pass
                    self.alg.actor_critic.update_distribution(obs_batch, use_privilege=True)
                    new_log_probs = self.alg.actor_critic.get_actions_log_prob(actions_batch)
                    new_values = self.alg.actor_critic.evaluate(obs_batch)
                    entropy = self.alg.actor_critic.entropy
                    
                    # PPO losses
                    if it < self.alg.teacher_only_interval:
                        surrogate_loss = torch.tensor(0.0, device=self.device)
                    else:
                        ratio = torch.exp(new_log_probs - old_log_probs_batch.squeeze(-1))
                        surr1 = ratio * advantages_batch.squeeze(-1)
                        surr2 = torch.clamp(ratio, 1.0 - self.alg.clip_param, 1.0 + self.alg.clip_param) * advantages_batch.squeeze(-1)
                        surrogate_loss = -torch.min(surr1, surr2).mean()
                    
                    if self.alg.use_clipped_value_loss:
                        value_clipped = values_batch + (new_values - values_batch).clamp(-self.alg.clip_param, self.alg.clip_param)
                        value_loss = torch.max((new_values - returns_batch).pow(2), (value_clipped - returns_batch).pow(2)).mean()
                    else:
                        value_loss = (new_values - returns_batch).pow(2).mean()
                    
                    loss = surrogate_loss + self.alg.value_loss_coef * value_loss - self.alg.entropy_coef * entropy.mean()
                    
                    # Privilege regularization
                    priv_reg_coef = self.alg.get_priv_reg_coef(it)
                    if priv_reg_coef > 0:
                        priv_latent = self.alg.actor_critic.actor.priv_encoder(obs_batch['priv_obs'])
                        with torch.no_grad():
                            hist_latent = self.alg.actor_critic.actor.history_encoder(obs_batch['prop_history'])
                        priv_reg_loss = (priv_latent - hist_latent).norm(p=2, dim=1).mean()
                        loss = loss + priv_reg_coef * priv_reg_loss
                        mean_priv_reg_loss += priv_reg_loss.item()
                    
                    # Optimizer step
                    self.alg.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.alg.actor_critic.parameters(), self.alg.max_grad_norm)
                    self.alg.optimizer.step()
                    
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()
                    mean_entropy += entropy.mean().item()
                
                # DAgger update
                if it % self.alg.dagger_update_freq == 0 and it >= self.alg.hybrid_training_intervals:
                    for start_idx in range(0, total_samples, batch_size):
                        end_idx = min(start_idx + batch_size, total_samples)
                        batch_indices = indices[start_idx:end_idx]
                        
                        obs_batch = {k: v[batch_indices] for k, v in flat_obs_dict.items()}
                        
                        with torch.no_grad():
                            priv_latent = self.alg.actor_critic.actor.priv_encoder(obs_batch['priv_obs'])
                        hist_latent = self.alg.actor_critic.actor.history_encoder(obs_batch['prop_history'])
                        hist_loss = (priv_latent - hist_latent).norm(p=2, dim=1).mean()
                        
                        self.alg.hist_encoder_optimizer.zero_grad()
                        hist_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.alg.actor_critic.actor.history_encoder.parameters(), self.alg.max_grad_norm)
                        self.alg.hist_encoder_optimizer.step()
                        
                        mean_hist_latent_loss += hist_loss.item()
            
            num_updates = self.alg.num_learning_epochs * self.alg.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_entropy /= num_updates
            mean_priv_reg_loss /= num_updates
            if mean_hist_latent_loss > 0:
                mean_hist_latent_loss /= num_updates
            
            self.alg.counter += 1
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # Logging
            if self.log_dir is not None:
                self.log({
                    'it': it,
                    'collection_time': collection_time,
                    'learn_time': learn_time,
                    'rewbuffer': rewbuffer,
                    'lenbuffer': lenbuffer,
                    'loss_dict': {
                        'mean_value_loss': mean_value_loss,
                        'mean_surrogate_loss': mean_surrogate_loss,
                        'mean_entropy': mean_entropy,
                        'mean_priv_reg_loss': mean_priv_reg_loss,
                        'mean_hist_latent_loss': mean_hist_latent_loss,
                    }
                })
                
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            
            ep_infos.clear()
        
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
    
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        
        mean_std = self.alg.actor_critic.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
        
        # Log to tensorboard
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        
        # Console output
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""Learning iteration {locs['it']}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                f"""{'Mean noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key}:':>{pad}} {value:.4f}\n"""
            log_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'-' * width}\n"""
                f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            )
            print(log_string)
    
    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["priv_obs_norm_state_dict"] = self.priv_obs_normalizer.state_dict()
        if self.alg.hist_encoder_optimizer is not None:
            saved_dict["hist_encoder_optimizer_state_dict"] = self.alg.hist_encoder_optimizer.state_dict()
        torch.save(saved_dict, path)
    
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.priv_obs_normalizer.load_state_dict(loaded_dict["priv_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "hist_encoder_optimizer_state_dict" in loaded_dict and self.alg.hist_encoder_optimizer is not None:
                self.alg.hist_encoder_optimizer.load_state_dict(loaded_dict["hist_encoder_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def get_inference_policy(self, device=None):
        self.alg.test_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
        
        def policy(obs_dict):
            # Use history encoder for deployment (no privilege)
            return self.alg.actor_critic.act_inference(obs_dict, use_privilege=False)
        
        return policy
