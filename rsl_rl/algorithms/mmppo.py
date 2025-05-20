# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Optimized from RSL_RL/PPO
# Created by Yifei Yao, 10/12/2024

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools

from rsl_rl.modules import ActorCriticMMTransformer, AMPNet
from rsl_rl.storage import RolloutStorageMM
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.utils import string_to_callable
import time

class SmoothL2Loss(nn.Module):
    """
    Smooth L2 loss (Huber loss), where:
    loss(x) = 0.5 * x^2               if |x| < delta
              delta * (|x| - 0.5*delta)  otherwise
    """

    def __init__(self, delta: float = 1.0):
        super(SmoothL2Loss, self).__init__()
        self.delta = delta

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        abs_diff = torch.abs(diff)

        mask = abs_diff < self.delta
        loss = torch.where(
            mask,
            0.5 * diff**2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )

        return loss.mean()
    
class L2Loss(nn.Module):
    """
    L2 loss function
    """

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        diff = input - target
        loss = torch.square(diff)
        return loss.mean()

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
        max_lr=1e-2,
        min_lr=1e-4,
        max_lr_after_certain_epoch=1e-3,
        max_lr_restriction_epoch=2500,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        ref_action_idx=0,
        # dagger
        # dagger parameters
        teacher_coef=None, # disable dagger if None
        teacher_coef_range=None,
        teacher_coef_decay=None,
        teacher_coef_decay_interval=100,
        # dagger imitation loss parameters
        teacher_loss_coef=None,
        teacher_loss_coef_range=None,
        teacher_loss_coef_decay=None,
        teacher_loss_coef_decay_interval=100,
        # dagger update parameters        
        teacher_supervising_intervals=0, # when epoch < teacher_supervising_intervals, PPO won't imitate dagger actions and dagger will not be updated
        teacher_apply_interval=5, # imitation loss will be add to PPO in * iters
        teacher_coef_mode="kl", # "kl" or "norm"
        teacher_update_interval=1,
        teacher_lr=5e-4,
        teacher_only_interval=0,
        default_action=None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,


        **kwargs # reserved for future use
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        
        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # AMP components
        if amp_cfg is not None:
            net_cfg = amp_cfg.get("net_cfg", {})
            self.amp = AMPNet(device=self.device, **net_cfg)
            self.amp_cfg = amp_cfg
            self.amp_optimizer = optim.Adam(self.amp.parameters(), lr=amp_cfg.get("learning_rate", 1e-3))
        else:
            self.amp = None
            self.amp_cfg = None
            self.amp_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])

            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_lr_after_certain_epoch = max_lr_after_certain_epoch
        self.max_lr_restriction_epoch = max_lr_restriction_epoch
        assert min_lr <= learning_rate <= max_lr, "learning_rate should be in range [min_lr, max_lr], and min_lr <= max_lr"
        self.ref_action_idx = ref_action_idx
        self.teacher_coef = teacher_coef
        self.teacher_coef_range = teacher_coef_range
        self.teacher_coef_decay = teacher_coef_decay
        self.teacher_coef_decay_interval = teacher_coef_decay_interval
        self.teacher_loss_coef = teacher_loss_coef
        self.teacher_loss_coef_range = teacher_loss_coef_range
        self.teacher_loss_coef_decay = teacher_loss_coef_decay
        self.teacher_loss_coef_decay_interval = teacher_loss_coef_decay_interval
        self.teacher_coef_mode = teacher_coef_mode
        self.teacher_only_interval = teacher_only_interval
        self.teacher_supervising_intervals = teacher_supervising_intervals
        self.teacher_apply_interval = teacher_apply_interval
        
        assert teacher_coef is not None or teacher_only_interval == 0, "teacher_only_interval should be 0 if teacher_coef is None"
        assert (teacher_coef is None and teacher_loss_coef is None) or (teacher_loss_coef is not None and teacher_coef is not None), "teacher_coef and teacher_loss_coef should be set together"

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.AdamW(
            itertools.chain(
                self.actor_critic.actor.parameters(),
                self.actor_critic.critic.parameters()
            ),
            lr=learning_rate) # Avoid optimizing dagger parameters if self.teacher_coef is None
        self.imitation_optimizer = optim.AdamW(
            itertools.chain(
                self.actor_critic.actor.parameters(),
                self.actor_critic.critic.parameters()
            ),
            lr=learning_rate)
        # self.optimizer = optim.AdamW(
        #     self.actor_critic.parameters(),
        #     lr=learning_rate
        # )
        self.teacher_update_interval = teacher_update_interval
        self.transition = RolloutStorageMM.Transition()

        if self.teacher_coef is not None and self.teacher_loss_coef is not None:
            assert self.teacher_coef_mode in ["kl", "norm"], "teacher_coef_mode should be either 'kl' or 'norm'"
            if self.teacher_coef_range is None:
                self.teacher_coef_range = (self.teacher_coef, self.teacher_coef)
            else:
                assert len(self.teacher_coef_range) == 2, "teacher_coef_range should be a tuple of (min, max)"
                assert self.teacher_coef_range[0] <= self.teacher_coef_range[1], "teacher_coef_range should be a tuple of (min, max)"
            
            if self.teacher_loss_coef_range is None:
                self.teacher_loss_coef_range = (self.teacher_loss_coef, self.teacher_loss_coef)
            else:
                assert len(self.teacher_loss_coef_range) == 2, "teacher_loss_coef_range should be a tuple of (min, max)"
                assert self.teacher_loss_coef_range[0] <= self.teacher_loss_coef_range[1], "teacher_loss_coef_range should be a tuple of (min, max)"
            
            if self.teacher_coef_decay is None:
                self.teacher_coef_decay = 0.0
            else:
                assert 0.0 <= self.teacher_coef_decay <= 1.0, "teacher_coef_decay should be in range [0.0, 1.0]"
                assert self.teacher_coef_decay_interval > 0, "teacher_coef_decay_interval should be greater than 0"
            
            if self.teacher_loss_coef_decay is None:
                self.teacher_loss_coef_decay = 0.0
            else:
                assert 0.0 <= self.teacher_loss_coef_decay <= 1.0, "teacher_loss_coef_decay should be in range [0.0, 1.0]"
                assert self.teacher_loss_coef_decay_interval > 0, "teacher_loss_coef_decay_interval should be greater than 0"    
            
            assert hasattr(self.actor_critic, "actor_dagger") and self.actor_critic.actor_dagger is not None, "cannot run Dagger mode without dagger actor, check your actor_critic initialization first"
            self.dagger_optimizer = optim.AdamW(self.actor_critic.actor_dagger.parameters(), lr=teacher_lr)
        else:
            self.dagger_optimizer = None
            
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
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, actor_ref_obs_shape, critic_obs_shape, critic_ref_obs_shape, action_shape):
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        self.storage = RolloutStorageMM(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape=actor_obs_shape,
            ref_obs_shape=actor_ref_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            privileged_ref_obs_shape=critic_ref_obs_shape,
            actions_shape=action_shape,
            rnd_state_shape=rnd_state_shape,
            apply_dagger_actions=(self.teacher_coef is not None),
            amp_cfg=self.amp_cfg,
            device=self.device,
        )


    def test_mode(self):
        self.actor_critic.test()
        if self.amp:
            self.amp.eval()

    def train_mode(self):
        self.actor_critic.train()
        if self.amp:
            self.amp.train()

    def act(self, obs, ref_obs, critic_obs, ref_critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, ref_observations=ref_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, ref_critic_observations=ref_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        if self.teacher_coef is not None:
            self.transition.dagger_actions = self.actor_critic.act_dagger(obs, ref_obs).detach()
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
        
        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            try:
                rnd_state = infos["observations"]["rnd_state"]
            except KeyError:
                rnd_state = infos["observations"]["policy"]
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            self.transition.rewards += self.intrinsic_rewards
            self.transition.rnd_state = rnd_state.clone()

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
        self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch)

    def update(self, epoch=0):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_imitation_loss = 0.0
        mean_dagger_loss = 0.0
        mean_entropy = 0.0

        if self.rnd:
            mean_rnd_loss = 0.0
        else:
            mean_rnd_loss = None

        if self.symmetry:
            mean_symmetry_loss = 0.0
        else:
            mean_symmetry_loss = None

        if self.amp:
            mean_amp_loss = 0.0
            mean_gradient_penalty = 0.0
            mean_pred_pos_acc = 0.0
            mean_pred_neg_acc = 0.0

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
            dagger_actions_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
            obs_prev_state,
            ref_obs_prev_state,
            ref_obs_prev_mask,
        ) in generator:
           

            num_aug = 1

            original_batch_size = obs_batch.shape[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, ref_obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, ref_obs=ref_obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )

                num_aug = int(obs_batch.shape[0] / original_batch_size)

                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.actor_critic.act(obs_batch, ref_observations=ref_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, ref_critic_observations=critic_ref_obs_batch
            )

            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            if self.amp:
                obs_cur_state = string_to_callable(self.amp_cfg["amp_obs_extractor"])(obs_batch, env=self.amp_cfg["_env"])
                ref_obs_cur_state, ref_obs_cur_mask = string_to_callable(self.amp_cfg["amp_ref_obs_extractor"])(ref_obs_batch, env=self.amp_cfg["_env"])

            # Imitation Entropy loss (optional, arxiv: 2409.08904)
            # This loss is calculated only when critic_ref_obs_batch is not None, otherwise 0.0
            # By default, we assume that ref_action_batch = critic_ref_obs_batch[0][:, ref_action_idx: num_actions + ref_action_idx] where ref_action_idx = 0
            # Loss: \sum_i [\sqrt(2 * \pi * \sigma_i^2) + (mu_i - ref_action_i)^2 / (2 * \sigma_i^2)]
            if self.max_lr_restriction_epoch != 0 and epoch > self.max_lr_restriction_epoch:
                self.max_lr = self.max_lr_after_certain_epoch
            
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
                    kl_mean = torch.mean(kl) #- imitation_loss # advanced kl

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    if self.gpu_global_rank == 0:

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.max_lr, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

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

            

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, ref_obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, ref_obs=ref_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                    mean_actions_batch = self.actor_critic.act_inference(obs_batch.detach().clone(), (ref_obs_batch[0].detach().clone(), ref_obs_batch[1].detach().clone()))

                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, ref_obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                    )

                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        mean_actions_batch[:original_batch_size], actions_mean_symm_batch.detach()[:original_batch_size]
                    )
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()
                    
                    mean_symmetry_loss += symmetry_loss.item()

            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mse_loss = torch.nn.MSELoss()
                rnd_loss = mse_loss(predicted_embedding, target_embedding) 

           
            

                    

            imitation_loss = self._imitation_loss(actions_batch=actions_batch, obs_batch=obs_batch, ref_obs_batch=ref_obs_batch, dagger_actions_batch=dagger_actions_batch)      
                    
            # if self.teacher_coef is not None:
            #     loss += imitation_loss * self.teacher_coef
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()

            if self.amp:
                amp_optimization_steps = self.amp_cfg.get("amp_optimization_steps", 3)

                for _ in range(amp_optimization_steps):
                    policy_score = self.amp.forward(torch.cat([obs_prev_state, obs_cur_state], dim=-1))
                    expert_score = self.amp.forward(torch.cat([ref_obs_prev_state, ref_obs_cur_state], dim=-1))
                    policy_loss = self.amp.policy_loss(policy_score)
                    expert_loss = self.amp.expert_loss(expert_score, ref_obs_cur_mask)
                    gradient_penalty = self.amp.expert_grad_penalty(obs_cur_state, ref_obs_cur_state, ref_obs_cur_mask * ref_obs_prev_mask)
                    amp_loss = 0.5 * (policy_loss + expert_loss) + gradient_penalty * self.amp_cfg["gradient_penalty_coeff"]
                    pred_pos_acc = self.amp.policy_acc(policy_score)
                    pred_neg_acc = self.amp.expert_acc(expert_score, ref_obs_cur_mask * ref_obs_prev_mask)
                    self.amp_optimizer.zero_grad()
                    amp_loss.backward()
                    
                mean_amp_loss += amp_loss.item()
                mean_gradient_penalty += gradient_penalty.item() * self.amp_cfg["gradient_penalty_coeff"]
                mean_pred_pos_acc += pred_pos_acc.item()
                mean_pred_neg_acc += pred_neg_acc.item()

            if self.teacher_loss_coef is not None and epoch > self.teacher_supervising_intervals and (epoch+1)%self.teacher_apply_interval == 0:
                backward_imitation_loss = imitation_loss * self.teacher_loss_coef
                self.imitation_optimizer.zero_grad()
                backward_imitation_loss.backward()

            # for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                mean_rnd_loss += rnd_loss.item()

            # if self.amp and isinstance(amp_loss, torch.Tensor): # skip first epoch where amp_loss is 0.0 (float)
            #     self.amp_optimizer.zero_grad()
            #     amp_loss.backward()
            #     mean_amp_loss += amp_loss.item()
            #     mean_gradient_penalty += gradient_penalty.item() * self.amp_cfg["gradient_penalty_coeff"]
            #     mean_pred_pos_acc += pred_pos_acc.item()
            #     mean_pred_neg_acc += pred_neg_acc.item()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            
            if self.teacher_loss_coef is not None and epoch > self.teacher_supervising_intervals and (epoch+1)%self.teacher_apply_interval == 0:
                # backward_imitation_loss = imitation_loss * self.teacher_loss_coef
                # self.imitation_optimizer.zero_grad()
                # backward_imitation_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.imitation_optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_imitation_loss += imitation_loss.item()
            mean_entropy += entropy_batch.mean().item()

            if (epoch+1) % self.teacher_update_interval == 0 and self.dagger_optimizer is not None and epoch > self.teacher_supervising_intervals:
                # dagger_loss = self.update_dagger(actions_batch, obs_batch, ref_obs_batch, epoch)
                dagger_loss = self.update_dagger(actions_batch, obs_batch, ref_obs_batch, dagger_actions_batch)
            else:
                dagger_loss = 0.0
            mean_dagger_loss += dagger_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates
        mean_dagger_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_gradient_penalty /= num_updates
        mean_pred_pos_acc /= num_updates
        mean_pred_neg_acc /= num_updates
        if epoch > 15000 and epoch % 200 == 0:
            self.min_lr = max(5e-6, self.min_lr / 1.5)
        
        # udpate teacher coef and teacher loss coef
        if self.teacher_coef is not None and (epoch+1) % self.teacher_coef_decay_interval == 0:
            self.teacher_coef = max(
                self.teacher_coef_range[0],
                self.teacher_coef - (1 - self.teacher_coef_decay) * (self.teacher_coef_range[1] - self.teacher_coef_range[0]),)
        if self.teacher_loss_coef is not None and (epoch+1) % self.teacher_loss_coef_decay_interval == 0:
            self.teacher_loss_coef = min(
                self.teacher_loss_coef_range[1],
                self.teacher_loss_coef + (1 - self.teacher_loss_coef_decay) * (self.teacher_loss_coef_range[1] - self.teacher_loss_coef_range[0]),)
        
        self.storage.clear()

        loss_dict = {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_imitation_loss": mean_imitation_loss,
            "mean_dagger_loss": mean_dagger_loss,	
            "mean_entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["mean_rnd_loss"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.amp:
            loss_dict["mean_amp_loss"] = mean_amp_loss
            loss_dict["mean_gradient_penalty"] = mean_gradient_penalty
            loss_dict["mean_pred_pos_acc"] = mean_pred_pos_acc
            loss_dict["mean_pred_neg_acc"] = mean_pred_neg_acc

        return loss_dict

        # return mean_value_loss.ite, mean_surrogate_loss.item(), mean_imitation_loss.item(), mean_dagger_loss.item()

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs. (Actor & Critic Only)"""

        actor_params = [self.actor_critic.actor.state_dict()]
        critic_params = [self.actor_critic.critic.state_dict()]
        model_params = actor_params + critic_params
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        if self.amp:
            model_params.append(self.amp.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.actor_critic.actor.load_state_dict(model_params[0])
        self.actor_critic.critic.load_state_dict(model_params[1])
        amp_idx = 2
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])
            amp_idx += 1
        if self.amp:
            self.amp.load_state_dict(model_params[amp_idx])

    def broadcast_parameters_dagger(self):
        """Broadcast model parameters to all GPUs. (Dagger Only)"""
        assert hasattr(self.actor_critic, "actor_dagger"), "actor_dagger not enabled in actor_critic, cannot broadcast parameters of None"
        model_params = [self.actor_critic.actor_dagger.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor_critic.actor_dagger.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        actor_grads = [param.grad.view(-1) for param in self.actor_critic.actor.parameters() if param.grad is not None]
        critic_grads = [param.grad.view(-1) for param in self.actor_critic.critic.parameters() if param.grad is not None]
        grads = actor_grads + critic_grads
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.predictor.parameters() if param.grad is not None]
        if self.amp:
            grads += [param.grad.view(-1) for param in self.amp.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = itertools.chain(
            self.actor_critic.actor.parameters(),
            self.actor_critic.critic.parameters(),
            self.rnd.predictor.parameters() if self.rnd else [],
            self.amp.parameters() if self.amp else []
        )
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                offset += numel

    def reduce_parameters_dagger(self):
        """Collect gradients from all GPUs and average them."""
        assert hasattr(self.actor_critic, "actor_dagger"), "actor_dagger not enabled in actor_critic, cannot reduce parameters of None"
        actor_grads = [param.grad.view(-1) for param in self.actor_critic.actor_dagger.parameters() if param.grad is not None]
        grads = actor_grads
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.actor_critic.actor_dagger.parameters()
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                offset += numel
        



    def _imitation_loss(self, actions_batch, obs_batch, ref_obs_batch, dagger_actions_batch):
        if dagger_actions_batch is not None:
            dagger_actions_batch = dagger_actions_batch.detach()
            if self.teacher_coef_mode == "kl":
                mu_batch = self.actor_critic.act_inference(observations=obs_batch, ref_observations=ref_obs_batch)
                sigma_batch = self.actor_critic.action_std.detach()
                imitation_loss = torch.sum(
                    (1 / actions_batch.shape[-1]) * (torch.sum((torch.square(mu_batch - dagger_actions_batch) / (2.0 * torch.square(sigma_batch) + 1e-5)), axis=-1) +  0.92 * torch.sum(torch.clamp_min(torch.log(sigma_batch), 0.0), axis=-1))
                )
            
            elif self.teacher_coef_mode == "norm":
                predicted_actions = self.actor_critic.act_inference(observations=obs_batch, ref_observations=ref_obs_batch)
                imitation_loss = torch.norm(
                    (predicted_actions - dagger_actions_batch)
                ).mean()
                    
        else:
            imitation_loss = torch.tensor(0.0).to(self.device)   
        return imitation_loss
    
    def _imitation_loss_original(self, actions_batch, critic_ref_obs_batch, mu_batch, sigma_batch, action_scale = 0.5):
        ref_action_batch = critic_ref_obs_batch[0][:, self.ref_action_idx: self.ref_action_idx + actions_batch.shape[-1]]
        ref_action_mask = critic_ref_obs_batch[1].long()
        action_processed = ref_action_batch / action_scale
        imitation_loss = torch.sum((
            (1 / actions_batch.shape[-1]) * (torch.sum((torch.square(mu_batch - action_processed) / (2.0 * torch.square(sigma_batch) + 1e-5)), axis=-1) +  0.92 * torch.sum(torch.clamp_min(torch.log(sigma_batch), 0.0), axis=-1))
        ) * ref_action_mask) / (ref_action_mask.sum() + 1e-5)
        return imitation_loss
    
    def update_dagger(self, actions_batch, obs_batch, ref_obs_batch, dagger_actions_batch):
        output = self.actor_critic.act_dagger_inference(obs_batch, ref_observations=ref_obs_batch)
        gt = self.teacher_coef * dagger_actions_batch + (1 - self.teacher_coef) * actions_batch
        loss = SmoothL2Loss()(output, gt)
        self.dagger_optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters_dagger()
        nn.utils.clip_grad_norm_(self.actor_critic.actor_dagger.parameters(), self.max_grad_norm)
        self.dagger_optimizer.step()
        return loss.item()
        