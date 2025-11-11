# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Optimized from RSL_RL/PPO
# Created by Yifei Yao, 10/12/2024

from __future__ import annotations
from weakref import ref

from numpy import std
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools

from torch.utils import weak
from torch.xpu import memory

from rsl_rl.modules import ActorCriticMMTransformer, AMPNet, ActorCriticMMTransformerV2, ActorCriticMMGPT
from rsl_rl.storage import RolloutStorageMM
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.utils import string_to_callable
import time

class DebugHook:
    def __init__(self, name):
        self.name = name

    def __call__(self, module, input):
        tensor = input[0]
        print(f"--- Hook at: {self.name} ---")
        print(f"dtype: {tensor.dtype}, shape: {tensor.shape}, device: {tensor.device}")
        print(f"has_nan: {torch.isnan(tensor).any().item()}")
        print(f"std: {tensor.std().item():.6f}, mean: {tensor.mean().item():.6f}")
        print(f"min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}")
        if tensor.std().item() == 0.0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!! WARNING: STANDARD DEVIATION IS ZERO !!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("-" * (len(self.name) + 14))
        
        
def print_gpu_memory_summary(point_in_code: str):
    """Prints a detailed summary of the VRAM usage."""
    return # disable debug print. Comment this out to enable debug print.
    # The 'abbreviated=True' version is a quick summary.
    # 'abbreviated=False' gives a detailed, multi-line report.
    print(f"--- GPU Memory Summary at: {point_in_code} ---")
    print(torch.cuda.memory_summary(abbreviated=False))
    print("--- End of Summary ---")

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
    actor_critic: ActorCriticMMTransformer | ActorCriticMMTransformerV2 | ActorCriticMMGPT

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
        max_lr_after_certain_epoch=5e-3,
        max_lr_restriction_epoch=25000,
        min_lr_after_certain_epoch=5e-5,
        min_lr_restriction_epoch=25000,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed", # "fixed", "adaptive", "linear", "cosine"
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
        teacher_updating_intervals=0, # when epoch > teacher_supervising_intervals, dagger will be updated
        teacher_apply_interval=5, # imitation loss will be add to PPO in * iters
        teacher_coef_mode="kl", # "kl" or "norm"
        teacher_update_interval=1,
        teacher_lr=5e-4,
        teacher_only_interval=0,  # Stage 1: Pure teacher output, student learns by imitation only
        hybrid_training_intervals=0,  # Stage 2: Weighted mix of teacher and student output
        hybrid_mix_coef=0.5,  # Coefficient for hybrid stage: output = hybrid_mix_coef * teacher + (1-hybrid_mix_coef) * student
        default_action=None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        auto_mix_precision: bool = False, # do not mix this with amp_cfg (that amp stands for Adversarial Motion Prior)


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

        self.auto_mix_precision = auto_mix_precision
        if auto_mix_precision:
            self.scaler = torch.amp.GradScaler(device=self.device)
        else:
            self.scaler = None


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
        self.min_lr_after_certain_epoch = min_lr_after_certain_epoch
        self.min_lr_restriction_epoch = min_lr_restriction_epoch
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
        self.hybrid_training_intervals = hybrid_training_intervals
        self.hybrid_mix_coef = hybrid_mix_coef
        self.teacher_supervising_intervals = teacher_supervising_intervals
        self.teacher_updating_intervals = teacher_updating_intervals
        self.teacher_apply_interval = teacher_apply_interval
        
        # Track current training stage
        self.current_stage = 1  # 1: teacher-only, 2: hybrid, 3: student
        
        # Validation for teacher_only and hybrid intervals
        assert teacher_coef is not None or teacher_only_interval == 0, "teacher_only_interval should be 0 if teacher_coef is None"
        assert (teacher_coef is None and teacher_loss_coef is None) or (teacher_loss_coef is not None and teacher_coef is not None), "teacher_coef and teacher_loss_coef should be set together"
        if teacher_only_interval > 0:
            assert hybrid_training_intervals >= teacher_only_interval, "hybrid_training_intervals must be >= teacher_only_interval"
        if hybrid_training_intervals > 0:
            assert teacher_only_interval > 0, "teacher_only_interval must be > 0 when using hybrid_training_intervals"
            assert 0.0 <= hybrid_mix_coef <= 1.0, "hybrid_mix_coef should be in range [0.0, 1.0]"

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        is_log_std = self.actor_critic.noise_std_type == "log"
        self.optimizer = optim.AdamW(
            itertools.chain(
                self.actor_critic.actor.parameters(),
                self.actor_critic.critic.parameters(),
                [self.actor_critic.std] if not is_log_std else [self.actor_critic.log_std],
            ),
            lr=learning_rate) # Avoid optimizing dagger parameters if self.teacher_coef is None
        self.imitation_optimizer = optim.AdamW(
            itertools.chain(
                self.actor_critic.actor.parameters(),
                [self.actor_critic.std_dagger] if not is_log_std else [self.actor_critic.log_std_dagger],
            ),
            lr=learning_rate) if self.teacher_coef is not None else None
        # self.optimizer = optim.AdamW(
        #     self.actor_critic.parameters(),
        #     lr=learning_rate
        # )
        self.teacher_update_interval = teacher_update_interval
        self.transition = RolloutStorageMM.Transition()
        
        if self.schedule == "linear":
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: 1 - epoch / num_learning_epochs
            )
        elif self.schedule == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_learning_epochs,# * num_mini_batches, # total number of updates
            )
        else:
            self.lr_scheduler = None

        if self.teacher_coef is not None and self.teacher_loss_coef is not None:
            assert self.teacher_coef_mode in ["kl", "norm", "original_kl", "mse", "huber"], \
                "teacher_coef_mode should be one of: 'kl', 'norm', 'original_kl', 'mse', 'huber'"
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
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Stage 1: Teacher-Only (pure imitation)
        # Stage 2: Hybrid (weighted mix of teacher and student)
        # Stage 3: Student (normal PPO with teacher guidance)
        
        # Compute student actions - always store student actions for training
        student_actions = self.actor_critic.act(obs, ref_observations=ref_obs).detach()
        self.transition.actions = student_actions  # Always use student actions for training
        
        # Compute values and other metrics based on student
        self.transition.values = self.actor_critic.evaluate(critic_obs, ref_critic_observations=ref_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(student_actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # Determine which actions to RETURN (for environment execution) based on current training stage
        if self.teacher_coef is not None:
            teacher_actions = self.actor_critic.act_dagger(obs, ref_observations=None).detach()
            _ = self.actor_critic.get_actions_log_prob_dagger(teacher_actions)
            self.transition.dagger_actions = teacher_actions
            
            # if self.current_stage == 1:
            #     # Stage 1: Return pure teacher actions to environment
            #     actions_to_execute = teacher_actions
            # elif self.current_stage == 2:
            #     # Stage 2: Return weighted mix to environment
            #     actions_to_execute = (self.hybrid_mix_coef * teacher_actions + 
            #                          (1 - self.hybrid_mix_coef) * student_actions)
            # else:
            #     # Stage 3: Return student actions to environment
            #     actions_to_execute = student_actions
            actions_to_execute = student_actions
        else:
            # No teacher, return student actions
            actions_to_execute = student_actions
        
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.reference_observations = ref_obs[0] if ref_obs is not None else None
        self.transition.reference_observations_mask = ref_obs[1] if ref_obs is not None else None
        self.transition.critic_observations = critic_obs 
        self.transition.critic_reference_observations = ref_critic_obs[0] if ref_critic_obs is not None else None
        self.transition.critic_reference_observations_mask = ref_critic_obs[1] if ref_critic_obs is not None else None
        if self.amp_cfg is not None and ref_critic_obs[0] is not None:
            self.transition.amp_observations = string_to_callable(self.amp_cfg["amp_obs_extractor"])(env=self.amp_cfg["_env"])
            ref_amp_observation, ref_amp_mask = string_to_callable(self.amp_cfg["amp_ref_obs_extractor"])(env=self.amp_cfg["_env"])
            self.transition.amp_reference_observations = ref_amp_observation
            self.transition.amp_reference_observations_mask = ref_amp_mask
        return actions_to_execute

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
        # Update current training stage based on epoch
        if epoch < self.teacher_only_interval:
            self.current_stage = 1  # Teacher-only stage
        elif epoch < self.hybrid_training_intervals:
            self.current_stage = 2  # Hybrid stage
        else:
            self.current_stage = 3  # Student stage
            
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
            mean_pred_pos_prob = 0.0
            mean_pred_neg_prob = 0.0

        if self.actor_critic.is_recurrent:
            # generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            generator = self.storage.buffer_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            assert self.actor_critic.is_recurrent == False, "MM-PPO does not support recurrent actor-critic networks."
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        assert self.storage is not None, "Storage is not initialized. Please call init_storage before update."
        print_gpu_memory_summary("before mini_batch_generator")
        
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
            obs_cur_state,
            ref_obs_cur_state,
            ref_obs_cur_mask,
        ) in generator:

            num_aug = 1

            original_batch_size = obs_batch.shape[-2]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, ref_obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, ref_obs=ref_obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )

                num_aug = int(obs_batch.shape[-2] / original_batch_size)

                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                self.actor_critic.act(obs_batch, ref_observations=ref_obs_batch, masks=masks_batch, memory=hid_states_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(
                    critic_obs_batch, ref_critic_observations=critic_ref_obs_batch,
                    masks=masks_batch, memory=hid_states_batch
                )

            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # Imitation Entropy loss (optional, arxiv: 2409.08904)
            # This loss is calculated only when critic_ref_obs_batch is not None, otherwise 0.0
            # By default, we assume that ref_action_batch = critic_ref_obs_batch[0][:, ref_action_idx: num_actions + ref_action_idx] where ref_action_idx = 0
            # Loss: \sum_i [\sqrt(2 * \pi * \sigma_i^2) + (mu_i - ref_action_i)^2 / (2 * \sigma_i^2)]
            # print("Inside update, current schedule is:", self.schedule)
            print_gpu_memory_summary("After forward pass")
            if self.max_lr_restriction_epoch != 0 and epoch > self.max_lr_restriction_epoch:
                self.max_lr = self.max_lr_after_certain_epoch
            
            if self.min_lr_restriction_epoch != 0 and epoch > self.min_lr_restriction_epoch:
                self.min_lr = self.min_lr_after_certain_epoch
            
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
                        
            elif self.schedule == "linear" or self.schedule == "cosine":
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()
                self.lr_scheduler.step()
                self.learning_rate = self.lr_scheduler.get_last_lr()[0]
                # print(f"Updated learning rate to {self.learning_rate} at epoch {epoch}.")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            # Surrogate loss
            # In teacher-only stage, skip PPO loss computation (student just imitates)
            if epoch < self.teacher_only_interval:
                # Teacher-only stage: no PPO update, pure imitation
                surrogate_loss = torch.tensor(0.0, device=self.device)
                value_loss = torch.tensor(0.0, device=self.device)
            else:
                # Normal PPO or Hybrid stage: compute surrogate and value loss
                with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
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

                    with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                        mean_actions_batch = self.actor_critic.act_inference(obs_batch, ref_observations=ref_obs_batch, masks=masks_batch)

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
                with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                    predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mse_loss = torch.nn.MSELoss()
                rnd_loss = mse_loss(predicted_embedding, target_embedding) 

           
            

                    

            imitation_loss = self._imitation_loss(actions_batch=actions_batch, obs_batch=obs_batch, ref_obs_batch=ref_obs_batch, dagger_actions_batch=dagger_actions_batch)      
            print_gpu_memory_summary("Before optimizer.zero_grad")
            
            # In teacher-only stage, imitation loss is the main loss
            if epoch < self.teacher_only_interval:
                # Pure imitation: only use imitation loss
                loss = imitation_loss
            elif self.teacher_only_interval <= epoch < self.hybrid_training_intervals:
                # Hybrid stage: combine PPO loss with stronger imitation loss
                # Use a higher weight for imitation during hybrid stage
                hybrid_imitation_weight = 2.0 * (self.teacher_loss_coef if self.teacher_loss_coef is not None else 0.1)
                loss = loss + hybrid_imitation_weight * imitation_loss
            # else: Stage 3 (student), imitation loss applied separately later
            
            # Gradient step
            self.optimizer.zero_grad()
            if not self.auto_mix_precision:
                try:
                    loss.backward()
                except RuntimeError as e:
                    print(f"Backward NaN detected, beginning debug hooks...")
                    
                    print("Saving the problematic batch to 'debug_batch.pt' for further analysis.")
                    torch.save({
                        'obs_batch': obs_batch,
                        'ref_obs_batch': ref_obs_batch,
                        'critic_obs_batch': critic_obs_batch,
                        'critic_ref_obs_batch': critic_ref_obs_batch,
                        'actions_batch': actions_batch,
                        'target_values_batch': target_values_batch,
                        'advantages_batch': advantages_batch,
                        'returns_batch': returns_batch,
                        'old_actions_log_prob_batch': old_actions_log_prob_batch,   
                        'old_mu_batch': old_mu_batch,
                        'old_sigma_batch': old_sigma_batch,
                        'dagger_actions_batch': dagger_actions_batch,
                        'hid_states_batch': hid_states_batch,
                        'masks_batch': masks_batch,
                        'rnd_state_batch': rnd_state_batch,
                        'obs_prev_state': obs_prev_state,
                        'ref_obs_prev_state': ref_obs_prev_state,
                        'ref_obs_prev_mask': ref_obs_prev_mask,
                        'obs_cur_state': obs_cur_state,
                        'ref_obs_cur_state': ref_obs_cur_state,
                        'ref_obs_cur_mask': ref_obs_cur_mask,
                    }, 'debug_batch.pt')
                    
                    print("Registering debug hooks to actor.decoder and critic.decoder...")
                    actor_hooks = []
                    critic_hooks = []
                    for i, layer in enumerate(self.actor_critic.actor.decoder.encoder.layers):
                        hook = layer.register_forward_pre_hook(DebugHook(f"actor.decoder.layer.{i}"))
                        actor_hooks.append(hook)
                    for i, layer in enumerate(self.actor_critic.critic.decoder.encoder.layers):
                        hook = layer.register_forward_pre_hook(DebugHook(f"critic.decoder.layer.{i}"))
                        critic_hooks.append(hook)
                    # Re-run forward pass to trigger hooks: act & evaluate
                    self.actor_critic.act(obs_batch, ref_observations=ref_obs_batch, masks=masks_batch, memory=hid_states_batch)
                    _ = self.actor_critic.evaluate(
                        critic_obs_batch, ref_critic_observations=critic_ref_obs_batch,
                        masks=masks_batch, memory=hid_states_batch
                    )
                    print("Debug hooks executed. Please check the outputs above for anomalies.")
                    raise e
            else:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            print_gpu_memory_summary("After backward pass")

            if self.amp and (obs_prev_state is not None and obs_cur_state is not None and ref_obs_prev_state is not None and ref_obs_cur_state is not None and ref_obs_prev_mask is not None and ref_obs_cur_mask is not None):

                # amp_optimization_steps = self.amp_cfg.get("amp_optimization_steps", 3)
                policy_score = self.amp.forward(torch.cat([obs_prev_state, obs_cur_state], dim=-1))
                expert_score = self.amp.forward(torch.cat([ref_obs_prev_state, ref_obs_cur_state], dim=-1))
                
                pred_pos_prob = self.amp.out_activation(policy_score).mean()
                pred_neg_prob = self.amp.out_activation(expert_score).mean()
                pred_pos_acc = self.amp.policy_acc(policy_score)
                pred_neg_acc = self.amp.expert_acc(expert_score, ref_obs_cur_mask * ref_obs_prev_mask)
                policy_loss = self.amp.policy_loss(policy_score)
                expert_loss = self.amp.expert_loss(expert_score, ref_obs_cur_mask)
                gradient_penalty = self.amp.expert_grad_penalty(obs_cur_state, ref_obs_cur_state, ref_obs_cur_mask * ref_obs_prev_mask)
                amp_loss = 0.5 * (policy_loss + expert_loss) + gradient_penalty * self.amp_cfg["gradient_penalty_coeff"]

                if (epoch % self.amp_cfg["amp_update_interval"] == 0) or (epoch < self.amp_cfg["amp_pretrain_steps"]):
                    self.amp_optimizer.zero_grad()
                    amp_loss.backward()
                
                mean_amp_loss += amp_loss.item()
                mean_gradient_penalty += gradient_penalty.item() * self.amp_cfg["gradient_penalty_coeff"]
                mean_pred_pos_acc += pred_pos_acc.item()
                mean_pred_neg_acc += pred_neg_acc.item()
                mean_pred_pos_prob += pred_pos_prob.item()
                mean_pred_neg_prob += pred_neg_prob.item()

            # Imitation loss applied separately only in Stage 3 (student stage)
            if (self.teacher_loss_coef is not None and 
                epoch >= self.hybrid_training_intervals and 
                epoch > self.teacher_updating_intervals and 
                (epoch+1) % self.teacher_apply_interval == 0):
                backward_imitation_loss = imitation_loss * self.teacher_loss_coef
                self.imitation_optimizer.zero_grad()
                if not self.auto_mix_precision:
                    backward_imitation_loss.backward()
                else:
                    self.scaler.scale(backward_imitation_loss).backward()
                    self.scaler.unscale_(self.imitation_optimizer)

            # for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                if not self.auto_mix_precision:
                    rnd_loss.backward()
                else:
                    self.scaler.scale(rnd_loss).backward()
                    self.scaler.unscale_(self.rnd_optimizer)
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
            if not self.auto_mix_precision:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)

            if self.amp and ((epoch % self.amp_cfg["amp_update_interval"] == 0) or (epoch < self.amp_cfg["amp_pretrain_steps"])):
                self.amp_optimizer.step()

            # Imitation optimizer step only in Stage 3
            if (self.teacher_loss_coef is not None and 
                epoch >= self.hybrid_training_intervals and
                epoch > self.teacher_supervising_intervals and 
                (epoch+1) % self.teacher_apply_interval == 0):
                nn.utils.clip_grad_norm_(self.actor_critic.actor_dagger.parameters(), self.max_grad_norm)
                if not self.auto_mix_precision:
                    self.imitation_optimizer.step()
                else:
                    self.scaler.step(self.imitation_optimizer)

            if self.auto_mix_precision:
                self.scaler.update()

            print_gpu_memory_summary("After optimizer.step")
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_imitation_loss += imitation_loss.item()
            mean_entropy += entropy_batch.mean().item()

            # DAgger update: start after hybrid stage
            if ((epoch+1) % self.teacher_update_interval == 0 and 
                self.dagger_optimizer is not None and 
                epoch >= self.hybrid_training_intervals and
                epoch > self.teacher_supervising_intervals):
                dagger_loss = self.update_dagger(actions_batch, obs_batch, ref_obs_batch, dagger_actions_batch)
            else:
                dagger_loss = 0.0
            mean_dagger_loss += dagger_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates
        mean_dagger_loss /= num_updates
        mean_entropy /= num_updates
        if self.amp:
            mean_amp_loss /= num_updates
            mean_gradient_penalty /= num_updates
            mean_pred_pos_acc /= num_updates
            mean_pred_neg_acc /= num_updates
            mean_pred_pos_prob /= num_updates
            mean_pred_neg_prob /= num_updates
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
        print_gpu_memory_summary("End of epoch")
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
            loss_dict["mean_pred_pos_prob"] = mean_pred_pos_prob
            loss_dict["mean_pred_neg_prob"] = mean_pred_neg_prob

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
        std_grads = []
        is_log_std = hasattr(self.actor_critic, "log_std")
        if not is_log_std:
            std_param = self.actor_critic.std
        else:
            std_param = self.actor_critic.log_std
        if std_param.grad is not None:
            std_grads.append(std_param.grad.view(-1))
        grads = actor_grads + critic_grads + std_grads
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.predictor.parameters() if param.grad is not None]
        if self.amp:
            grads += [param.grad.view(-1) for param in self.amp.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        is_log_std = hasattr(self.actor_critic, "log_std")
        all_params = itertools.chain(
            self.actor_critic.actor.parameters(),
            self.actor_critic.critic.parameters(),
            [self.actor_critic.std] if not is_log_std else [self.actor_critic.log_std],
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
        """
        Compute imitation loss between student and teacher outputs.
        
        Modes:
        - "kl": KL-divergence with entropy regularization (original)
        - "original_kl": Standard KL divergence between Gaussian distributions
        - "norm": L2 norm distance
        - "mse": Mean Squared Error (simple and effective)
        - "huber": Huber loss (robust to outliers)
        """
        if dagger_actions_batch is not None:
            dagger_actions_batch = dagger_actions_batch.detach()
            
            if self.teacher_coef_mode == "kl":
                with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                    mu_batch = self.actor_critic.act_inference(observations=obs_batch, ref_observations=ref_obs_batch)
                sigma_batch = self.actor_critic.action_std.detach()
                imitation_loss = torch.mean(
                    (1 / actions_batch.shape[-1]) * (torch.sum((torch.square(mu_batch - dagger_actions_batch) / (2.0 * torch.square(sigma_batch) + 1e-5)), axis=-1) +  0.92 * torch.sum(torch.clamp_min(torch.log(sigma_batch), 0.0), axis=-1))
                )

            elif self.teacher_coef_mode == "original_kl":
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                with torch.no_grad():
                    self.actor_critic.act_dagger(obs_batch, None)
                    dagger_mu_batch = self.actor_critic.action_mean_dagger.detach()
                    dagger_sigma_batch = self.actor_critic.action_std_dagger.detach()
                imitation_loss = torch.sum(
                    torch.log(sigma_batch / dagger_sigma_batch + 1.0e-5)
                    + (torch.square(dagger_sigma_batch) + torch.square(dagger_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                imitation_loss = imitation_loss.mean()
                
            elif self.teacher_coef_mode == "norm":
                with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                    predicted_actions = self.actor_critic.act_inference(observations=obs_batch, ref_observations=ref_obs_batch)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                        dagger_actions_batch = self.actor_critic.act_dagger_inference(observations=obs_batch, ref_observations=None)
                imitation_loss = torch.norm(
                    (predicted_actions - dagger_actions_batch)
                ).mean()
            
            elif self.teacher_coef_mode == "mse":
                # Mean Squared Error: simple and effective
                with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                    predicted_actions = self.actor_critic.act_inference(observations=obs_batch, ref_observations=ref_obs_batch)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                        dagger_actions_batch = self.actor_critic.act_dagger_inference(observations=obs_batch, ref_observations=None)
                
                # MSE per action dimension, then mean
                diff = predicted_actions - dagger_actions_batch
                imitation_loss = (diff ** 2).mean()
            
            elif self.teacher_coef_mode == "huber":
                # Huber loss: robust to outliers, smooth transition
                with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                    predicted_actions = self.actor_critic.act_inference(observations=obs_batch, ref_observations=ref_obs_batch)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", enabled=self.auto_mix_precision):
                        dagger_actions_batch = self.actor_critic.act_dagger_inference(observations=obs_batch, ref_observations=None)
                
                # Huber loss with delta=1.0 (standard)
                delta = 1.0
                diff = predicted_actions - dagger_actions_batch
                abs_diff = torch.abs(diff)
                
                # Huber: 0.5*x^2 if |x| < delta, else delta*(|x| - 0.5*delta)
                huber = torch.where(
                    abs_diff < delta,
                    0.5 * diff ** 2,
                    delta * (abs_diff - 0.5 * delta)
                )
                imitation_loss = huber.mean()
            
            else:
                raise ValueError(f"Unknown teacher_coef_mode: {self.teacher_coef_mode}. "
                               f"Choose from ['kl', 'original_kl', 'norm', 'mse', 'huber']")
                    
        else:
            imitation_loss = torch.tensor(0.0).to(self.device)   
        return imitation_loss
    
    # def _imitation_loss_original(self, actions_batch, critic_ref_obs_batch, mu_batch, sigma_batch, action_scale = 0.5):
    #     ref_action_batch = critic_ref_obs_batch[0][:, self.ref_action_idx: self.ref_action_idx + actions_batch.shape[-1]]
    #     ref_action_mask = critic_ref_obs_batch[1].long()
    #     action_processed = ref_action_batch / action_scale
    #     imitation_loss = torch.sum((
    #         (1 / actions_batch.shape[-1]) * (torch.sum((torch.square(mu_batch - action_processed) / (2.0 * torch.square(sigma_batch) + 1e-5)), axis=-1) +  0.92 * torch.sum(torch.clamp_min(torch.log(sigma_batch), 0.0), axis=-1))
    #     ) * ref_action_mask) / (ref_action_mask.sum() + 1e-5)
    #     return imitation_loss
    
    def update_dagger(self, actions_batch, obs_batch, ref_obs_batch, dagger_actions_batch): # for this function, we haven't figured out how to use autocast. Maybe a seperate scaler is needed.
        output = self.actor_critic.act_dagger_inference(obs_batch, ref_observations=None)
        with torch.no_grad():
            gt = self.teacher_coef * output + (1 - self.teacher_coef) * self.actor_critic.act_inference(obs_batch, ref_observations=ref_obs_batch)
        gt = gt.detach()
        loss = SmoothL2Loss()(output, gt)
        self.dagger_optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters_dagger()
        nn.utils.clip_grad_norm_(self.actor_critic.actor_dagger.parameters(), self.max_grad_norm)
        self.dagger_optimizer.step()
        return loss.item()
        