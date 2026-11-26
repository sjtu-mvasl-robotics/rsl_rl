# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import MMPPO, MMDistillation
from rsl_rl.env import MMVecEnv
from rsl_rl.modules import *
from rsl_rl.utils import store_code_state, string_to_callable


def _compose_amp_training_rewards(
    base_rewards: torch.Tensor,
    amp_rewards: torch.Tensor,
    *,
    scale: float,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the PPO reward and the AMP shaping term actually applied to it."""

    applied_amp_rewards = (
        amp_rewards * scale if enabled else torch.zeros_like(amp_rewards)
    )
    return base_rewards + applied_amp_rewards, applied_amp_rewards


def _amp_reward_enabled(iteration: int, pretrain_steps: int) -> bool:
    """Use AMP shaping starting at the configured integer iteration."""

    return iteration >= pretrain_steps


def _unpack_observations_result(observation_result):
    """Normalize old and current Isaac Lab observation-wrapper contracts.

    Older Isaac Lab wrappers return ``(policy_obs, extras)`` while current
    wrappers return a TensorDict containing the ``policy`` and optional
    privileged groups directly.  Keep the raw dict-like object in ``extras``
    so MMGPT self-supervision and the privileged critic see the same sample.
    """

    if isinstance(observation_result, tuple):
        if len(observation_result) != 2:
            raise RuntimeError(
                "legacy get_observations() must return (policy_obs, extras)"
            )
        return observation_result

    try:
        policy_obs = observation_result["policy"]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError(
            "get_observations() must return either (policy_obs, extras) or "
            "a dict-like object containing a 'policy' group"
        ) from exc
    return policy_obs, {"observations": observation_result}


def _restore_sonic_adaptive_sampler_state(env, loaded_dict) -> bool:
    """Restore curriculum state only when the active task owns that state.

    Adaptive checkpoints remain valid policy checkpoints for the ordinary
    frame-zero evaluation task.  That task deliberately ignores curriculum
    statistics, while an adaptive resume must advertise support and restore
    the payload strictly.
    """

    adaptive_state = loaded_dict.get("sonic_adaptive_sampler_state_dict")
    if adaptive_state is None:
        return False
    adaptive_env = getattr(env, "unwrapped", env)
    adaptive_enabled = getattr(
        adaptive_env, "sonic_adaptive_sampling_enabled", None
    )
    if not callable(adaptive_enabled) or not adaptive_enabled():
        return False
    if os.environ.get("MMGPT_RESET_EXPANDED_SAMPLER") == "1":
        previous_count = int(adaptive_state["motion_num_frames"].numel())
        current_count = len(adaptive_env.ref_observation_manager.files)
        if previous_count != current_count:
            if previous_count != 65 or current_count not in (126, 196):
                raise ValueError(f"unexpected expanded sampler transition: {previous_count}->{current_count}")
            print(f"[ExpandedDAgger] fresh sampler: {previous_count}->{current_count}; model/optimizer resume unchanged", flush=True)
            return False
    load_adaptive_state = getattr(
        adaptive_env, "load_sonic_adaptive_state_dict", None
    )
    if not callable(load_adaptive_state):
        raise RuntimeError(
            "adaptive task cannot restore the checkpoint's SONIC sampler state"
        )
    load_adaptive_state(adaptive_state)
    return True


class OnPolicyRunnerMM:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: MMVecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.metrics_jsonl_path = os.environ.get("GBC_TRAIN_METRICS_JSONL")

        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] == "MMPPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        obs, extras = _unpack_observations_result(self.env.get_observations())
        ref_obs_tuple, ref_extras = self.env.get_reference_observations()
            
        num_obs = obs.shape[1]
        if ref_obs_tuple is not None:
            num_ref_obs = ref_obs_tuple[0].shape[1]
        else:
            num_ref_obs = 0

        if self.training_type == "rl":
            self.privileged_obs_type = "critic"
        elif self.training_type == "distillation":
            self.privileged_obs_type = "teacher"
        else:
            self.privileged_obs_type = None

        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                num_critic_obs = extras["observations"]["critic"].shape[1]
            else:
                num_critic_obs = num_obs
            if "critic" in ref_extras["ref_observations"]:
                num_critic_ref_obs = ref_extras["ref_observations"]["critic"][0].shape[1]
            else:
                num_critic_ref_obs = num_ref_obs
        elif self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                num_critic_obs = extras["observations"]["teacher"].shape[1]
            else:
                num_critic_obs = num_obs
            if "teacher" in ref_extras["ref_observations"]:
                num_critic_ref_obs = ref_extras["ref_observations"]["teacher"][0].shape[1]
            else:
                num_critic_ref_obs = num_ref_obs

        else:
            num_critic_obs = num_obs
            num_critic_ref_obs = num_ref_obs

        actor_critic_dim = {
            "num_actor_obs": num_obs,
            "num_actor_ref_obs": num_ref_obs,
            "num_critic_obs": num_critic_obs,
            "num_critic_ref_obs": num_critic_ref_obs,
        }

        actor_critic_name = self.policy_cfg.pop("class_name")
        actor_critic_class = eval(actor_critic_name)  # ActorCritic
        self._is_mmgpt = actor_critic_name == "ActorCriticMMGPT"
        self._mmgpt_dreamer_target_term_name = self.policy_cfg.get(
            "dreamer_target_term_name", "target_actions"
        )
        self._mmgpt_external_target_enabled = self._is_mmgpt and callable(
            getattr(self.env, "get_mmgpt_self_supervision_observations", None)
        )
        if actor_critic_name in ["ActorCriticMMTransformerV2", "ActorCriticMMGPT", "ActorCriticMLPV2", "ActorCriticMLPV2MoE"]:
            def get_term_dict(dict_names, dict_dims):
                ret = dict(
                    (key, dict((name, dim[0]) for name, dim in zip(dict_names[key], dict_dims[key]) if dim[0] > 0))
                    for key in dict_names.keys()
                )
                if "critic" not in ret:
                    ret["critic"] = ret["policy"]
                return ret
            
            def get_term_steps(dict_names, dict_dims, dict_cfgs):
                ret = dict(
                    (
                        key,
                        dict(
                            (
                                name,
                                int(cfg.encoding_steps)
                                if getattr(cfg, "encoding_steps", None) is not None
                                else cfg.steps // cfg.sample_rate,
                            )
                            for name, dim, cfg in zip(
                                dict_names[key], dict_dims[key], dict_cfgs[key]
                            )
                            if dim[0] > 0
                        ),
                    )
                    for key in dict_names.keys()
                )
                if "critic" not in ret:
                    ret["critic"] = ret["policy"]
                return ret
            # actor_critic_dim += {
            #     "term_dict": get_term_dict(
            #         self.env.unwrapped.observation_manager.active_terms,
            #         self.env.unwrapped.observation_manager.group_obs_term_dim,
            #     ),
            #     "ref_term_dict": get_term_dict(
            #         self.env.unwrapped.ref_observation_manager.active_terms,
            #         self.env.unwrapped.ref_observation_manager.group_ref_obs_term_dim,
            #     ),
            # }
            actor_critic_dim["term_dict"] = get_term_dict(
                self.env.unwrapped.observation_manager.active_terms,
                self.env.unwrapped.observation_manager.group_obs_term_dim,
            )
            actor_critic_dim["ref_term_dict"] = get_term_dict(
                self.env.unwrapped.ref_observation_manager.active_terms,
                self.env.unwrapped.ref_observation_manager.group_ref_obs_term_dim,
            )
            if actor_critic_name == "ActorCriticMMGPT":
                actor_critic_dim["ref_term_steps"] = get_term_steps(
                    self.env.unwrapped.ref_observation_manager.active_terms,
                    self.env.unwrapped.ref_observation_manager.group_ref_obs_term_dim,
                    self.env.unwrapped.ref_observation_manager.group_ref_obs_term_cfgs,
                )
                actor_critic_dim["default_joint_pos"] = self.env.unwrapped.scene['robot'].data.default_joint_pos[0] # shape: (num_joints,)
                actor_critic_dim["action_is_delta"] = bool(
                    getattr(self.env.unwrapped, "is_delta", False)
                )
                actor_critic_dim["action_clip_value"] = getattr(
                    self.env.unwrapped, "action_clip_value", None
                )
        elif actor_critic_name == "StudentTeacherMMTransformer":
            actor_critic_dim = {
                "num_teacher_obs": num_critic_obs,
                "num_teacher_ref_obs": num_critic_ref_obs,
                "num_student_obs": num_obs,
                "num_student_ref_obs": num_ref_obs,
            }
        actor_critic = actor_critic_class(
            **actor_critic_dim,
            num_actions=self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        actor_parameter_count = sum(
            parameter.numel() for parameter in actor_critic.actor.parameters()
        )
        critic_parameter_count = sum(
            parameter.numel() for parameter in actor_critic.critic.parameters()
        )
        recurrent_contract = "none"
        if self._is_mmgpt and actor_critic.actor.ttt_memory is not None:
            ttt_memory = actor_critic.actor.ttt_memory
            recurrent_contract = (
                f"ttt_shape={ttt_memory.state_shape}, "
                f"ttt_scalars_per_env="
                f"{ttt_memory.state_numel_per_environment:,}, "
                f"ttt_slow_parameters="
                f"{sum(parameter.numel() for parameter in ttt_memory.parameters()):,}"
            )
        print(
            "[OnPolicyRunnerMM] Student network contract: "
            f"actor_parameters={actor_parameter_count:,}, "
            f"critic_parameters={critic_parameter_count:,}, "
            f"recurrent_state={recurrent_contract}"
        )

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            rnd_state = extras["observations"].get("rnd_state")
            # for debug use only

            if rnd_state is None:
                rnd_state = extras["observations"].get("policy")
                if rnd_state is None:
                    raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            num_rnd_state = rnd_state.shape[1]
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            self.alg_cfg["rnd_cfg"]["weight"] *= self.env.unwrapped.step_dt

        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = self.env

        if "amp_cfg" in self.alg_cfg and self.alg_cfg["amp_cfg"] is not None:
            self.alg_cfg["amp_cfg"]["_env"] = self.env
            self.amp_cfg = self.alg_cfg["amp_cfg"]
        else:
            self.amp_cfg = None

        # Keep the declarative teacher contract in alg_cfg for logs and saved
        # run metadata; MMPPO accepts unknown future kwargs but receives the
        # instantiated frozen module separately below.
        fixed_teacher_cfg = self.alg_cfg.get("fixed_teacher_cfg", None)
        fixed_teacher = None
        self.fixed_teacher_reference_group = "policy"
        if fixed_teacher_cfg is not None:
            if self.training_type != "rl" or not self._is_mmgpt:
                raise ValueError(
                    "fixed SONIC expert distillation requires MMGPT RL"
                )
            fixed_teacher_cfg = dict(fixed_teacher_cfg)
            self.fixed_teacher_reference_group = str(
                fixed_teacher_cfg.pop("reference_group", "policy")
            )
            expected_policy_terms = tuple(
                fixed_teacher_cfg.pop("expected_policy_terms", ())
            )
            expected_ref_terms = tuple(
                fixed_teacher_cfg.pop("expected_ref_terms", ())
            )
            policy_term_dims = actor_critic_dim["term_dict"]["policy"]
            if self.fixed_teacher_reference_group not in actor_critic_dim[
                "ref_term_dict"
            ]:
                raise ValueError(
                    "fixed SONIC expert reference group is missing: "
                    f"{self.fixed_teacher_reference_group!r}; available="
                    f"{tuple(actor_critic_dim['ref_term_dict'])}"
                )
            policy_ref_term_dims = actor_critic_dim["ref_term_dict"][
                self.fixed_teacher_reference_group
            ]
            if expected_policy_terms and tuple(policy_term_dims) != (
                expected_policy_terms
            ):
                raise ValueError(
                    "fixed SONIC expert policy terms differ from its checkpoint "
                    f"contract: expected {expected_policy_terms}, got "
                    f"{tuple(policy_term_dims)}"
                )
            if expected_ref_terms and tuple(policy_ref_term_dims) != (
                expected_ref_terms
            ):
                raise ValueError(
                    "fixed SONIC expert reference terms differ from its "
                    f"checkpoint contract: expected {expected_ref_terms}, got "
                    f"{tuple(policy_ref_term_dims)}"
                )
            fixed_teacher = FixedSonicMLPTeacher(
                policy_term_dims=policy_term_dims,
                policy_ref_term_dims=policy_ref_term_dims,
                num_actions=self.env.num_actions,
                student_is_delta=bool(
                    getattr(self.env.unwrapped, "is_delta", False)
                ),
                action_clip_value=getattr(
                    self.env.unwrapped, "action_clip_value", None
                ),
                **fixed_teacher_cfg,
            )
            print(
                "[OnPolicyRunnerMM] Loaded frozen SONIC expert "
                f"iteration={fixed_teacher.checkpoint_iteration}, "
                f"actor_parameters={fixed_teacher.actor_parameter_count:,}, "
                f"reference_group={self.fixed_teacher_reference_group}, "
                f"checkpoint={fixed_teacher.checkpoint_path}"
            )

        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: MMPPO = alg_class(
            actor_critic,
            device=self.device,
            fixed_teacher=fixed_teacher,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        # ``expert_freeze_loaded_normalization`` applies only after a
        # checkpoint's empirical statistics have actually been restored.
        # Keep this runner-side latch separate from the algorithm config so a
        # fresh run with the same config can still fit its normalizers.
        self._loaded_normalization_frozen = False
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.ref_obs_normalizer = EmpiricalNormalization(shape=[num_ref_obs], until=1.0e8).to(self.device) if num_ref_obs > 0 else None
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
            self.critic_ref_obs_normalizer = EmpiricalNormalization(shape=[num_critic_ref_obs], until=1.0e8).to(self.device) if num_critic_ref_obs > 0 else None
            
            # Load normalizer from dagger path if available
            if "load_dagger_path" in self.policy_cfg and self.policy_cfg["load_dagger_path"] is not None:
                dagger_path = self.policy_cfg["load_dagger_path"]
                print(f"Loading observation normalizer from dagger path: {dagger_path}")
                try:
                    loaded_dict = torch.load(dagger_path, map_location=self.device)
                    if "obs_norm_state_dict" in loaded_dict:
                        self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                        print("Loaded obs_norm_state_dict from dagger checkpoint.")
                    if "critic_obs_norm_state_dict" in loaded_dict:
                        self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
                    if self.ref_obs_normalizer is not None and "ref_obs_norm_state_dict" in loaded_dict:
                        self.ref_obs_normalizer.load_state_dict(loaded_dict["ref_obs_norm_state_dict"])
                    if self.critic_ref_obs_normalizer is not None and "critic_ref_obs_norm_state_dict" in loaded_dict:
                        self.critic_ref_obs_normalizer.load_state_dict(loaded_dict["critic_ref_obs_norm_state_dict"])
                except Exception as e:
                    print(f"Failed to load normalizer from dagger path: {e}")
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.ref_obs_normalizer = torch.nn.Identity().to(self.device) if num_ref_obs > 0 else None
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_ref_obs_normalizer = torch.nn.Identity().to(self.device) if num_critic_ref_obs > 0 else None
        # init storage and model
        # self.alg.init_storage(
        #     self.env.num_envs,
        #     self.num_steps_per_env,
        #     [num_obs],
        #     [num_critic_obs],
        #     [self.env.num_actions],
        # )
        self.alg.init_storage(
            training_type=self.training_type,
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            actor_ref_obs_shape=[num_ref_obs] if num_ref_obs > 0 else [None],
            critic_obs_shape=[num_critic_obs],
            critic_ref_obs_shape=[num_critic_ref_obs] if num_critic_ref_obs > 0 else [None],
            action_shape=[self.env.num_actions],
            **(
                {
                    "store_raw_observations": self._is_mmgpt,
                    "dreamer_target_shape": (
                        [self.env.num_actions]
                        if self._mmgpt_external_target_enabled
                        else None
                    ),
                }
                if self.training_type == "rl"
                else {}
            ),
        )

        # --- Planner RL setup ---
        self._planner_rl_enabled = getattr(self.env, 'planner_rl_enabled', False)
        self._planner_rl_episode_mode = False  # episode-level planner RL (autoregressive multi-regen)
        if self._planner_rl_enabled:
            diffusion_buffer = self.env.diffusion_buffer
            planner_params = diffusion_buffer.get_planner_trainable_params()
            planner_lr = self.cfg.get("planner_learning_rate", 1e-5)
            planner_max_grad_norm = self.cfg.get("planner_max_grad_norm", 1.0)
            planner_use_muon = self.cfg.get("planner_use_muon", False)
            self.alg.init_planner_optimizer(
                planner_params, lr=planner_lr, max_grad_norm=planner_max_grad_norm,
                use_muon=planner_use_muon, planner_model=diffusion_buffer.model,
                muon_lr=self.cfg.get("planner_muon_lr", 0.001),
                muon_momentum=self.cfg.get("planner_muon_momentum", 0.95),
                muon_weight_decay=self.cfg.get("planner_muon_weight_decay", 0.0),
                muon_backend=self.cfg.get("planner_muon_backend", "torch"),
                muon_adjust_lr_fn=self.cfg.get("planner_muon_adjust_lr_fn", None),
            )
            print(f"[OnPolicyRunnerMM] Planner RL enabled: lr={planner_lr}, params={len(planner_params)}, muon={planner_use_muon}")

            # Stage 1: Freeze controller (actor-critic) so only planner is trained
            freeze_controller = self.cfg.get("freeze_controller", False)
            if freeze_controller:
                for param in self.alg.actor_critic.parameters():
                    param.requires_grad = False
                self.alg.actor_critic.eval()
                self.alg.skip_controller_update = True
                print(f"[OnPolicyRunnerMM] Controller (actor-critic) FROZEN for planner RL Stage 1")

            # Episode-level planner RL: autoregressive multi-regen mode
            self._planner_rl_episode_mode = self.cfg.get("planner_episode_mode", False)
            if self._planner_rl_episode_mode:
                print(f"[OnPolicyRunnerMM] Episode-level planner RL enabled (autoregressive multi-regen)")

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        # Keep the selected logger available even when logging is disabled on
        # a non-zero distributed rank or when ``log_dir`` is intentionally
        # omitted (for example, a checkpoint-only continuation).  ``save`` is
        # still called for the final checkpoint in those modes.
        self.logger_type = str(self.cfg.get("logger", "tensorboard")).lower()
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _get_mmgpt_self_supervision_inputs(self, observations_dict):
        """Preserve pre-normalizer state and query a true one-step target."""

        if not self._is_mmgpt:
            return None, None
        raw_obs_dict = observations_dict
        dreamer_target = None
        if self._mmgpt_external_target_enabled:
            raw_obs_dict, dreamer_target = (
                self.env.get_mmgpt_self_supervision_observations(
                    observations_dict,
                    self._mmgpt_dreamer_target_term_name,
                )
            )
        if not isinstance(raw_obs_dict, dict) or "policy" not in raw_obs_dict:
            raise RuntimeError(
                "MMGPT self-supervision requires observations['policy'] "
                "before empirical normalization"
            )
        raw_obs = raw_obs_dict["policy"]
        if not isinstance(raw_obs, torch.Tensor) or raw_obs.ndim != 2:
            raise ValueError(
                "raw policy observations must have shape (num_envs, obs_dim)"
            )
        raw_obs = raw_obs.to(self.device).detach().clone()

        if dreamer_target is None:
            return raw_obs, None
        target, target_mask = dreamer_target
        if target.shape != (self.env.num_envs, self.env.num_actions):
            raise ValueError(
                "adjacent Dreamer target must match the controller action "
                f"shape {(self.env.num_envs, self.env.num_actions)}, got "
                f"{tuple(target.shape)}"
            )
        if target_mask.shape != (self.env.num_envs,):
            raise ValueError(
                "adjacent Dreamer target mask must have shape "
                f"{(self.env.num_envs,)}, got {tuple(target_mask.shape)}"
            )
        return raw_obs, (
            target.to(self.device).detach().clone(),
            target_mask.to(self.device).bool().detach().clone(),
        )

    def _clone_raw_reference_observations(self, ref_obs_tuple):
        """Preserve the teacher's pre-normalizer command tensor and mask."""

        if ref_obs_tuple is None:
            return None
        return tuple(
            value.to(self.device).detach().clone() for value in ref_obs_tuple
        )

    def _fixed_teacher_reference_observations(
        self,
        reference_groups,
        policy_reference,
    ):
        """Select the frozen expert's raw command independently of the actor."""

        if not self.alg.fixed_expert_enabled:
            return None
        group_name = self.fixed_teacher_reference_group
        if group_name == "policy":
            # The wrapper may interrupt only the student's reference tokens.
            # Raw environment groups remain complete for expert supervision.
            selected = reference_groups.get("policy", policy_reference)
        else:
            selected = reference_groups.get(group_name)
        if selected is None:
            raise RuntimeError(
                f"fixed expert reference group {group_name!r} returned no data"
            )
        return self._clone_raw_reference_observations(selected)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True):
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if self.training_type == "distillation" and not self.alg.actor_critic.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = _unpack_observations_result(self.env.get_observations())
        ref_obs_tuple, ref_extras = self.env.get_reference_observations()
        raw_obs, dreamer_target = self._get_mmgpt_self_supervision_inputs(
            extras["observations"]
        )
        critic_obs = extras["observations"].get("critic", obs)
        critic_ref_obs_tuple = ref_extras["ref_observations"].get("critic", ref_obs_tuple) if ref_obs_tuple is not None else None
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if ref_obs_tuple is not None:
            ref_obs_tuple = tuple(ref_obs.to(self.device) for ref_obs in ref_obs_tuple)
        raw_ref_obs_tuple = self._clone_raw_reference_observations(ref_obs_tuple)
        raw_expert_ref_obs_tuple = self._fixed_teacher_reference_observations(
            ref_extras["ref_observations"], ref_obs_tuple
        )
        if critic_ref_obs_tuple is not None:
            critic_ref_obs_tuple = tuple(critic_ref_obs.to(self.device) for critic_ref_obs in critic_ref_obs_tuple)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Normalize the initial transition as well.  Previously only values
        # returned by env.step() were normalized, so the first action after a
        # fresh learn()/resume call saw a different observation contract.
        obs = self.obs_normalizer(obs)
        if ref_obs_tuple is not None and self.ref_obs_normalizer is not None:
            ref_obs_tuple = (
                self.ref_obs_normalizer(ref_obs_tuple[0]),
                ref_obs_tuple[1],
            )
        if "critic" in extras["observations"]:
            critic_obs = self.critic_obs_normalizer(critic_obs)
        else:
            critic_obs = obs
        if (
            critic_ref_obs_tuple is not None
            and self.critic_ref_obs_normalizer is not None
            and "critic" in ref_extras["ref_observations"]
        ):
            critic_ref_obs_tuple = (
                self.critic_ref_obs_normalizer(critic_ref_obs_tuple[0]),
                critic_ref_obs_tuple[1],
            )
        else:
            critic_ref_obs_tuple = ref_obs_tuple

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        if self.amp_cfg:
            amp_rewbuffer = deque(maxlen=100)
            amp_prev_neg_score = 0.45
            amp_scorebuffer = deque(maxlen=100)
            amp_score_avgbuffer = deque(maxlen=100)
            cur_amp_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_amp_score_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_amp_score_avg = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            # Per-env ring buffer for AMP observation history
            amp_history_len = self.amp_cfg["net_cfg"]["amp_history_length"]
            amp_obs_dim = self.amp_cfg["net_cfg"]["backbone_input_dim"]
            amp_obs_buffer = torch.zeros(self.env.num_envs, amp_history_len, amp_obs_dim, device=self.device)
            amp_obs_fill_count = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)  # how many obs each env has collected
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            try:
                self.alg.broadcast_parameters_dagger()
            except:
                pass

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            self.alg.set_training_iteration(it)

            # ====== Episode-level planner RL: separate code path ======
            if self._planner_rl_episode_mode:
                episode_cache = self._collect_planner_episode()
                stop = time.time()
                collection_time = stop - start

                start = stop
                planner_metrics = self.alg.planner_episode_update(
                    self.env.diffusion_buffer, episode_cache,
                )
                # Create a minimal loss_dict for logging compatibility
                loss_dict = {}
                stop = time.time()
                learn_time = stop - start
                self.current_learning_iteration = it

                if self.log_dir is not None and not self.disable_logs:
                    self.log(locals())
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                ep_infos.clear()
                continue

            # ====== Standard RL training path ======
            # Rollout
            with torch.inference_mode():
                # Planner RL: sample all envs simultaneously before rollout
                if self._planner_rl_enabled:
                    self.env.diffusion_buffer.planner_rl_sample_all()

                # ref_obss = []
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(#obs, critic_obs)
                        obs=obs,
                        ref_obs=ref_obs_tuple,
                        critic_obs=critic_obs,
                        ref_critic_obs=critic_ref_obs_tuple,
                        raw_obs=raw_obs,
                        raw_ref_obs=raw_ref_obs_tuple,
                        expert_ref_obs=raw_expert_ref_obs_tuple,
                        dreamer_target=dreamer_target,
                    )
                    if self.amp_cfg: # Get amp observation before step and shift into per-env ring buffer
                        amp_obs = string_to_callable(self.amp_cfg["amp_obs_extractor"])(env=self.amp_cfg["_env"])
                        # Shift buffer left and insert new obs at the end
                        amp_obs_buffer[:, :-1] = amp_obs_buffer[:, 1:].clone()
                        amp_obs_buffer[:, -1] = amp_obs
                        amp_obs_fill_count = (amp_obs_fill_count + 1).clamp(max=amp_history_len)

                    obs, ref_obs_tuple, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # move to the right device
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    ref_obs_tuple = tuple(ref_obs.to(self.device) for ref_obs in ref_obs_tuple) if ref_obs_tuple is not None else None
                    raw_ref_obs_tuple = self._clone_raw_reference_observations(
                        ref_obs_tuple
                    )
                    raw_expert_ref_obs_tuple = (
                        self._fixed_teacher_reference_observations(
                            infos["ref_observations"], ref_obs_tuple
                        )
                    )
                    raw_obs, dreamer_target = (
                        self._get_mmgpt_self_supervision_inputs(
                            infos["observations"]
                        )
                    )
                    # ref_obss.append(ref_obs_tuple[0])
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if ref_obs_tuple is not None and self.ref_obs_normalizer is not None:
                        ref_obs_tuple = (self.ref_obs_normalizer(ref_obs_tuple[0]), ref_obs_tuple[1])
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    if "critic" in infos["ref_observations"]:
                        critic_ref_obs_tuple = infos["ref_observations"]["critic"]
                        if self.critic_ref_obs_normalizer is not None:
                            critic_ref_obs_tuple = (self.critic_ref_obs_normalizer(critic_ref_obs_tuple[0]), critic_ref_obs_tuple[1])
                    else:
                        critic_ref_obs_tuple = ref_obs_tuple

                    # The reward passed to process_env_step is the sole source
                    # for GAE/returns.  Compose AMP shaping before storage; the
                    # previous order added it only to logging after storage.
                    base_rewards = rewards
                    amp_rewards = None
                    amp_score = None
                    if self.amp_cfg:
                        ready_mask = amp_obs_fill_count >= amp_history_len
                        amp_input = amp_obs_buffer.flatten(-2, -1)
                        if ready_mask.any():
                            amp_rewards = self.alg.amp.amp_reward(
                                amp_input, epsilon=self.amp_cfg["epsilon"]
                            )
                            amp_score = self.alg.amp.amp_score(amp_input)
                            amp_rewards = amp_rewards * ready_mask.float()
                            amp_score = amp_score * ready_mask.float()
                        else:
                            amp_rewards = torch.zeros(
                                self.env.num_envs, device=self.device
                            )
                            amp_score = torch.zeros(
                                self.env.num_envs, device=self.device
                            )

                        rewards, applied_amp_rewards = _compose_amp_training_rewards(
                            base_rewards,
                            amp_rewards,
                            scale=self.amp_cfg["amp_reward_scale"],
                            enabled=_amp_reward_enabled(
                                it, self.amp_cfg["amp_pretrain_steps"]
                            ),
                        )

                        # A done transition still uses the completed history
                        # above; clear only the next episode's history state.
                        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                        if done_ids.numel() > 0:
                            amp_obs_buffer[done_ids] = 0
                            amp_obs_fill_count[done_ids] = 0

                    # Store the fully composed controller-training reward.
                    self.alg.process_env_step(rewards, dones, infos)

                    # Planner RL retains its environment reward contract; AMP
                    # is a controller-only learned shaping term.
                    if self._planner_rl_enabled:
                        self.env.diffusion_buffer.accumulate_planner_reward(
                            base_rewards.squeeze()
                        )

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        # clip rewards < 0 to 0
                        # rewards = torch.clamp(rewards, min=0.0)

                        if self.amp_cfg:
                            # Log the shaping term that was actually stored for
                            # PPO.  During AMP pretraining this is exactly zero,
                            # rather than the discriminator's unapplied candidate.
                            cur_amp_reward_sum += applied_amp_rewards
                            cur_amp_score_sum += amp_score
                            cur_amp_score_avg = cur_amp_score_sum / (cur_episode_length + 1)
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0
                        if self.amp_cfg:
                            amp_rewbuffer.extend(cur_amp_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_amp_reward_sum[new_ids] = 0
                        
                            amp_scorebuffer.extend(cur_amp_score_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_amp_score_sum[new_ids] = 0
                            amp_score_avgbuffer.extend(cur_amp_score_avg[new_ids][:, 0].cpu().numpy().tolist())
                            cur_amp_score_avg[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                # ref_obss = torch.stack(ref_obss, dim=1) if ref_obss else None
                # if ref_obss is not None:
                #     torch.save(ref_obss, os.path.join(self.log_dir, f"ref_obs_{it}.pkl"))

                # Learning step
                start = stop
                if self.training_type == "rl":
                    self.alg.compute_returns(critic_obs, critic_ref_obs_tuple)

            loss_dict = self.alg.update(epoch=it)
            # Match SONIC's curriculum boundary: accumulate motion/bin
            # occupancy during all 24 rollout steps, then recompute exactly
            # once so the sampling distribution is frozen within a rollout.
            adaptive_env = getattr(self.env, "unwrapped", self.env)
            update_adaptive_sampling = getattr(
                adaptive_env, "update_sonic_adaptive_sampling", None
            )
            if callable(update_adaptive_sampling):
                update_adaptive_sampling()
            if self.amp_cfg is not None:
                amp_prev_neg_score = loss_dict["mean_pred_pos_prob"]

            # Planner RL update (after controller PPO update)
            planner_metrics = {}
            if self._planner_rl_enabled:
                planner_metrics = self.alg.planner_update(self.env.diffusion_buffer)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.gpu_global_rank == 0 and self.metrics_jsonl_path:
                metrics_path = Path(self.metrics_jsonl_path)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "iteration": int(it),
                    "collection_time_s": float(collection_time),
                    "learning_time_s": float(learn_time),
                    "loss": {
                        key: float(value)
                        for key, value in loss_dict.items()
                    },
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload, sort_keys=True) + "\n")
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            # if it == start_iter and not self.disable_logs:
            #     # obtain all the diff files
            #     git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
            #     # if possible store them to wandb
            #     if self.logger_type in ["wandb", "neptune"] and git_file_paths:
            #         for path in git_file_paths:
            #             self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            flush = getattr(self.writer, "flush", None)
            if callable(flush):
                flush()

    def _collect_planner_episode(self):
        """Collect a full episode for episode-level planner RL.

        Runs the entire episode (500 steps at 50Hz = 10s) with frozen controller,
        allowing natural autoregressive regen (~6 times). Regen snapshots are
        cached by the diffusion buffer for gradient replay.

        Returns:
            EpisodeRegenCache with all regen snapshots and accumulated rewards.
        """
        db = self.env.diffusion_buffer
        episode_steps = int(self.env.max_episode_length)

        # Sync all envs to same initial state + text prompt, then begin episode
        db.sync_episode_start()
        db.begin_planner_rl_episode()

        # Get initial observations
        obs, extras = _unpack_observations_result(self.env.get_observations())
        ref_obs_tuple, ref_extras = self.env.get_reference_observations()
        critic_obs = extras["observations"].get("critic", obs)
        critic_ref_obs_tuple = ref_extras["ref_observations"].get("critic", ref_obs_tuple) if ref_obs_tuple is not None else None
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if ref_obs_tuple is not None:
            ref_obs_tuple = tuple(r.to(self.device) for r in ref_obs_tuple)
        if critic_ref_obs_tuple is not None:
            critic_ref_obs_tuple = tuple(r.to(self.device) for r in critic_ref_obs_tuple)

        obs = self.obs_normalizer(obs)
        if ref_obs_tuple is not None and self.ref_obs_normalizer is not None:
            ref_obs_tuple = (
                self.ref_obs_normalizer(ref_obs_tuple[0]),
                ref_obs_tuple[1],
            )
        if "critic" in extras["observations"]:
            critic_obs = self.critic_obs_normalizer(critic_obs)
        else:
            critic_obs = obs
        if (
            critic_ref_obs_tuple is not None
            and self.critic_ref_obs_normalizer is not None
            and "critic" in ref_extras["ref_observations"]
        ):
            critic_ref_obs_tuple = (
                self.critic_ref_obs_normalizer(critic_ref_obs_tuple[0]),
                critic_ref_obs_tuple[1],
            )
        else:
            critic_ref_obs_tuple = ref_obs_tuple

        with torch.inference_mode():
            for step in range(episode_steps):
                actions = self.alg.act(
                    obs=obs,
                    ref_obs=ref_obs_tuple,
                    critic_obs=critic_obs,
                    ref_critic_obs=critic_ref_obs_tuple,
                )

                obs, ref_obs_tuple, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                obs, critic_obs, rewards, dones = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )
                if ref_obs_tuple is not None:
                    ref_obs_tuple = tuple(r.to(self.device) for r in ref_obs_tuple)

                # Normalize observations
                obs = self.obs_normalizer(obs)
                if ref_obs_tuple is not None and self.ref_obs_normalizer is not None:
                    ref_obs_tuple = (self.ref_obs_normalizer(ref_obs_tuple[0]), ref_obs_tuple[1])
                if "critic" in infos["observations"]:
                    critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                else:
                    critic_obs = obs
                if "critic" in infos["ref_observations"]:
                    critic_ref_obs_tuple = infos["ref_observations"]["critic"]
                    if self.critic_ref_obs_normalizer is not None:
                        critic_ref_obs_tuple = (
                            self.critic_ref_obs_normalizer(critic_ref_obs_tuple[0]),
                            critic_ref_obs_tuple[1],
                        )
                else:
                    critic_ref_obs_tuple = ref_obs_tuple

                # Accumulate episode-level reward
                db.accumulate_planner_episode_reward(rewards.squeeze())

        return db.end_planner_rl_episode()

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # self.writer.add_scalar("Loss/value_function", locs["loss_dict"]["mean_value_loss"], locs["it"])
        # self.writer.add_scalar("Loss/surrogate", locs["loss_dict"]["mean_surrogate_loss"], locs["it"])
        # self.writer.add_scalar("Loss/imitation", locs["loss_dict"]["mean_imitation_loss"], locs["it"])
        # if locs["loss_dict"]["mean_dagger_loss"] > 0:
        #     self.writer.add_scalar("Loss/dagger", locs["loss_dict"]["mean_dagger_loss"], locs["it"])
        # if self.alg.rnd:
        #     self.writer.add_scalar("Loss/rnd", locs["loss_dict"]["mean_rnd_loss"], locs["it"])
        # if self.alg.symmetry:
        #     self.writer.add_scalar("Loss/symmetry", locs["loss_dict"]["symmetry"], locs["it"])
        if self.amp_cfg:
            self.writer.add_scalar("Loss/amp", locs["loss_dict"]["mean_amp_loss"], locs["it"])
        for key, value in locs["loss_dict"].items():
            if 'amp' in key:
                continue
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        

        
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Planner RL metrics
        if self._planner_rl_enabled and locs.get("planner_metrics"):
            for key, value in locs["planner_metrics"].items():
                self.writer.add_scalar(f"PlannerRL/{key}", value, locs["it"])
        elif self._planner_rl_enabled:
            print(f"[PlannerRL] WARNING: planner_metrics empty at iter {locs['it']}")

        if self.amp_cfg:
            self.writer.add_scalar("AMP/mean_gradient_penalty", locs["loss_dict"]["mean_gradient_penalty"], locs["it"])
            self.writer.add_scalar("AMP/mean_pred_pos_acc", locs["loss_dict"]["mean_pred_pos_acc"], locs["it"])
            self.writer.add_scalar("AMP/mean_pred_neg_acc", locs["loss_dict"]["mean_pred_neg_acc"], locs["it"])
            self.writer.add_scalar("AMP/mean_pred_pos_prob", locs["loss_dict"]["mean_pred_pos_prob"], locs["it"])
            self.writer.add_scalar("AMP/mean_pred_neg_prob", locs["loss_dict"]["mean_pred_neg_prob"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])

            if self.amp_cfg:
                self.writer.add_scalar("AMP/mean_reward", statistics.mean(locs["amp_rewbuffer"]), locs["it"])
                self.writer.add_scalar("AMP/mean_score", statistics.mean(locs["amp_score_avgbuffer"]), locs["it"])
          
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f"       Learning iteration {locs['it']}/{locs['tot_iter']}       "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # f"""{'Value function loss:':>{pad}} {locs['loss_dict']['mean_value_loss']:.4f}\n"""
                # f"""{'Surrogate loss:':>{pad}} {locs['loss_dict']['mean_surrogate_loss']:.4f}\n"""
                # f"""{'Imitation loss:':>{pad}} {locs['loss_dict']['mean_imitation_loss']:.4f}\n"""
                
                )
            for key, value in locs["loss_dict"].items():
                if 'amp' in key:
                    continue
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
                # f"""{'Dagger loss:':>{pad}} {locs['loss_dict']['mean_dagger_loss']:.4f}\n""" if locs['loss_dict']['mean_dagger_loss'] > 0 else ""
                # f"""{'Rnd loss:':>{pad}} {locs['loss_dict']['mean_rnd_loss']:.4f}\n""" if self.alg.rnd else ""
                # f"""{'Entropy:':>{pad}} {locs['loss_dict']['mean_entropy']:.4f}\n"""
                # f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n""" if self.alg.rnd else ""
                # f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n""" if self.alg.rnd else ""
                # f"""{'Symmetry loss:':>{pad}} {locs['loss_dict']['symmetry']:.4f}\n""" if self.alg.symmetry else ""

            # if locs['loss_dict']['mean_dagger_loss'] > 0:
            #     log_string += (f"""{'Dagger loss:':>{pad}} {locs['loss_dict']['mean_dagger_loss']:.4f}\n""")
            # if self.alg.rnd:
            #     log_string += (f"""{'Rnd loss:':>{pad}} {locs['loss_dict']['mean_rnd_loss']:.4f}\n"""
            #                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
            #                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n""")
            # if self.alg.symmetry:
            #     log_string += (f"""{'Symmetry loss:':>{pad}} {locs['loss_dict']['symmetry']:.4f}\n""")

            if self.amp_cfg:
                log_string += (f"""{'Mean amp loss:':>{pad}} {locs['loss_dict']['mean_amp_loss']:.4f}\n""")
                log_string += (f"""{'Mean gradient penalty:':>{pad}} {locs['loss_dict']['mean_gradient_penalty']:.4f}\n""")
                log_string += (f"""{'Mean pred pos acc:':>{pad}} {locs['loss_dict']['mean_pred_pos_acc']:.4f}\n""")
                log_string += (f"""{'Mean pred neg acc:':>{pad}} {locs['loss_dict']['mean_pred_neg_acc']:.4f}\n""")
                log_string += (f"""{'Mean pred pos prob:':>{pad}} {locs['loss_dict']['mean_pred_pos_prob']:.4f}\n""")
                log_string += (f"""{'Mean pred neg prob:':>{pad}} {locs['loss_dict']['mean_pred_neg_prob']:.4f}\n""")
                log_string += (f"""{'Mean amp reward:':>{pad}} {statistics.mean(locs['amp_rewbuffer']):.4f}\n""")
                log_string += (f"""{'Mean amp score:':>{pad}} {statistics.mean(locs['amp_score_avgbuffer']):.4f}\n""")

            log_string += (f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['loss_dict']['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['loss_dict']['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        # Planner RL console logging
        if self._planner_rl_enabled and locs.get("planner_metrics"):
            log_string += f"""{'-' * width}\n"""
            for key, value in locs["planner_metrics"].items():
                log_string += f"""{f'PlannerRL/{key}:':>{pad}} {value:.4f}\n"""
        # log_string += (
        #     f"""{'-' * width}\n"""
        #     f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
        #     f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
        #     f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
        #     f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
        #                        locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        # )
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "next_iter": self.current_learning_iteration + 1,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
            if self.ref_obs_normalizer is not None:
                saved_dict["ref_obs_norm_state_dict"] = self.ref_obs_normalizer.state_dict()
            if self.critic_ref_obs_normalizer is not None:
                saved_dict["critic_ref_obs_norm_state_dict"] = self.critic_ref_obs_normalizer.state_dict()
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        if self.alg.amp is not None:
            saved_dict["amp_state_dict"] = self.alg.amp.state_dict()
            saved_dict["amp_optimizer_state_dict"] = self.alg.amp_optimizer.state_dict()
        if self.alg.scaler is not None:
            saved_dict["scaler_state_dict"] = self.alg.scaler.state_dict()
        fixed_expert_state_fn = getattr(
            self.alg, "fixed_expert_state_dict", None
        )
        fixed_expert_state = (
            fixed_expert_state_fn()
            if callable(fixed_expert_state_fn)
            else None
        )
        if fixed_expert_state is not None:
            saved_dict["fixed_expert_state_dict"] = fixed_expert_state
        
        if self.alg.teacher_coef is not None:
            saved_dict["imitation_optimizer_state_dict"] = self.alg.imitation_optimizer.state_dict()
            saved_dict["dagger_optimizer_state_dict"] = self.alg.dagger_optimizer.state_dict()

        # Planner RL checkpoint
        if self._planner_rl_enabled:
            diffusion_buffer = self.env.diffusion_buffer
            if diffusion_buffer is not None:
                saved_dict["planner_model_state_dict"] = diffusion_buffer.model.state_dict()
            if hasattr(self.alg, 'planner_optimizer') and self.alg.planner_optimizer is not None:
                saved_dict["planner_optimizer_state_dict"] = self.alg.planner_optimizer.state_dict()

        adaptive_env = getattr(self.env, "unwrapped", self.env)
        adaptive_state_fn = getattr(
            adaptive_env, "sonic_adaptive_state_dict", None
        )
        if callable(adaptive_state_fn):
            adaptive_state = adaptive_state_fn()
            if adaptive_state is not None:
                saved_dict["sonic_adaptive_sampler_state_dict"] = adaptive_state

        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and self.writer is not None:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        _restore_sonic_adaptive_sampler_state(self.env, loaded_dict)
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
            if self.ref_obs_normalizer is not None:
                self.ref_obs_normalizer.load_state_dict(loaded_dict["ref_obs_norm_state_dict"])
            if self.critic_ref_obs_normalizer is not None:
                self.critic_ref_obs_normalizer.load_state_dict(loaded_dict["critic_ref_obs_norm_state_dict"])
            if getattr(
                self.alg, "expert_freeze_loaded_normalization", False
            ):
                # Aggregated DAgger sequences are stored after normalization.
                # Freeze the mature BC checkpoint statistics so old and new
                # sequences retain one stable input coordinate system.
                for normalizer in (
                    self.obs_normalizer,
                    self.ref_obs_normalizer,
                    self.critic_obs_normalizer,
                    self.critic_ref_obs_normalizer,
                ):
                    if normalizer is not None:
                        normalizer.eval()
                self._loaded_normalization_frozen = True
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if self.alg.teacher_coef is not None:
            self.alg.imitation_optimizer.load_state_dict(loaded_dict["imitation_optimizer_state_dict"])
            self.alg.dagger_optimizer.load_state_dict(loaded_dict["dagger_optimizer_state_dict"])
        if self.alg.scaler is not None:
            self.alg.scaler.load_state_dict(loaded_dict["scaler_state_dict"])
        if self.alg.amp is not None:
            try:
                self.alg.amp.load_state_dict(loaded_dict["amp_state_dict"])
                self.alg.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
            except:
                print("AMP state dict not available or mismatch in the loaded file. Skipping AMP loading.")
        load_fixed_expert_state = getattr(
            self.alg, "load_fixed_expert_state_dict", None
        )
        if callable(load_fixed_expert_state):
            load_fixed_expert_state(
                loaded_dict.get("fixed_expert_state_dict")
            )
        if load_optimizer:
            if "next_iter" in loaded_dict:
                self.current_learning_iteration = int(loaded_dict["next_iter"])
            elif getattr(self.alg, "fixed_expert_enabled", False):
                # Legacy fixed-expert checkpoints stored the iteration that
                # had already completed.  Repeating it can silently shift a
                # BC/DAgger boundary, so continue at the next iteration.
                self.current_learning_iteration = int(loaded_dict["iter"]) + 1
            else:
                # Preserve the historical resume convention for unrelated
                # runners whose schedules may already rely on it.
                self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
                if self.ref_obs_normalizer is not None:
                    self.ref_obs_normalizer.to(device)
                    # policy = lambda x, ref_x: self.alg.actor_critic.act_inference(self.obs_normalizer(x), ref_x)  # noqa: E731
                    def pol(x, ref_x):
                        if ref_x is not None:
                            assert self.ref_obs_normalizer is not None
                            return self.alg.actor_critic.act_inference(self.obs_normalizer(x), (self.ref_obs_normalizer(ref_x[0]), ref_x[1]))
                        else:
                            return self.alg.actor_critic.act_inference(self.obs_normalizer(x), ref_x)
                    policy = pol
                
                else:
                    policy = lambda x, ref_x: self.alg.actor_critic.act_inference(self.obs_normalizer(x), ref_x)
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()

        if self.alg.rnd:
            self.alg.rnd.train()
        if self.amp_cfg:
            self.alg.amp.train()
        if self.empirical_normalization:
            # DAgger replay stores already-normalized sequence tensors.  Once
            # a mature BC checkpoint's statistics are loaded, train_mode must
            # not silently put the empirical normalizers back into update
            # mode, otherwise old replay and new rollout samples use drifting
            # coordinate systems.
            normalizer_training = not self._loaded_normalization_frozen
            self.obs_normalizer.train(normalizer_training)
            self.critic_obs_normalizer.train(normalizer_training)
            if self.ref_obs_normalizer is not None:
                self.ref_obs_normalizer.train(normalizer_training)
            if self.critic_ref_obs_normalizer is not None:
                self.critic_ref_obs_normalizer.train(normalizer_training)

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.alg.rnd:
            self.alg.rnd.eval()
        if self.amp_cfg:
            self.alg.amp.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()
            if self.ref_obs_normalizer is not None:
                self.ref_obs_normalizer.eval()
            if self.critic_ref_obs_normalizer is not None:
                self.critic_ref_obs_normalizer.eval()

    def collect_dataset(self, num_episodes: int, save_path: str = None, max_steps_per_episode: int = None):
        """
        采集当前 policy 的执行轨迹数据集
        
        Args:
            num_episodes: 采集的 episode 数量
            save_path: 保存路径（如果为 None 则不保存，只返回数据）
            max_steps_per_episode: 每个 episode 最大步数（None 则使用环境默认）
        
        Returns:
            dataset: {
                'observations': List[np.ndarray],
                'ref_observations': List[Tuple[np.ndarray, np.ndarray]],
                'actions': List[np.ndarray],
                'rewards': List[np.ndarray],
                'dones': List[np.ndarray],
                'episode_lengths': List[int],
                'episode_rewards': List[float],
            }
        """
        import numpy as np
        import pickle
        from pathlib import Path
        from tqdm import tqdm
        
        self.eval_mode()
        policy = self.get_inference_policy(device=self.device)
        
        # 重置环境
        self.env.reset()
        obs, extras = _unpack_observations_result(self.env.get_observations())
        ref_obs_tuple, ref_extras = self.env.get_reference_observations()
        
        obs = obs.to(self.device)
        if ref_obs_tuple is not None:
            ref_obs_tuple = tuple(ref_obs.to(self.device) for ref_obs in ref_obs_tuple)
        
        # 数据存储
        dataset = {
            'observations': [],
            'ref_observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'episode_lengths': [],
            'episode_rewards': [],
        }
        
        # 当前 episode 的缓存
        current_obs = []
        current_ref_obs = []
        current_actions = []
        current_rewards = []
        current_dones = []
        
        num_envs = self.env.num_envs
        episode_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        episode_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        episode_rewards_sum = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        
        max_steps = max_steps_per_episode if max_steps_per_episode else self.env.max_episode_length
        
        print(f"开始采集 {num_episodes} 个 episodes 的数据...")
        print(f"并行环境数: {num_envs}")
        print(f"每个 episode 最大步数: {max_steps}")
        
        pbar = tqdm(total=num_episodes, desc="采集进度", disable=self.disable_logs)
        
        with torch.no_grad():
            step_count = 0
            while episode_counts.sum() < num_episodes:
                # 获取 action
                actions = policy(obs, ref_obs_tuple)
                
                # 存储当前状态
                current_obs.append(obs.cpu().numpy())
                if ref_obs_tuple is not None:
                    current_ref_obs.append((ref_obs_tuple[0].cpu().numpy(), ref_obs_tuple[1].cpu().numpy()))
                else:
                    current_ref_obs.append(None)
                current_actions.append(actions.cpu().numpy())
                
                # 执行 action
                obs, ref_obs_tuple, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                
                # 移动到正确的设备
                obs = obs.to(self.device)
                if ref_obs_tuple is not None:
                    ref_obs_tuple = tuple(ref_obs.to(self.device) for ref_obs in ref_obs_tuple)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                
                # Keep dataset observations raw.  The inference-policy wrapper
                # owns normalization; normalizing here would feed the next
                # action through the same normalizer a second time.
                
                current_rewards.append(rewards.cpu().numpy())
                current_dones.append(dones.cpu().numpy())
                
                episode_steps += 1
                episode_rewards_sum += rewards
                
                # 检查哪些环境 episode 结束
                done_indices = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                
                if len(done_indices) > 0:
                    for env_idx in done_indices:
                        env_idx = env_idx.item()
                        
                        # 只保存还没达到目标数量的 episode
                        if episode_counts[env_idx] < num_episodes // num_envs + 1:
                            # 提取该环境的完整轨迹
                            ep_len = episode_steps[env_idx].item()
                            
                            obs_traj = np.stack([step[env_idx] for step in current_obs[-ep_len:]], axis=0)
                            actions_traj = np.stack([step[env_idx] for step in current_actions[-ep_len:]], axis=0)
                            rewards_traj = np.stack([step[env_idx] for step in current_rewards[-ep_len:]], axis=0)
                            dones_traj = np.stack([step[env_idx] for step in current_dones[-ep_len:]], axis=0)
                            
                            if current_ref_obs[-1] is not None:
                                ref_obs_traj = (
                                    np.stack([step[0][env_idx] for step in current_ref_obs[-ep_len:]], axis=0),
                                    np.stack([step[1][env_idx] for step in current_ref_obs[-ep_len:]], axis=0)
                                )
                            else:
                                ref_obs_traj = None
                            
                            dataset['observations'].append(obs_traj)
                            dataset['ref_observations'].append(ref_obs_traj)
                            dataset['actions'].append(actions_traj)
                            dataset['rewards'].append(rewards_traj)
                            dataset['dones'].append(dones_traj)
                            dataset['episode_lengths'].append(ep_len)
                            dataset['episode_rewards'].append(episode_rewards_sum[env_idx].item())
                            
                            episode_counts[env_idx] += 1
                            pbar.update(1)
                        
                        # 重置该环境的计数
                        episode_steps[env_idx] = 0
                        episode_rewards_sum[env_idx] = 0
                
                step_count += 1
                
                # 防止无限循环
                if step_count > num_episodes * max_steps * 2:
                    print("警告: 达到最大步数限制，提前结束采集")
                    break
        
        pbar.close()
        
        # 统计信息
        actual_episodes = len(dataset['observations'])
        print(f"\n采集完成!")
        print(f"实际采集 episodes: {actual_episodes}")
        print(f"平均 episode 长度: {np.mean(dataset['episode_lengths']):.2f} ± {np.std(dataset['episode_lengths']):.2f}")
        print(f"平均 episode reward: {np.mean(dataset['episode_rewards']):.2f} ± {np.std(dataset['episode_rewards']):.2f}")
        print(f"Episode 长度范围: [{np.min(dataset['episode_lengths'])}, {np.max(dataset['episode_lengths'])}]")
        
        # 保存数据
        if save_path is not None:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            
            import os
            print(f"\n数据集已保存到: {save_path}")
            print(f"文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
        
        return dataset

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
