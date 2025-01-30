#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import MMPPO
from rsl_rl.env import MMVecEnv
from rsl_rl.modules import ActorCriticMMTransformer, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class OnPolicyRunnerMM:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: MMVecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        ref_obs_tuple, ref_extras = self.env.get_reference_observations()
        num_obs = obs.shape[1]
        if ref_obs_tuple is not None:
            num_ref_obs = ref_obs_tuple[0].shape[1]
        else:
            num_ref_obs = 0
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        if "critic" in ref_extras["ref_observations"]:
            num_critic_ref_obs = ref_extras["ref_observations"]["critic"][0].shape[1]
        else:
            num_critic_ref_obs = num_ref_obs
            
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCriticMMTransformer = actor_critic_class(
            num_actor_obs=num_obs,
            num_actor_ref_obs=num_ref_obs,
            num_critic_obs=num_critic_obs,
            num_critic_ref_obs=num_critic_ref_obs,
            num_actions=self.env.num_actions,
             **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: MMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.ref_obs_normalizer = EmpiricalNormalization(shape=[num_ref_obs], until=1.0e8).to(self.device) if num_ref_obs > 0 else None
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
            self.critic_ref_obs_normalizer = EmpiricalNormalization(shape=[num_critic_ref_obs], until=1.0e8).to(self.device) if num_critic_ref_obs > 0 else None
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
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            actor_ref_obs_shape=[num_ref_obs] if num_ref_obs > 0 else [None],
            critic_obs_shape=[num_critic_obs],
            critic_ref_obs_shape=[num_critic_ref_obs] if num_critic_ref_obs > 0 else [None],
            action_shape=[self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
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

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        ref_obs_tuple, ref_extras = self.env.get_reference_observations()
        critic_obs = extras["observations"].get("critic", obs)
        critic_ref_obs_tuple = ref_extras["ref_observations"].get("critic", ref_obs_tuple) if ref_obs_tuple is not None else None
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if ref_obs_tuple is not None:
            ref_obs_tuple = tuple(ref_obs.to(self.device) for ref_obs in ref_obs_tuple)
        if critic_ref_obs_tuple is not None:
            critic_ref_obs_tuple = tuple(critic_ref_obs.to(self.device) for critic_ref_obs in critic_ref_obs_tuple)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                # ref_obss = []
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(#obs, critic_obs)
                        obs=obs,
                        ref_obs=ref_obs_tuple,
                        critic_obs=critic_obs,
                        ref_critic_obs=critic_ref_obs_tuple,
                    )
                    obs, ref_obs_tuple, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # move to the right device
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    ref_obs_tuple = tuple(ref_obs.to(self.device) for ref_obs in ref_obs_tuple) if ref_obs_tuple is not None else None
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

                    
                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

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
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                # ref_obss = torch.stack(ref_obss, dim=1) if ref_obss else None
                # if ref_obss is not None:
                #     torch.save(ref_obss, os.path.join(self.log_dir, f"ref_obs_{it}.pkl"))

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_imitation_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

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

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/imitation", locs["mean_imitation_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
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
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Imitation loss:':>{pad}} {locs['mean_imitation_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
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
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
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
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
            if self.ref_obs_normalizer is not None:
                self.ref_obs_normalizer.load_state_dict(loaded_dict["ref_obs_norm_state_dict"])
            if self.critic_ref_obs_normalizer is not None:
                self.critic_ref_obs_normalizer.load_state_dict(loaded_dict["critic_ref_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
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
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
