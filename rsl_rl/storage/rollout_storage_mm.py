# Copyright (c) 2025, Shanghai Jiao Tong University, MVASL Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
from calendar import c

import torch

from rsl_rl.utils import split_and_pad_trajectories_front as split_and_pad_trajectories


class RolloutStorageMM:
    class Transition:
        def __init__(self):
            self.observations = None
            self.reference_observations = None
            self.reference_observations_mask = None
            self.critic_observations = None
            self.critic_reference_observations = None
            self.critic_reference_observations_mask = None
            self.amp_observations = None
            self.amp_reference_observations = None
            self.amp_reference_observations_mask = None
            self.actions = None
            self.privileged_actions = None
            self.dagger_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def __init__(self, training_type, num_envs, num_transitions_per_env, obs_shape, ref_obs_shape, privileged_obs_shape, privileged_ref_obs_shape, actions_shape, apply_dagger_actions = False, rnd_state_shape = None, amp_cfg = None, device="cpu"):
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.obs_shape = obs_shape
        self.ref_obs_shape = ref_obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.privileged_ref_obs_shape = privileged_ref_obs_shape
        self.actions_shape = actions_shape
        self.rnd_state_shape = rnd_state_shape
        self.amp_cfg = amp_cfg
        self.amp_shape = amp_cfg["net_cfg"]["backbone_input_dim"] // 2 if amp_cfg is not None else None
     
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        if ref_obs_shape[0] is not None:
            self.reference_observations = torch.zeros(num_transitions_per_env, num_envs, *ref_obs_shape, device=self.device)
            self.reference_observations_mask = torch.zeros(num_transitions_per_env, num_envs, device=self.device).bool() # single stage mask shape: (num_envs,) with num_transitions: (num_transitions_per_env, num_envs)
            if privileged_ref_obs_shape[0] is not None:
                self.privileged_reference_observations = torch.zeros(
                    num_transitions_per_env, num_envs, *privileged_ref_obs_shape, device=self.device
                )
                self.privileged_reference_observations_mask = torch.zeros(num_transitions_per_env, num_envs, device=self.device).bool()
            else:
                self.privileged_reference_observations = None
                self.privileged_reference_observations_mask = None
        else:
            self.reference_observations = None
            self.reference_observations_mask = None
            self.privileged_reference_observations = None
            self.privileged_reference_observations_mask = None
        
        # for amp
        if self.amp_cfg is not None:
            assert ref_obs_shape[0] is not None, "AMP requires reference observations."
            self.amp_observations = torch.zeros(num_transitions_per_env, num_envs, self.amp_shape, device=self.device)
            self.amp_reference_observations = torch.zeros(num_transitions_per_env, num_envs, self.amp_shape, device=self.device)
            self.amp_reference_observations_mask = torch.zeros(num_transitions_per_env, num_envs, device=self.device).bool()
        else:
            self.amp_observations = None
            self.amp_reference_observations = None
            self.amp_reference_observations_mask = None
        
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.dagger_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device) if apply_dagger_actions else None

        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For PPO
        if training_type == "rl":
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)

        if self.reference_observations is not None:
            self.reference_observations[self.step].copy_(transition.reference_observations)
            self.reference_observations_mask[self.step].copy_(transition.reference_observations_mask)
            if self.privileged_reference_observations is not None:
                self.privileged_reference_observations[self.step].copy_(transition.critic_reference_observations)
                self.privileged_reference_observations_mask[self.step].copy_(transition.critic_reference_observations_mask)
                
        if self.dagger_actions is not None:
            self.dagger_actions[self.step].copy_(transition.dagger_actions)           
        
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)
            
        if self.amp_cfg is not None:
            self.amp_observations[self.step].copy_(transition.amp_observations)
            self.amp_reference_observations[self.step].copy_(transition.amp_reference_observations)
            self.amp_reference_observations_mask[self.step].copy_(transition.amp_reference_observations_mask)
        
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()
    
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]

            if self.privileged_reference_observations is not None:
                privileged_reference_observations = self.privileged_reference_observations[i]
                privileged_reference_observations_mask = self.privileged_reference_observations_mask[i]
                privileged_ref_obs_batch_rtn = (privileged_reference_observations, privileged_reference_observations_mask)
            else:
                privileged_ref_obs_batch_rtn = None
            ref_obs_batch_rtn = (self.reference_observations[i], self.reference_observations_mask[i])

            yield self.observations[i], privileged_observations, ref_obs_batch_rtn, privileged_ref_obs_batch_rtn, self.actions[i], self.privileged_actions[i], self.dones[i]

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        # indices = torch.arange(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        
        if self.reference_observations is not None:
            reference_observations = self.reference_observations.flatten(0, 1)
            reference_observations_mask = self.reference_observations_mask.flatten(0, 1)
            if self.privileged_reference_observations is not None:
                critic_reference_observations = self.privileged_reference_observations.flatten(0, 1)
                critic_reference_observations_mask = self.privileged_reference_observations_mask.flatten(0, 1)
            else:
                critic_reference_observations = reference_observations
                critic_reference_observations_mask = reference_observations_mask
        else:
            reference_observations = None
            reference_observations_mask = None
            critic_reference_observations = None
            critic_reference_observations_mask = None
        
        if self.amp_observations is not None:
            amp_observations = self.amp_observations.flatten(0, 1)
            amp_reference_observations = self.amp_reference_observations.flatten(0, 1)
            amp_reference_observations_mask = self.amp_reference_observations_mask.flatten(0, 1)
        else:
            amp_observations = None
            amp_reference_observations = None
            amp_reference_observations_mask = None

        actions = self.actions.flatten(0, 1)
        dagger_actions = self.dagger_actions.flatten(0, 1) if self.dagger_actions is not None else None
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                prev_batch_idx = torch.where(batch_idx > self.num_envs, batch_idx - self.num_envs, batch_idx) # since the batch is not circular, we don't allow index < 0. The first batch will be ignored.

                obs_batch = observations[batch_idx] # shape: (mini_batch_size, num_envs, *obs_shape)
                ref_obs_batch = reference_observations[batch_idx] if reference_observations is not None else None
                ref_obs_mask_batch = reference_observations_mask[batch_idx] if reference_observations_mask is not None else None
                ref_obs_batch_rtn = (ref_obs_batch, ref_obs_mask_batch) if ref_obs_batch is not None else None
                critic_observations_batch = critic_observations[batch_idx]
                critic_ref_obs_batch = critic_reference_observations[batch_idx] if critic_reference_observations is not None else None
                critic_ref_obs_mask_batch = critic_reference_observations_mask[batch_idx] if critic_reference_observations_mask is not None else None
                critic_ref_obs_batch_rtn = (critic_ref_obs_batch, critic_ref_obs_mask_batch) if critic_ref_obs_batch is not None else None
                actions_batch = actions[batch_idx]
                dagger_actions_batch = dagger_actions[batch_idx] if dagger_actions is not None else None
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # get amp new
                if self.amp_cfg and amp_observations is not None:
                    obs_prev_state = amp_observations[prev_batch_idx]
                    ref_obs_prev_state = amp_reference_observations[prev_batch_idx]
                    ref_obs_prev_mask = amp_reference_observations_mask[prev_batch_idx].squeeze(-1).bool()
                    obs_cur_state = amp_observations[batch_idx]
                    ref_obs_cur_state = amp_reference_observations[batch_idx]
                    ref_obs_cur_mask = amp_reference_observations_mask[batch_idx].squeeze(-1).bool()

                else:
                    obs_prev_state = None
                    ref_obs_prev_state = None
                    ref_obs_prev_mask = None
                    obs_cur_state = None
                    ref_obs_cur_state = None
                    ref_obs_cur_mask = None
                
                yield obs_batch, ref_obs_batch_rtn, critic_observations_batch, critic_ref_obs_batch_rtn, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, dagger_actions_batch, (
                    None,
                    None,
                ), None, rnd_state_batch, obs_prev_state, ref_obs_prev_state, ref_obs_prev_mask, obs_cur_state, ref_obs_cur_state, ref_obs_cur_mask

    def _create_obs_buffer_slice(self, obs_sequence, idx, dones_sequence):
        """
        Create a special obs buffer slice for buffer_mini_batch_generator.
        Optimized version using PyTorch parallel operations instead of loops.
        
        Args:
            obs_sequence: Flattened observation sequence of shape (total_steps, *obs_shape)
            idx: Current indices tensor of shape (mini_batch_size,)
            dones_sequence: Flattened dones sequence of shape (total_steps, 1) 
            num_steps_per_env: Number of steps to look back per environment
            num_envs: Number of environments
            
        Returns:
            obs_buffer: Buffer of shape (num_steps_per_env, mini_batch_size, *obs_shape)
            mask: Mask of shape (num_steps_per_env, mini_batch_size) indicating valid entries
        """
        num_steps_per_env = self.num_transitions_per_env
        num_envs = self.num_envs
        device = obs_sequence.device
        obs_shape = obs_sequence.shape[1:]
        mini_batch_size = idx.shape[0]
        
        # Flatten dones for easier indexing
        dones_flat = dones_sequence.squeeze(-1) if dones_sequence.dim() > 1 else dones_sequence
        
        # Create step offsets for all timesteps: [num_steps_per_env-1, num_steps_per_env-2, ..., 0]
        step_offsets = torch.arange(num_steps_per_env - 1, -1, -1, device=device)  # [3, 2, 1, 0]
        
        # Broadcast to create target indices for all batch samples and all steps
        # idx: (mini_batch_size,) -> (1, mini_batch_size)
        # step_offsets: (num_steps_per_env,) -> (num_steps_per_env, 1)
        # target_indices: (num_steps_per_env, mini_batch_size)
        idx_expanded = idx.unsqueeze(0)  # (1, mini_batch_size)
        step_offsets_expanded = step_offsets.unsqueeze(1)  # (num_steps_per_env, 1)
        target_indices = idx_expanded - step_offsets_expanded * num_envs  # (num_steps_per_env, mini_batch_size)
        
        # Create validity mask for indices (True where indices are valid)
        valid_mask = (target_indices >= 0) & (target_indices < obs_sequence.shape[0])
        
        # Clamp indices to valid range to avoid indexing errors
        clamped_indices = torch.clamp(target_indices, 0, obs_sequence.shape[0] - 1)
        
        # Gather observations using advanced indexing
        # obs_sequence: (total_steps, *obs_shape)
        # clamped_indices: (num_steps_per_env, mini_batch_size)
        obs_buffer = obs_sequence[clamped_indices]  # (num_steps_per_env, mini_batch_size, *obs_shape)
        
        # Zero out invalid entries
        obs_buffer = obs_buffer * valid_mask.unsqueeze(-1).expand_as(obs_buffer)
        
        # Apply done masking
        # Get dones for all target indices
        dones_buffer = dones_flat[clamped_indices]  # (num_steps_per_env, mini_batch_size)
        dones_buffer = dones_buffer * valid_mask  # Zero out invalid entries
        
        # Find the latest done position for each batch sample
        # We need to find the last True value in each column (batch sample)
        # First, create a mask for done positions
        done_positions = dones_buffer.bool()  # (num_steps_per_env, mini_batch_size)
        
        # For each batch sample, find the latest done step
        # We'll use a clever approach: multiply done positions by step indices and take max
        step_indices = torch.arange(num_steps_per_env, device=device).unsqueeze(1)  # (num_steps_per_env, 1)
        done_step_indices = done_positions.float() * step_indices.float()  # (num_steps_per_env, mini_batch_size)
        
        # Get the latest done step for each batch sample (-1 if no done found)
        latest_done_steps, _ = torch.max(done_step_indices, dim=0)  # (mini_batch_size,)
        has_done = torch.any(done_positions, dim=0)  # (mini_batch_size,)
        latest_done_steps = torch.where(has_done, latest_done_steps, -1)  # Set to -1 where no done
        
        # Create done masking
        # For each batch sample, mask out steps from 0 to latest_done_step (inclusive)
        step_range = torch.arange(num_steps_per_env, device=device).unsqueeze(1)  # (num_steps_per_env, 1)
        latest_done_expanded = latest_done_steps.unsqueeze(0)  # (1, mini_batch_size)
        
        # Create mask: False where step <= latest_done_step, True otherwise
        done_mask = step_range > latest_done_expanded  # (num_steps_per_env, mini_batch_size)
        
        # Where there's no done (latest_done_steps == -1), all steps should be True
        no_done_mask = latest_done_steps == -1  # (mini_batch_size,)
        done_mask = torch.where(no_done_mask.unsqueeze(0), True, done_mask)
        
        # Combine validity mask and done mask
        final_mask = valid_mask & done_mask
        
        return obs_buffer, final_mask, clamped_indices, valid_mask
    
    def _buffer_sample(self, obs_sequence, indices, mask):
        seq = obs_sequence[indices]  # (num_steps_per_env, mini_batch_size, *obs_shape)
        seq = seq * mask.unsqueeze(-1).expand_as(seq)  # Zero out invalid entries
        return seq
    # for MMGPT (non recurrent)
    def buffer_mini_batch_generator(self, num_mini_batches, num_epochs=8, num_steps_per_env=None):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        # indices = torch.arange(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        
        if self.reference_observations is not None:
            reference_observations = self.reference_observations.flatten(0, 1)
            reference_observations_mask = self.reference_observations_mask.flatten(0, 1)
            if self.privileged_reference_observations is not None:
                critic_reference_observations = self.privileged_reference_observations.flatten(0, 1)
                critic_reference_observations_mask = self.privileged_reference_observations_mask.flatten(0, 1)
            else:
                critic_reference_observations = reference_observations
                critic_reference_observations_mask = reference_observations_mask
        else:
            reference_observations = None
            reference_observations_mask = None
            critic_reference_observations = None
            critic_reference_observations_mask = None
        
        if self.amp_observations is not None:
            amp_observations = self.amp_observations.flatten(0, 1)
            amp_reference_observations = self.amp_reference_observations.flatten(0, 1)
            amp_reference_observations_mask = self.amp_reference_observations_mask.flatten(0, 1)
        else:
            amp_observations = None
            amp_reference_observations = None
            amp_reference_observations_mask = None

        actions = self.actions.flatten(0, 1)
        dagger_actions = self.dagger_actions.flatten(0, 1) if self.dagger_actions is not None else None
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                prev_batch_idx = torch.where(batch_idx > self.num_envs, batch_idx - self.num_envs, batch_idx) # since the batch is not circular, we don't allow index < 0. The first batch will be ignored.

                # obs_batch = observations[batch_idx] # shape: (mini_batch_size, num_envs, *obs_shape)
                obs_batch, masks_batch, clamped_indices, valid_mask = self._create_obs_buffer_slice(observations, batch_idx, self.dones.flatten(0, 1)) # shape: (num_steps_per_env, mini_batch_size, *obs_shape)
                # obs_batch: (num_steps_per_env, mini_batch_size, *obs_shape)
                # ref_obs_batch = reference_observations[batch_idx] if reference_observations is not None else None
                # ref_obs_batch, _ = self._create_obs_buffer_slice(reference_observations, batch_idx, self.dones.flatten(0, 1)) if reference_observations is not None else (None, None)
                ref_obs_batch = self._buffer_sample(reference_observations, clamped_indices, valid_mask) if reference_observations is not None else None
                # ref_obs_mask_batch = reference_observations_mask[batch_idx] if reference_observations_mask is not None else None
                # ref_obs_mask_batch, _ = self._create_obs_buffer_slice(reference_observations_mask.unsqueeze(-1).float(), batch_idx, self.dones.flatten(0, 1)) if reference_observations_mask is not None else (None, None)
                ref_obs_mask_batch = self._buffer_sample(reference_observations_mask.unsqueeze(-1).float(), clamped_indices, valid_mask).squeeze(-1) if reference_observations_mask is not None else None
                # ref_obs_mask_batch = ref_obs_mask_batch.squeeze(-1).bool() if ref_obs_mask_batch is not None else None
                ref_obs_batch_rtn = (ref_obs_batch, ref_obs_mask_batch) if ref_obs_batch is not None else None
                # critic_observations_batch = critic_observations[batch_idx]
                # critic_observations_batch, _ = self._create_obs_buffer_slice(critic_observations, batch_idx, self.dones.flatten(0, 1))
                critic_observations_batch = self._buffer_sample(critic_observations, clamped_indices, valid_mask)
                # critic_ref_obs_batch = critic_reference_observations[batch_idx] if critic_reference_observations is not None else None
                # critic_ref_obs_batch, _ = self._create_obs_buffer_slice(critic_reference_observations, batch_idx, self.dones.flatten(0, 1)) if critic_reference_observations is not None else (None, None)
                critic_ref_obs_batch = self._buffer_sample(critic_reference_observations, clamped_indices, valid_mask) if critic_reference_observations is not None else None
                # critic_ref_obs_mask_batch, _ = self._create_obs_buffer_slice(critic_reference_observations_mask.unsqueeze(-1).float(), batch_idx, self.dones.flatten(0, 1)) if critic_reference_observations_mask is not None else (None, None)
                critic_ref_obs_mask_batch = self._buffer_sample(critic_reference_observations_mask.unsqueeze(-1).float(), clamped_indices, valid_mask).squeeze(-1) if critic_reference_observations_mask is not None else None
                # critic_ref_obs_mask_batch = critic_ref_obs_mask_batch.squeeze(-1).bool() if critic_ref_obs_mask_batch is not None else None
                critic_ref_obs_batch_rtn = (critic_ref_obs_batch, critic_ref_obs_mask_batch) if critic_ref_obs_batch is not None else None
                actions_batch = actions[batch_idx]
                dagger_actions_batch = dagger_actions[batch_idx] if dagger_actions is not None else None
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # get amp new
                if self.amp_cfg and amp_observations is not None:
                    obs_prev_state = amp_observations[prev_batch_idx]
                    ref_obs_prev_state = amp_reference_observations[prev_batch_idx]
                    ref_obs_prev_mask = amp_reference_observations_mask[prev_batch_idx].squeeze(-1).bool()
                    obs_cur_state = amp_observations[batch_idx]
                    ref_obs_cur_state = amp_reference_observations[batch_idx]
                    ref_obs_cur_mask = amp_reference_observations_mask[batch_idx].squeeze(-1).bool()

                else:
                    obs_prev_state = None
                    ref_obs_prev_state = None
                    ref_obs_prev_mask = None
                    obs_cur_state = None
                    ref_obs_cur_state = None
                    ref_obs_cur_mask = None
                
                yield obs_batch, ref_obs_batch_rtn, critic_observations_batch, critic_ref_obs_batch_rtn, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, dagger_actions_batch, (
                    None,
                    None,
                ), masks_batch, rnd_state_batch, obs_prev_state, ref_obs_prev_state, ref_obs_prev_mask, obs_cur_state, ref_obs_cur_state, ref_obs_cur_mask
    
    

    # for MMGPT
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        
        # batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = self.num_envs // num_mini_batches
        padded_obs_trajectories, obs_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_ref_obs_trajectories, padded_ref_obs_masks = split_and_pad_trajectories(self.reference_observations, self.dones) if self.reference_observations is not None else (None, None)
        padded_ref_observation_masks, _ = split_and_pad_trajectories(self.reference_observations_mask.unsqueeze(-1).float(), self.dones) if self.reference_observations_mask is not None else (None, None)
        
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, critic_obs_masks = split_and_pad_trajectories(self.privileged_observations, self.dones)
            padded_critic_ref_obs_trajectories, padded_critic_ref_obs_masks = split_and_pad_trajectories(self.privileged_reference_observations, self.dones) if self.privileged_reference_observations is not None else (None, None)
            padded_critic_ref_observation_masks, _ = split_and_pad_trajectories(self.privileged_reference_observations_mask.unsqueeze(-1).float(), self.dones) if self.privileged_reference_observations_mask is not None else (None, None)
        else:
            padded_critic_obs_trajectories, critic_obs_masks = padded_obs_trajectories, obs_masks
            padded_critic_ref_obs_trajectories, padded_critic_ref_obs_masks = padded_ref_obs_trajectories, padded_ref_obs_masks
            padded_critic_ref_observation_masks = padded_ref_observation_masks
            
        if self.amp_observations is not None:
            padded_amp_obs_trajectories, amp_obs_masks = split_and_pad_trajectories(self.amp_observations, self.dones)
            padded_amp_ref_obs_trajectories, padded_amp_ref_obs_masks = split_and_pad_trajectories(self.amp_reference_observations, self.dones) if self.amp_reference_observations is not None else (None, None)
            padded_amp_ref_observation_masks, _ = split_and_pad_trajectories(self.amp_reference_observations_mask.unsqueeze(-1).float(), self.dones) if self.amp_reference_observations_mask is not None else (None, None)
            
        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        
        else:
            padded_rnd_state_trajectories = None

        
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = 1
                trajectories_batch_size = torch.sum(last_was_done[:,start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = obs_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                ref_obs_batch = padded_ref_obs_trajectories[:, first_traj:last_traj] if padded_ref_obs_trajectories is not None else None
                ref_obs_mask_batch = padded_ref_observation_masks[:, first_traj:last_traj].squeeze(-1) if padded_ref_observation_masks is not None else None
                ref_obs_batch_rtn = (ref_obs_batch, ref_obs_mask_batch)
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                critic_ref_obs_batch = padded_critic_ref_obs_trajectories[:, first_traj:last_traj] if padded_critic_ref_obs_trajectories is not None else None
                critic_ref_obs_mask_batch = padded_critic_ref_observation_masks[:, first_traj:last_traj].squeeze(-1) if padded_critic_ref_observation_masks is not None else None
                critic_ref_obs_batch_rtn = (critic_ref_obs_batch, critic_ref_obs_mask_batch)
                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None
                    
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                target_values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                dagger_actions_batch = self.dagger_actions[:, start:stop] if self.dagger_actions is not None else None
                last_was_done = last_was_done.permute(1, 0)
                
                # get amp new states
                if self.amp_cfg and ref_obs_batch is not None and ref_obs_mask_batch is not None:
                    obs_prev_state = padded_amp_obs_trajectories[:, first_traj-1:last_traj-1] if first_traj > 0 else None # (T, N, D)
                    ref_obs_prev_state = padded_amp_ref_obs_trajectories[:, first_traj-1:last_traj-1] if first_traj > 0 else None # (T, N, D)
                    ref_obs_prev_mask = padded_amp_ref_observation_masks[:, first_traj-1:last_traj-1].squeeze(-1) if (padded_amp_ref_observation_masks is not None and first_traj > 0) else None # (T, N)
                    obs_cur_state = padded_amp_obs_trajectories[:, first_traj:last_traj] # (T, N, D)
                    ref_obs_cur_state = padded_amp_ref_obs_trajectories[:, first_traj:last_traj] if padded_amp_ref_obs_trajectories is not None else None # (T, N, D)
                    ref_obs_cur_mask = padded_amp_ref_observation_masks[:, first_traj:last_traj].squeeze(-1) if padded_amp_ref_observation_masks is not None else None # (T, N)
                    
                    # reshape to (T*N, D) and (T*N,)
                    obs_prev_state = obs_prev_state.reshape(-1, self.amp_shape) if obs_prev_state is not None else None
                    ref_obs_prev_state = ref_obs_prev_state.reshape(-1, self.amp_shape) if ref_obs_prev_state is not None else None
                    ref_obs_prev_mask = ref_obs_prev_mask.reshape(-1).bool() if ref_obs_prev_mask is not None else None
                    obs_cur_state = obs_cur_state.reshape(-1, self.amp_shape)
                    ref_obs_cur_state = ref_obs_cur_state.reshape(-1, self.amp_shape) if ref_obs_cur_state is not None else None
                    ref_obs_cur_mask = ref_obs_cur_mask.reshape(-1).bool() if ref_obs_cur_mask is not None else None
                else:
                    obs_prev_state = None
                    ref_obs_prev_state = None
                    ref_obs_prev_mask = None
                    obs_cur_state = None
                    ref_obs_cur_state = None
                    ref_obs_cur_mask = None


                yield obs_batch, ref_obs_batch_rtn, critic_obs_batch, critic_ref_obs_batch_rtn, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, dagger_actions_batch, (
                    None,
                    None,
                ), masks_batch, rnd_state_batch, obs_prev_state, ref_obs_prev_state, ref_obs_prev_mask, obs_cur_state, ref_obs_cur_state, ref_obs_cur_mask
                
                first_traj = last_traj

                
                    
                
                
                
                
                
