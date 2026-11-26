import inspect

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.algorithms.mmppo import MMPPO, _actor_ppo_enabled
from rsl_rl.modules.actor_critic_mm_gpt import ActorCriticMMGPT
from rsl_rl.runners.on_policy_runner_mm import (
    OnPolicyRunnerMM,
    _amp_reward_enabled,
    _compose_amp_training_rewards,
    _restore_sonic_adaptive_sampler_state,
)
import rsl_rl


def test_source_checkout_exposes_distribution_version():
    assert rsl_rl.__version__ == "2.3.1.dev3"


def test_plain_ppo_never_waits_for_teacher_critic_gate():
    assert _actor_ppo_enabled(
        teacher_enabled=False,
        epoch=0,
        teacher_only_interval=0,
        teacher_supervising_intervals=40_000,
        critic_win_streak=0,
    )


def test_teacher_warmup_keeps_critic_readiness_gate():
    common = {
        "teacher_enabled": True,
        "teacher_only_interval": 10,
        "teacher_supervising_intervals": 5,
    }
    assert not _actor_ppo_enabled(
        **common, epoch=4, critic_win_streak=3
    )
    assert not _actor_ppo_enabled(
        **common, epoch=10, critic_win_streak=2
    )
    assert _actor_ppo_enabled(
        **common, epoch=10, critic_win_streak=3
    )


def test_amp_reward_is_composed_without_mutating_environment_reward():
    base = torch.tensor([1.0, -0.5])
    amp = torch.tensor([0.4, 0.2])

    disabled, disabled_amp = _compose_amp_training_rewards(
        base, amp, scale=0.5, enabled=False
    )
    enabled, enabled_amp = _compose_amp_training_rewards(
        base, amp, scale=0.5, enabled=True
    )

    torch.testing.assert_close(disabled, base)
    torch.testing.assert_close(disabled_amp, torch.zeros_like(amp))
    torch.testing.assert_close(enabled, torch.tensor([1.2, -0.4]))
    torch.testing.assert_close(enabled_amp, torch.tensor([0.2, 0.1]))
    torch.testing.assert_close(base, torch.tensor([1.0, -0.5]))
    assert disabled.data_ptr() != base.data_ptr()


def test_amp_pretrain_boundary_is_explicit():
    assert not _amp_reward_enabled(2_999, 3_000)
    assert _amp_reward_enabled(3_000, 3_000)


def test_runner_stores_amp_reward_before_process_env_step():
    source = inspect.getsource(OnPolicyRunnerMM.learn)
    compose_at = source.index("rewards, applied_amp_rewards = _compose_amp_training_rewards(")
    clear_done_at = source.index("amp_obs_buffer[done_ids] = 0", compose_at)
    storage_at = source.index("self.alg.process_env_step(rewards, dones, infos)")
    planner_at = source.index("accumulate_planner_reward(\n                            base_rewards.squeeze()")
    logging_at = source.index("if self.log_dir is not None:", compose_at)
    applied_logging_at = source.index(
        "cur_amp_reward_sum += applied_amp_rewards", logging_at
    )

    assert compose_at < clear_done_at < storage_at < planner_at < logging_at < applied_logging_at


def test_sonic_adaptive_distribution_updates_only_after_each_rollout():
    source = inspect.getsource(OnPolicyRunnerMM.learn)
    rollout_at = source.index("for i in range(self.num_steps_per_env):")
    update_at = source.index("loss_dict = self.alg.update(epoch=it)")
    adaptive_at = source.index(
        'getattr(\n                adaptive_env, "update_sonic_adaptive_sampling", None'
    )

    assert rollout_at < update_at < adaptive_at
    assert source.count('"update_sonic_adaptive_sampling"') == 1


def test_sonic_adaptive_statistics_are_checkpointed_backward_compatibly():
    save_source = inspect.getsource(OnPolicyRunnerMM.save)
    load_source = inspect.getsource(OnPolicyRunnerMM.load)

    assert '"sonic_adaptive_sampler_state_dict"' in save_source
    assert "sonic_adaptive_state_dict" in save_source
    assert "_restore_sonic_adaptive_sampler_state" in load_source


class _SamplerStateEnv:
    def __init__(self, enabled: bool, expose_loader: bool = True):
        self.enabled = enabled
        self.loaded_state = None
        if expose_loader:
            self.load_sonic_adaptive_state_dict = self._load

    def sonic_adaptive_sampling_enabled(self):
        return self.enabled

    def _load(self, state):
        self.loaded_state = state


def test_adaptive_checkpoint_is_policy_loadable_by_strict_base_eval():
    state = {"num_episodes": torch.tensor([3.0])}
    base_env = _SamplerStateEnv(enabled=False)

    restored = _restore_sonic_adaptive_sampler_state(
        base_env, {"sonic_adaptive_sampler_state_dict": state}
    )

    assert restored is False
    assert base_env.loaded_state is None


def test_adaptive_resume_strictly_restores_sampler_state():
    state = {"num_episodes": torch.tensor([3.0])}
    adaptive_env = _SamplerStateEnv(enabled=True)

    restored = _restore_sonic_adaptive_sampler_state(
        adaptive_env, {"sonic_adaptive_sampler_state_dict": state}
    )

    assert restored is True
    assert adaptive_env.loaded_state is state

    broken_adaptive_env = _SamplerStateEnv(
        enabled=True, expose_loader=False
    )
    with pytest.raises(RuntimeError, match="cannot restore"):
        _restore_sonic_adaptive_sampler_state(
            broken_adaptive_env,
            {"sonic_adaptive_sampler_state_dict": state},
        )


class _TinyActorCritic(nn.Module):
    """Small CPU policy exposing the contract MMPPO uses in one update."""

    is_recurrent = False
    noise_std_type = "scalar"
    supports_latent_alignment = False

    def __init__(self, obs_dim=3, action_dim=2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, action_dim)
        self.critic = nn.Linear(obs_dim, 1)
        self.std = nn.Parameter(torch.full((action_dim,), 0.5))
        self._distribution = None

    def act(self, observations, **_kwargs):
        mean = self.actor(observations)
        self._distribution = Normal(mean, self.std.clamp_min(0.05).expand_as(mean))
        return self._distribution.sample()

    def evaluate(self, observations, *_args, **_kwargs):
        return self.critic(observations)

    def get_actions_log_prob(self, actions):
        return self._distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self._distribution.mean

    @property
    def action_std(self):
        return self._distribution.stddev

    @property
    def entropy(self):
        return self._distribution.entropy().sum(dim=-1)

    def reset(self, _dones):
        pass


def test_plain_ppo_first_update_changes_actor_with_cold_critic():
    torch.manual_seed(7)
    actor_critic = _TinyActorCritic()
    algorithm = MMPPO(
        actor_critic,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="fixed",
        teacher_coef=None,
        teacher_loss_coef=None,
        teacher_supervising_intervals=40_000,
        normalize_advantage_per_mini_batch=True,
        device="cpu",
    )
    algorithm.init_storage(
        "rl",
        num_envs=8,
        num_transitions_per_env=4,
        actor_obs_shape=(3,),
        actor_ref_obs_shape=(None,),
        critic_obs_shape=(3,),
        critic_ref_obs_shape=(None,),
        action_shape=(2,),
    )

    observations = torch.randn(8, 3)
    for step in range(4):
        actions = algorithm.act(observations, None, observations, None)
        rewards = -torch.square(actions - 0.25).sum(dim=-1)
        dones = torch.zeros(8, dtype=torch.bool)
        if step == 3:
            dones[0] = True
        algorithm.process_env_step(rewards, dones, {})
        observations = torch.randn(8, 3)
    algorithm.compute_returns(observations, None)

    before = [parameter.detach().clone() for parameter in actor_critic.actor.parameters()]
    metrics = algorithm.update(epoch=0)
    changed = [
        not torch.equal(old, new.detach())
        for old, new in zip(before, actor_critic.actor.parameters())
    ]

    assert metrics["actor_ppo_update_fraction"] == 1.0
    assert metrics["mean_actor_grad_norm"] > 0.0
    assert all(changed)


def test_streaming_mmgpt_first_ppo_update_changes_backbone(capsys):
    torch.manual_seed(11)
    terms = {
        "joint_pos": 2,
        "joint_vel": 2,
        "projected_gravity": 3,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "actions": 2,
    }
    reference_terms = {"target_actions": 2}
    actor_critic = ActorCriticMMGPT(
        num_actor_obs=sum(terms.values()),
        num_actor_ref_obs=2,
        num_critic_obs=sum(terms.values()),
        num_critic_ref_obs=2,
        num_actions=2,
        term_dict={"policy": terms, "critic": terms},
        ref_term_dict={
            "policy": reference_terms,
            "critic": reference_terms,
        },
        ref_term_steps={"policy": None, "critic": None},
        concatenate_term_names={"policy": None, "critic": None},
        concatenate_ref_term_names={"policy": None, "critic": None},
        pred_obs_term_names=["joint_pos"],
        pred_obs_term_weights=[1.0],
        dim_model=16,
        num_heads=2,
        num_layers=2,
        num_steps_per_env=3,
        max_seq_len=3,
        critic_history_length=1,
        critic_hidden_dims=[32, 16],
        critic_fusion_mode="direct_concat",
        apply_rope=True,
        latest_reference_only=False,
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=3,
        default_joint_pos=torch.zeros(2),
    )
    algorithm = MMPPO(
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="fixed",
        teacher_coef=None,
        teacher_loss_coef=None,
        normalize_advantage_per_mini_batch=True,
        aux_loss_coef=1.0,
        dreamer_loss_coef=0.1,
        device="cpu",
    )
    num_envs, rollout_steps = 4, 4
    obs_dim = sum(terms.values())
    algorithm.init_storage(
        "rl",
        num_envs=num_envs,
        num_transitions_per_env=rollout_steps,
        actor_obs_shape=(obs_dim,),
        actor_ref_obs_shape=(2,),
        critic_obs_shape=(obs_dim,),
        critic_ref_obs_shape=(2,),
        action_shape=(2,),
        store_raw_observations=True,
        dreamer_target_shape=(2,),
    )

    observations = torch.randn(num_envs, obs_dim)
    for step in range(rollout_steps):
        references = torch.randn(num_envs, 2)
        reference_masks = torch.ones(num_envs, dtype=torch.bool)
        actions = algorithm.act(
            observations,
            (references, reference_masks),
            observations,
            (references, reference_masks),
            raw_obs=observations,
            dreamer_target=(
                torch.randn(num_envs, 2),
                torch.ones(num_envs, dtype=torch.bool),
            ),
        )
        rewards = -torch.square(actions - 0.1).sum(dim=-1)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        dones[0] = step == 1
        algorithm.process_env_step(rewards, dones, {})
        actor_critic.reset(dones)
        observations = torch.randn_like(observations)
    final_references = torch.randn(num_envs, 2)
    algorithm.compute_returns(
        observations,
        (final_references, torch.ones(num_envs, dtype=torch.bool)),
    )

    backbone_before = [
        parameter.detach().clone()
        for parameter in actor_critic.actor.decoder.parameters()
    ]
    metrics = algorithm.update(epoch=0)
    backbone_changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(
            backbone_before, actor_critic.actor.decoder.parameters()
        )
    )
    capsys.readouterr()

    assert metrics["actor_ppo_update_fraction"] == 1.0
    assert metrics["mean_actor_grad_norm"] > 0.0
    assert metrics["mean_aux_loss"] > 0.0
    assert metrics["mean_dreamer_loss"] > 0.0
    assert backbone_changed


def test_chunk5_first_ppo_update_uses_stored_phase_and_changes_slow_backbone(
    capsys,
):
    torch.manual_seed(13)
    terms = {
        "joint_pos": 2,
        "joint_vel": 2,
        "projected_gravity": 3,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "actions": 2,
    }
    reference_terms = {"target_actions": 2}
    actor_critic = ActorCriticMMGPT(
        num_actor_obs=sum(terms.values()),
        num_actor_ref_obs=2,
        num_critic_obs=sum(terms.values()),
        num_critic_ref_obs=2,
        num_actions=2,
        term_dict={"policy": terms, "critic": terms},
        ref_term_dict={
            "policy": reference_terms,
            "critic": reference_terms,
        },
        ref_term_steps={"policy": None, "critic": None},
        concatenate_term_names={"policy": None, "critic": None},
        concatenate_ref_term_names={"policy": None, "critic": None},
        pred_obs_term_names=["joint_pos"],
        pred_obs_term_weights=[1.0],
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=5,
        max_seq_len=5,
        critic_history_length=1,
        critic_hidden_dims=[32, 16],
        critic_fusion_mode="direct_concat",
        apply_rope=True,
        latest_reference_only=False,
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=5,
        temporal_chunk_size=5,
        default_joint_pos=torch.zeros(2),
    )
    algorithm = MMPPO(
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="fixed",
        teacher_coef=None,
        teacher_loss_coef=None,
        normalize_advantage_per_mini_batch=True,
        aux_loss_coef=1.0,
        dreamer_loss_coef=0.1,
        device="cpu",
    )
    num_envs, rollout_steps = 4, 6
    obs_dim = sum(terms.values())
    algorithm.init_storage(
        "rl",
        num_envs=num_envs,
        num_transitions_per_env=rollout_steps,
        actor_obs_shape=(obs_dim,),
        actor_ref_obs_shape=(2,),
        critic_obs_shape=(obs_dim,),
        critic_ref_obs_shape=(2,),
        action_shape=(2,),
        store_raw_observations=True,
        dreamer_target_shape=(2,),
    )
    assert algorithm.storage.actor_history_length == 9
    assert algorithm.storage.actor_temporal_phases is not None

    observations = torch.randn(num_envs, obs_dim)
    for step in range(rollout_steps):
        references = torch.randn(num_envs, 2)
        reference_masks = torch.ones(num_envs, dtype=torch.bool)
        actions = algorithm.act(
            observations,
            (references, reference_masks),
            observations,
            (references, reference_masks),
            raw_obs=observations,
            dreamer_target=(
                torch.randn(num_envs, 2),
                torch.ones(num_envs, dtype=torch.bool),
            ),
        )
        rewards = -torch.square(actions - 0.1).sum(dim=-1)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        dones[0] = step == 2
        algorithm.process_env_step(rewards, dones, {})
        observations = torch.randn_like(observations)
    torch.testing.assert_close(
        algorithm.storage.actor_temporal_phases[:, 1],
        torch.tensor([0, 1, 2, 3, 4, 0]),
    )
    torch.testing.assert_close(
        algorithm.storage.actor_temporal_phases[:, 0],
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )
    final_references = torch.randn(num_envs, 2)
    algorithm.compute_returns(
        observations,
        (final_references, torch.ones(num_envs, dtype=torch.bool)),
    )

    slow_before = actor_critic.actor.decoder.encoder.layers[
        0
    ].self_attn.qkv_proj.weight.detach().clone()
    metrics = algorithm.update(epoch=0)
    capsys.readouterr()

    assert metrics["actor_ppo_update_fraction"] == 1.0
    assert metrics["mean_actor_grad_norm"] > 0.0
    assert metrics["mean_aux_loss"] > 0.0
    assert metrics["mean_dreamer_loss"] > 0.0
    assert not torch.equal(
        slow_before,
        actor_critic.actor.decoder.encoder.layers[
            0
        ].self_attn.qkv_proj.weight.detach(),
    )


def test_composed_amp_reward_precedes_timeout_bootstrap_in_storage():
    actor_critic = _TinyActorCritic()
    algorithm = MMPPO(actor_critic, device="cpu")
    algorithm.init_storage(
        "rl",
        num_envs=1,
        num_transitions_per_env=1,
        actor_obs_shape=(3,),
        actor_ref_obs_shape=(None,),
        critic_obs_shape=(3,),
        critic_ref_obs_shape=(None,),
        action_shape=(2,),
    )
    observation = torch.tensor([[0.2, -0.1, 0.4]])
    algorithm.act(observation, None, observation, None)
    bootstrap_value = algorithm.transition.values.squeeze().clone()
    training_reward, applied_amp_reward = _compose_amp_training_rewards(
        torch.tensor([1.0]),
        torch.tensor([0.4]),
        scale=0.5,
        enabled=True,
    )
    torch.testing.assert_close(applied_amp_reward, torch.tensor([0.2]))
    algorithm.process_env_step(
        training_reward,
        torch.tensor([False]),
        {"time_outs": torch.tensor([1.0])},
    )

    expected = training_reward.squeeze() + algorithm.gamma * bootstrap_value
    torch.testing.assert_close(algorithm.storage.rewards[0, 0, 0], expected)
