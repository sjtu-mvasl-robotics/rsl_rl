import inspect

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.algorithms.mmppo import MMPPO, _actor_ppo_enabled
from rsl_rl.runners.on_policy_runner_mm import (
    OnPolicyRunnerMM,
    _amp_reward_enabled,
    _compose_amp_training_rewards,
)


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

    disabled = _compose_amp_training_rewards(
        base, amp, scale=0.5, enabled=False
    )
    enabled = _compose_amp_training_rewards(
        base, amp, scale=0.5, enabled=True
    )

    torch.testing.assert_close(disabled, base)
    torch.testing.assert_close(enabled, torch.tensor([1.2, -0.4]))
    torch.testing.assert_close(base, torch.tensor([1.0, -0.5]))
    assert disabled.data_ptr() != base.data_ptr()


def test_amp_pretrain_boundary_is_explicit():
    assert not _amp_reward_enabled(2_999, 3_000)
    assert _amp_reward_enabled(3_000, 3_000)


def test_runner_stores_amp_reward_before_process_env_step():
    source = inspect.getsource(OnPolicyRunnerMM.learn)
    compose_at = source.index("rewards = _compose_amp_training_rewards(")
    clear_done_at = source.index("amp_obs_buffer[done_ids] = 0", compose_at)
    storage_at = source.index("self.alg.process_env_step(rewards, dones, infos)")
    planner_at = source.index("accumulate_planner_reward(\n                            base_rewards.squeeze()")
    logging_at = source.index("if self.log_dir is not None:", compose_at)

    assert compose_at < clear_done_at < storage_at < planner_at < logging_at


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
    training_reward = _compose_amp_training_rewards(
        torch.tensor([1.0]),
        torch.tensor([0.4]),
        scale=0.5,
        enabled=True,
    )
    algorithm.process_env_step(
        training_reward,
        torch.tensor([False]),
        {"time_outs": torch.tensor([1.0])},
    )

    expected = training_reward.squeeze() + algorithm.gamma * bootstrap_value
    torch.testing.assert_close(algorithm.storage.rewards[0, 0, 0], expected)
