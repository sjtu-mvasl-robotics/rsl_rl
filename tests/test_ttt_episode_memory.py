from types import SimpleNamespace

import torch

from rsl_rl.algorithms.mmppo import FixedExpertSequenceReplayBuffer
from rsl_rl.modules.actor_critic_mm_gpt import MMGPT, TTTKVBLinearMemory


def _terminal_input_gradients(
    memory: TTTKVBLinearMemory,
    hidden: torch.Tensor,
    *,
    reset_at: int | None = None,
) -> torch.Tensor:
    masks = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    reset_masks = torch.zeros_like(masks)
    if reset_at is not None:
        reset_masks[reset_at] = True
    output, _ = memory.forward_sequence(
        hidden,
        masks,
        state_reset_masks=reset_masks,
    )
    terminal_objective = output[-1].square().sum()
    return torch.autograd.grad(terminal_objective, hidden)[0]


def test_terminal_objective_backpropagates_through_long_fast_weight_history():
    """The final read must retain a differentiable path to the first write."""

    torch.manual_seed(901)
    time_steps = 800
    # One 32-dimensional head has the same per-head rank as the production
    # d_model=256, num_heads=8 memory while keeping this CPU check inexpensive.
    memory = TTTKVBLinearMemory(
        32,
        1,
        inner_lr=0.1,
        gate_init=0.25,
    ).double()
    hidden = torch.randn(
        time_steps,
        1,
        32,
        dtype=torch.double,
        requires_grad=True,
    )

    gradients = _terminal_input_gradients(memory, hidden)

    assert torch.isfinite(gradients).all()
    assert gradients[0].norm().item() > 0.0
    assert gradients[24].norm().item() > 0.0
    assert gradients[249].norm().item() > 0.0
    assert gradients[499].norm().item() > 0.0


def test_explicit_reset_removes_only_pre_reset_credit_assignment():
    torch.manual_seed(902)
    reset_at = 96
    memory = TTTKVBLinearMemory(
        32,
        1,
        inner_lr=0.1,
        gate_init=0.25,
    ).double()
    hidden = torch.randn(
        192,
        1,
        32,
        dtype=torch.double,
        requires_grad=True,
    )

    gradients = _terminal_input_gradients(
        memory,
        hidden,
        reset_at=reset_at,
    )

    torch.testing.assert_close(
        gradients[:reset_at],
        torch.zeros_like(gradients[:reset_at]),
        atol=0.0,
        rtol=0.0,
    )
    assert gradients[reset_at].norm().item() > 0.0
    assert gradients[-2].norm().item() > 0.0


def test_long_terminal_objective_reaches_every_slow_memory_parameter_group():
    torch.manual_seed(903)
    memory = TTTKVBLinearMemory(
        32,
        1,
        inner_lr=0.1,
        gate_init=0.25,
    ).double()
    hidden = torch.randn(192, 2, 32, dtype=torch.double)
    masks = torch.ones(192, 2, dtype=torch.bool)

    output, _ = memory.forward_sequence(hidden, masks)
    output[-1].square().mean().backward()

    expected = {
        "initial_fast_weights",
        "inner_lr_logit",
        "gate_logit",
        "input_norm.weight",
        "input_norm.bias",
        "qkv_proj.weight",
        "out_proj.weight",
    }
    assert set(dict(memory.named_parameters())) == expected
    for name, parameter in memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm().item() > 0.0, name


def _rollout_storage(
    observations: torch.Tensor,
    expert_actions: torch.Tensor,
    *,
    starts_at_initial_state: bool,
    terminal: bool,
) -> SimpleNamespace:
    time_steps, num_envs = observations.shape[:2]
    dones = torch.zeros(time_steps, num_envs, 1, dtype=torch.bool)
    if terminal:
        dones[-1] = True
    return SimpleNamespace(
        step=time_steps,
        num_transitions_per_env=time_steps,
        observations=observations,
        reference_observations=None,
        reference_observations_mask=None,
        dones=dones,
        actor_temporal_phases=None,
        actor_recurrent_reset_events=torch.zeros(
            time_steps, num_envs, dtype=torch.bool
        ),
        dagger_actions=expert_actions,
        store_actor_recurrent_state=True,
        actor_initial_recurrent_state_reset_mask=torch.full(
            (num_envs,), starts_at_initial_state, dtype=torch.bool
        ),
        actor_history_length=1,
        actor_history_prefix_observations=None,
        actor_history_prefix_reference_observations=None,
        actor_history_prefix_reference_masks=None,
        actor_history_prefix_valid=None,
        actor_history_prefix_dones=None,
        actor_history_prefix_temporal_phases=None,
    )


def test_sequence_replay_preserves_one_episode_across_rollout_boundaries():
    torch.manual_seed(904)
    rollout_steps = 24
    num_rollouts = 4
    hidden_dim = 32
    replay = FixedExpertSequenceReplayBuffer(
        capacity=8,
        history_length=1,
        storage_dtype=torch.float32,
    )

    for rollout_index in range(num_rollouts):
        observations = torch.randn(rollout_steps, 1, hidden_dim)
        expert_actions = torch.randn(rollout_steps, 1, 3)
        replay.add_rollout(
            _rollout_storage(
                observations,
                expert_actions,
                starts_at_initial_state=rollout_index == 0,
                terminal=rollout_index == num_rollouts - 1,
            )
        )

    assert len(replay) == 1
    assert replay.buffer[0]["expert_actions"].shape[0] == (
        rollout_steps * num_rollouts
    )
    batch = replay.sample_batch(1, device="cpu", dtype=torch.double)
    observations = batch["observations"].detach().requires_grad_(True)
    memory = TTTKVBLinearMemory(
        hidden_dim,
        1,
        inner_lr=0.1,
        gate_init=0.25,
    ).double()
    output, _ = memory.forward_sequence(
        observations,
        batch["masks"],
        episode_ids=batch["episode_ids"],
        state_reset_masks=batch["ttt_state_reset_masks"],
    )
    terminal_objective = output[-1].square().sum()
    terminal_objective.backward()

    assert observations.grad[0].norm().item() > 0.0
    assert observations.grad[rollout_steps - 1].norm().item() > 0.0
    assert observations.grad[rollout_steps].norm().item() > 0.0
    assert memory.qkv_proj.weight.grad.norm().item() > 0.0


def _small_streaming_ttt_actor() -> MMGPT:
    term_dims = {
        "joint_pos": 2,
        "joint_vel": 2,
        "projected_gravity": 3,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "actions": 2,
    }
    return MMGPT(
        obs_size=sum(term_dims.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        num_steps_per_env=4,
        max_seq_len=4,
        term_dict=term_dims,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=None,
        pred_obs_term_weights=None,
        default_joint_pos=torch.zeros(2),
        apply_rope=True,
        latest_reference_only=False,
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=4,
        ttt_memory_enabled=True,
        ttt_num_heads=2,
        ttt_inner_lr=0.1,
        ttt_gate_init=1.0e-3,
    ).eval()


def _terminal_action_input_gradients(
    model: MMGPT,
    observations: torch.Tensor,
    references: torch.Tensor,
) -> torch.Tensor:
    time_steps, batch_size = observations.shape[:2]
    masks = torch.ones(time_steps, batch_size, dtype=torch.bool)
    episode_ids = torch.zeros(time_steps, batch_size, dtype=torch.long)
    initial_state = model.ttt_memory.initial_state(
        batch_size,
        device=observations.device,
        dtype=observations.dtype,
    )
    actions = model(
        observations,
        (references, masks),
        masks,
        memory={
            "ttt_initial_state": initial_state,
            "ttt_initial_state_reset_mask": torch.ones(
                batch_size, dtype=torch.bool
            ),
        },
        episode_ids=episode_ids,
        sequence_output_start=0,
        compute_aux_loss=False,
        compute_dreamer_loss=False,
    ).reshape(time_steps, batch_size, -1)
    return torch.autograd.grad(actions[-1].square().sum(), observations)[0]


def test_full_actor_terminal_action_uses_ttt_path_beyond_short_context():
    """Early action credit must pass through TTT, not the four-step KV cache."""

    torch.manual_seed(905)
    model = _small_streaming_ttt_actor()
    time_steps = 96
    observations_base = torch.randn(time_steps, 1, model.obs_dim)
    references = torch.randn(time_steps, 1, model.ref_obs_dim)

    observations = observations_base.clone().requires_grad_(True)
    with_memory = _terminal_action_input_gradients(
        model,
        observations,
        references,
    )
    assert with_memory[0].norm().item() > 0.0
    assert with_memory[24].norm().item() > 0.0
    assert with_memory[50].norm().item() > 0.0

    with torch.no_grad():
        model.ttt_memory.gate_logit.zero_()
    control_observations = observations_base.clone().requires_grad_(True)
    without_memory_read = _terminal_action_input_gradients(
        model,
        control_observations,
        references,
    )
    # Two local-attention layers with a four-token window have at most a
    # 1 + 2 * (4 - 1) = 7 token receptive field without the fast weights.
    short_context_receptive_field = 7
    torch.testing.assert_close(
        without_memory_read[:-short_context_receptive_field],
        torch.zeros_like(without_memory_read[:-short_context_receptive_field]),
        atol=0.0,
        rtol=0.0,
    )
    assert without_memory_read[-2].norm().item() > 0.0
