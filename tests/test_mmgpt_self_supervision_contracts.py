import inspect

import pytest
import torch

from rsl_rl.algorithms import chunk_q_bootstrap_target
from rsl_rl.algorithms.mmppo import MMPPO
from rsl_rl.modules import PrivilegedChunkQ
from rsl_rl.modules.actor_critic_mm_gpt import (
    ActorCriticMMGPT,
    MMGPT,
)
from rsl_rl.modules.actor_critic_mlp_v2 import (
    DirectConcatMLP,
    FusedMultiModalMLP,
)


_TERM_DIMS = {
    "joint_pos": 2,
    "joint_vel": 2,
    "projected_gravity": 3,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "actions": 2,
}
_PREDICTED_TERMS = [
    "joint_pos",
    "joint_vel",
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
]


def _make_model(
    *,
    aux_backbone_grad=True,
    dreamer_gradient_mode="backbone_only",
    **overrides,
):
    torch.manual_seed(11)
    return MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=4,
        max_seq_len=4,
        term_dict=_TERM_DIMS,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=_PREDICTED_TERMS,
        pred_obs_term_weights=[1.0, 1.0, 0.5, 0.5, 0.5],
        default_joint_pos=torch.zeros(2),
        apply_rope=False,
        aux_backbone_grad=aux_backbone_grad,
        dreamer_gradient_mode=dreamer_gradient_mode,
        **overrides,
    )


def _forward(model):
    torch.manual_seed(17)
    observations = torch.randn(4, 3, sum(_TERM_DIMS.values()))
    reference = torch.randn(4, 3, 2)
    masks = torch.ones(4, 3, dtype=torch.bool)
    output = model(observations, (reference, masks), masks)
    return output


def _grad_norm(parameter):
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm())


def _backbone_parameter(model):
    return model.decoder.layers[0].self_attn.in_proj_weight


def test_policy_action_path_updates_mmgpt_backbone_and_action_head():
    model = _make_model()
    output = _forward(model)

    output.square().mean().backward()

    assert _grad_norm(_backbone_parameter(model)) > 0.0
    assert _grad_norm(model.fc[-1].weight) > 0.0


def test_ppo_action_path_updates_ttt_memory_and_shared_backbone():
    torch.manual_seed(12)
    model = MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        num_steps_per_env=4,
        max_seq_len=4,
        term_dict=_TERM_DIMS,
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
    )
    time_steps = 6
    batch_size = 2
    observations = torch.randn(
        time_steps, batch_size, sum(_TERM_DIMS.values())
    )
    references = torch.randn(time_steps, batch_size, 2)
    masks = torch.ones(time_steps, batch_size, dtype=torch.bool)
    initial_state = model.ttt_memory.initial_state(
        batch_size,
        device=observations.device,
        dtype=observations.dtype,
    ).detach()

    output = model(
        observations,
        (references, masks),
        masks,
        memory={
            "ttt_initial_state": initial_state,
            "ttt_initial_state_reset_mask": torch.ones(
                batch_size, dtype=torch.bool
            ),
        },
        episode_ids=torch.zeros_like(masks, dtype=torch.long),
        sequence_output_start=0,
    )
    output.square().mean().backward()

    assert _grad_norm(
        model.decoder.encoder.layers[0].self_attn.qkv_proj.weight
    ) > 0.0
    assert _grad_norm(model.ttt_memory.qkv_proj.weight) > 0.0
    assert _grad_norm(model.ttt_memory.out_proj.weight) > 0.0
    assert _grad_norm(model.ttt_memory.gate_logit) > 0.0
    assert _grad_norm(model.fc[-1].weight) > 0.0


def test_chunk5_partial_frame_policy_aux_and_dreamer_share_trainable_latent():
    torch.manual_seed(13)
    model = MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        num_steps_per_env=10,
        max_seq_len=10,
        term_dict=_TERM_DIMS,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=_PREDICTED_TERMS,
        pred_obs_term_weights=[1.0, 1.0, 0.5, 0.5, 0.5],
        default_joint_pos=torch.zeros(2),
        apply_rope=True,
        aux_backbone_grad=True,
        dreamer_gradient_mode="backbone_only",
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=10,
        temporal_chunk_size=5,
    )
    time_steps = 9  # ends at phase 3: the current chunk is intentionally open
    batch_size = 2
    observations = torch.randn(
        time_steps, batch_size, sum(_TERM_DIMS.values())
    )
    references = torch.randn(time_steps, batch_size, 2)
    masks = torch.ones(time_steps, batch_size, dtype=torch.bool)
    phases = torch.arange(time_steps).remainder(5).view(-1, 1).expand(-1, 2)
    output = model(
        observations,
        (references, masks),
        masks,
        raw_observations=observations,
        dreamer_target=(
            torch.randn(batch_size, 2),
            torch.ones(batch_size, dtype=torch.bool),
        ),
        episode_ids=torch.zeros_like(masks, dtype=torch.long),
        temporal_phases=phases,
    )

    total_loss = output.square().mean() + model.aux_loss + model.dreamer_loss
    total_loss.backward()

    assert _grad_norm(model.temporal_chunk_encoder.chunk_encoder[0].weight) > 0.0
    assert _grad_norm(model.temporal_chunk_encoder.frame_encoder[0].weight) > 0.0
    assert _grad_norm(
        model.decoder.encoder.layers[0].self_attn.qkv_proj.weight
    ) > 0.0
    assert _grad_norm(model.fc[-1].weight) > 0.0
    assert _grad_norm(model.aux_heads[0].weight) > 0.0


def test_fast_slow_dreamer_control_uses_only_the_non_detached_action_path():
    torch.manual_seed(14)
    model = MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        num_steps_per_env=10,
        max_seq_len=10,
        term_dict=_TERM_DIMS,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=_PREDICTED_TERMS,
        pred_obs_term_weights=[1.0, 1.0, 0.5, 0.5, 0.5],
        default_joint_pos=torch.zeros(2),
        apply_rope=True,
        aux_backbone_grad=True,
        dreamer_gradient_mode="action_path_only",
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=10,
        temporal_chunk_size=5,
        temporal_fast_slow_enabled=True,
    )
    time_steps = 9
    batch_size = 2
    observations = torch.randn(
        time_steps, batch_size, sum(_TERM_DIMS.values())
    )
    references = torch.randn(time_steps, batch_size, 2)
    masks = torch.ones(time_steps, batch_size, dtype=torch.bool)
    phases = torch.arange(time_steps).remainder(5).view(-1, 1).expand(-1, 2)

    model(
        observations,
        (references, masks),
        masks,
        raw_observations=observations,
        dreamer_target=(
            torch.randn(batch_size, 2),
            torch.ones(batch_size, dtype=torch.bool),
        ),
        episode_ids=torch.zeros_like(masks, dtype=torch.long),
        temporal_phases=phases,
    )
    model.dreamer_loss.backward()

    # The frozen dynamics readout transmits control gradients through the
    # resolved policy action into both rates of the actor.
    assert _grad_norm(model.temporal_plan_head[-1].weight) > 0.0
    assert _grad_norm(model.temporal_fast_controller[-1].weight) > 0.0
    assert _grad_norm(
        model.decoder.encoder.layers[0].self_attn.qkv_proj.weight
    ) > 0.0
    # Dreamer control must not refit the supervised dynamics readout itself.
    assert _grad_norm(model.aux_heads[0].weight) == 0.0
    assert _grad_norm(model.action_film.weight) == 0.0


@pytest.mark.parametrize(
    ("aux_backbone_grad", "expect_backbone_grad"),
    [(True, True), (False, False)],
)
def test_future_prediction_backbone_gradient_is_an_explicit_control(
    aux_backbone_grad, expect_backbone_grad
):
    model = _make_model(aux_backbone_grad=aux_backbone_grad)
    _forward(model)

    model.aux_loss.backward()

    assert (_grad_norm(_backbone_parameter(model)) > 0.0) is expect_backbone_grad
    assert _grad_norm(model.aux_heads[0].weight) > 0.0
    assert _grad_norm(model.action_film.weight) > 0.0
    assert _grad_norm(model.fc[-1].weight) == 0.0


def test_prediction_heads_and_weights_follow_config_term_order():
    term_dict = {
        "gravity_dir": 3,
        "base_ang_vel": 3,
        "joint_pos": 2,
        "joint_vel": 2,
        "actions": 2,
    }
    prediction_order = [
        "joint_pos",
        "joint_vel",
        "gravity_dir",
        "base_ang_vel",
    ]
    model = MMGPT(
        obs_size=sum(term_dict.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=4,
        max_seq_len=4,
        term_dict=term_dict,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=prediction_order,
        pred_obs_term_weights=[1.0, 0.8, 0.5, 0.25],
        default_joint_pos=torch.zeros(2),
    )

    assert model.obs_term_names == prediction_order
    assert model.obs_dims == [2, 2, 3, 3]
    assert [head.out_features for head in model.aux_heads] == [2, 2, 3, 3]
    assert model.pred_obs_term_weights == [1.0, 0.8, 0.5, 0.25]


def test_future_prediction_detach_changes_only_autograd_not_loss_values():
    attached = _make_model(aux_backbone_grad=True).eval()
    detached = _make_model(aux_backbone_grad=False).eval()

    attached_output = _forward(attached)
    detached_output = _forward(detached)

    torch.testing.assert_close(attached_output, detached_output)
    torch.testing.assert_close(attached.aux_loss, detached.aux_loss)
    torch.testing.assert_close(attached.dreamer_loss, detached.dreamer_loss)


@pytest.mark.parametrize(
    ("gradient_mode", "expect_action_head_grad"),
    [
        ("backbone_only", False),
        ("backbone_and_action_head", True),
        ("action_path_only", True),
    ],
)
def test_dreamer_gradient_destination_is_an_explicit_control(
    gradient_mode, expect_action_head_grad
):
    model = _make_model(dreamer_gradient_mode=gradient_mode)
    _forward(model)

    model.dreamer_loss.backward()

    assert _grad_norm(_backbone_parameter(model)) > 0.0
    assert (_grad_norm(model.fc[-1].weight) > 0.0) is expect_action_head_grad
    # The prediction decoder is a fixed differentiable readout for this loss.
    assert _grad_norm(model.aux_heads[0].weight) == 0.0
    assert _grad_norm(model.action_film.weight) == 0.0


def test_dreamer_action_detach_changes_only_autograd_not_loss_values():
    detached = _make_model(dreamer_gradient_mode="backbone_only").eval()
    attached = _make_model(
        dreamer_gradient_mode="backbone_and_action_head"
    ).eval()

    detached_output = _forward(detached)
    attached_output = _forward(attached)

    torch.testing.assert_close(detached_output, attached_output)
    torch.testing.assert_close(detached.aux_loss, attached.aux_loss)
    torch.testing.assert_close(detached.dreamer_loss, attached.dreamer_loss)


@pytest.mark.parametrize(
    ("gradient_mode", "expect_direct_latent_grad"),
    [
        ("backbone_and_action_head", True),
        ("action_path_only", False),
    ],
)
def test_action_path_only_blocks_the_direct_latent_shortcut(
    gradient_mode, expect_direct_latent_grad
):
    model = _make_model(dreamer_gradient_mode=gradient_mode)
    # Remove the action-to-backbone Jacobian without removing the action-head
    # parameter Jacobian. Any remaining backbone gradient must then have come
    # from the predictor's direct latent input.
    with torch.no_grad():
        model.fc[-1].weight.zero_()
    _forward(model)

    model.dreamer_loss.backward()

    has_backbone_grad = _grad_norm(_backbone_parameter(model)) > 0.0
    assert has_backbone_grad is expect_direct_latent_grad
    assert _grad_norm(model.fc[-1].weight) > 0.0
    assert _grad_norm(model.aux_heads[0].weight) == 0.0
    assert _grad_norm(model.action_film.weight) == 0.0


def test_dreamer_gradient_mode_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="dreamer_gradient_mode"):
        _make_model(dreamer_gradient_mode="action_head_only")


class _RecordingMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.targets = []

    def forward(self, prediction, target):
        self.targets.append(target.detach().clone())
        return torch.nn.functional.mse_loss(prediction, target)


def test_future_prediction_uses_t_plus_one_targets_without_crossing_reset():
    torch.manual_seed(23)
    model = MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=4,
        max_seq_len=4,
        term_dict=_TERM_DIMS,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=["joint_pos"],
        pred_obs_term_weights=[1.0],
        default_joint_pos=torch.zeros(2),
        apply_rope=False,
        aux_backbone_grad=True,
    ).eval()
    recorder = _RecordingMSE()
    model.aux_loss_fn = recorder

    observations = torch.randn(4, 2, sum(_TERM_DIMS.values()))
    joint_positions = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 1.5]],
            [[10.0, 10.5], [11.0, 11.5]],
            [[20.0, 20.5], [21.0, 21.5]],
            [[30.0, 30.5], [31.0, 31.5]],
        ]
    )
    observations[..., :2] = joint_positions
    masks = torch.tensor(
        [
            [False, False],
            [False, True],
            [True, True],
            [True, True],
        ]
    )
    reference = torch.randn(4, 2, 2)

    model._forward_batch(
        observations,
        (reference, masks),
        masks,
        unpad_output=False,
        compute_aux_loss=True,
        compute_dreamer_loss=False,
    )

    # Valid pairs are (t=1 -> 2, env=1), then both t=2 -> 3 pairs.
    expected = torch.stack(
        (joint_positions[2, 1], joint_positions[3, 0], joint_positions[3, 1])
    )
    assert len(recorder.targets) == 1
    torch.testing.assert_close(recorder.targets[0], expected)


def test_future_prediction_uses_raw_state_and_resolved_action_coordinates():
    model = _make_model().eval()
    recorder = _RecordingMSE()
    model.aux_loss_fn = recorder
    film_inputs = []

    def record_film_input(_module, args):
        film_inputs.append(args[0].detach().clone())

    handle = model.action_film.register_forward_pre_hook(record_film_input)
    normalized = torch.randn(4, 2, sum(_TERM_DIMS.values()))
    raw = normalized + 100.0
    masks = torch.ones(4, 2, dtype=torch.bool)
    reference = torch.randn(4, 2, 2)
    try:
        model._forward_batch(
            normalized,
            (reference, masks),
            masks,
            raw_observations=raw,
            compute_aux_loss=True,
            compute_dreamer_loss=False,
            unpad_output=False,
        )
    finally:
        handle.remove()

    action_start = sum(
        dim for name, dim in _TERM_DIMS.items() if name != "actions"
    )
    torch.testing.assert_close(
        film_inputs[0],
        raw[1:, :, action_start:action_start + _TERM_DIMS["actions"]],
    )
    torch.testing.assert_close(
        recorder.targets[0],
        raw[1:, :, :_TERM_DIMS["joint_pos"]].reshape(-1, 2),
    )


@pytest.mark.parametrize(
    ("is_delta", "clip_value", "expected"),
    [
        (False, 1.0, [[0.5, 0.0]]),
        (True, 1.0, [[1.0, -0.75]]),
        (True, None, [[1.25, -0.75]]),
    ],
)
def test_dreamer_action_resolution_matches_environment_contract(
    is_delta, clip_value, expected
):
    model = _make_model(
        action_is_delta=is_delta,
        action_clip_value=clip_value,
    )
    previous = torch.tensor([[0.75, -0.75]])
    policy = torch.tensor([[0.5, float("nan")]])

    resolved = model._resolve_prediction_action(previous, policy)

    torch.testing.assert_close(resolved, torch.tensor(expected))


def test_external_adjacent_target_replaces_strided_reference_command():
    model = MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=8,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=4,
        max_seq_len=4,
        term_dict=_TERM_DIMS,
        ref_term_dict={"sonic_release_command": 8},
        pred_obs_term_names=["joint_pos"],
        pred_obs_term_weights=[1.0],
        default_joint_pos=torch.zeros(2),
        dreamer_target_term_name="target_actions",
    ).eval()
    assert model.target_action_slice is None
    recorder = _RecordingMSE()
    model.aux_loss_fn = recorder
    normalized = torch.randn(4, 3, sum(_TERM_DIMS.values()))
    raw = normalized + 10.0
    references = torch.randn(4, 3, 8)
    masks = torch.ones(4, 3, dtype=torch.bool)
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target_mask = torch.tensor([True, False, True])

    model._forward_batch(
        normalized,
        (references, masks),
        masks,
        raw_observations=raw,
        dreamer_target=(target, target_mask),
        compute_aux_loss=False,
        compute_dreamer_loss=True,
        unpad_output=False,
    )

    assert len(recorder.targets) == 1
    torch.testing.assert_close(recorder.targets[0], target[target_mask])


def test_dreamer_mask_reduction_is_not_diluted_by_invalid_environments():
    model = _make_model().eval()
    torch.manual_seed(29)
    observations = torch.randn(4, 3, sum(_TERM_DIMS.values()))
    references = torch.randn(4, 3, 2)
    observation_masks = torch.ones(4, 3, dtype=torch.bool)
    reference_masks = torch.zeros(4, 3, dtype=torch.bool)
    reference_masks[:, 0] = True

    model(
        observations,
        (references, reference_masks),
        observation_masks,
    )
    batched_loss = model.dreamer_loss.detach().clone()

    model(
        observations[:, :1],
        (references[:, :1], reference_masks[:, :1]),
        observation_masks[:, :1],
    )
    valid_only_loss = model.dreamer_loss.detach().clone()

    torch.testing.assert_close(batched_loss, valid_only_loss)


def test_score_prediction_stays_disabled_during_cold_start_ablation():
    model = _make_model()
    _forward(model)

    assert model.score_prediction_enabled is False
    assert model.dreamer_reward == 0.0


def test_mmppo_scales_and_averages_both_mmgpt_losses():
    source = inspect.getsource(MMPPO.update)

    assert "if self.aux_loss_coef > 0.0:" in source
    assert "weighted_aux_loss = self.aux_loss_coef * aux_loss" in source
    assert "if self.dreamer_loss_coef > 0.0:" in source
    assert "weighted_dreamer_loss = self.dreamer_loss_coef * dreamer_loss" in source
    assert "mean_aux_loss /= num_updates" in source
    assert "mean_dreamer_loss /= num_updates" in source


def test_a_disabled_auxiliary_loss_leaves_prediction_heads_out_of_adamw():
    model = _make_model(dreamer_gradient_mode="action_path_only")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(("aux_heads.", "action_film."))
    }
    output = _forward(model)

    # This is the gradient topology of PPO with both auxiliary coefficients
    # disabled: prediction losses may be measured, but are not added to loss.
    output.square().mean().backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name in frozen_before
    )
    optimizer.step()

    for name, parameter in model.named_parameters():
        if name in frozen_before:
            assert torch.equal(parameter.detach(), frozen_before[name])


def test_single_frame_privileged_critic_has_scalar_value_per_environment():
    critic = FusedMultiModalMLP(
        term_dict={"critic": {"privileged_state": 5}},
        ref_term_dict={"critic": {"privileged_reference": 3}},
        output_size=1,
        hidden_dims=[8],
        activation="elu",
        history_length=1,
        encoder_latent_dim=4,
        encoder_compress_threshold=32,
        use_layer_norm=False,
        fusion_mode="gated",
        reorgnize_obs=True,
    )
    observations = torch.randn(4, 5)
    references = torch.randn(4, 3)
    reference_mask = torch.ones(4, dtype=torch.bool)

    inference_values = critic(
        observations,
        (references, reference_mask),
    )
    critic.reset()
    batch_values = critic(
        observations.unsqueeze(0),
        (references.unsqueeze(0), reference_mask.unsqueeze(0)),
    )

    assert inference_values.shape == (4, 1)
    assert batch_values.shape == (4, 1)


def test_chunk_q_bootstrap_stops_after_terminal_transition():
    rewards = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [4.0, 4.0],
            [8.0, 8.0],
            [16.0, 16.0],
        ]
    )
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    dones[1, 0] = True
    bootstrap = torch.tensor([100.0, 100.0])

    target = chunk_q_bootstrap_target(rewards, dones, bootstrap, gamma=0.5)

    expected_terminal = 1.0 + 0.5 * 2.0
    expected_continuing = (
        1.0
        + 0.5 * 2.0
        + 0.25 * 4.0
        + 0.125 * 8.0
        + 0.0625 * 16.0
        + 0.03125 * 100.0
    )
    torch.testing.assert_close(
        target, torch.tensor([expected_terminal, expected_continuing])
    )


def test_privileged_chunk_q_masks_reference_and_has_scalar_output():
    torch.manual_seed(31)
    chunk_q = PrivilegedChunkQ(
        critic_obs_dim=5,
        critic_ref_obs_dim=3,
        action_dim=2,
        chunk_size=5,
        hidden_dims=[16, 8],
        reference_first=True,
    )
    states = torch.randn(4, 5)
    action_chunks = torch.randn(4, 5, 2)
    references = torch.randn(4, 3)
    reference_mask = torch.tensor([True, False, True, False])

    baseline = chunk_q(
        states,
        action_chunks,
        (references, reference_mask),
    )
    changed_reference = references.clone()
    changed_reference[~reference_mask] += 1000.0
    masked_change = chunk_q(
        states,
        action_chunks,
        (changed_reference, reference_mask),
    )
    torch.testing.assert_close(baseline, masked_change)
    assert baseline.shape == (4, 1)

    baseline.square().mean().backward()
    assert any(parameter.grad is not None for parameter in chunk_q.parameters())


def test_sonic_command_can_supply_the_dreamer_action_target():
    model = MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=643,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=4,
        max_seq_len=4,
        term_dict=_TERM_DIMS,
        ref_term_dict={"prefix": 3, "sonic_release_command": 640},
        pred_obs_term_names=_PREDICTED_TERMS,
        pred_obs_term_weights=[1.0, 1.0, 0.5, 0.5, 0.5],
        default_joint_pos=torch.zeros(2),
        dreamer_target_term_name="sonic_release_command",
        dreamer_target_action_offset=0,
    )

    # Reference terms are concatenated in manager order. The first future q
    # sample starts at the beginning of the SONIC command term.
    assert model.target_action_slice == (3, 5)


def test_sonic_alignment_keeps_only_the_current_reference_command():
    model = _make_model()
    model.latest_reference_only = True
    reference = torch.randn(4, 3, 2)
    reference_mask = torch.ones(4, 3, dtype=torch.bool)

    aligned_reference, aligned_mask = model._apply_reference_history_contract(
        (reference, reference_mask)
    )

    assert not aligned_mask[:-1].any()
    assert aligned_mask[-1].all()
    assert torch.count_nonzero(aligned_reference[:-1]) == 0
    torch.testing.assert_close(aligned_reference[-1], reference[-1])

    # The first inference call has nine padded proprio frames and nine masked
    # reference tokens. It must remain numerically valid under causal RoPE.
    first_action = model(
        torch.randn(3, sum(_TERM_DIMS.values())),
        (torch.randn(3, 2), torch.ones(3, dtype=torch.bool)),
    )
    assert torch.isfinite(first_action).all()


def _make_actor_critic_with_sonic_critic(**overrides):
    kwargs = dict(
        num_actor_obs=12,
        num_actor_ref_obs=8,
        num_critic_obs=5,
        num_critic_ref_obs=3,
        num_actions=2,
        term_dict={
            "policy": {
                "joint_pos": 2,
                "joint_vel": 2,
                "gravity_dir": 3,
                "base_ang_vel": 3,
                "actions": 2,
            },
            "critic": {"privileged_state": 5},
        },
        ref_term_dict={
            "policy": {"sonic_release_command": 8},
            "critic": {"privileged_reference": 3},
        },
        ref_term_steps={
            "policy": {"sonic_release_command": 1},
            "critic": {},
        },
        concatenate_term_names={"policy": [], "critic": []},
        concatenate_ref_term_names={"policy": [], "critic": []},
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=4,
        max_seq_len=4,
        init_noise_std=0.8,
        min_noise_std=0.01,
        max_noise_std=0.5,
        critic_history_length=1,
        critic_hidden_dims=[32, 16],
        activation="silu",
        critic_fusion_mode="direct_concat_ref_first",
        default_joint_pos=torch.zeros(2),
    )
    kwargs.update(overrides)
    return ActorCriticMMGPT(**kwargs)


def test_chunk_q_calibration_gradient_is_separate_from_actor_and_value_critic():
    model = _make_actor_critic_with_sonic_critic(
        num_steps_per_env=5,
        max_seq_len=5,
        apply_rope=True,
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=5,
        temporal_chunk_size=5,
        chunk_q_enabled=True,
        chunk_q_hidden_dims=[16, 8],
    )
    chunk_values = model.evaluate_chunk_q(
        torch.randn(4, 5),
        torch.randn(4, 5, 2),
        (torch.randn(4, 3), torch.ones(4, dtype=torch.bool)),
    )

    chunk_values.square().mean().backward()

    assert any(parameter.grad is not None for parameter in model.chunk_q.parameters())
    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert all(parameter.grad is None for parameter in model.critic.parameters())


def test_mmgpt_uses_sonic_direct_privileged_critic_for_inference_and_batches():
    model = _make_actor_critic_with_sonic_critic()

    assert isinstance(model.critic, DirectConcatMLP)
    assert model.critic.reference_first is True
    assert model.critic.fusion_mode == "direct_concat_ref_first"
    assert [
        layer.out_features
        for layer in model.critic.network
        if isinstance(layer, torch.nn.Linear)
    ] == [32, 16, 1]

    inference_values = model.evaluate(
        torch.randn(4, 5),
        (torch.randn(4, 3), torch.ones(4, dtype=torch.bool)),
    )
    batch_values = model.evaluate(
        torch.randn(4, 3, 5),
        (
            torch.randn(4, 3, 3),
            torch.ones(4, 3, dtype=torch.bool),
        ),
        masks=torch.ones(4, 3, dtype=torch.bool),
    )

    assert inference_values.shape == (4, 1)
    assert batch_values.shape == (3, 1)


def test_mmgpt_applies_sonic_action_noise_bounds():
    model = _make_actor_critic_with_sonic_critic()
    with torch.no_grad():
        model.std.fill_(2.0)

    model.update_distribution(
        torch.randn(3, 12),
        (torch.randn(3, 8), torch.ones(3, dtype=torch.bool)),
    )

    assert torch.all(model.action_std == 0.5)
