import inspect

import pytest
import torch

from rsl_rl.algorithms.mmppo import MMPPO
from rsl_rl.modules.actor_critic_mm_gpt import ActorCriticMMGPT, MMGPT
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


def _make_model(*, aux_backbone_grad=True, dreamer_gradient_mode="backbone_only"):
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
