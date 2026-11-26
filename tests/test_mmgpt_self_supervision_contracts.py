import inspect

import pytest
import torch

from rsl_rl.algorithms.mmppo import MMPPO
from rsl_rl.modules.actor_critic_mm_gpt import MMGPT
from rsl_rl.modules.actor_critic_mlp_v2 import FusedMultiModalMLP


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
