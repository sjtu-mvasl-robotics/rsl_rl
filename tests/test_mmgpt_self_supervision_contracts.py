import inspect

import pytest
import torch

from rsl_rl.algorithms.mmppo import MMPPO
from rsl_rl.modules.actor_critic_mm_gpt import MMGPT


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

    assert "weighted_aux_loss = self.aux_loss_coef * aux_loss" in source
    assert "weighted_dreamer_loss = self.dreamer_loss_coef * dreamer_loss" in source
    assert "mean_aux_loss /= num_updates" in source
    assert "mean_dreamer_loss /= num_updates" in source
