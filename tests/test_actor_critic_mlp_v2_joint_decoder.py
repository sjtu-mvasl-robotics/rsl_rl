from __future__ import annotations

import inspect

import torch
from torch import nn

from rsl_rl.modules.actor_critic_mlp_v2 import (
    ActorCriticMLPV2,
    DirectConcatMLP,
    FusedMultiModalMLP,
)
from rsl_rl.algorithms.mmppo import MMPPO


def _make_model(fusion_mode: str) -> FusedMultiModalMLP:
    return FusedMultiModalMLP(
        term_dict={"policy": {"state": 5}},
        ref_term_dict={"policy": {"target": 7}},
        output_size=3,
        hidden_dims=[16, 8],
        activation="silu",
        history_length=1,
        encoder_latent_dim=4,
        encoder_compress_threshold=32,
        use_layer_norm=False,
        fusion_mode=fusion_mode,
        joint_decoder_hidden_dims=[16, 8],
    )


def test_joint_decoder_conditions_on_state_and_reference_features():
    model = _make_model("joint_decoder")
    observations = torch.randn(4, 5, requires_grad=True)
    references = torch.randn(4, 7, requires_grad=True)
    mask = torch.ones(4, dtype=torch.bool)

    output = model(observations, (references, mask))
    output.sum().backward()

    assert output.shape == (4, 3)
    assert observations.grad is not None
    assert observations.grad.abs().sum() > 0
    assert references.grad is not None
    assert references.grad.abs().sum() > 0


def test_sonic_joint_decoder_uses_raw_state_and_encoded_reference():
    model = _make_model("sonic_joint_decoder")
    first_linear = model.joint_decoder[0]
    assert first_linear.in_features == 5 + 8

    observations = torch.randn(4, 5, requires_grad=True)
    references = torch.randn(4, 7, requires_grad=True)
    output = model(
        observations,
        (references, torch.ones(4, dtype=torch.bool)),
    )
    output.sum().backward()

    assert observations.grad is not None
    assert observations.grad.abs().sum() > 0
    assert references.grad is not None
    assert references.grad.abs().sum() > 0


def test_invalid_reference_is_exact_main_branch_passthrough():
    observations = torch.randn(4, 5)
    references = torch.randn(4, 7)
    invalid_mask = torch.zeros(4, dtype=torch.bool)

    for fusion_mode in (
        "gated",
        "add",
        "concat",
        "joint_decoder",
        "sonic_joint_decoder",
    ):
        model = _make_model(fusion_mode)
        output = model(observations, (references, invalid_mask))
        expected = model.main_net(observations)
        torch.testing.assert_close(output, expected)


def test_mixed_reference_mask_preserves_each_invalid_main_row():
    observations = torch.randn(4, 5)
    references = torch.randn(4, 7)
    mask = torch.tensor([True, False, True, False])

    for fusion_mode in ("joint_decoder", "sonic_joint_decoder"):
        model = _make_model(fusion_mode)
        model.fuse_activation = True
        output = model(observations, (references, mask))
        expected = model.main_net(observations)
        torch.testing.assert_close(output[~mask], expected[~mask])


def test_sonic_decoder_gradients_reach_ref_encoder_and_joint_decoder():
    model = _make_model("sonic_joint_decoder")
    output = model(
        torch.randn(4, 5),
        (torch.randn(4, 7), torch.ones(4, dtype=torch.bool)),
    )
    output.square().mean().backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.ref_net.main_mlp.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.joint_decoder.parameters()
    )


def test_action_noise_matches_sonic_clamp_contract():
    model = ActorCriticMLPV2(
        term_dict={"policy": {"state": 5}, "critic": {"state": 5}},
        ref_term_dict={"policy": {}, "critic": {}},
        num_actions=3,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        init_noise_std=0.05,
        min_noise_std=0.001,
        max_noise_std=0.5,
    )
    with torch.no_grad():
        model.std.copy_(torch.tensor([-1.0, 0.05, 2.0]))

    model.update_distribution(torch.randn(4, 5))
    expected = torch.tensor([0.001, 0.05, 0.5]).expand(4, 3)
    torch.testing.assert_close(model.action_std, expected)


def test_direct_concat_critic_has_one_mlp_and_respects_reference_mask():
    critic = DirectConcatMLP(
        term_dict={"critic": {"state": 5}},
        ref_term_dict={"critic": {"target": 7}},
        output_size=1,
        hidden_dims=[16, 8],
        activation="silu",
        use_layer_norm=False,
    )
    assert critic.network[0].in_features == 12
    observations = torch.randn(4, 5, requires_grad=True)
    references = torch.randn(4, 7, requires_grad=True)
    mask = torch.tensor([True, False, True, False])
    output = critic(observations, (references, mask))
    output.sum().backward()
    assert output.shape == (4, 1)
    assert observations.grad is not None
    assert references.grad is not None
    torch.testing.assert_close(references.grad[~mask], torch.zeros_like(references.grad[~mask]))


def test_sonic_direct_concat_critic_can_preserve_reference_first_order():
    critic = DirectConcatMLP(
        term_dict={"critic": {"state": 2}},
        ref_term_dict={"critic": {"reference": 2}},
        output_size=1,
        hidden_dims=[],
        activation="silu",
        use_layer_norm=False,
        reference_first=True,
    )
    critic.network = nn.Identity()

    state = torch.tensor([[1.0, 2.0]])
    reference = torch.tensor([[3.0, 4.0]])
    output = critic(state, (reference, torch.tensor([True])))

    torch.testing.assert_close(output, torch.tensor([[3.0, 4.0, 1.0, 2.0]]))
    assert critic.fusion_mode == "direct_concat_ref_first"


def test_actor_critic_can_use_sonic_actor_and_direct_critic_topologies():
    model = ActorCriticMLPV2(
        term_dict={"policy": {"state": 5}, "critic": {"state": 9}},
        ref_term_dict={"policy": {"target": 7}, "critic": {"target": 11}},
        num_actions=3,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[32, 16, 8],
        joint_decoder_hidden_dims=[16, 8],
        fusion_mode="sonic_joint_decoder",
        critic_fusion_mode="direct_concat",
        activation="silu",
    )
    assert model.actor.fusion_mode == "sonic_joint_decoder"
    assert model.critic.fusion_mode == "direct_concat"
    assert model.critic.network[0].in_features == 20


def test_mmppo_preserves_separate_sonic_actor_and_critic_learning_rates():
    model = ActorCriticMLPV2(
        term_dict={"policy": {"state": 5}, "critic": {"state": 9}},
        ref_term_dict={"policy": {"target": 7}, "critic": {"target": 11}},
        num_actions=3,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[32, 16, 8],
        joint_decoder_hidden_dims=[16, 8],
        fusion_mode="sonic_joint_decoder",
        critic_fusion_mode="direct_concat",
        activation="silu",
    )
    algorithm = MMPPO(
        model,
        learning_rate=2.0e-5,
        actor_learning_rate=2.0e-5,
        critic_learning_rate=1.0e-3,
        min_lr=1.0e-5,
        max_lr=2.0e-4,
        desired_kl=0.01,
        schedule="adaptive",
    )
    learning_rates = {
        group.get("name"): group["lr"]
        for group in algorithm.optimizer.param_groups
    }
    assert learning_rates == {"actor": 2.0e-5, "critic": 1.0e-3}

    algorithm.learning_rate = 3.0e-5
    algorithm._apply_adaptive_learning_rates()
    learning_rates = {
        group.get("name"): group["lr"]
        for group in algorithm.optimizer.param_groups
    }
    assert learning_rates == {"actor": 3.0e-5, "critic": 1.0e-3}


def test_mmppo_entropy_loss_updates_trainable_sonic_std():
    model = ActorCriticMLPV2(
        term_dict={"policy": {"state": 2}, "critic": {"state": 2}},
        ref_term_dict={
            "policy": {"command": 2},
            "critic": {"command": 2},
        },
        num_actions=2,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        activation="silu",
        init_noise_std=0.05,
        min_noise_std=0.001,
        max_noise_std=0.5,
        history_length=1,
        fusion_mode="concat",
        critic_fusion_mode="direct_concat_ref_first",
    )
    algorithm = MMPPO(
        model,
        entropy_coef=1.0e-2,
        learning_rate=1.0e-2,
        actor_learning_rate=1.0e-2,
        critic_learning_rate=1.0e-2,
        min_lr=1.0e-5,
        max_lr=1.0,
        fix_sigma=False,
    )

    optimized_parameters = [
        parameter
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert any(parameter is model.std for parameter in optimized_parameters)
    assert algorithm.entropy_coef == 1.0e-2
    assert "- self.entropy_coef * entropy_batch.mean()" in inspect.getsource(
        MMPPO.update
    )

    state = torch.zeros(8, 2)
    reference = torch.zeros(8, 2)
    model.update_distribution(
        state, (reference, torch.ones(8, dtype=torch.bool))
    )
    before = model.std.detach().clone()
    entropy_loss = -algorithm.entropy_coef * model.entropy.mean()
    algorithm.optimizer.zero_grad()
    entropy_loss.backward()

    assert model.std.grad is not None
    assert torch.all(model.std.grad < 0.0)
    algorithm.optimizer.step()
    assert torch.all(model.std.detach() > before)
