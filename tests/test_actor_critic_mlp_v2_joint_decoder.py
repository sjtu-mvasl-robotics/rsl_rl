from __future__ import annotations

import torch

from rsl_rl.modules.actor_critic_mlp_v2 import ActorCriticMLPV2, FusedMultiModalMLP


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
