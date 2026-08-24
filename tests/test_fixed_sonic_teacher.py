from collections import OrderedDict
import os
from pathlib import Path

import pytest
import torch

from rsl_rl.algorithms.mmppo import (
    MMPPO,
    _fixed_expert_actor_ppo_enabled,
    _fixed_expert_bc_loss,
    _fixed_expert_stage,
)
from rsl_rl.modules.fixed_sonic_teacher import (
    FixedSonicMLPTeacher,
    SonicHistoryAssembler,
    convert_teacher_action_coordinates,
)
from rsl_rl.modules import ActorCriticMMGPT
from rsl_rl.storage import RolloutStorageMM


def test_sonic_history_is_term_major_and_repeats_first_reset_frame():
    history = SonicHistoryAssembler(
        OrderedDict((('gravity', 1), ('actions', 2))),
        history_length=3,
    )

    first = history.append(torch.tensor([[1.0, 10.0, 11.0]]))
    second = history.append(torch.tensor([[2.0, 20.0, 21.0]]))

    torch.testing.assert_close(
        first,
        torch.tensor([[1.0, 1.0, 1.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0]]),
    )
    torch.testing.assert_close(
        second,
        torch.tensor([[1.0, 1.0, 2.0, 10.0, 11.0, 10.0, 11.0, 20.0, 21.0]]),
    )


def test_sonic_history_reset_is_per_environment_and_has_no_leak():
    history = SonicHistoryAssembler(
        OrderedDict((('state', 1), ('actions', 1))),
        history_length=2,
    )
    history.append(torch.tensor([[1.0, 11.0], [2.0, 12.0]]))
    history.append(torch.tensor([[3.0, 13.0], [4.0, 14.0]]))
    history.reset(torch.tensor([True, False]))
    output = history.append(torch.tensor([[5.0, 15.0], [6.0, 16.0]]))

    torch.testing.assert_close(output[0], torch.tensor([5.0, 5.0, 15.0, 15.0]))
    torch.testing.assert_close(output[1], torch.tensor([4.0, 6.0, 14.0, 16.0]))


def test_sonic_history_allocated_in_rollout_inference_mode_remains_resettable():
    history = SonicHistoryAssembler(
        OrderedDict((('state', 1), ('actions', 1))),
        history_length=2,
    )
    with torch.inference_mode():
        history.append(torch.ones(2, 2))
    assert history.buffer is not None
    assert not history.buffer.is_inference()
    history.reset(torch.tensor([True, False]))


@pytest.mark.parametrize(
    ("teacher_is_delta", "student_is_delta", "expected"),
    [
        (False, False, (1.0, -2.0)),
        (False, True, (0.5, -1.0)),
        (True, False, (1.5, -1.0)),
        (True, True, (1.0, 0.0)),
    ],
)
def test_teacher_action_coordinate_conversion(
    teacher_is_delta, student_is_delta, expected
):
    converted = convert_teacher_action_coordinates(
        torch.tensor([[1.0, float("nan") if teacher_is_delta else -2.0]]),
        torch.tensor([[0.5, -1.0]]),
        teacher_is_delta=teacher_is_delta,
        student_is_delta=student_is_delta,
        clip_value=2.0,
    )
    torch.testing.assert_close(converted, torch.tensor([expected]))


def test_fixed_expert_schedule_never_enables_ppo_before_student_only_rl():
    common = {
        "bc_only_iterations": 2,
        "dagger_plateau_iterations": (3, 4),
        "dagger_student_probs": (0.1, 0.5),
    }
    assert _fixed_expert_stage(iteration=0, **common) == ("bc", 0.0, -1)
    assert _fixed_expert_stage(iteration=2, **common) == ("dagger", 0.1, 0)
    assert _fixed_expert_stage(iteration=5, **common) == ("dagger", 0.5, 1)
    assert _fixed_expert_stage(iteration=9, **common) == ("rl", 1.0, -1)

    assert not _fixed_expert_actor_ppo_enabled(
        stage="bc", require_critic_ready=False, critic_win_streak=99
    )
    assert not _fixed_expert_actor_ppo_enabled(
        stage="dagger", require_critic_ready=False, critic_win_streak=99
    )
    assert not _fixed_expert_actor_ppo_enabled(
        stage="rl", require_critic_ready=True, critic_win_streak=2
    )
    assert _fixed_expert_actor_ppo_enabled(
        stage="rl", require_critic_ready=True, critic_win_streak=3
    )


def test_fixed_expert_bc_target_is_detached_but_student_receives_gradient():
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    expert = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = _fixed_expert_bc_loss(student, expert, mode="mse")
    loss.backward()

    torch.testing.assert_close(student.grad, torch.tensor([[1.0, 2.0]]))
    assert expert.grad is None


def test_dagger_intervention_is_one_step_then_fixed_expert_recovers():
    algorithm = MMPPO.__new__(MMPPO)
    algorithm.fixed_expert_stage = "dagger"
    algorithm.fixed_expert_student_prob = 1.0
    algorithm.expert_recovery_steps = 2
    algorithm._expert_recovery_remaining = None
    algorithm._fixed_expert_student_execute_count = 0
    algorithm._fixed_expert_total_execute_count = 0
    student = torch.ones(3, 2)
    expert = torch.zeros(3, 2)

    first = algorithm._select_fixed_expert_actions(student, expert)
    second = algorithm._select_fixed_expert_actions(student, expert)
    third = algorithm._select_fixed_expert_actions(student, expert)
    fourth = algorithm._select_fixed_expert_actions(student, expert)

    torch.testing.assert_close(first, student)
    torch.testing.assert_close(second, expert)
    torch.testing.assert_close(third, expert)
    torch.testing.assert_close(fourth, student)


def test_streaming_sequence_batch_retains_fixed_expert_labels():
    storage = RolloutStorageMM(
        training_type="rl",
        num_envs=2,
        num_transitions_per_env=3,
        obs_shape=(1,),
        ref_obs_shape=(1,),
        privileged_obs_shape=(1,),
        privileged_ref_obs_shape=(1,),
        actions_shape=(1,),
        apply_dagger_actions=True,
        device="cpu",
        actor_history_length=2,
        critic_history_length=1,
    )
    for step in range(3):
        transition = RolloutStorageMM.Transition()
        current = torch.tensor([[float(step)], [float(step + 10)]])
        transition.observations = current
        transition.reference_observations = current
        transition.reference_observations_mask = torch.ones(2, dtype=torch.bool)
        transition.critic_observations = current
        transition.critic_reference_observations = current
        transition.critic_reference_observations_mask = torch.ones(
            2, dtype=torch.bool
        )
        transition.actions = torch.zeros(2, 1)
        transition.dagger_actions = current + 100.0
        transition.rewards = torch.zeros(2)
        transition.dones = torch.zeros(2, dtype=torch.bool)
        transition.values = torch.zeros(2, 1)
        transition.actions_log_prob = torch.zeros(2)
        transition.action_mean = torch.zeros(2, 1)
        transition.action_sigma = torch.ones(2, 1)
        storage.add_transitions(transition)

    batch = next(storage.streaming_sequence_mini_batch_generator(1, 1))
    labels = batch[11]
    assert labels is not None
    assert sorted(labels.squeeze(-1).tolist()) == [100.0, 101.0, 102.0, 110.0, 111.0, 112.0]


def test_real_sonic_checkpoint_crosses_bc_to_dagger_with_frozen_labels():
    """Exercise the complete fixed-expert path with a real SONIC checkpoint.

    This test is opt-in so the portable unit-test suite does not depend on a
    user-specific checkpoint.  CI or a local contract check enables it through
    ``RSL_RL_FIXED_SONIC_TEST_CHECKPOINT``.
    """

    checkpoint_value = os.environ.get("RSL_RL_FIXED_SONIC_TEST_CHECKPOINT")
    if checkpoint_value is None:
        pytest.skip("real SONIC checkpoint integration test was not requested")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        pytest.fail(f"SONIC checkpoint does not exist: {checkpoint}")

    torch.manual_seed(20260824)
    policy_terms = OrderedDict(
        (
            ("gravity_dir", 3),
            ("base_ang_vel", 3),
            ("joint_pos", 29),
            ("joint_vel", 29),
            ("actions", 29),
        )
    )
    reference_terms = OrderedDict((("sonic_release_command", 640),))
    observation_dim = sum(policy_terms.values())
    reference_dim = sum(reference_terms.values())
    num_actions = 29
    num_envs = 2
    rollout_steps = 2

    teacher = FixedSonicMLPTeacher(
        checkpoint_path=str(checkpoint),
        policy_term_dims=policy_terms,
        policy_ref_term_dims=reference_terms,
        num_actions=num_actions,
        history_length=10,
        teacher_is_delta=False,
        student_is_delta=False,
        action_clip_value=20.0,
    )
    actor_critic = ActorCriticMMGPT(
        num_actor_obs=observation_dim,
        num_actor_ref_obs=reference_dim,
        num_critic_obs=observation_dim,
        num_critic_ref_obs=reference_dim,
        num_actions=num_actions,
        term_dict={"policy": policy_terms, "critic": policy_terms},
        ref_term_dict={"policy": reference_terms, "critic": reference_terms},
        ref_term_steps={"policy": None, "critic": None},
        concatenate_term_names={"policy": None, "critic": None},
        concatenate_ref_term_names={"policy": None, "critic": None},
        pred_obs_term_names=None,
        pred_obs_term_weights=None,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        num_steps_per_env=rollout_steps,
        max_seq_len=rollout_steps,
        critic_history_length=1,
        critic_hidden_dims=[32, 16],
        critic_fusion_mode="direct_concat",
        apply_rope=True,
        latest_reference_only=False,
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=rollout_steps,
        ttt_memory_enabled=True,
        ttt_num_heads=2,
        default_joint_pos=torch.zeros(num_actions),
    )
    algorithm = MMPPO(
        actor_critic,
        fixed_teacher=teacher,
        expert_bc_loss_coef=1.0,
        expert_bc_only_iterations=1,
        expert_dagger_plateau_iterations=[1],
        expert_dagger_student_probs=[1.0],
        expert_recovery_steps=1,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        schedule="fixed",
        teacher_coef=None,
        teacher_loss_coef=None,
        normalize_advantage_per_mini_batch=True,
        aux_loss_coef=0.0,
        dreamer_loss_coef=0.0,
        device="cpu",
    )
    algorithm.init_storage(
        "rl",
        num_envs=num_envs,
        num_transitions_per_env=rollout_steps,
        actor_obs_shape=(observation_dim,),
        actor_ref_obs_shape=(reference_dim,),
        critic_obs_shape=(observation_dim,),
        critic_ref_obs_shape=(reference_dim,),
        action_shape=(num_actions,),
        store_raw_observations=True,
    )

    def make_inputs(step: int):
        observations = torch.zeros(num_envs, observation_dim)
        observations[:, 2] = -1.0
        observations[:, 6:35] = 0.01 * step
        observations[:, 35:64] = 0.005 * step
        references = torch.linspace(-0.2, 0.2, reference_dim).repeat(num_envs, 1)
        references = references + 0.001 * step
        return observations, (references, torch.ones(num_envs, dtype=torch.bool))

    def collect_step(step: int):
        observations, references = make_inputs(step)
        executed = algorithm.act(
            observations,
            references,
            observations,
            references,
            raw_obs=observations,
            raw_ref_obs=references,
        )
        student = algorithm.transition.actions.detach().clone()
        expert = algorithm.transition.dagger_actions.detach().clone()
        assert torch.isfinite(expert).all()
        algorithm.process_env_step(
            torch.ones(num_envs), torch.zeros(num_envs, dtype=torch.bool), {}
        )
        return executed, student, expert

    def stored_label_error(sequence_batch):
        observations = sequence_batch[0]
        references = sequence_batch[1]
        expert_labels = sequence_batch[11]
        sequence_metadata = sequence_batch[12]
        masks = sequence_batch[13]
        raw_observations = sequence_batch[19]
        assert expert_labels is not None
        with torch.no_grad():
            actor_critic.act(
                observations,
                ref_observations=references,
                masks=masks,
                memory=sequence_metadata,
                raw_observations=raw_observations,
                episode_ids=sequence_metadata["episode_ids"],
                sequence_output_start=sequence_metadata[
                    "sequence_output_start"
                ],
            )
            return (
                actor_critic.action_mean - expert_labels
            ).square().mean().item()

    teacher_parameter = next(teacher.parameters())
    teacher_parameter_before = teacher_parameter.detach().clone()
    action_head_before = [
        parameter.detach().clone() for parameter in actor_critic.actor.fc.parameters()
    ]

    algorithm.set_training_iteration(0)
    assert algorithm.fixed_expert_stage == "bc"
    for step in range(rollout_steps):
        executed, _, expert = collect_step(step)
        torch.testing.assert_close(executed, expert)
    final_observations, final_references = make_inputs(rollout_steps)
    algorithm.compute_returns(final_observations, final_references)
    bc_sequence_batch = next(
        algorithm.storage.streaming_sequence_mini_batch_generator(1, 1)
    )
    bc_error_before = stored_label_error(bc_sequence_batch)
    bc_metrics = algorithm.update(epoch=0)
    bc_error_after = stored_label_error(bc_sequence_batch)

    assert bc_metrics["fixed_expert_stage"] == 0.0
    assert bc_metrics["fixed_expert_student_execute_fraction"] == 0.0
    assert bc_metrics["actor_ppo_update_fraction"] == 0.0
    assert bc_metrics["mean_weighted_expert_bc_loss"] > 0.0
    assert bc_error_after < bc_error_before
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(action_head_before, actor_critic.actor.fc.parameters())
    )

    algorithm.set_training_iteration(1)
    assert algorithm.fixed_expert_stage == "dagger"
    executed, student, expert = collect_step(rollout_steps + 1)
    torch.testing.assert_close(executed, student)
    assert not torch.equal(student, expert)
    executed, _, expert = collect_step(rollout_steps + 2)
    torch.testing.assert_close(executed, expert)
    final_observations, final_references = make_inputs(rollout_steps + 3)
    algorithm.compute_returns(final_observations, final_references)
    dagger_sequence_batch = next(
        algorithm.storage.streaming_sequence_mini_batch_generator(1, 1)
    )
    dagger_error_before = stored_label_error(dagger_sequence_batch)
    dagger_metrics = algorithm.update(epoch=1)
    dagger_error_after = stored_label_error(dagger_sequence_batch)

    assert dagger_metrics["fixed_expert_stage"] == 1.0
    assert dagger_metrics["fixed_expert_student_execute_fraction"] == 0.5
    assert dagger_metrics["actor_ppo_update_fraction"] == 0.0
    assert dagger_metrics["mean_weighted_expert_bc_loss"] > 0.0
    assert dagger_error_after < dagger_error_before
    torch.testing.assert_close(teacher_parameter, teacher_parameter_before)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    optimizer_parameter_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    assert not any(
        id(parameter) in optimizer_parameter_ids for parameter in teacher.parameters()
    )
