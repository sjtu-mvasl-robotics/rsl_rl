from collections import OrderedDict

import pytest
import torch

from rsl_rl.algorithms.mmppo import (
    MMPPO,
    _fixed_expert_actor_ppo_enabled,
    _fixed_expert_bc_loss,
    _fixed_expert_stage,
)
from rsl_rl.modules.fixed_sonic_teacher import (
    SonicHistoryAssembler,
    convert_teacher_action_coordinates,
)
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
