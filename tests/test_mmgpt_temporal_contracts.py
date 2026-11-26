import copy

import pytest
import torch

from rsl_rl.modules.actor_critic_mm_gpt import MMGPT
from rsl_rl.networks.rope_transformer import RoPEMultiHeadAttention
from rsl_rl.storage.rollout_storage_mm import RolloutStorageMM


def _causal_mask(length):
    return torch.nn.Transformer.generate_square_subsequent_mask(
        length, dtype=torch.bool
    )


def test_rope_sdpa_matches_manual_attention_with_causal_and_padding_masks():
    torch.manual_seed(3)
    attention = RoPEMultiHeadAttention(
        d_model=16,
        num_heads=2,
        dropout=0.0,
        use_sdpa=True,
    ).eval()
    inputs = torch.randn(3, 5, 16)
    key_padding_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False],
        ]
    )

    sdpa_output, _ = attention(
        inputs,
        inputs,
        inputs,
        attn_mask=_causal_mask(5),
        key_padding_mask=key_padding_mask,
    )
    attention.use_sdpa = False
    manual_output, _ = attention(
        inputs,
        inputs,
        inputs,
        attn_mask=_causal_mask(5),
        key_padding_mask=key_padding_mask,
    )

    assert torch.isfinite(sdpa_output).all()
    torch.testing.assert_close(sdpa_output, manual_output, atol=2.0e-6, rtol=2.0e-5)
    assert torch.count_nonzero(sdpa_output[key_padding_mask]) == 0


def test_rope_causal_mask_blocks_future_and_padding_but_keeps_past_visible():
    torch.manual_seed(5)
    attention = RoPEMultiHeadAttention(
        d_model=16,
        num_heads=2,
        dropout=0.0,
        use_sdpa=True,
    ).eval()
    inputs = torch.randn(1, 5, 16)
    causal_mask = _causal_mask(5)

    baseline, _ = attention(inputs, inputs, inputs, attn_mask=causal_mask)

    future_changed = inputs.clone()
    future_changed[:, 3:] += 20.0
    future_output, _ = attention(
        future_changed,
        future_changed,
        future_changed,
        attn_mask=causal_mask,
    )
    torch.testing.assert_close(baseline[:, :3], future_output[:, :3])

    past_changed = inputs.clone()
    past_changed[:, 0] += 20.0
    past_output, _ = attention(
        past_changed,
        past_changed,
        past_changed,
        attn_mask=causal_mask,
    )
    assert not torch.allclose(baseline[:, -1], past_output[:, -1])

    padding_mask = torch.tensor([[True, True, False, False, False]])
    padded_baseline, _ = attention(
        inputs,
        inputs,
        inputs,
        attn_mask=causal_mask,
        key_padding_mask=padding_mask,
    )
    padded_changed = inputs.clone()
    padded_changed[:, :2] += 20.0
    padded_output, _ = attention(
        padded_changed,
        padded_changed,
        padded_changed,
        attn_mask=causal_mask,
        key_padding_mask=padding_mask,
    )
    torch.testing.assert_close(
        padded_baseline[:, 2:], padded_output[:, 2:]
    )


def _make_storage(
    *,
    num_envs=2,
    rollout_steps=4,
    history_length=3,
    obs_dim=1,
    ref_dim=1,
    action_dim=1,
    self_supervision=False,
    store_temporal_phases=False,
):
    return RolloutStorageMM(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=rollout_steps,
        obs_shape=(obs_dim,),
        ref_obs_shape=(ref_dim,),
        privileged_obs_shape=(obs_dim,),
        privileged_ref_obs_shape=(ref_dim,),
        actions_shape=(action_dim,),
        device="cpu",
        actor_history_length=history_length,
        critic_history_length=1,
        raw_obs_shape=(obs_dim,) if self_supervision else None,
        dreamer_target_shape=(action_dim,) if self_supervision else None,
        store_actor_temporal_phases=store_temporal_phases,
    )


def _add_storage_step(
    storage,
    observations,
    references,
    dones,
    action_mean=None,
    temporal_phase=None,
):
    num_envs = observations.shape[0]
    if action_mean is None:
        action_mean = observations[:, :1]
    action_dim = storage.actions.shape[-1]
    transition = RolloutStorageMM.Transition()
    transition.observations = observations
    if storage.raw_observations is not None:
        transition.raw_observations = observations + 10_000.0
        transition.dreamer_target_observations = (
            observations[:, :action_dim] + 20_000.0
        )
        transition.dreamer_target_observations_mask = torch.ones(
            num_envs, dtype=torch.bool
        )
    transition.reference_observations = references
    transition.reference_observations_mask = torch.ones(
        num_envs, dtype=torch.bool
    )
    transition.critic_observations = observations
    transition.critic_reference_observations = references
    transition.critic_reference_observations_mask = torch.ones(
        num_envs, dtype=torch.bool
    )
    transition.actions = torch.zeros(num_envs, action_dim)
    transition.rewards = torch.zeros(num_envs)
    transition.dones = dones
    transition.values = torch.zeros(num_envs, 1)
    transition.actions_log_prob = torch.zeros(num_envs)
    transition.action_mean = action_mean
    transition.action_sigma = torch.ones(num_envs, action_dim)
    transition.hidden_states = None
    if storage.actor_temporal_phases is not None:
        if temporal_phase is None:
            raise ValueError("test transition requires an explicit temporal phase")
        transition.actor_temporal_phase = temporal_phase
    storage.add_transitions(transition)


def test_rollout_prefix_is_visible_and_done_cuts_only_the_previous_episode():
    storage = _make_storage()

    for step in range(4):
        observations = torch.tensor(
            [[10.0 * step], [10.0 * step + 1.0]]
        )
        references = observations + 1_000.0
        dones = torch.tensor([step == 2, False])
        _add_storage_step(storage, observations, references, dones)
    storage.clear()

    for step in range(4):
        observations = torch.tensor(
            [[100.0 + 10.0 * step], [101.0 + 10.0 * step]]
        )
        references = observations + 1_000.0
        # A done on the sampled transition must not remove that transition's
        # own observation. It affects only the following observation.
        dones = torch.tensor([False, step == 0])
        _add_storage_step(storage, observations, references, dones)

    batch = next(storage.buffer_mini_batch_generator(1, 1))
    obs_batch, ref_batch, critic_obs_batch = batch[0], batch[1], batch[2]
    old_mu_batch, masks_batch = batch[9], batch[13]

    env0_first = torch.nonzero(old_mu_batch[:, 0] == 100.0).item()
    env1_first = torch.nonzero(old_mu_batch[:, 0] == 101.0).item()

    torch.testing.assert_close(
        obs_batch[:, env0_first, 0], torch.tensor([0.0, 30.0, 100.0])
    )
    torch.testing.assert_close(
        masks_batch[:, env0_first], torch.tensor([False, True, True])
    )
    torch.testing.assert_close(
        ref_batch[0][:, env0_first, 0],
        torch.tensor([0.0, 1_030.0, 1_100.0]),
    )

    torch.testing.assert_close(
        obs_batch[:, env1_first, 0], torch.tensor([21.0, 31.0, 101.0])
    )
    assert masks_batch[:, env1_first].all()
    assert critic_obs_batch.shape[0] == 1


def test_streaming_storage_carries_episode_local_chunk_phase_across_rollouts():
    storage = _make_storage(
        num_envs=2,
        rollout_steps=4,
        history_length=6,
        store_temporal_phases=True,
    )
    for step in range(4):
        observations = torch.full((2, 1), float(step))
        _add_storage_step(
            storage,
            observations,
            observations + 100.0,
            torch.zeros(2, dtype=torch.bool),
            temporal_phase=torch.full((2,), step, dtype=torch.long),
        )
    storage.clear()

    second_rollout_phases = (4, 0, 1, 2)
    for step, phase in enumerate(second_rollout_phases):
        observations = torch.full((2, 1), float(10 + step))
        _add_storage_step(
            storage,
            observations,
            observations + 100.0,
            torch.zeros(2, dtype=torch.bool),
            temporal_phase=torch.full((2,), phase, dtype=torch.long),
        )

    batch = next(
        storage.streaming_sequence_mini_batch_generator(
            num_mini_batches=1,
            num_epochs=1,
            num_steps_per_env=6,
        )
    )
    metadata = batch[12]
    phases = metadata["temporal_phases"]
    masks = batch[13]
    torch.testing.assert_close(
        phases[:, 0], torch.tensor([0, 0, 1, 2, 3, 4, 0, 1, 2])
    )
    assert not masks[0].any()
    assert masks[1:].all()


_TERM_DIMS = {
    "joint_pos": 2,
    "joint_vel": 2,
    "projected_gravity": 3,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "actions": 2,
}


def _make_rope_mmgpt(history_length):
    torch.manual_seed(17)
    model = MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        num_steps_per_env=history_length,
        max_seq_len=history_length,
        term_dict=_TERM_DIMS,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=None,
        pred_obs_term_weights=None,
        default_joint_pos=torch.zeros(2),
        apply_rope=True,
        latest_reference_only=True,
    )
    return model.eval()


def _make_streaming_rope_mmgpt(history_length, num_layers=2):
    torch.manual_seed(17)
    return MMGPT(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=num_layers,
        dropout=0.0,
        num_steps_per_env=history_length,
        max_seq_len=history_length,
        term_dict=_TERM_DIMS,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=None,
        pred_obs_term_weights=None,
        default_joint_pos=torch.zeros(2),
        apply_rope=True,
        latest_reference_only=False,
        streaming_kv_cache=True,
        streaming_context_steps=history_length,
    ).eval()


def _make_chunked_streaming_mmgpt(
    history_length=10, num_layers=2, **overrides
):
    torch.manual_seed(17)
    kwargs = dict(
        obs_size=sum(_TERM_DIMS.values()),
        ref_obs_size=2,
        dim_out=2,
        dim_model=16,
        num_heads=2,
        num_layers=num_layers,
        dropout=0.0,
        num_steps_per_env=history_length,
        max_seq_len=history_length,
        term_dict=_TERM_DIMS,
        ref_term_dict={"target_actions": 2},
        pred_obs_term_names=None,
        pred_obs_term_weights=None,
        default_joint_pos=torch.zeros(2),
        apply_rope=True,
        latest_reference_only=False,
        streaming_kv_cache=True,
        streaming_sequence_training=True,
        streaming_context_steps=history_length,
        temporal_chunk_size=5,
    )
    kwargs.update(overrides)
    return MMGPT(**kwargs).eval()


def _episode_ids_from_dones(dones):
    episode_ids = torch.zeros_like(dones, dtype=torch.long)
    if dones.shape[0] > 1:
        episode_ids[1:] = torch.cumsum(dones[:-1].long(), dim=0)
    return episode_ids


def test_chunk5_emits_actions_each_frame_but_caches_only_completed_chunks():
    model = _make_chunked_streaming_mmgpt()
    torch.manual_seed(101)
    observations = torch.randn(6, 2, sum(_TERM_DIMS.values()))
    references = torch.randn(6, 2, 2)
    ref_masks = torch.ones(6, 2, dtype=torch.bool)

    outputs = []
    phases = []
    with torch.no_grad():
        for step in range(4):
            outputs.append(
                model(observations[step], (references[step], ref_masks[step]))
            )
            phases.append(model.last_inference_temporal_phase.clone())
            assert model.inference_kv_cache is None
        outputs.append(model(observations[4], (references[4], ref_masks[4])))
        phases.append(model.last_inference_temporal_phase.clone())

    assert torch.isfinite(torch.stack(outputs)).all()
    torch.testing.assert_close(
        torch.stack(phases)[:, 0], torch.arange(5, dtype=torch.long)
    )
    assert model.inference_kv_cache["key_padding_mask"].shape == (2, 1)
    assert not model.inference_kv_cache["key_padding_mask"].any()
    assert model.inference_chunk_phase.tolist() == [0, 0]


def test_chunk5_streaming_matches_batch_across_async_resets():
    time_steps = 17
    num_envs = 3
    online_model = _make_chunked_streaming_mmgpt()
    batch_model = copy.deepcopy(online_model)
    torch.manual_seed(103)
    observations = torch.randn(
        time_steps, num_envs, sum(_TERM_DIMS.values())
    )
    references = torch.randn(time_steps, num_envs, 2)
    ref_masks = torch.ones(time_steps, num_envs, dtype=torch.bool)
    masks = torch.ones_like(ref_masks)
    dones = torch.zeros_like(ref_masks)
    dones[2, 0] = True
    dones[6, 1] = True
    dones[10, 2] = True
    dones[12, 0] = True
    episode_ids = _episode_ids_from_dones(dones)

    online_outputs = []
    temporal_phases = []
    with torch.no_grad():
        for step in range(time_steps):
            online_outputs.append(
                online_model(
                    observations[step],
                    (references[step], ref_masks[step]),
                )
            )
            temporal_phases.append(
                online_model.last_inference_temporal_phase.clone()
            )
            online_model.reset(dones[step])

        hidden, _, _ = batch_model._chunked_streaming_sequence_hidden(
            observations,
            (references, ref_masks),
            masks,
            episode_ids=episode_ids,
            temporal_phases=torch.stack(temporal_phases),
        )
        batch_outputs = batch_model.fc(hidden)

    torch.testing.assert_close(
        torch.stack(online_outputs),
        batch_outputs,
        atol=3.0e-7,
        rtol=3.0e-6,
    )


def test_fast_slow_plan_is_persistent_while_current_frame_closes_the_loop():
    online_a = _make_chunked_streaming_mmgpt(
        temporal_fast_slow_enabled=True,
    )
    torch.manual_seed(105)
    with torch.no_grad():
        torch.nn.init.normal_(
            online_a.temporal_fast_controller[-1].weight, std=0.1
        )
    online_b = copy.deepcopy(online_a)
    plan_batch_sizes = []
    plan_hook = online_a.temporal_plan_head.register_forward_hook(
        lambda _module, args, _output: plan_batch_sizes.append(args[0].shape[0])
    )

    first_obs = torch.randn(1, sum(_TERM_DIMS.values()))
    first_ref = torch.randn(1, 2)
    valid_ref = torch.ones(1, dtype=torch.bool)
    with torch.no_grad():
        first_a = online_a(first_obs, (first_ref, valid_ref))
        first_b = online_b(first_obs, (first_ref, valid_ref))
        plan_at_start = online_a.inference_chunk_plan.clone()
        torch.testing.assert_close(first_a, first_b)
        torch.testing.assert_close(
            plan_at_start, online_b.inference_chunk_plan
        )

        second_obs = torch.randn_like(first_obs)
        second_ref = torch.randn_like(first_ref)
        second_a = online_a(second_obs, (second_ref, valid_ref))
        second_b = online_b(
            second_obs + 10.0, (second_ref - 10.0, valid_ref)
        )
    plan_hook.remove()

    # The slow proposal is committed only at phase zero.  Later observations
    # change the feedback correction, not the nominal plan.
    torch.testing.assert_close(online_a.inference_chunk_plan, plan_at_start)
    torch.testing.assert_close(online_b.inference_chunk_plan, plan_at_start)
    assert plan_batch_sizes == [1]
    assert not torch.allclose(second_a, second_b)


def test_fast_slow_streaming_matches_differentiable_batch_across_resets():
    time_steps = 17
    num_envs = 3
    online_model = _make_chunked_streaming_mmgpt(
        temporal_fast_slow_enabled=True,
    )
    batch_model = copy.deepcopy(online_model)
    torch.manual_seed(106)
    observations = torch.randn(
        time_steps, num_envs, sum(_TERM_DIMS.values())
    )
    references = torch.randn(time_steps, num_envs, 2)
    ref_masks = torch.ones(time_steps, num_envs, dtype=torch.bool)
    masks = torch.ones_like(ref_masks)
    dones = torch.zeros_like(ref_masks)
    dones[2, 0] = True
    dones[6, 1] = True
    dones[10, 2] = True
    dones[12, 0] = True
    episode_ids = _episode_ids_from_dones(dones)

    online_outputs = []
    temporal_phases = []
    with torch.no_grad():
        for step in range(time_steps):
            online_outputs.append(
                online_model(
                    observations[step],
                    (references[step], ref_masks[step]),
                )
            )
            temporal_phases.append(
                online_model.last_inference_temporal_phase.clone()
            )
            online_model.reset(dones[step])

        phases = torch.stack(temporal_phases)
        hidden, obs_embedding, ref_embedding = (
            batch_model._chunked_streaming_sequence_hidden(
                observations,
                (references, ref_masks),
                masks,
                episode_ids=episode_ids,
                temporal_phases=phases,
            )
        )
        batch_outputs = batch_model._decode_temporal_sequence_actions(
            hidden,
            obs_embedding,
            masks,
            episode_ids,
            phases,
            ref_embedding,
            ref_masks,
        )

    torch.testing.assert_close(
        torch.stack(online_outputs),
        batch_outputs,
        atol=3.0e-7,
        rtol=3.0e-6,
    )


def test_fast_slow_policy_gradient_reaches_slow_plan_and_fast_controller():
    model = _make_chunked_streaming_mmgpt(
        history_length=10,
        num_layers=1,
        temporal_fast_slow_enabled=True,
    ).train()
    torch.manual_seed(108)
    observations = torch.randn(10, 2, sum(_TERM_DIMS.values()))
    references = torch.randn(10, 2, 2)
    masks = torch.ones(10, 2, dtype=torch.bool)
    phases = torch.arange(10).remainder(5).view(-1, 1).expand(-1, 2)
    plan_batch_sizes = []
    plan_hook = model.temporal_plan_head.register_forward_hook(
        lambda _module, args, _output: plan_batch_sizes.append(args[0].shape[0])
    )

    output = model(
        observations,
        (references, masks),
        masks,
        episode_ids=torch.zeros_like(phases),
        temporal_phases=phases,
    )
    plan_hook.remove()
    output.square().mean().backward()

    assert plan_batch_sizes == [4]
    assert model.temporal_plan_head[-1].weight.grad.norm() > 0.0
    assert model.temporal_fast_controller[-1].weight.grad.norm() > 0.0
    assert (
        model.decoder.encoder.layers[0].self_attn.qkv_proj.weight.grad.norm()
        > 0.0
    )


def test_chunk5_partial_tokens_are_causal_and_use_seen_frames_in_chunk():
    model = _make_chunked_streaming_mmgpt()
    torch.manual_seed(107)
    observations = torch.randn(10, 1, sum(_TERM_DIMS.values()))
    references = torch.randn(10, 1, 2)
    masks = torch.ones(10, 1, dtype=torch.bool)
    phases = torch.arange(10).remainder(5).view(10, 1)
    episode_ids = torch.zeros_like(phases)

    with torch.no_grad():
        baseline, _, _ = model._chunked_streaming_sequence_hidden(
            observations,
            (references, masks),
            masks,
            episode_ids=episode_ids,
            temporal_phases=phases,
        )
        future_observations = observations.clone()
        future_references = references.clone()
        future_observations[7:] += 50.0
        future_references[7:] -= 50.0
        future_changed, _, _ = model._chunked_streaming_sequence_hidden(
            future_observations,
            (future_references, masks),
            masks,
            episode_ids=episode_ids,
            temporal_phases=phases,
        )
        torch.testing.assert_close(baseline[:7], future_changed[:7])

        seen_observations = observations.clone()
        seen_observations[0] += 50.0
        seen_changed, _, _ = model._chunked_streaming_sequence_hidden(
            seen_observations,
            (references, masks),
            masks,
            episode_ids=episode_ids,
            temporal_phases=phases,
        )
    assert not torch.allclose(baseline[2], seen_changed[2])


@pytest.mark.parametrize("fast_slow_enabled", [False, True])
def test_chunk5_cache_rebuild_discards_only_the_unrecoverable_prefix_chunk(
    fast_slow_enabled,
):
    model = _make_chunked_streaming_mmgpt(
        temporal_fast_slow_enabled=fast_slow_enabled,
    )
    torch.manual_seed(109)
    num_envs = 2
    observations = torch.randn(25, num_envs, sum(_TERM_DIMS.values()))
    references = torch.randn(25, num_envs, 2)
    ref_masks = torch.ones(25, num_envs, dtype=torch.bool)

    with torch.no_grad():
        for step in range(24):
            model(observations[step], (references[step], ref_masks[step]))

        train_masks = torch.ones(10, num_envs, dtype=torch.bool)
        train_phases = torch.arange(10).remainder(5).view(10, 1).expand(-1, 2)
        model(
            observations[:10],
            (references[:10], ref_masks[:10]),
            train_masks,
            episode_ids=torch.zeros_like(train_masks, dtype=torch.long),
            temporal_phases=train_phases,
        )
        model.decoder.encoder.layers[0].self_attn.qkv_proj.weight.add_(0.01)
        rebuilt_output = model(
            observations[24], (references[24], ref_masks[24])
        )

        replay_model = copy.deepcopy(model)
        replay_masks = model.generate_mask()
        end_position = model.inference_position_ids[0].item()
        valid_length = replay_masks[:, 0].long().sum().item()
        start_position = end_position - valid_length
        replay_phases = torch.arange(
            start_position, end_position
        ).remainder(5).view(-1, 1).expand(-1, num_envs)
        replay_hidden, replay_obs_embedding, replay_ref_embedding = (
            replay_model._chunked_streaming_sequence_hidden(
                model.inference_buffer_obs,
                (
                    model.inference_buffer_ref_obs,
                    model.inference_buffer_ref_obs_mask,
                ),
                replay_masks,
                episode_ids=torch.zeros_like(replay_masks, dtype=torch.long),
                temporal_phases=replay_phases,
            )
        )
        if fast_slow_enabled:
            replay_output = replay_model._decode_temporal_sequence_actions(
                replay_hidden,
                replay_obs_embedding,
                replay_masks,
                torch.zeros_like(replay_masks, dtype=torch.long),
                replay_phases,
                replay_ref_embedding,
                model.inference_buffer_ref_obs_mask,
            )[-1]
        else:
            replay_output = replay_model.fc(replay_hidden[-1])

    assert not model.inference_kv_cache_dirty
    torch.testing.assert_close(
        rebuilt_output, replay_output, atol=3.0e-7, rtol=3.0e-6
    )


@pytest.mark.parametrize("fast_slow_enabled", [False, True])
def test_chunk5_rollout_update_recomputes_means_across_rollout_boundary(
    fast_slow_enabled,
):
    torch.manual_seed(113)
    model = _make_chunked_streaming_mmgpt(
        temporal_fast_slow_enabled=fast_slow_enabled,
    )
    num_envs = 2
    rollout_steps = 6
    obs_dim = sum(_TERM_DIMS.values())
    storage = _make_storage(
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        history_length=model.rollout_history_frames,
        obs_dim=obs_dim,
        ref_dim=2,
        action_dim=2,
        store_temporal_phases=True,
    )

    def collect_rollout(done_schedule):
        for step in range(rollout_steps):
            observations = torch.randn(num_envs, obs_dim)
            references = torch.randn(num_envs, 2)
            reference_masks = torch.ones(num_envs, dtype=torch.bool)
            with torch.no_grad():
                action_mean = model(
                    observations, (references, reference_masks)
                )
            phase = model.last_inference_temporal_phase.clone()
            dones = done_schedule(step)
            _add_storage_step(
                storage,
                observations,
                references,
                dones,
                action_mean=action_mean,
                temporal_phase=phase,
            )
            model.reset(dones)

    collect_rollout(lambda step: torch.tensor([step == 2, False]))
    storage.clear()
    collect_rollout(lambda step: torch.tensor([False, step == 3]))

    batch = next(
        storage.streaming_sequence_mini_batch_generator(
            num_mini_batches=1,
            num_epochs=1,
            num_steps_per_env=model.rollout_history_frames,
        )
    )
    obs_batch, ref_batch = batch[0], batch[1]
    old_mu_batch, metadata, masks_batch = batch[9], batch[12], batch[13]
    with torch.no_grad():
        recomputed_mu = model(
            obs_batch,
            ref_batch,
            masks_batch,
            episode_ids=metadata["episode_ids"],
            temporal_phases=metadata["temporal_phases"],
            sequence_output_start=metadata["sequence_output_start"],
        )

    torch.testing.assert_close(
        recomputed_mu, old_mu_batch, atol=3.0e-6, rtol=3.0e-5
    )


def test_streaming_kv_matches_full_local_sequence_across_async_resets():
    history_length = 4
    time_steps = 9
    num_envs = 3
    online_model = _make_streaming_rope_mmgpt(history_length)
    batch_model = copy.deepcopy(online_model)

    torch.manual_seed(41)
    observations = torch.randn(
        time_steps, num_envs, sum(_TERM_DIMS.values())
    )
    references = torch.randn(time_steps, num_envs, 2)
    reference_masks = torch.ones(time_steps, num_envs, dtype=torch.bool)
    masks = torch.ones_like(reference_masks)
    dones = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, False],
            [False, True, False],
            [False, False, False],
            [False, False, False],
            [False, False, True],
            [False, False, False],
        ]
    )
    episode_ids = torch.zeros(time_steps, num_envs, dtype=torch.long)
    for step in range(1, time_steps):
        episode_ids[step] = episode_ids[step - 1] + dones[step - 1].long()

    online_actions = []
    with torch.no_grad():
        for step in range(time_steps):
            online_actions.append(
                online_model(
                    observations[step],
                    (references[step], reference_masks[step]),
                )
            )
            online_model.reset(dones[step])

        hidden, _, _ = batch_model._streaming_sequence_hidden(
            observations,
            (references, reference_masks),
            masks,
            episode_ids=episode_ids,
        )
        batch_actions = batch_model.fc(hidden)

    torch.testing.assert_close(
        torch.stack(online_actions), batch_actions, atol=3.0e-7, rtol=3.0e-6
    )
    assert online_model.inference_kv_cache["key_padding_mask"].shape[-1] == 6


def test_streaming_history_keeps_past_reference_but_reset_removes_it():
    model_a = _make_streaming_rope_mmgpt(history_length=4)
    model_b = copy.deepcopy(model_a)
    torch.manual_seed(42)
    observations = torch.randn(2, 1, sum(_TERM_DIMS.values()))
    references = torch.randn(2, 1, 2)
    changed_references = references.clone()
    changed_references[0] += 50.0
    valid = torch.ones(2, 1, dtype=torch.bool)

    with torch.no_grad():
        hidden_a, _, _ = model_a._streaming_sequence_hidden(
            observations, (references, valid), valid
        )
        hidden_b, _, _ = model_b._streaming_sequence_hidden(
            observations, (changed_references, valid), valid
        )
    assert not torch.allclose(hidden_a[1], hidden_b[1])

    online_a = copy.deepcopy(model_a)
    online_b = copy.deepcopy(model_a)
    with torch.no_grad():
        online_a(observations[0], (references[0], valid[0]))
        online_b(observations[0], (changed_references[0], valid[0]))
        done = torch.ones(1, dtype=torch.bool)
        online_a.reset(done)
        online_b.reset(done)
        after_reset_a = online_a(
            observations[1], (references[1], valid[1])
        )
        after_reset_b = online_b(
            observations[1], (references[1], valid[1])
        )
    torch.testing.assert_close(after_reset_a, after_reset_b)


def test_streaming_kv_is_rebuilt_after_a_training_forward_changes_weights():
    history_length = 4
    num_envs = 2
    model = _make_streaming_rope_mmgpt(history_length)
    torch.manual_seed(43)
    observations = torch.randn(
        history_length + 1, num_envs, sum(_TERM_DIMS.values())
    )
    references = torch.randn(history_length + 1, num_envs, 2)
    reference_masks = torch.ones(history_length + 1, num_envs, dtype=torch.bool)

    with torch.no_grad():
        for step in range(history_length):
            model(observations[step], (references[step], reference_masks[step]))

    # Batch mode marks rollout K/V stale because an optimizer step may follow.
    masks = torch.ones(history_length, num_envs, dtype=torch.bool)
    model(
        observations[:history_length],
        (references[:history_length], reference_masks[:history_length]),
        masks,
    )
    assert model.inference_kv_cache_dirty
    with torch.no_grad():
        model.decoder.encoder.layers[0].self_attn.qkv_proj.weight.add_(0.01)
        rebuilt_action = model(
            observations[-1], (references[-1], reference_masks[-1])
        )

        replay_model = copy.deepcopy(model)
        retained_obs = observations[-history_length:]
        retained_ref = references[-history_length:]
        retained_ref_mask = reference_masks[-history_length:]
        hidden, _, _ = replay_model._streaming_sequence_hidden(
            retained_obs,
            (retained_ref, retained_ref_mask),
            masks,
        )
        replay_action = replay_model.fc(hidden[-1])

    assert not model.inference_kv_cache_dirty
    torch.testing.assert_close(
        rebuilt_action, replay_action, atol=3.0e-7, rtol=3.0e-6
    )


def test_mmgpt_action_uses_valid_history_and_current_reference_only():
    model = _make_rope_mmgpt(history_length=4)
    torch.manual_seed(19)
    observations = torch.randn(4, 2, sum(_TERM_DIMS.values()))
    references = torch.randn(4, 2, 2)
    masks = torch.tensor(
        [[False, False], [False, False], [True, True], [True, True]]
    )
    reference_masks = masks.clone()

    with torch.no_grad():
        baseline = model(observations, (references, reference_masks), masks)

        masked_changed = observations.clone()
        masked_changed[:2] += 50.0
        masked_references_changed = references.clone()
        masked_references_changed[:2] += 50.0
        masked_output = model(
            masked_changed,
            (masked_references_changed, reference_masks),
            masks,
        )
        torch.testing.assert_close(baseline, masked_output)

        valid_history_changed = observations.clone()
        valid_history_changed[-2] += 50.0
        history_output = model(
            valid_history_changed, (references, reference_masks), masks
        )
        assert not torch.allclose(baseline, history_output)

        current_reference_changed = references.clone()
        current_reference_changed[-1] += 50.0
        reference_output = model(
            observations,
            (current_reference_changed, reference_masks),
            masks,
        )
        assert not torch.allclose(baseline, reference_output)


def test_mmgpt_online_inference_matches_batch_history_with_async_resets():
    """Lock the no-cache rollout semantics before adding a KV cache."""

    history_length = 4
    num_envs = 3
    online_model = _make_rope_mmgpt(history_length)
    batch_model = copy.deepcopy(online_model)
    observation_histories = [[] for _ in range(num_envs)]
    reference_histories = [[] for _ in range(num_envs)]
    done_schedule = (
        torch.tensor([False, False, False]),
        torch.tensor([True, False, False]),
        torch.tensor([False, False, True]),
        torch.tensor([False, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([False, False, False]),
    )

    torch.manual_seed(31)
    with torch.no_grad():
        for dones in done_schedule:
            observations = torch.randn(
                num_envs, sum(_TERM_DIMS.values())
            )
            references = torch.randn(num_envs, 2)
            reference_mask = torch.ones(num_envs, dtype=torch.bool)
            online_actions = online_model(
                observations,
                (references, reference_mask),
            )

            observation_sequence = torch.zeros(
                history_length,
                num_envs,
                sum(_TERM_DIMS.values()),
            )
            reference_sequence = torch.zeros(
                history_length, num_envs, 2
            )
            masks = torch.zeros(
                history_length, num_envs, dtype=torch.bool
            )
            reference_masks = torch.zeros_like(masks)
            for env_id in range(num_envs):
                observation_histories[env_id].append(
                    observations[env_id].clone()
                )
                reference_histories[env_id].append(
                    references[env_id].clone()
                )
                observation_histories[env_id] = observation_histories[
                    env_id
                ][-history_length:]
                reference_histories[env_id] = reference_histories[env_id][
                    -history_length:
                ]
                valid_length = len(observation_histories[env_id])
                observation_sequence[-valid_length:, env_id] = torch.stack(
                    observation_histories[env_id]
                )
                masks[-valid_length:, env_id] = True
                # latest_reference_only deliberately exposes only the current
                # SONIC command, even though proprioceptive history is kept.
                reference_sequence[-1, env_id] = reference_histories[env_id][
                    -1
                ]
                reference_masks[-1, env_id] = True

            batch_actions = batch_model(
                observation_sequence,
                (reference_sequence, reference_masks),
                masks,
                unpad_output=False,
            )
            torch.testing.assert_close(online_actions, batch_actions)

            online_model.reset(dones)
            for env_id in dones.nonzero(as_tuple=False).flatten().tolist():
                observation_histories[env_id].clear()
                reference_histories[env_id].clear()


def test_rollout_update_recomputes_the_same_mmgpt_context_across_boundary():
    torch.manual_seed(23)
    num_envs = 2
    rollout_steps = 5
    history_length = 4
    obs_dim = sum(_TERM_DIMS.values())
    model = _make_rope_mmgpt(history_length)
    storage = _make_storage(
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        history_length=history_length,
        obs_dim=obs_dim,
        ref_dim=2,
        action_dim=2,
    )

    def collect_rollout(done_schedule):
        for step in range(rollout_steps):
            observations = torch.randn(num_envs, obs_dim)
            references = torch.randn(num_envs, 2)
            reference_masks = torch.ones(num_envs, dtype=torch.bool)
            with torch.no_grad():
                action_mean = model(
                    observations, (references, reference_masks)
                )
            dones = done_schedule(step)
            _add_storage_step(
                storage,
                observations,
                references,
                dones,
                action_mean=action_mean,
            )
            model.reset(dones)

    collect_rollout(lambda step: torch.tensor([step == 2, False]))
    storage.clear()
    collect_rollout(lambda step: torch.tensor([False, step == 1]))

    batch = next(storage.buffer_mini_batch_generator(1, 1))
    obs_batch, ref_batch = batch[0], batch[1]
    old_mu_batch, masks_batch = batch[9], batch[13]
    with torch.no_grad():
        recomputed_mu = model(obs_batch, ref_batch, masks_batch)

    torch.testing.assert_close(
        recomputed_mu, old_mu_batch, atol=3.0e-6, rtol=3.0e-5
    )


def test_streaming_sequence_batch_recomputes_collected_policy_means():
    torch.manual_seed(47)
    num_envs = 2
    rollout_steps = 4
    history_length = 3
    obs_dim = sum(_TERM_DIMS.values())
    model = _make_streaming_rope_mmgpt(history_length)
    model.streaming_sequence_training = True
    storage = _make_storage(
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        history_length=history_length,
        obs_dim=obs_dim,
        ref_dim=2,
        action_dim=2,
    )

    def collect(done_schedule):
        for step in range(rollout_steps):
            observations = torch.randn(num_envs, obs_dim)
            references = torch.randn(num_envs, 2)
            reference_masks = torch.ones(num_envs, dtype=torch.bool)
            with torch.no_grad():
                action_mean = model(
                    observations, (references, reference_masks)
                )
            dones = done_schedule(step)
            _add_storage_step(
                storage,
                observations,
                references,
                dones,
                action_mean=action_mean,
            )
            model.reset(dones)

    collect(lambda step: torch.tensor([step == 2, False]))
    storage.clear()
    # PPO changes model parameters between rollouts, so rollout inference starts
    # by prefilling the cache from the retained raw history.
    model.invalidate_inference_cache()
    collect(lambda step: torch.tensor([False, step == 1]))

    batch = next(storage.streaming_sequence_mini_batch_generator(1, 1))
    obs_batch, ref_batch, old_mu_batch = batch[0], batch[1], batch[9]
    metadata, masks_batch = batch[12], batch[13]
    with torch.no_grad():
        recomputed_mu = model(
            obs_batch,
            ref_batch,
            masks_batch,
            episode_ids=metadata["episode_ids"],
            sequence_output_start=metadata["sequence_output_start"],
        )

    assert obs_batch.shape[:2] == (history_length - 1 + rollout_steps, num_envs)
    assert recomputed_mu.shape == (rollout_steps * num_envs, 2)
    torch.testing.assert_close(
        recomputed_mu, old_mu_batch, atol=3.0e-7, rtol=3.0e-6
    )


def test_raw_history_and_adjacent_targets_follow_the_same_sample_indices():
    storage = _make_storage(
        num_envs=2,
        rollout_steps=4,
        history_length=3,
        self_supervision=True,
    )
    for step in range(4):
        observations = torch.tensor(
            [[10.0 * step], [10.0 * step + 1.0]]
        )
        _add_storage_step(
            storage,
            observations,
            observations + 1_000.0,
            torch.zeros(2, dtype=torch.bool),
        )

    batch = next(storage.buffer_mini_batch_generator(1, 1))
    obs_batch, old_mu_batch, masks_batch = batch[0], batch[9], batch[13]
    raw_obs_batch, dreamer_target_batch = batch[18], batch[19]

    assert raw_obs_batch is not None
    assert dreamer_target_batch is not None
    torch.testing.assert_close(
        raw_obs_batch[masks_batch],
        obs_batch[masks_batch] + 10_000.0,
    )
    torch.testing.assert_close(
        raw_obs_batch[~masks_batch],
        torch.zeros_like(raw_obs_batch[~masks_batch]),
    )
    torch.testing.assert_close(
        dreamer_target_batch[0], old_mu_batch + 20_000.0
    )
    assert dreamer_target_batch[1].all()
