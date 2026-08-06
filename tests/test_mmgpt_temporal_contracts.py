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
    )


def _add_storage_step(storage, observations, references, dones, action_mean=None):
    num_envs = observations.shape[0]
    if action_mean is None:
        action_mean = observations[:, :1]
    action_dim = storage.actions.shape[-1]
    transition = RolloutStorageMM.Transition()
    transition.observations = observations
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
