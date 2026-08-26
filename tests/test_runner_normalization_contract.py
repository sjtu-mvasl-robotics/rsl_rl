from __future__ import annotations

import ast
from pathlib import Path

import torch

from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.runners.on_policy_runner_mm import OnPolicyRunnerMM


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "rsl_rl"
    / "runners"
    / "on_policy_runner_mm.py"
)


def _method_source(name: str) -> str:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunnerMM"
    )
    method = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(method)


def test_all_four_normalizers_follow_train_and_eval_mode():
    train_source = _method_source("train_mode")
    eval_source = _method_source("eval_mode")
    assert "normalizer_training = not self._loaded_normalization_frozen" in train_source
    for normalizer_name in (
        "obs_normalizer",
        "critic_obs_normalizer",
        "ref_obs_normalizer",
        "critic_ref_obs_normalizer",
    ):
        assert f"self.{normalizer_name}.train(normalizer_training)" in train_source
        assert f"self.{normalizer_name}.eval()" in eval_source


def test_loaded_dagger_normalizers_remain_frozen_in_train_mode():
    class _Algorithm:
        actor_critic = torch.nn.Identity()
        rnd = None

    runner = OnPolicyRunnerMM.__new__(OnPolicyRunnerMM)
    runner.alg = _Algorithm()
    runner.amp_cfg = None
    runner.empirical_normalization = True
    runner._loaded_normalization_frozen = True
    runner.obs_normalizer = EmpiricalNormalization(2)
    runner.critic_obs_normalizer = EmpiricalNormalization(2)
    runner.ref_obs_normalizer = EmpiricalNormalization(2)
    runner.critic_ref_obs_normalizer = EmpiricalNormalization(2)

    runner.train_mode()
    for normalizer in (
        runner.obs_normalizer,
        runner.critic_obs_normalizer,
        runner.ref_obs_normalizer,
        runner.critic_ref_obs_normalizer,
    ):
        normalizer(torch.ones(4, 2))

    assert runner.alg.actor_critic.training
    assert runner.obs_normalizer.count.item() == 0
    assert runner.critic_obs_normalizer.count.item() == 0
    assert runner.ref_obs_normalizer.count.item() == 0
    assert runner.critic_ref_obs_normalizer.count.item() == 0


def test_fresh_run_normalizers_still_train():
    class _Algorithm:
        actor_critic = torch.nn.Identity()
        rnd = None

    runner = OnPolicyRunnerMM.__new__(OnPolicyRunnerMM)
    runner.alg = _Algorithm()
    runner.amp_cfg = None
    runner.empirical_normalization = True
    runner._loaded_normalization_frozen = False
    runner.obs_normalizer = EmpiricalNormalization(2)
    runner.critic_obs_normalizer = EmpiricalNormalization(2)
    runner.ref_obs_normalizer = None
    runner.critic_ref_obs_normalizer = None

    runner.train_mode()
    runner.obs_normalizer(torch.ones(4, 2))
    runner.critic_obs_normalizer(torch.ones(4, 2))

    assert runner.obs_normalizer.count.item() == 4
    assert runner.critic_obs_normalizer.count.item() == 4


def test_dataset_collection_does_not_double_normalize_policy_inputs():
    source = _method_source("collect_dataset")
    assert "policy = self.get_inference_policy" in source
    assert "obs = self.obs_normalizer(obs)" not in source
    assert "self.ref_obs_normalizer(ref_obs_tuple[0])" not in source


def test_mmgpt_self_supervision_preserves_pre_normalizer_policy_observations():
    extraction_source = _method_source("_get_mmgpt_self_supervision_inputs")
    learn_source = _method_source("learn")

    assert "raw_obs_dict['policy']" in extraction_source
    assert "observations_dict," in extraction_source
    assert "raw_obs.to(self.device).detach().clone()" in extraction_source
    first_extract = learn_source.index("_get_mmgpt_self_supervision_inputs")
    first_normalize = learn_source.index("self.obs_normalizer(obs)")
    assert first_extract < first_normalize
    assert "raw_obs=raw_obs" in learn_source
    assert "dreamer_target=dreamer_target" in learn_source
