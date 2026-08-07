from __future__ import annotations

import ast
from pathlib import Path


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
    for method_name, mode_call in (("train_mode", "train()"), ("eval_mode", "eval()")):
        source = _method_source(method_name)
        for normalizer_name in (
            "obs_normalizer",
            "critic_obs_normalizer",
            "ref_obs_normalizer",
            "critic_ref_obs_normalizer",
        ):
            assert f"self.{normalizer_name}.{mode_call}" in source


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
