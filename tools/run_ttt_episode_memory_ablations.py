"""Run matched TTT episode-memory ablations and preserve raw JSON results."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from evaluate_ttt_episode_memory import (
    EvaluationConfig,
    evaluate_length,
    memory_step_latency,
    train_model,
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    changes: dict[str, object]


def _base_config(*, seed: int, device: str) -> EvaluationConfig:
    return EvaluationConfig(
        task="persistent",
        model_dim=256,
        memory_heads=8,
        values=16,
        slots=8,
        reactive_dim=16,
        train_length=128,
        train_steps=700,
        batch_size=64,
        eval_lengths=(800, 1600),
        eval_batch_size=256,
        learning_rate=1.0e-3,
        gate_init=1.0e-3,
        inner_lr=0.1,
        reactive_loss_weight=0.1,
        memory_only_steps=0,
        reactive_head_only_steps=0,
        reactive_update_interval=1,
        preserve_reactive_gradient_budget=False,
        compare_stateless=False,
        seed=seed,
        device=device,
    )


def objective_specs() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec("joint", {}),
        AblationSpec("memory_only", {"memory_only_steps": 700}),
        AblationSpec("memory_first_300", {"memory_only_steps": 300}),
        AblationSpec(
            "reactive_head_only_300",
            {"reactive_head_only_steps": 300},
        ),
        AblationSpec(
            "reactive_head_only_600",
            {"reactive_head_only_steps": 600},
        ),
    )


def _set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def _summary(
    raw: dict[str, object],
    *,
    longest: int,
) -> dict[str, object]:
    evaluation = raw["evaluation"]
    assert isinstance(evaluation, dict)
    longest_evaluation = evaluation[str(longest)]
    assert isinstance(longest_evaluation, dict)
    accuracy = longest_evaluation["accuracy"]
    intervention = longest_evaluation["state_intervention"]
    assert isinstance(accuracy, dict)
    assert isinstance(intervention, dict)
    training = raw["training"]
    assert isinstance(training, dict)
    return {
        "name": raw["name"],
        "seed": raw["seed"],
        "final_accuracy": accuracy[f"step_{longest - 1}"],
        "late_half_accuracy": accuracy["late_half"],
        "reactive_mse": longest_evaluation["final_reactive_mse"],
        "normal_accuracy": intervention["normal_final_accuracy"],
        "cleared_accuracy": intervention["cleared_final_accuracy"],
        "permuted_original_accuracy": intervention[
            "permuted_vs_original_accuracy"
        ],
        "permuted_donor_accuracy": intervention[
            "permuted_vs_donor_accuracy"
        ],
        "permuted_logit_delta_l2": intervention[
            "permuted_logit_delta_l2"
        ],
        "elapsed_seconds": training["elapsed_seconds"],
        "peak_memory_gib": training["peak_memory_gib"],
        "read_scale_abs_mean": raw["read_scale_abs_mean"],
        "inner_lr_mean": raw["inner_lr_mean"],
        "slow_parameter_count": raw["slow_parameter_count"],
        "fast_state_scalars_per_environment": raw[
            "fast_state_scalars_per_environment"
        ],
    }


def run_one(
    spec: AblationSpec,
    *,
    seed: int,
    device: str,
    output_dir: Path,
) -> dict[str, object]:
    output_path = output_dir / f"{spec.name}_seed{seed}.json"
    config = replace(_base_config(seed=seed, device=device), **spec.changes)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        expected_config = json.loads(json.dumps(asdict(config)))
        expected_changes = json.loads(json.dumps(spec.changes))
        if raw.get("config") != expected_config:
            raise ValueError(
                f"{output_path}: existing result has a different config"
            )
        if raw.get("changes") != expected_changes:
            raise ValueError(
                f"{output_path}: existing result has different changes"
            )
        return raw

    _set_seed(seed, device)
    started = time.perf_counter()
    model, training = train_model(config, use_memory=True, verbose=False)
    model.eval()

    evaluation: dict[str, object] = {}
    for length in config.eval_lengths:
        # Evaluation episodes are independent of how many random numbers each
        # training condition consumed.
        _set_seed(100_000 + seed + length, device)
        evaluation[str(length)] = evaluate_length(
            model,
            config,
            length=length,
        )

    assert model.memory is not None
    raw: dict[str, object] = {
        "name": spec.name,
        "seed": seed,
        "config": asdict(config),
        "changes": spec.changes,
        "training": training,
        "evaluation": evaluation,
        "wall_seconds": time.perf_counter() - started,
        "slow_parameter_count": sum(
            parameter.numel() for parameter in model.memory.parameters()
        ),
        "fast_state_scalars_per_environment": math.prod(
            model.memory.state_shape
        ),
        "read_scale_abs_mean": float(
            model.memory.gate.detach().abs().mean()
        ),
        "inner_lr_mean": float(model.memory.inner_lr.detach().mean()),
        "latency": memory_step_latency(model, config),
    }
    raw["summary"] = _summary(raw, longest=max(config.eval_lengths))

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(output_path)

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("objective",), default="objective")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds:
        raise ValueError("at least one seed is required")
    specs = objective_specs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            raw = run_one(
                spec,
                seed=seed,
                device=args.device,
                output_dir=args.output_dir,
            )
            summary = raw["summary"]
            assert isinstance(summary, dict)
            summaries.append(summary)
            print("ABLATION_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)

    summary_path = args.output_dir / "objective_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(summary_path)


if __name__ == "__main__":
    main()
