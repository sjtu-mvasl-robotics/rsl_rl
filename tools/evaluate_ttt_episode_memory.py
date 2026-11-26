"""Measure differentiable and functional episode memory in TTTKVBLinearMemory.

The tasks deliberately remove the persistent command/value from later inputs
while retaining a dense reactive signal on every step.  A model must therefore
use its recurrent fast weights for the delayed information and its residual
path for the current signal.  State reset and cross-sample state permutation
provide causal interventions on the learned memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules.actor_critic_mm_gpt import TTTKVBLinearMemory


@dataclass(frozen=True)
class EvaluationConfig:
    task: str
    model_dim: int
    memory_heads: int
    values: int
    slots: int
    reactive_dim: int
    train_length: int
    train_steps: int
    batch_size: int
    eval_lengths: tuple[int, ...]
    eval_batch_size: int
    learning_rate: float
    gate_init: float
    inner_lr: float
    reactive_loss_weight: float
    memory_only_steps: int
    reactive_head_only_steps: int
    reactive_update_interval: int
    preserve_reactive_gradient_budget: bool
    compare_stateless: bool
    seed: int
    device: str


@dataclass
class EpisodeBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    reactive_targets: torch.Tensor
    loss_start: int
    persistent_values: torch.Tensor | None = None
    value_table: torch.Tensor | None = None
    query_slots: torch.Tensor | None = None

    def donor_targets(self, permutation: torch.Tensor) -> torch.Tensor:
        if self.persistent_values is not None:
            return self.persistent_values[permutation]
        if self.value_table is None or self.query_slots is None:
            raise RuntimeError("episode metadata is incomplete")
        final_queries = self.query_slots[-1]
        donor_table = self.value_table[permutation]
        return donor_table.gather(1, final_queries.unsqueeze(1)).squeeze(1)


class EpisodeMemoryModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int,
        memory_heads: int,
        values: int,
        reactive_dim: int,
        gate_init: float,
        inner_lr: float,
        use_memory: bool,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.memory = (
            TTTKVBLinearMemory(
                model_dim,
                memory_heads,
                gate_init=gate_init,
                inner_lr=inner_lr,
            )
            if use_memory
            else None
        )
        self.value_head = nn.Linear(model_dim, values)
        self.reactive_head = nn.Linear(model_dim, reactive_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        reset_masks: torch.Tensor | None = None,
        detach_reactive_backbone: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden = self.encoder(inputs)
        final_state = None
        if self.memory is not None:
            masks = torch.ones(
                hidden.shape[:2], dtype=torch.bool, device=hidden.device
            )
            hidden, final_state = self.memory.forward_sequence(
                hidden,
                masks,
                initial_state=initial_state,
                state_reset_masks=reset_masks,
            )
        reactive_hidden = hidden.detach() if detach_reactive_backbone else hidden
        return (
            self.value_head(hidden),
            self.reactive_head(reactive_hidden),
            final_state,
        )


def _input_dim(config: EvaluationConfig) -> int:
    if config.task == "persistent":
        return 1 + config.values + config.reactive_dim
    if config.task == "associative":
        return (
            1
            + config.slots
            + config.values
            + config.slots
            + config.reactive_dim
        )
    raise ValueError(f"unsupported task: {config.task}")


def _sample_persistent(
    config: EvaluationConfig,
    *,
    length: int,
    batch_size: int,
) -> EpisodeBatch:
    device = torch.device(config.device)
    values = torch.randint(config.values, (batch_size,), device=device)
    reactive = torch.randn(length, batch_size, config.reactive_dim, device=device)
    inputs = torch.zeros(
        length,
        batch_size,
        _input_dim(config),
        device=device,
    )
    inputs[0, :, 0] = 1.0
    inputs[0, :, 1 : 1 + config.values] = F.one_hot(
        values, config.values
    ).float()
    inputs[:, :, -config.reactive_dim :] = reactive
    targets = values.unsqueeze(0).expand(length, -1)
    return EpisodeBatch(
        inputs=inputs,
        targets=targets,
        reactive_targets=reactive,
        loss_start=1,
        persistent_values=values,
    )


def _sample_associative(
    config: EvaluationConfig,
    *,
    length: int,
    batch_size: int,
) -> EpisodeBatch:
    if length <= config.slots:
        raise ValueError("associative sequences must extend beyond all writes")
    device = torch.device(config.device)
    table = torch.randint(
        config.values,
        (batch_size, config.slots),
        device=device,
    )
    queries = torch.randint(
        config.slots,
        (length, batch_size),
        device=device,
    )
    reactive = torch.randn(length, batch_size, config.reactive_dim, device=device)
    inputs = torch.zeros(
        length,
        batch_size,
        _input_dim(config),
        device=device,
    )
    inputs[: config.slots, :, 0] = 1.0
    value_start = 1 + config.slots
    query_start = value_start + config.values
    for slot in range(config.slots):
        inputs[slot, :, 1 + slot] = 1.0
        inputs[slot, :, value_start:query_start] = F.one_hot(
            table[:, slot], config.values
        ).float()
    inputs.scatter_(2, (query_start + queries).unsqueeze(-1), 1.0)
    inputs[:, :, -config.reactive_dim :] = reactive
    targets = table.gather(1, queries.transpose(0, 1)).transpose(0, 1)
    return EpisodeBatch(
        inputs=inputs,
        targets=targets,
        reactive_targets=reactive,
        loss_start=config.slots,
        value_table=table,
        query_slots=queries,
    )


def sample_batch(
    config: EvaluationConfig,
    *,
    length: int,
    batch_size: int,
) -> EpisodeBatch:
    if config.task == "persistent":
        return _sample_persistent(config, length=length, batch_size=batch_size)
    return _sample_associative(config, length=length, batch_size=batch_size)


def _loss(
    logits: torch.Tensor,
    reactive_prediction: torch.Tensor,
    batch: EpisodeBatch,
    reactive_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_loss = F.cross_entropy(
        logits[batch.loss_start :].flatten(0, 1),
        batch.targets[batch.loss_start :].flatten(),
    )
    reactive_loss = F.mse_loss(
        reactive_prediction,
        batch.reactive_targets,
    )
    return (
        value_loss + reactive_weight * reactive_loss,
        value_loss,
        reactive_loss,
    )


def _objective_gradient_geometry(
    model: EpisodeMemoryModel,
    value_loss: torch.Tensor,
    reactive_loss: torch.Tensor,
    *,
    reactive_weight: float,
) -> dict[str, dict[str, float]]:
    shared_parameters = list(model.encoder.parameters())
    group_slices = {"encoder": slice(0, len(shared_parameters))}
    if model.memory is not None:
        memory_start = len(shared_parameters)
        shared_parameters.extend(model.memory.parameters())
        group_slices["memory"] = slice(memory_start, len(shared_parameters))
    group_slices["all_shared"] = slice(0, len(shared_parameters))
    value_gradients = torch.autograd.grad(
        value_loss,
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    reactive_gradients = torch.autograd.grad(
        reactive_loss,
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def group_metrics(parameter_slice: slice) -> dict[str, float]:
        value_norm_sq = value_loss.new_zeros(())
        reactive_norm_sq = value_loss.new_zeros(())
        dot = value_loss.new_zeros(())
        for value_gradient, reactive_gradient in zip(
            value_gradients[parameter_slice],
            reactive_gradients[parameter_slice],
        ):
            if value_gradient is not None:
                value_norm_sq = value_norm_sq + value_gradient.float().square().sum()
            if reactive_gradient is not None:
                reactive_norm_sq = (
                    reactive_norm_sq + reactive_gradient.float().square().sum()
                )
            if value_gradient is not None and reactive_gradient is not None:
                dot = dot + (
                    value_gradient.float() * reactive_gradient.float()
                ).sum()
        value_norm = value_norm_sq.sqrt()
        reactive_norm = reactive_norm_sq.sqrt()
        denominator = value_norm * reactive_norm
        cosine = (
            dot / denominator
            if float(denominator) > 0.0
            else value_loss.new_zeros(())
        )
        return {
            "value_grad_norm": float(value_norm),
            "reactive_grad_norm_raw": float(reactive_norm),
            "reactive_grad_norm_weighted": float(reactive_norm * reactive_weight),
            "cosine": float(cosine),
        }

    return {
        group_name: group_metrics(parameter_slice)
        for group_name, parameter_slice in group_slices.items()
    }


def _effective_reactive_weight(
    config: EvaluationConfig,
    *,
    update: int,
    use_memory: bool,
) -> float:
    if use_memory and update < config.memory_only_steps:
        return 0.0
    if update % config.reactive_update_interval != 0:
        return 0.0
    if config.preserve_reactive_gradient_budget:
        return config.reactive_loss_weight * config.reactive_update_interval
    return config.reactive_loss_weight


def train_model(
    config: EvaluationConfig,
    *,
    use_memory: bool,
    verbose: bool = True,
) -> tuple[EpisodeMemoryModel, dict[str, object]]:
    model = EpisodeMemoryModel(
        input_dim=_input_dim(config),
        model_dim=config.model_dim,
        memory_heads=config.memory_heads,
        values=config.values,
        reactive_dim=config.reactive_dim,
        gate_init=config.gate_init,
        inner_lr=config.inner_lr,
        use_memory=use_memory,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1.0e-4,
    )
    history: list[dict[str, float]] = []
    objective_geometry: list[dict[str, object]] = []
    started = time.perf_counter()
    if torch.cuda.is_available() and str(config.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(config.device))

    for update in range(config.train_steps + 1):
        batch = sample_batch(
            config,
            length=config.train_length,
            batch_size=config.batch_size,
        )
        detach_reactive_backbone = (
            use_memory and update < config.reactive_head_only_steps
        )
        logits, reactive_prediction, _ = model(
            batch.inputs,
            detach_reactive_backbone=detach_reactive_backbone,
        )
        effective_reactive_weight = _effective_reactive_weight(
            config,
            update=update,
            use_memory=use_memory,
        )
        total_loss, value_loss, reactive_loss = _loss(
            logits,
            reactive_prediction,
            batch,
            effective_reactive_weight,
        )
        if update in {
            0,
            config.memory_only_steps,
            config.reactive_head_only_steps,
        }:
            objective_geometry.append(
                {
                    "update": update,
                    "reactive_loss_weight": effective_reactive_weight,
                    "reactive_backbone_detached": detach_reactive_backbone,
                    "groups": _objective_gradient_geometry(
                        model,
                        value_loss,
                        reactive_loss,
                        reactive_weight=effective_reactive_weight,
                    ),
                }
            )
        optimizer.zero_grad()
        total_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update % max(config.train_steps // 10, 1) == 0:
            with torch.no_grad():
                prediction = logits.argmax(dim=-1)
                late_accuracy = (
                    prediction[config.train_length // 2 :]
                    == batch.targets[config.train_length // 2 :]
                ).float().mean()
                final_accuracy = (
                    prediction[-1] == batch.targets[-1]
                ).float().mean()
            point = {
                "update": float(update),
                "value_loss": float(value_loss.detach()),
                "reactive_loss": float(reactive_loss.detach()),
                "late_accuracy": float(late_accuracy),
                "final_accuracy": float(final_accuracy),
                "gradient_norm": float(gradient_norm),
                "reactive_loss_weight": effective_reactive_weight,
                "reactive_backbone_detached": float(
                    detach_reactive_backbone
                ),
            }
            history.append(point)
            if verbose:
                print(
                    f"train memory={int(use_memory)} update={update} "
                    f"value_loss={point['value_loss']:.6f} "
                    f"late_accuracy={point['late_accuracy']:.4f} "
                    f"final_accuracy={point['final_accuracy']:.4f}",
                    flush=True,
                )

    if str(config.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(config.device))
        peak_memory_gib = torch.cuda.max_memory_allocated(
            torch.device(config.device)
        ) / (1024**3)
    else:
        peak_memory_gib = 0.0
    return model, {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gib": peak_memory_gib,
        "history": history,
        "objective_gradient_geometry": objective_geometry,
    }


def _accuracy_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    correct = logits.argmax(dim=-1).eq(targets).float()
    length = correct.shape[0]
    indices = sorted(
        {
            min(1, length - 1),
            length // 4,
            length // 2,
            length - 1,
        }
    )
    return {
        **{f"step_{index}": float(correct[index].mean()) for index in indices},
        "late_half": float(correct[length // 2 :].mean()),
    }


@torch.no_grad()
def evaluate_length(
    model: EpisodeMemoryModel,
    config: EvaluationConfig,
    *,
    length: int,
) -> dict[str, object]:
    batch = sample_batch(
        config,
        length=length,
        batch_size=config.eval_batch_size,
    )
    logits, reactive_prediction, final_state = model(batch.inputs)
    result: dict[str, object] = {
        "accuracy": _accuracy_summary(logits, batch.targets),
        "final_reactive_mse": float(
            F.mse_loss(reactive_prediction[-1], batch.reactive_targets[-1])
        ),
    }
    if final_state is not None:
        result["final_state_norm"] = float(
            final_state.flatten(1).norm(dim=1).mean()
        )

        reset_masks = torch.zeros(
            length,
            config.eval_batch_size,
            device=config.device,
            dtype=torch.bool,
        )
        reset_masks[length // 2] = True
        reset_logits, _, _ = model(batch.inputs, reset_masks=reset_masks)
        result["mid_episode_reset_final_accuracy"] = float(
            reset_logits[-1].argmax(dim=-1).eq(batch.targets[-1]).float().mean()
        )

        periodic_masks = torch.zeros_like(reset_masks)
        periodic_masks[50::50] = True
        periodic_logits, _, _ = model(batch.inputs, reset_masks=periodic_masks)
        result["periodic_50_final_accuracy"] = float(
            periodic_logits[-1].argmax(dim=-1).eq(batch.targets[-1]).float().mean()
        )

        prefix_length = min(max(batch.loss_start + 1, 64), length // 2)
        _, _, prefix_state = model(batch.inputs[:prefix_length])
        assert prefix_state is not None
        tail_inputs = batch.inputs[prefix_length:]
        normal_tail, _, _ = model(tail_inputs, initial_state=prefix_state)
        permutation = torch.randperm(config.eval_batch_size, device=config.device)
        permuted_tail, _, _ = model(
            tail_inputs,
            initial_state=prefix_state[permutation],
        )
        reset_state = model.memory.initial_state(
            config.eval_batch_size,
            device=torch.device(config.device),
            dtype=batch.inputs.dtype,
        )
        cleared_tail, _, _ = model(tail_inputs, initial_state=reset_state)
        normal_prediction = normal_tail[-1].argmax(dim=-1)
        permuted_prediction = permuted_tail[-1].argmax(dim=-1)
        cleared_prediction = cleared_tail[-1].argmax(dim=-1)
        donor_targets = batch.donor_targets(permutation)
        result["state_intervention"] = {
            "prefix_length": prefix_length,
            "normal_final_accuracy": float(
                normal_prediction.eq(batch.targets[-1]).float().mean()
            ),
            "permuted_vs_original_accuracy": float(
                permuted_prediction.eq(batch.targets[-1]).float().mean()
            ),
            "permuted_vs_donor_accuracy": float(
                permuted_prediction.eq(donor_targets).float().mean()
            ),
            "cleared_final_accuracy": float(
                cleared_prediction.eq(batch.targets[-1]).float().mean()
            ),
            "permuted_logit_delta_l2": float(
                (normal_tail[-1] - permuted_tail[-1])
                .norm(dim=-1)
                .mean()
            ),
        }
    return result


def gradient_by_lag(
    model: EpisodeMemoryModel,
    config: EvaluationConfig,
    *,
    length: int,
) -> dict[str, object]:
    if model.memory is None:
        return {}
    batch = sample_batch(config, length=length, batch_size=4)
    inputs = batch.inputs.detach().requires_grad_(True)
    logits, _, _ = model(inputs)
    terminal_score = logits[-1].gather(
        1, batch.targets[-1].unsqueeze(1)
    ).sum()
    gradients = torch.autograd.grad(terminal_score, inputs)[0]
    indices = sorted(
        {
            0,
            min(24, length - 1),
            min(50, length - 1),
            min(187, length - 1),
            min(250, length - 1),
            min(500, length - 1),
            length - 1,
        }
    )

    reset_inputs = batch.inputs.detach().requires_grad_(True)
    reset_masks = torch.zeros(
        length, 4, dtype=torch.bool, device=config.device
    )
    reset_masks[length // 2] = True
    reset_logits, _, _ = model(reset_inputs, reset_masks=reset_masks)
    reset_score = reset_logits[-1].gather(
        1, batch.targets[-1].unsqueeze(1)
    ).sum()
    reset_gradients = torch.autograd.grad(reset_score, reset_inputs)[0]
    return {
        "input_gradient_norm": {
            f"step_{index}": float(
                gradients[index].flatten(1).norm(dim=1).mean()
            )
            for index in indices
        },
        "pre_reset_max_abs_gradient": float(
            reset_gradients[: length // 2].abs().max()
        ),
    }


@torch.no_grad()
def memory_step_latency(
    model: EpisodeMemoryModel,
    config: EvaluationConfig,
) -> dict[str, float]:
    if model.memory is None or not str(config.device).startswith("cuda"):
        return {}
    device = torch.device(config.device)
    result: dict[str, float] = {}
    for batch_size in (1, 4096):
        hidden = torch.randn(batch_size, config.model_dim, device=device)
        state = model.memory.initial_state(
            batch_size,
            device=device,
            dtype=hidden.dtype,
        )
        for _ in range(20):
            _, state = model.memory.step(hidden, state)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _, state = model.memory.step(hidden, state)
        end.record()
        torch.cuda.synchronize(device)
        result[f"batch_{batch_size}_milliseconds"] = start.elapsed_time(end) / 100
    return result


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("persistent", "associative"), default="persistent")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--memory-heads", type=int, default=8)
    parser.add_argument("--values", type=int, default=16)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--reactive-dim", type=int, default=16)
    parser.add_argument("--train-length", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=650)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-lengths", default="50,128,187,250,500,800")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--gate-init", type=float, default=1.0e-3)
    parser.add_argument("--inner-lr", type=float, default=0.1)
    parser.add_argument("--reactive-loss-weight", type=float, default=0.1)
    parser.add_argument("--memory-only-steps", type=int, default=0)
    parser.add_argument("--reactive-head-only-steps", type=int, default=0)
    parser.add_argument("--reactive-update-interval", type=int, default=1)
    parser.add_argument(
        "--preserve-reactive-gradient-budget",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--compare-stateless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    eval_lengths = tuple(int(value) for value in args.eval_lengths.split(","))
    if not eval_lengths or any(length < 2 for length in eval_lengths):
        raise ValueError("eval lengths must be comma-separated integers >= 2")
    if args.model_dim % args.memory_heads != 0:
        raise ValueError("model dim must be divisible by memory heads")
    if not 0 <= args.memory_only_steps <= args.train_steps:
        raise ValueError("memory-only steps must lie in [0, train steps]")
    if not 0 <= args.reactive_head_only_steps <= args.train_steps:
        raise ValueError("reactive head-only steps must lie in [0, train steps]")
    if args.memory_only_steps > 0 and args.reactive_head_only_steps > 0:
        raise ValueError(
            "memory-only and reactive head-only schedules are mutually exclusive"
        )
    if args.reactive_update_interval < 1:
        raise ValueError("reactive update interval must be >= 1")
    return EvaluationConfig(
        task=args.task,
        model_dim=args.model_dim,
        memory_heads=args.memory_heads,
        values=args.values,
        slots=args.slots,
        reactive_dim=args.reactive_dim,
        train_length=args.train_length,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_lengths=eval_lengths,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        gate_init=args.gate_init,
        inner_lr=args.inner_lr,
        reactive_loss_weight=args.reactive_loss_weight,
        memory_only_steps=args.memory_only_steps,
        reactive_head_only_steps=args.reactive_head_only_steps,
        reactive_update_interval=args.reactive_update_interval,
        preserve_reactive_gradient_budget=args.preserve_reactive_gradient_budget,
        compare_stateless=args.compare_stateless,
        seed=args.seed,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    if str(config.device).startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    memory_model, memory_training = train_model(config, use_memory=True)
    memory_model.eval()
    evaluation = {
        str(length): evaluate_length(memory_model, config, length=length)
        for length in config.eval_lengths
    }
    longest = max(config.eval_lengths)
    results: dict[str, object] = {
        "config": asdict(config),
        "chance_accuracy": 1.0 / config.values,
        "memory_training": memory_training,
        "memory_evaluation": evaluation,
        "gradient_by_lag": gradient_by_lag(
            memory_model, config, length=longest
        ),
        "memory_step_latency": memory_step_latency(memory_model, config),
    }
    assert memory_model.memory is not None
    results["memory_parameters"] = {
        "slow_parameter_count": sum(
            parameter.numel() for parameter in memory_model.memory.parameters()
        ),
        "fast_state_scalars_per_environment": math.prod(
            memory_model.memory.state_shape
        ),
        "head_dim": memory_model.memory.head_dim,
        "gate_abs_mean": float(memory_model.memory.gate.detach().abs().mean()),
        "inner_lr": memory_model.memory.inner_lr.detach().cpu().tolist(),
    }

    if config.compare_stateless:
        # Reset the random stream so both models see the same distribution.
        torch.manual_seed(config.seed)
        if str(config.device).startswith("cuda"):
            torch.cuda.manual_seed_all(config.seed)
        stateless_model, stateless_training = train_model(
            config, use_memory=False
        )
        stateless_model.eval()
        results["stateless_training"] = stateless_training
        results["stateless_evaluation"] = {
            str(length): evaluate_length(stateless_model, config, length=length)
            for length in config.eval_lengths
        }

    print("RESULT_JSON=" + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
