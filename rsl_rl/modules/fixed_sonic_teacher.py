"""Frozen SONIC actor adapter for sequence-aware policy distillation.

The deployed MMGPT actor receives one 93-D proprioceptive frame at a time,
whereas the retained SONIC MLP checkpoint receives the same five terms with a
10-frame history flattened term-by-term.  This module owns that conversion and
the checkpoint's frozen empirical normalizers.  It deliberately contains no
optimizer-facing parameters: the teacher labels the states visited by the
student, but is never adapted toward the student.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .actor_critic_mlp_v2 import FusedMultiModalMLP
from .normalizer import EmpiricalNormalization


def convert_teacher_action_coordinates(
    teacher_action: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    teacher_is_delta: bool,
    student_is_delta: bool,
    clip_value: Optional[float],
) -> torch.Tensor:
    """Convert a teacher policy output into the student's policy coordinates.

    GBC stores ``last_action`` as the resolved raw controller command.  An
    absolute policy emits that command directly, while a delta policy emits an
    increment relative to it.  Clipping applies to the resolved command in
    both cases, matching the MMGPT future-prediction action contract.
    """

    if teacher_action.shape != previous_action.shape:
        raise ValueError(
            "teacher and previous actions must have identical shapes, got "
            f"{tuple(teacher_action.shape)} and {tuple(previous_action.shape)}"
        )
    if clip_value is not None and clip_value <= 0.0:
        raise ValueError("clip_value must be positive or None")

    finite_teacher_action = torch.where(
        torch.isfinite(teacher_action),
        teacher_action,
        torch.zeros_like(teacher_action),
    )
    resolved = (
        previous_action + finite_teacher_action
        if teacher_is_delta
        else finite_teacher_action
    )
    if clip_value is not None:
        resolved = resolved.clamp(min=-clip_value, max=clip_value)
    return resolved - previous_action if student_is_delta else resolved


class SonicHistoryAssembler:
    """Reset-safe conversion from current frames to SONIC term-major history."""

    def __init__(
        self,
        term_dims: Mapping[str, int],
        history_length: int,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if not term_dims:
            raise ValueError("term_dims must not be empty")
        if any(int(dim) <= 0 for dim in term_dims.values()):
            raise ValueError("all observation term dimensions must be positive")

        self.term_dims = OrderedDict(
            (str(name), int(dim)) for name, dim in term_dims.items()
        )
        self.history_length = int(history_length)
        self.frame_dim = sum(self.term_dims.values())
        self._term_slices: OrderedDict[str, slice] = OrderedDict()
        start = 0
        for name, dim in self.term_dims.items():
            self._term_slices[name] = slice(start, start + dim)
            start += dim

        self.buffer: Optional[torch.Tensor] = None
        self.num_pushes: Optional[torch.Tensor] = None

    @property
    def term_slices(self) -> Mapping[str, slice]:
        return self._term_slices

    def _allocate(self, frame: torch.Tensor) -> None:
        # The runner collects rollouts under torch.inference_mode().  These
        # tensors are persistent mutable state, so force ordinary tensors here
        # rather than inference tensors that cannot later be reset in-place.
        with torch.inference_mode(False):
            self.buffer = torch.empty(
                self.history_length,
                frame.shape[0],
                self.frame_dim,
                dtype=frame.dtype,
                device=frame.device,
            )
            self.num_pushes = torch.zeros(
                frame.shape[0], dtype=torch.long, device=frame.device
            )

    def append(self, frame: torch.Tensor) -> torch.Tensor:
        """Append one frame and return ``[B, sum(T * term_dim)]``.

        Isaac Lab's circular observation buffer repeats the first observation
        across the complete history after a reset.  Reproducing that detail is
        required for exact SONIC checkpoint inputs; zero-padding would create
        a different reset policy.
        """

        if frame.ndim != 2 or frame.shape[1] != self.frame_dim:
            raise ValueError(
                "SONIC teacher frame must have shape "
                f"(B, {self.frame_dim}), got {tuple(frame.shape)}"
            )
        if self.buffer is None:
            self._allocate(frame)
        assert self.buffer is not None
        assert self.num_pushes is not None
        if (
            self.buffer.shape[1] != frame.shape[0]
            or self.buffer.device != frame.device
            or self.buffer.dtype != frame.dtype
        ):
            raise ValueError(
                "SONIC teacher history batch/device/dtype changed without reset"
            )

        if self.history_length > 1:
            self.buffer[:-1].copy_(self.buffer[1:].clone())
        self.buffer[-1].copy_(frame)
        first_push = self.num_pushes == 0
        if first_push.any():
            self.buffer[:, first_push] = frame[first_push]
        self.num_pushes += 1
        return self.term_major_history()

    def term_major_history(self) -> torch.Tensor:
        if self.buffer is None:
            raise RuntimeError("history is empty; append a frame first")
        batch_size = self.buffer.shape[1]
        return torch.cat(
            [
                self.buffer[..., term_slice]
                .permute(1, 0, 2)
                .reshape(batch_size, -1)
                for term_slice in self._term_slices.values()
            ],
            dim=-1,
        )

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        if self.buffer is None:
            return
        assert self.num_pushes is not None
        if dones is None:
            self.buffer.zero_()
            self.num_pushes.zero_()
            return
        if dones.numel() != self.num_pushes.numel():
            raise ValueError(
                "teacher reset mask must have shape "
                f"{tuple(self.num_pushes.shape)}, got {tuple(dones.shape)}"
            )
        reset_mask = dones.bool().reshape_as(self.num_pushes)
        if reset_mask.any():
            self.buffer[:, reset_mask] = 0.0
            self.num_pushes[reset_mask] = 0


class FixedSonicMLPTeacher(nn.Module):
    """Frozen exact-SONIC actor with its own history and normalizers."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        policy_term_dims: Mapping[str, int],
        policy_ref_term_dims: Mapping[str, int],
        num_actions: int,
        history_length: int = 10,
        teacher_is_delta: bool = False,
        student_is_delta: bool = False,
        action_clip_value: Optional[float] = 20.0,
        actor_hidden_dims: Sequence[int] = (2048, 1024, 512, 512),
        joint_decoder_hidden_dims: Sequence[int] = (
            2048,
            2048,
            1024,
            1024,
            512,
            512,
        ),
        reference_encoder_output_dim: int = 64,
        activation: str = "silu",
        use_layer_norm: bool = False,
        encoder_latent_dim: int = 128,
        encoder_compress_threshold: int = 32,
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SONIC teacher checkpoint not found: {checkpoint}")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        self.checkpoint_path = str(checkpoint)
        self.num_actions = int(num_actions)
        self.history = SonicHistoryAssembler(policy_term_dims, history_length)
        if "actions" not in self.history.term_slices:
            raise ValueError("SONIC teacher policy terms must include 'actions'")
        action_slice = self.history.term_slices["actions"]
        if action_slice.stop - action_slice.start != self.num_actions:
            raise ValueError(
                "the current-frame 'actions' term must match num_actions"
            )
        self._previous_action_slice = action_slice
        self.teacher_is_delta = bool(teacher_is_delta)
        self.student_is_delta = bool(student_is_delta)
        self.action_clip_value = action_clip_value

        expanded_terms = OrderedDict(
            (name, int(dim) * int(history_length))
            for name, dim in policy_term_dims.items()
        )
        ref_terms = OrderedDict(
            (str(name), int(dim)) for name, dim in policy_ref_term_dims.items()
        )
        if not ref_terms:
            raise ValueError("SONIC teacher requires the released reference command")
        self.reference_dim = sum(ref_terms.values())

        self.actor = FusedMultiModalMLP(
            term_dict={"policy": expanded_terms},
            ref_term_dict={"policy": ref_terms},
            output_size=self.num_actions,
            hidden_dims=list(actor_hidden_dims),
            activation=activation,
            history_length=int(history_length),
            encoder_latent_dim=int(encoder_latent_dim),
            encoder_compress_threshold=int(encoder_compress_threshold),
            use_layer_norm=bool(use_layer_norm),
            fusion_mode="sonic_joint_decoder",
            joint_decoder_hidden_dims=list(joint_decoder_hidden_dims),
            reference_history_length=1,
            reference_encoder_output_dim=int(reference_encoder_output_dim),
            project_reference_features=True,
            name="fixed_sonic_teacher",
            reorgnize_obs=False,
        )
        self.obs_normalizer = EmpiricalNormalization(
            shape=[self.history.frame_dim * int(history_length)], until=0
        )
        self.ref_normalizer = EmpiricalNormalization(
            shape=[self.reference_dim], until=0
        )

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older supported PyTorch
            payload = torch.load(checkpoint, map_location="cpu")
        if "model_state_dict" not in payload:
            raise KeyError("SONIC teacher checkpoint lacks model_state_dict")
        actor_state = {
            key[len("actor.") :]: value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
        self.actor.load_state_dict(actor_state, strict=True)
        for key, normalizer in (
            ("obs_norm_state_dict", self.obs_normalizer),
            ("ref_obs_norm_state_dict", self.ref_normalizer),
        ):
            if key not in payload:
                raise KeyError(f"SONIC teacher checkpoint lacks {key}")
            normalizer.load_state_dict(payload[key], strict=True)

        self.checkpoint_iteration = int(payload.get("iter", -1))
        self.actor_parameter_count = sum(
            parameter.numel() for parameter in self.actor.parameters()
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """Keep the fixed expert and its empirical normalizers in eval mode."""

        del mode
        return super().train(False)

    # ``no_grad`` rather than ``inference_mode`` is intentional: the history
    # buffer is persistent mutable rollout state and must remain writable when
    # reset() is called from an ordinary autograd context.
    @torch.no_grad()
    def act(
        self,
        raw_observations: torch.Tensor,
        raw_ref_observations: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if raw_ref_observations is None:
            raise ValueError("SONIC teacher requires raw reference observations")
        raw_ref, ref_mask = raw_ref_observations
        if raw_ref.ndim != 2 or raw_ref.shape[1] != self.reference_dim:
            raise ValueError(
                "SONIC teacher reference must have shape "
                f"(B, {self.reference_dim}), got {tuple(raw_ref.shape)}"
            )
        if ref_mask.numel() != raw_ref.shape[0]:
            raise ValueError(
                "SONIC teacher reference mask must have shape "
                f"{(raw_ref.shape[0],)}, got {tuple(ref_mask.shape)}"
            )
        ref_mask = ref_mask.reshape(raw_ref.shape[0])

        history = self.history.append(raw_observations)
        normalized_history = self.obs_normalizer(history)
        normalized_ref = self.ref_normalizer(raw_ref)
        teacher_action = self.actor(
            normalized_history,
            (normalized_ref, ref_mask.bool()),
        )
        previous_action = raw_observations[:, self._previous_action_slice]
        return convert_teacher_action_coordinates(
            teacher_action,
            previous_action,
            teacher_is_delta=self.teacher_is_delta,
            student_is_delta=self.student_is_delta,
            clip_value=self.action_clip_value,
        )

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        self.history.reset(dones)
