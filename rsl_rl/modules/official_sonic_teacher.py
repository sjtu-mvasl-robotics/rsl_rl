"""Frozen official NVIDIA SONIC v1.1 actor as a DAgger expert inside GBC.

The released checkpoint (GEAR-SONIC ``sonic_v1_1/last.pt``) is a universal-token
encoder/decoder policy, not the local MLP replica wrapped by
``FixedSonicMLPTeacher``.  This module builds the actor with the official
``gear_sonic`` classes from the checkpoint's exported ``model_config.yaml``,
loads it strictly, and converts GBC inputs into the native observation layout:

* proprioception: GBC frames are ``[gravity, ang_vel, q_rel, dq_rel, action]``;
  the native ``policy`` group is ``[ang_vel, q_rel, dq_rel, action, gravity]``,
  each term holding 10 frames oldest-to-newest (term-major).
* tokenizer: only the g1 encoder is used.  Its inputs are the one-hot encoder
  index ``[1, 0, 0]``, ``command_multi_future_nonflat`` (10 x 58, released
  q/dq packing) and ``motion_anchor_ori_heading_mf_nonflat`` (10 x 6, reference
  root orientation relative to the robot heading).  GBC supplies both through
  the ``sonic_v11_g1_command`` reference term as ten ``[58 | 6]`` frames.

The teacher has no trainable state and is never adapted toward the student.
"""

from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .fixed_sonic_teacher import SonicHistoryAssembler, convert_teacher_action_coordinates

GBC_FRAME_TERMS = ("gravity_dir", "base_ang_vel", "joint_pos", "joint_vel", "actions")
NATIVE_POLICY_TERMS = ("base_ang_vel", "joint_pos", "joint_vel", "actions", "gravity_dir")
NUM_FUTURE_FRAMES = 10
G1_COMMAND_FRAME_DIM = 58 + 6
G1_COMMAND_DIM = NUM_FUTURE_FRAMES * G1_COMMAND_FRAME_DIM

# Native tokenizer group layout (runtime "Active Observation Terms in Group:
# 'tokenizer'" table of sonic_v1_1, 1761 dims).  Only the g1 slices are filled.
TOKENIZER_DIM = 1761
TOKENIZER_ENCODER_INDEX = slice(0, 3)
TOKENIZER_G1_JOINT_FUTURE = slice(3, 583)
TOKENIZER_G1_ANCHOR_ORI_HEADING = slice(594, 654)
G1_ENCODER_ONE_HOT = (1.0, 0.0, 0.0)


def build_official_sonic_actor(checkpoint_path: str) -> Tuple[nn.Module, dict]:
    """Instantiate and strictly load the official actor from ``last.pt``.

    ``model_config.yaml`` must sit next to the checkpoint (as released on the
    Hugging Face hub).  Module paths are rewritten exactly like the official
    ``eval_agent_trl.py`` does for released configs.
    """

    import yaml
    from omegaconf import OmegaConf

    import trl.trainer.utils as trl_trainer_utils

    if not hasattr(trl_trainer_utils, "OnlineTrainerState"):
        try:
            from trl.experimental.ppo.ppo_trainer import OnlineTrainerState
        except ImportError:  # pragma: no cover - older trl
            from trl.trainer.ppo_trainer import OnlineTrainerState
        trl_trainer_utils.OnlineTrainerState = OnlineTrainerState
    from gear_sonic.trl.utils.common import custom_instantiate

    checkpoint = Path(checkpoint_path).expanduser()
    config_path = checkpoint.parent / "model_config.yaml"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"official SONIC checkpoint and model_config.yaml required: {checkpoint}, {config_path}"
        )

    class _Loader(yaml.SafeLoader):
        pass

    # The exported config serializes one output directory as PosixPath; accept
    # only that inert tag rather than enabling arbitrary YAML objects.
    _Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
        lambda loader, node: str(Path(*loader.construct_sequence(node))),
    )
    raw = config_path.read_text()
    for old, new in (
        ("groot.rl.trl.", "gear_sonic.trl."),
        ("groot.rl.envs.", "gear_sonic.envs."),
        ("groot.rl.utils.", "gear_sonic.utils."),
    ):
        raw = raw.replace(old, new)
    cfg = OmegaConf.create(yaml.load(io.StringIO(raw), Loader=_Loader))
    actor = custom_instantiate(
        cfg.algo_config.actor,
        env_config=cfg.env_config,
        algo_config=cfg.algo_config,
        _resolve=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["policy_state_dict"], strict=True)
    actor.eval().requires_grad_(False)
    obs_dims = OmegaConf.to_container(cfg.env_config.robot.algo_obs_dim_dict)
    info = dict(
        global_step=int(payload["state"].global_step),
        obs_dims={str(k): int(v) for k, v in obs_dims.items()},
    )
    return actor, info


class OfficialSonicV11Teacher(nn.Module):
    """Frozen official SONIC v1.1 g1-encoder policy with its own H10 history."""

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
    ) -> None:
        super().__init__()
        terms = OrderedDict((str(k), int(v)) for k, v in policy_term_dims.items())
        if tuple(terms) != GBC_FRAME_TERMS:
            raise ValueError(f"expected GBC policy terms {GBC_FRAME_TERMS}, got {tuple(terms)}")
        ref_terms = OrderedDict((str(k), int(v)) for k, v in policy_ref_term_dims.items())
        if tuple(ref_terms.values()) != (G1_COMMAND_DIM,):
            raise ValueError(
                f"official SONIC teacher needs one {G1_COMMAND_DIM}-D sonic_v11_g1_command term, got {dict(ref_terms)}"
            )
        if int(history_length) != 10:
            raise ValueError("the released sonic_v1_1 policy group uses history_length=10")
        if terms["actions"] != num_actions:
            raise ValueError("GBC 'actions' term must match num_actions")

        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.num_actions = int(num_actions)
        self.teacher_is_delta = bool(teacher_is_delta)
        self.student_is_delta = bool(student_is_delta)
        self.action_clip_value = action_clip_value
        self.reference_dim = G1_COMMAND_DIM

        self._gbc_slices = OrderedDict()
        start = 0
        for name, dim in terms.items():
            self._gbc_slices[name] = slice(start, start + dim)
            start += dim
        self.frame_dim = start
        self.history = SonicHistoryAssembler(
            OrderedDict((name, terms[name]) for name in NATIVE_POLICY_TERMS), int(history_length)
        )

        self.actor, info = build_official_sonic_actor(self.checkpoint_path)
        self.obs_dims = info["obs_dims"]
        if self.obs_dims.get("actor_obs") != self.history.frame_dim * int(history_length):
            raise ValueError(f"checkpoint actor_obs dim {self.obs_dims.get('actor_obs')} != 930")
        if self.obs_dims.get("tokenizer") != TOKENIZER_DIM:
            raise ValueError(f"checkpoint tokenizer dim {self.obs_dims.get('tokenizer')} != {TOKENIZER_DIM}")
        self.invalid_reference_rows = 0
        self.checkpoint_iteration = info["global_step"]
        self.actor_parameter_count = sum(p.numel() for p in self.actor.parameters())
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    def native_frame(self, gbc_frame: torch.Tensor) -> torch.Tensor:
        if gbc_frame.ndim != 2 or gbc_frame.shape[1] != self.frame_dim:
            raise ValueError(f"GBC frame must be (B, {self.frame_dim}), got {tuple(gbc_frame.shape)}")
        return torch.cat([gbc_frame[:, self._gbc_slices[name]] for name in NATIVE_POLICY_TERMS], dim=-1)

    @staticmethod
    def tokenizer_from_g1_command(command: torch.Tensor) -> torch.Tensor:
        """Place GBC ``sonic_v11_g1_command`` (B, 640) into the native tokenizer layout."""

        if command.ndim != 2 or command.shape[1] != G1_COMMAND_DIM:
            raise ValueError(f"g1 command must be (B, {G1_COMMAND_DIM}), got {tuple(command.shape)}")
        frames = command.reshape(command.shape[0], NUM_FUTURE_FRAMES, G1_COMMAND_FRAME_DIM)
        tokenizer = command.new_zeros(command.shape[0], TOKENIZER_DIM)
        tokenizer[:, TOKENIZER_ENCODER_INDEX] = command.new_tensor(G1_ENCODER_ONE_HOT)
        tokenizer[:, TOKENIZER_G1_JOINT_FUTURE] = frames[..., :58].reshape(command.shape[0], -1)
        tokenizer[:, TOKENIZER_G1_ANCHOR_ORI_HEADING] = frames[..., 58:].reshape(command.shape[0], -1)
        return tokenizer

    def native_action(self, actor_obs: torch.Tensor, tokenizer: torch.Tensor) -> torch.Tensor:
        """Deterministic native action mean for native-layout inputs."""

        obs = {
            key: actor_obs.new_zeros(actor_obs.shape[0], dim) for key, dim in self.obs_dims.items()
        }
        obs["actor_obs"] = actor_obs
        obs["tokenizer"] = tokenizer
        action = self.actor.act_pure_inference({k: v.unsqueeze(1) for k, v in obs.items()})
        return action[:, -1] if action.ndim == 3 else action

    # no_grad rather than inference_mode: the history buffer is persistent
    # mutable rollout state that reset() later edits in an ordinary context.
    @torch.no_grad()
    def act(
        self,
        raw_observations: torch.Tensor,
        raw_ref_observations: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if raw_ref_observations is None:
            raise ValueError("official SONIC teacher requires raw reference observations")
        raw_ref, ref_mask = raw_ref_observations
        if ref_mask.numel() != raw_ref.shape[0]:
            raise ValueError("official SONIC teacher reference mask must have one entry per env")
        # The native policy has no reference-validity input: its command is
        # always clamped to the final motion frame.  Count invalid rows so a
        # caller can verify they never reach supervision.
        self.invalid_reference_rows += int((~ref_mask.reshape(-1).bool()).sum())
        actor_obs = self.history.append(self.native_frame(raw_observations))
        action = self.native_action(actor_obs, self.tokenizer_from_g1_command(raw_ref))
        previous_action = raw_observations[:, self._gbc_slices["actions"]]
        return convert_teacher_action_coordinates(
            action,
            previous_action,
            teacher_is_delta=self.teacher_is_delta,
            student_is_delta=self.student_is_delta,
            clip_value=self.action_clip_value,
        )

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        self.history.reset(dones)
