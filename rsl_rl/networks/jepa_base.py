# Ctrl JEPA: Control-Enhanced Joint Embedding Predictive Architecture
# Basic Modules for JEPA-based Control Learning
# These components are served for both CtrlJEPA and TD-JEPA implementations.


from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal, Any, Sequence
from rsl_rl.networks.rope import RotaryEmbedding
from rsl_rl.networks.rope_transformer import RoPETransformerEncoder, RoPETransformerEncoderLayer
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import typing as tp
import numbers
import gymnasium
import numpy as np
import pydantic

##########################
# TD-JEPA Base Classes
##########################
class BaseConfig(pydantic.BaseModel):
    """Base class for model configurations."""

    # extra="forbid" -- prevent adding new fields (e.g., accidentally typo-ing argument model-dimm=256 )
    # strict=True -- be strict on types, do not try to cast e.g., "True" --> "1" (or vice versa)
    # use_enum_values=True -- fixes issues with serializing/deserializing enums
    # frozen=True -- make the model immutable (to match the behavior of exca scripts)
    model_config = pydantic.ConfigDict(extra="forbid", strict=True, use_enum_values=True, frozen=True)

    name: Literal["BaseConfig"] = "BaseConfig"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # We want to make sure `class_name` is always a valid name.
        # So we will add it to the list of Literals (if user set custom name), or create one.
        # We will default using user's name if one provided, otherwise the class name.
        if not hasattr(cls, "name") or cls.name == "BaseConfig":
            # Add field consistent with `name: Literal["BaseConfig"] = "BaseConfig"`
            cls.name = cls.__name__
            cls.__annotations__["name"] = Literal[cls.__name__]
        else:
            # Append the name to the list of Literals
            existing_args = tp.get_args(cls.__annotations__.get("name"))
            if not existing_args:
                existing_args = (cls.name,)
            cls.__annotations__["name"] = Literal[(cls.__name__,) + tuple(existing_args)]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in config {self.__class__.__name__}")

    def build(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"The object {self} did not have valid build function. Did you forget to define it?")


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [
                normalized_shape,
            ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class IdentityNNConfig(BaseConfig):
    name: tp.Literal["Identity"] = "Identity"

    def build(self, obs_space, *args) -> nn.Module:
        return IdentityNN(obs_space)


class IdentityNN(nn.Identity):
    def __init__(self, obs_space):
        super().__init__()
        self.obs_space = obs_space

    @property
    def output_space(self):
        return self.obs_space

##########################
# Initialization utils
##########################


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Update utils
##########################


def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)

# TD-JEPA Core
##########################
# Creation utils
##########################


class ForwardArchiConfig(BaseConfig):
    name: tp.Literal["ForwardArchi"] = "ForwardArchi"
    hidden_dim: int = 1024
    model: tp.Literal["simple"] = "simple"
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: tp.Literal["batch"] = "batch"

    def build(self, obs_space, z_dim: int, action_dim, output_dim=None) -> torch.nn.Module:
        """Note: Forward model is also used for critics"""

        if self.ensemble_mode == "batch":
            return _build_batch_forward(self, obs_space, z_dim, action_dim, output_dim)
        else:
            raise ValueError(f"Unsupported ensemble_mode {self.ensemble_mode}")


def _build_batch_forward(cfg, obs_space, z_dim, action_dim, output_dim=None):
    if cfg.model == "simple":
        forward_cls = ForwardMap
    else:
        raise ValueError(f"Unsupported forward_map model {cfg.model}")
    return forward_cls(obs_space, z_dim, action_dim, cfg, output_dim=output_dim)


class ActorArchiConfig(BaseConfig):
    name: tp.Literal["actor"] = "actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim, action_dim):
        if self.model == "simple":
            return Actor(obs_space, z_dim, action_dim, self)
        else:
            raise ValueError(f"Unsupported actor model {self.model}. Define 'model' or use other configs explicitely")


def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################


class BackwardArchiConfig(BaseConfig):
    name: tp.Literal["BackwardArchi"] = "BackwardArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True

    def build(self, obs_space, z_dim: int):
        return BackwardMap(obs_space, z_dim, self)


class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg

        self.output_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(z_dim,), dtype=np.float32)

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        seq = [nn.Linear(obs_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, z_dim)]
        if cfg.hidden_layers == 0:
            seq = [nn.Linear(obs_space.shape[0], z_dim)]
        if cfg.norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class ForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        action_dim,
        cfg: ForwardArchiConfig,
        output_dim=None,
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg = cfg
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class SimpleActorArchiConfig(ActorArchiConfig):
    name: tp.Literal["simple"] = "simple"
    model: tp.Literal["simple"] = "simple"

    def build(self, obs_space, z_dim: int, action_dim: int) -> "Actor":
        return Actor(obs_space, z_dim, action_dim, self)


class Actor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: SimpleActorArchiConfig) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: SimpleActorArchiConfig = cfg

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class VForwardArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2

    def build(self, obs_space, z_dim: int, output_dim=None) -> torch.nn.Module:
        return VForwardMap(obs_space, z_dim, output_dim, self)


class VForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        output_dim=None,
        cfg: VForwardArchiConfig = VForwardArchiConfig(),
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([s_embedding, z_embedding], dim=-1))


##########################
# Visual modules
##########################


class DrQEncoderArchiConfig(BaseConfig):
    name: tp.Literal["drq"] = "drq"
    feature_dim: int | None = None  # if not None, linearly project the output to feature_dim

    def build(self, obs_space):
        return DrQEncoder(obs_space, self)


class DrQEncoder(nn.Module):
    """RGB encoder from the DrQ-v2 paper"""

    def __init__(self, obs_space, cfg: DrQEncoderArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

        # courtesy of https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.repr_dim = np.prod(self.trunk(torch.zeros(1, *obs_space.shape)).shape)

        if self.cfg.feature_dim is not None:
            self.proj = nn.Sequential(nn.Linear(self.repr_dim, self.cfg.feature_dim), nn.LayerNorm(self.cfg.feature_dim), nn.Tanh())
            self.repr_dim = self.cfg.feature_dim
        else:
            self.proj = nn.Identity()
            print(
                "WARNING: using a DrQ encoder with feature_dim=None. This yields very large feature vectors that are fed as input to other networks"
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(obs))

    @property
    def output_space(self):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.repr_dim,), dtype=np.float32)


class AugmentatorArchiConfig(BaseConfig):
    name: tp.Literal["random_shifts"] = "random_shifts"
    pad: int = 4

    def build(self, obs_space):
        return Augmentator(obs_space, self)


class Augmentator(nn.Module):
    """Image augmentations from DrQ-v2"""

    def __init__(self, obs_space, cfg: AugmentatorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n, _, h, w = obs.size()
        assert h == w, "Augmentator only supports square images"
        padding = tuple([self.cfg.pad] * 4)
        obs = F.pad(obs, padding, "replicate")
        eps = 1.0 / (h + 2 * self.cfg.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.cfg.pad, device=obs.device, dtype=obs.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.cfg.pad + 1, size=(n, 1, 1, 2), device=obs.device, dtype=obs.dtype)
        shift *= 2.0 / (h + 2 * self.cfg.pad)
        grid = base_grid + shift
        return F.grid_sample(obs, grid, padding_mode="zeros", align_corners=False)


##########################
# FlowQ modules
##########################


class NoiseConditionedActorArchiConfig(BaseConfig):
    name: tp.Literal["noise_conditioned_actor"] = "noise_conditioned_actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim: int, action_dim: int) -> "NoiseConditionedActor":
        return NoiseConditionedActor(obs_space, z_dim, action_dim, self)


class NoiseConditionedActor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: NoiseConditionedActorArchiConfig) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: NoiseConditionedActorArchiConfig = cfg
        self.embed_z = simple_embedding(obs_dim + z_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z_embedding = self.embed_z(torch.cat([obs, z, noise], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(torch.cat([obs, noise], dim=-1))  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        actions = torch.tanh(self.policy(embedding))
        return actions


# Ctrl-JEPA-specific modules
from dataclasses import dataclass, field

ModalityRole = Literal[
    "base_obs", 
    "privileged_obs",
    "history_obs",
    "history_privileged_obs",
    "vision_obs",
    "history_vision_obs", # video, reserved for video-wm integration
    "history_action", # normally, at-1 only
    "action_target", # PD target. Can be either 1 frame or action chunks for multi-step prediction
    "condition", # condition injections, e.g. text tokens, skill types etc.
    "state_target", # for state prediction, reserved for world-modeling
    "reward_target" # for reward prediction, reserved for world-modeling and reward modeling
]

class ModProcessorBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim: int = -1  # to be set by subclass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ModProcessorBase is an abstract class, please implement forward method in subclass")



Branch = Literal["context_deploy", "context_full", "target", "target_full"]

CONTEXT_DEPLOY_ALLOWED = {
    "base_obs",
    "history_obs",
    "vision_obs",
    "history_vision_obs",
    "history_action",
    "condition",
}

CONTEXT_FULL_ALLOWED = {
    "base_obs",
    "history_obs",
    "privileged_obs",
    "history_privileged_obs",
    "vision_obs",
    "history_vision_obs",
    "history_action",
    "condition",
}

TARGET_ALLOWED = {
    "action_target",
    "state_target",
    "reward_target",
    "privileged_obs",
    "history_privileged_obs",
}

TARGET_ONLY_ROLES = {"action_target", "state_target", "reward_target"}



@dataclass
class TokenFeatures:
    """
    Processor output.

    feat accepted shapes before canonicalization:
        (B, C)
        (B, T, C)
        (B, T, G, C)

    After canonicalization:
        feat:      (B, T, G, C)
        valid:     (B, T, G)
        time_ids:  (B, T, G)
        group_ids: (B, T, G)
    """
    feat: torch.Tensor
    valid: Optional[torch.Tensor] = None
    time_ids: Optional[torch.Tensor] = None
    group_ids: Optional[torch.Tensor] = None


class ModProcessorBase(nn.Module):
    out_dim: int

    def __init__(self):
        super().__init__()
        self.out_dim = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor | TokenFeatures:
        raise NotImplementedError


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: Optional[int] = None,
    layers: int = 2,
    norm: bool = True,
    activation: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    if layers <= 1:
        mods: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if norm:
            mods.append(nn.LayerNorm(out_dim))
        return nn.Sequential(*mods)

    hidden_dim = hidden_dim or out_dim
    mods = []
    dim = in_dim
    for _ in range(layers - 1):
        mods.append(nn.Linear(dim, hidden_dim))
        if norm:
            mods.append(nn.LayerNorm(hidden_dim))
        mods.append(activation())
        dim = hidden_dim
    mods.append(nn.Linear(dim, out_dim))
    if norm:
        mods.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*mods)


class VectorProcessor(ModProcessorBase):
    """Vector or temporal-vector modality processor.

    Input:
      x: (B, D) or (B, T, D)

    Output:
      feat: (B, T, out_dim), where T=1 for (B,D).
    """

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 2, norm: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.net = make_mlp(in_dim, out_dim, layers=num_layers, norm=norm)

    def forward(self, x: torch.Tensor) -> TokenFeatures:
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError(f"VectorProcessor expects (B,D) or (B,T,D), got {tuple(x.shape)}")
        B, T, D = x.shape
        feat = self.net(x.reshape(B * T, D)).reshape(B, T, self.out_dim)
        time_ids = torch.arange(T, device=x.device, dtype=torch.long)[None, :].expand(B, T)
        group_ids = torch.zeros(B, T, device=x.device, dtype=torch.long)
        valid = torch.ones(B, T, device=x.device, dtype=torch.bool)
        return TokenFeatures(feat=feat, valid=valid, time_ids=time_ids, group_ids=group_ids)


class GroupedVectorProcessor(ModProcessorBase):
    """Split vector into semantic groups and produce group tokens.

    Input:
      x: (B, D) or (B, T, D)

    Output:
      feat: (B, T*G, out_dim)
    """

    def __init__(
        self,
        group_indices: dict[str, Sequence[int]],
        out_dim: int,
        num_layers: int = 2,
        norm: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.group_names = list(group_indices.keys())
        self.group_indices = {k: torch.as_tensor(v, dtype=torch.long) for k, v in group_indices.items()}
        self.group_mlps = nn.ModuleDict(
            {
                name: make_mlp(len(indices), out_dim, layers=num_layers, norm=norm)
                for name, indices in self.group_indices.items()
            }
        )

    def forward(self, x: torch.Tensor) -> TokenFeatures:
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError(f"GroupedVectorProcessor expects (B,D) or (B,T,D), got {tuple(x.shape)}")
        B, T, _ = x.shape
        feats = []
        for name in self.group_names:
            idx = self.group_indices[name].to(x.device)
            x_g = x.index_select(dim=-1, index=idx)
            Dg = x_g.shape[-1]
            f_g = self.group_mlps[name](x_g.reshape(B * T, Dg)).reshape(B, T, self.out_dim)
            feats.append(f_g)
        # (B,T,G,D)
        feat = torch.stack(feats, dim=2)
        G = feat.shape[2]
        feat = feat.reshape(B, T * G, self.out_dim)
        time_ids = torch.arange(T, device=x.device)[None, :, None].expand(B, T, G).reshape(B, T * G)
        group_ids = torch.arange(G, device=x.device)[None, None, :].expand(B, T, G).reshape(B, T * G)
        valid = torch.ones(B, T * G, device=x.device, dtype=torch.bool)
        return TokenFeatures(feat=feat, valid=valid, time_ids=time_ids, group_ids=group_ids)


@dataclass
class ModalitySpec:
    name: str
    processor: ModProcessorBase
    role: ModalityRole
    out_dim: int
    # Optional visibility overrides. Target-only hard rule still wins.
    visibility: Optional[dict[Branch, bool]] = None
    # Optional source modality whose processor/projection and default token
    # embeddings should be reused. Example: action_target -> history_action.
    representation_source: Optional[str] = None
    embedding_source: Optional[str] = None


@dataclass
class TokenPack:
    tokens: torch.Tensor          # (B, N, D)
    valid: torch.Tensor           # (B, N)
    modality_ids: torch.Tensor    # (B, N)
    role_ids: torch.Tensor        # (B, N)
    time_ids: torch.Tensor        # (B, N)
    group_ids: torch.Tensor       # (B, N)
    names: list[str]


class CtrlJEPAEncoder(nn.Module):
    """Online/target encoder E_theta / E_bar.

    This is a complete encoder, not just a tokenizer:
      raw modalities -> tokens -> embeddings -> transformer -> hidden tokens.

    For context branch, real action_target is excluded.
    For target branch, real action_target is visible and its hidden token is used as target latent.
    """

    TARGET_ONLY = {"action_target", "state_target", "reward_target"}
    PRIVILEGED = {"privileged_obs", "history_privileged_obs"}

    def __init__(
        self,
        specs: dict[str, ModalitySpec],
        token_dim: int = 256,
        max_time: int = 1,
        max_groups: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        proj_layers: int = 2,
        use_modality_embedding: bool = True,
        use_role_embedding: bool = True,
        use_time_embedding: bool = False,
        use_group_embedding: bool = False,
        use_causal_time_mask: bool = False,
    ):
        super().__init__()
        self.specs = specs
        self.token_dim = token_dim
        self.max_time = max_time
        self.max_groups = max_groups
        self.use_causal_time_mask = use_causal_time_mask
        self.use_rope_time = use_time_embedding or use_causal_time_mask

        self.representation_sources: dict[str, str] = {}
        self.embedding_sources: dict[str, str] = {}
        for name, spec in specs.items():
            source = spec.representation_source or name
            if source not in specs:
                raise KeyError(f"Modality {name} representation_source={source!r} is not in specs.")
            embedding_source = spec.embedding_source or source
            if embedding_source not in specs:
                raise KeyError(f"Modality {name} embedding_source={embedding_source!r} is not in specs.")
            self.representation_sources[name] = source
            self.embedding_sources[name] = embedding_source

        processor_modules = {n: specs[self.representation_sources[n]].processor for n in specs}
        projection_by_source: dict[str, nn.Module] = {}
        projection_modules = {}
        for name in specs:
            source = self.representation_sources[name]
            if source not in projection_by_source:
                projection_by_source[source] = make_mlp(specs[source].out_dim, token_dim, layers=proj_layers, norm=True)
            projection_modules[name] = projection_by_source[source]

        self.processors = nn.ModuleDict(processor_modules)
        self.projections = nn.ModuleDict(projection_modules)

        self.modality_to_id = {name: i for i, name in enumerate(specs.keys())}
        roles = sorted({s.role for s in specs.values()} | {"action_target", "state_target", "reward_target"})
        self.role_to_id = {r: i for i, r in enumerate(roles)}

        self.modality_embedding = nn.Embedding(len(self.modality_to_id), token_dim) if use_modality_embedding else None
        self.role_embedding = nn.Embedding(len(self.role_to_id), token_dim) if use_role_embedding else None
        self.time_embedding = nn.Embedding(max_time, token_dim) if use_time_embedding else None
        self.group_embedding = nn.Embedding(max_groups, token_dim) if use_group_embedding else None

        if self.use_rope_time:
            layer = RoPETransformerEncoderLayer(
                d_model=token_dim,
                num_heads=num_heads,
                dim_feedforward=int(token_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                use_sdpa=False,
            )
            self.blocks = RoPETransformerEncoder(layer, num_layers=depth)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=int(token_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, batch: dict[str, torch.Tensor], branch: Branch, return_pack: bool = False) -> dict[str, torch.Tensor | TokenPack]:
        pack = self.build_tokens(batch, branch=branch)
        attn_mask = None
        if self.use_causal_time_mask:
            attn_mask = self._build_causal_time_attn_mask(pack.time_ids)
        if isinstance(self.blocks, RoPETransformerEncoder):
            hidden = self.blocks(
                pack.tokens,
                mask=attn_mask,
                src_key_padding_mask=~pack.valid,
                position_ids=pack.time_ids.clamp(0, self.max_time - 1),
            )
        else:
            hidden = self.blocks(pack.tokens, mask=attn_mask, src_key_padding_mask=~pack.valid)
        hidden = self.norm(hidden)
        out: dict[str, torch.Tensor | TokenPack] = {"hidden": hidden, "valid": pack.valid}
        if return_pack:
            out["pack"] = pack
        return out

    def build_tokens(self, batch: dict[str, torch.Tensor], branch: Branch) -> TokenPack:
        all_tokens = []
        all_valid = []
        all_mod_ids = []
        all_role_ids = []
        all_time_ids = []
        all_group_ids = []
        names = []
        B_ref = None
        device_ref = None

        for name, spec in self.specs.items():
            if name not in batch:
                continue
            if not self._allowed(spec, branch):
                continue

            tf = self.processors[name](batch[name])
            feat = tf.feat
            if feat.ndim != 3:
                raise ValueError(f"Processor {name} must return feat (B,N,D), got {tuple(feat.shape)}")
            B, N, Din = feat.shape
            source = self.representation_sources[name]
            source_spec = self.specs[source]
            if Din != source_spec.out_dim:
                raise ValueError(f"Processor {name} returned D={Din}, source {source}.out_dim={source_spec.out_dim}")
            B_ref = B if B_ref is None else B_ref
            device_ref = feat.device if device_ref is None else device_ref

            valid = tf.valid if tf.valid is not None else torch.ones(B, N, device=feat.device, dtype=torch.bool)
            time_ids = tf.time_ids if tf.time_ids is not None else torch.zeros(B, N, device=feat.device, dtype=torch.long)
            group_ids = tf.group_ids if tf.group_ids is not None else torch.zeros(B, N, device=feat.device, dtype=torch.long)

            tok = self.projections[name](feat)
            embedding_source = self.embedding_sources[name]
            embedding_spec = self.specs[embedding_source]
            mid = self.modality_to_id[name]
            embed_mid = self.modality_to_id[embedding_source]
            rid = self.role_to_id[spec.role]
            if self.modality_embedding is not None:
                tok = tok + self.modality_embedding(torch.full((B, N), embed_mid, device=feat.device, dtype=torch.long))
            if self.role_embedding is not None:
                embed_rid = self.role_to_id[embedding_spec.role]
                tok = tok + self.role_embedding(torch.full((B, N), embed_rid, device=feat.device, dtype=torch.long))
            if self.time_embedding is not None:
                tok = tok + self.time_embedding(time_ids.clamp(0, self.max_time - 1))
            if self.group_embedding is not None:
                tok = tok + self.group_embedding(group_ids.clamp(0, self.max_groups - 1))

            all_tokens.append(tok)
            all_valid.append(valid)
            all_mod_ids.append(torch.full((B, N), mid, device=feat.device, dtype=torch.long))
            all_role_ids.append(torch.full((B, N), rid, device=feat.device, dtype=torch.long))
            all_time_ids.append(time_ids)
            all_group_ids.append(group_ids)
            names.extend([name] * N)

        if not all_tokens:
            raise RuntimeError(f"No tokens for branch={branch}. Check batch/spec visibility.")

        return TokenPack(
            tokens=torch.cat(all_tokens, dim=1),
            valid=torch.cat(all_valid, dim=1),
            modality_ids=torch.cat(all_mod_ids, dim=1),
            role_ids=torch.cat(all_role_ids, dim=1),
            time_ids=torch.cat(all_time_ids, dim=1),
            group_ids=torch.cat(all_group_ids, dim=1),
            names=names,
        )

    def extract_role_tokens(self, hidden: torch.Tensor, pack: TokenPack, role: ModalityRole) -> torch.Tensor:
        rid = self.role_to_id[role]
        mask = pack.role_ids == rid  # (B,N)
        # For Stage-1, action_target should be exactly one token per sample.
        counts = mask.sum(dim=1)
        if not torch.all(counts == counts[0]):
            raise RuntimeError(f"Different number of role={role} tokens across batch: {counts}")
        if counts[0].item() < 1:
            raise RuntimeError(f"No tokens found for role={role}")
        return hidden[mask].reshape(hidden.shape[0], counts[0].item(), hidden.shape[-1])

    def _allowed(self, spec: ModalitySpec, branch: Branch) -> bool:
        # Hard target-only rule: target roles never enter context.
        if spec.role in self.TARGET_ONLY:
            return branch == "target"
        if spec.visibility is not None and branch in spec.visibility:
            return spec.visibility[branch]
        if branch == "context_deploy":
            if spec.role in self.PRIVILEGED:
                return False
            return spec.role in {
                "base_obs",
                "history_obs",
                "vision_obs",
                "history_vision_obs",
                "history_action",
                "condition",
            }
        if branch == "context_full":
            return spec.role in {
                "base_obs",
                "history_obs",
                "privileged_obs",
                "history_privileged_obs",
                "vision_obs",
                "history_vision_obs",
                "history_action",
                "condition",
            }
        if branch == "target":
            # Target encoder should see context + actual target tokens.
            # This is the full target view for action-slot target generation.
            return spec.role in {
                "base_obs",
                "history_obs",
                "privileged_obs",
                "history_privileged_obs",
                "vision_obs",
                "history_vision_obs",
                "history_action",
                "condition",
                "action_target",
                "state_target",
                "reward_target",
            }
        raise ValueError(f"Unknown branch: {branch}")

    @staticmethod
    def _build_causal_time_attn_mask(time_ids_1d: torch.Tensor) -> torch.Tensor:
        # PyTorch Transformer mask: True means disallow attention.
        # Token i can attend token j iff time_j <= time_i.
        if time_ids_1d.ndim == 1:
            ti = time_ids_1d[:, None]
            tj = time_ids_1d[None, :]
        elif time_ids_1d.ndim == 2:
            ti = time_ids_1d[:, :, None]
            tj = time_ids_1d[:, None, :]
        else:
            raise ValueError(f"time_ids must be (N,) or (B,N), got {tuple(time_ids_1d.shape)}")
        return tj > ti

    def target_only_representation_issues(self) -> list[str]:
        issues = []
        for name, spec in self.specs.items():
            if spec.role not in self.TARGET_ONLY:
                continue
            source = self.representation_sources[name]
            if source == name:
                issues.append(f"{name} is target-only but has no visible representation_source")
                continue
            source_spec = self.specs[source]
            if not (self._allowed(source_spec, "context_deploy") or self._allowed(source_spec, "context_full")):
                issues.append(f"{name} source {source} is not visible to an online context branch")
        return issues


class TargetQueryBank(nn.Module):
    """Learned target query tokens for predictor.

    The query has the same semantic role as the target slot, but contains no real target value.
    """

    def __init__(self, token_dim: int, roles: Sequence[str], max_time: int = 16):
        super().__init__()
        self.token_dim = token_dim
        self.role_to_id = {r: i for i, r in enumerate(roles)}
        self.base_query = nn.ParameterDict({r: nn.Parameter(torch.zeros(1, 1, token_dim)) for r in roles})
        self.role_embedding = nn.Embedding(len(roles), token_dim)
        self.time_embedding = nn.Embedding(max_time, token_dim)
        for p in self.base_query.values():
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, role: str, time_ids: torch.Tensor) -> torch.Tensor:
        if role not in self.role_to_id:
            raise KeyError(f"Unknown query role {role}")
        B, Nq = time_ids.shape
        q = self.base_query[role].expand(B, Nq, -1)
        r = self.role_embedding(torch.full_like(time_ids, self.role_to_id[role]))
        t = self.time_embedding(time_ids)
        return q + r + t


class CtrlJEPAPredictor(nn.Module):
    """Predictor P_phi(context tokens, target query) -> target latent tokens.

    This mirrors the JEPA predictor role: it is separate from the online/target encoders.
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int = 256,
        out_dim: Optional[int] = None,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_dim = out_dim or encoder_dim
        self.context_proj = nn.Linear(encoder_dim, predictor_dim)
        self.query_proj = nn.Linear(encoder_dim, predictor_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, out_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_valid: torch.Tensor,
        query_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nc, _ = context_tokens.shape
        Nq = query_tokens.shape[1]
        ctx = self.context_proj(context_tokens)
        qry = self.query_proj(query_tokens)
        x = torch.cat([ctx, qry], dim=1)
        # Query tokens are always valid.
        q_valid = torch.ones(B, Nq, device=context_valid.device, dtype=torch.bool)
        valid = torch.cat([context_valid, q_valid], dim=1)
        h = self.blocks(x, mask=attn_mask, src_key_padding_mask=~valid)
        h = self.norm(h)
        pred = self.out_proj(h[:, Nc:, :])
        return pred


class ActionDecoder(nn.Module):
    """Decode JEPA action-slot latent to PD target.

    This is not an action autoencoder. It has no action encoder.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.net = make_mlp(latent_dim, action_dim, hidden_dim=hidden_dim, layers=num_layers, norm=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """V-JEPA style EMA: target = m * target + (1-m) * online."""
    online_params = list(online.parameters())
    target_params = list(target.parameters())
    if len(online_params) != len(target_params):
        raise RuntimeError("Online and target encoder have different parameter counts.")
    torch._foreach_mul_(target_params, momentum)
    torch._foreach_add_(target_params, online_params, alpha=1.0 - momentum)


class LinearEMAScheduler:
    """Linear EMA momentum ramp."""

    def __init__(self, start: float = 0.99, end: float = 0.9995, total_steps: int = 100_000):
        self.start = start
        self.end = end
        self.total_steps = max(int(total_steps), 1)
        self.step_idx = 0

    def step(self) -> float:
        r = min(self.step_idx / self.total_steps, 1.0)
        m = self.start + r * (self.end - self.start)
        self.step_idx += 1
        return m


class CtrlJEPAStage1(nn.Module):
    """Stage-1 true JEPA wrapper.

    Components:
      online_encoder E_theta: context visible only
      target_encoder E_bar: EMA target, context + real action_target
      predictor P_phi: context tokens + action query -> predicted action latent
      optional action_decoder D: predicted latent -> PD target
    """

    def __init__(
        self,
        online_encoder: CtrlJEPAEncoder,
        predictor: CtrlJEPAPredictor,
        query_bank: TargetQueryBank,
        action_decoder: Optional[ActionDecoder] = None,
        target_encoder: Optional[CtrlJEPAEncoder] = None,
        ema_momentum: float = 0.99,
        normalize_targets: bool = True,
        z_dim: int = 0,
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.target_encoder = copy.deepcopy(online_encoder) if target_encoder is None else target_encoder
        self.predictor = predictor
        self.query_bank = query_bank
        self.action_decoder = action_decoder
        self.ema_momentum = ema_momentum
        self.normalize_targets = normalize_targets
        # Optional decoupled action-latent bottleneck: when z_dim > 0 the JEPA loss is
        # computed in a z_dim subspace and the decoder consumes z_dim, decoupling "how
        # few latent channels" from transformer width (token_dim). z_dim == 0 keeps the
        # original token_dim latent untouched (bit-identical / checkpoint-compatible).
        self.z_dim = int(z_dim)
        if self.z_dim > 0:
            self.z_head = nn.Linear(self.online_encoder.token_dim, self.z_dim)
            self.z_head_target = copy.deepcopy(self.z_head)
            for p in self.z_head_target.parameters():
                p.requires_grad_(False)
        issues = self.online_encoder.target_only_representation_issues()
        if issues:
            raise RuntimeError("target-only EMA transfer requires visible representation_source: " + "; ".join(issues))
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def project_z(self, z: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Project a token_dim action latent into the z_dim bottleneck. Identity when
        the bottleneck is off (z_dim == 0). `target=True` uses the EMA head so the
        target representation stays stable, mirroring the EMA target encoder."""
        if self.z_dim <= 0:
            return z
        return self.z_head_target(z) if target else self.z_head(z)

    def forward(self, batch: dict[str, torch.Tensor], context_branch: Branch = "context_deploy") -> dict[str, torch.Tensor]:
        # 1) Online context encoder: visible context only. No action_target here.
        ctx_out = self.online_encoder(batch, branch=context_branch, return_pack=True)
        ctx_hidden = ctx_out["hidden"]  # (B,Nc,D)
        ctx_pack = ctx_out["pack"]
        assert isinstance(ctx_pack, TokenPack)

        # 2) Predictor query for action target.
        B = ctx_hidden.shape[0]
        device = ctx_hidden.device
        # Stage-1 N=1: one action query at current time.
        query_time = torch.zeros(B, 1, device=device, dtype=torch.long)
        q_action = self.query_bank("action_target", query_time)  # (B,1,D)

        z_pred = self.predictor(
            context_tokens=ctx_hidden,
            context_valid=ctx_pack.valid,
            query_tokens=q_action,
        )[:, 0, :]  # (B,D)

        # 3) Target encoder: EMA encoder sees real action_target.
        with torch.no_grad():
            tgt_out = self.target_encoder(batch, branch="target", return_pack=True)
            tgt_hidden = tgt_out["hidden"]
            tgt_pack = tgt_out["pack"]
            assert isinstance(tgt_pack, TokenPack)
            z_tgt_tokens = self.target_encoder.extract_role_tokens(tgt_hidden, tgt_pack, "action_target")
            # Stage-1 H_a=1: exactly one action target token.
            z_tgt = z_tgt_tokens[:, 0, :]
            z_tgt = self.project_z(z_tgt, target=True)  # identity when z_dim == 0
            if self.normalize_targets:
                z_tgt = F.layer_norm(z_tgt, (z_tgt.shape[-1],))

        # Bottleneck the predicted latent into z space; the decoder consumes z too so the
        # compression is a real information bottleneck (identity when z_dim == 0).
        z_pred_z = self.project_z(z_pred)
        if self.normalize_targets:
            z_pred_for_loss = F.layer_norm(z_pred_z, (z_pred_z.shape[-1],))
        else:
            z_pred_for_loss = z_pred_z

        out = {
            "z_pred": z_pred_z,
            "z_pred_loss": z_pred_for_loss,
            "z_tgt": z_tgt,
        }
        if self.action_decoder is not None:
            out["action_pred"] = self.action_decoder(z_pred_z)
        return out

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        m = self.ema_momentum if momentum is None else momentum
        update_ema(self.online_encoder, self.target_encoder, m)
        if self.z_dim > 0:
            update_ema(self.z_head, self.z_head_target, m)

    @staticmethod
    def loss_fn(
        out: dict[str, torch.Tensor],
        action_target: Optional[torch.Tensor] = None,
        lambda_action: float = 0.0,
        jepa_loss: Literal["smooth_l1", "mse", "l1"] = "smooth_l1",
    ) -> dict[str, torch.Tensor]:
        if jepa_loss == "smooth_l1":
            loss_jepa = F.smooth_l1_loss(out["z_pred_loss"], out["z_tgt"].detach())
        elif jepa_loss == "mse":
            loss_jepa = F.mse_loss(out["z_pred_loss"], out["z_tgt"].detach())
        elif jepa_loss == "l1":
            loss_jepa = F.l1_loss(out["z_pred_loss"], out["z_tgt"].detach())
        else:
            raise ValueError(jepa_loss)
        loss_action = out["z_pred"].new_zeros(())
        if lambda_action != 0.0:
            if action_target is None:
                raise ValueError("action_target is required when lambda_action != 0.")
            if "action_pred" not in out:
                raise ValueError("action_pred is missing. Build CtrlJEPAStage1 with_action_decoder=True.")
            loss_action = F.mse_loss(out["action_pred"], action_target)
        loss = loss_jepa + lambda_action * loss_action
        return {"loss": loss, "loss_jepa": loss_jepa, "loss_action": loss_action}


# -----------------------------
# Minimal factory + smoke test
# -----------------------------

def make_stage1_specs(obs_dim: int, action_dim: int, cond_dim: int, proc_dim: int = 128) -> dict[str, ModalitySpec]:
    return {
        "base_obs": ModalitySpec(
            name="base_obs",
            role="base_obs",
            processor=VectorProcessor(obs_dim, proc_dim),
            out_dim=proc_dim,
        ),
        "history_action": ModalitySpec(
            name="history_action",
            role="history_action",
            processor=VectorProcessor(action_dim, proc_dim),
            out_dim=proc_dim,
        ),
        "condition": ModalitySpec(
            name="condition",
            role="condition",
            processor=VectorProcessor(cond_dim, proc_dim),
            out_dim=proc_dim,
        ),
        "action_target": ModalitySpec(
            name="action_target",
            role="action_target",
            processor=VectorProcessor(action_dim, proc_dim),
            out_dim=proc_dim,
            representation_source="history_action",
        ),
    }


def build_stage1_model(
    obs_dim: int,
    action_dim: int,
    cond_dim: int,
    proc_dim: int = 128,
    token_dim: int = 256,
    predictor_dim: int = 256,
    depth: int = 2,
    predictor_depth: int = 2,
    num_heads: int = 4,
    with_action_decoder: bool = False,
    z_dim: int = 0,
) -> CtrlJEPAStage1:
    specs = make_stage1_specs(obs_dim, action_dim, cond_dim, proc_dim=proc_dim)
    online_encoder = CtrlJEPAEncoder(
        specs=specs,
        token_dim=token_dim,
        max_time=1,
        depth=depth,
        num_heads=num_heads,
        use_time_embedding=False,
        use_group_embedding=False,
        use_causal_time_mask=False,
    )
    predictor = CtrlJEPAPredictor(
        encoder_dim=token_dim,
        predictor_dim=predictor_dim,
        out_dim=token_dim,
        depth=predictor_depth,
        num_heads=num_heads,
    )
    query_bank = TargetQueryBank(token_dim=token_dim, roles=["action_target"], max_time=1)
    # The decoder consumes the bottlenecked latent when z_dim > 0, else the token latent.
    decoder_latent_dim = z_dim if z_dim > 0 else token_dim
    decoder = ActionDecoder(latent_dim=decoder_latent_dim, action_dim=action_dim) if with_action_decoder else None
    return CtrlJEPAStage1(online_encoder, predictor, query_bank, decoder, z_dim=z_dim)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, obs_dim, action_dim, cond_dim = 2, 16, 4, 2
    model = build_stage1_model(obs_dim, action_dim, cond_dim, proc_dim=16, token_dim=32, predictor_dim=32, depth=1, predictor_depth=1, num_heads=4)
    batch = {
        "base_obs": torch.randn(B, obs_dim),
        "history_action": torch.randn(B, action_dim),
        "condition": torch.randn(B, cond_dim),
        "action_target": torch.randn(B, action_dim),
    }
    out = model(batch)
    losses = model.loss_fn(out)
    losses["loss"].backward()
    model.update_target_encoder(momentum=0.99)
    print({k: tuple(v.shape) for k, v in out.items()})
    print({k: float(v.detach()) for k, v in losses.items()})
