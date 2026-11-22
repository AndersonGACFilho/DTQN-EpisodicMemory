"""Scientific parameters shared by the single-run CLIs and the sweep runner."""

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass
class RunConfig:
    model: str = "DTQN"
    env: str = "DiscreteCarFlag-v0"
    corridor_length: int = 10
    noise_level: float = 0.5
    noise_std: float = 0.1
    num_steps: int = 2_000_000
    prefill_steps: int = 50_000
    tuf: int = 10_000
    lr: float = 3e-4
    batch: int = 32
    buf_size: int = 500_000
    context: int = 50
    obs_embed: int = 8
    a_embed: int = 0
    in_embed: int = 128
    max_episode_steps: int = -1
    training_seed: int = 1
    history: int = 50
    heads: int = 8
    layers: int = 2
    dropout: float = 0.0
    discount: float = 0.99
    gate: str = "res"
    identity: bool = False
    pos: str = "learned"
    bag_size: int = 0
    eval_episodes: int = 100
    eval_seed: int = 10_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        for key, value in self.to_dict().items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.model not in {"DTQN", "DTQN-bag", "ADRQN", "DRQN", "DARQN", "DQN"}:
            raise ValueError(f"Unknown model: {self.model}")
        if not self.env:
            raise ValueError("env must not be empty")
        for key in (
            "num_steps",
            "tuf",
            "lr",
            "batch",
            "buf_size",
            "context",
            "obs_embed",
            "in_embed",
            "history",
            "heads",
            "layers",
            "eval_episodes",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in (
            "prefill_steps",
            "a_embed",
            "bag_size",
            "training_seed",
            "eval_seed",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.max_episode_steps != -1 and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be -1 (infer) or positive")
        if not 0 <= self.dropout < 1 or not 0 <= self.discount <= 1:
            raise ValueError("dropout must be in [0, 1); discount in [0, 1]")
        if self.gate not in {"res", "gru"} or self.pos not in {
            "learned",
            "sin",
            "none",
        }:
            raise ValueError("Invalid gate or positional encoding")
        if self.model.startswith("DTQN"):
            if self.in_embed % self.heads:
                raise ValueError("in_embed must be divisible by heads")
            if self.a_embed >= self.in_embed:
                raise ValueError("a_embed must be smaller than in_embed")
        if self.model in {"DQN", "DRQN", "DARQN", "ADRQN"} and self.bag_size:
            raise ValueError("bag_size is supported only for DTQN models")
        if self.env in {"PassiveTMaze-v0", "NoisyTMaze-v0", "AssociativeCorridor-v0"}:
            minimum = 8 if self.env == "AssociativeCorridor-v0" else 1
            if self.corridor_length < minimum:
                raise ValueError(f"corridor_length must be >= {minimum} for {self.env}")
        if self.env == "NoisyTMaze-v0":
            if not 0 <= self.noise_level <= 1 or self.noise_std <= 0:
                raise ValueError("noise_level must be in [0, 1]; noise_std positive")


DEFAULTS = RunConfig().to_dict()
REQUIRED_GRID_KEYS = {
    "model",
    "env",
    "training_seed",
    "num_steps",
    "eval_episodes",
    "eval_seed",
}


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    for field in fields(RunConfig):
        default = DEFAULTS[field.name]
        kwargs: Dict[str, Any] = {
            "default": default,
            "help": f"{field.name.replace('_', ' ')} (default: {default})",
        }
        if isinstance(default, bool):
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = type(default)
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(**{key: getattr(args, key) for key in DEFAULTS})
    config.validate()
    return config


def config_cli(params: Mapping[str, Any]) -> List[str]:
    """Concrete CLI arguments; train/eval never read the sweep JSON."""
    result: List[str] = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                result.append("--" + key.replace("_", "-"))
        else:
            result.extend(["--" + key.replace("_", "-"), str(value)])
    return result


def normalize_saved_parameters(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep pre-rename model bundles and saved results readable."""
    result = dict(params)
    if "seed" in result:
        if "training_seed" in result:
            raise ValueError("Saved parameters contain both seed and training_seed")
        result["training_seed"] = result.pop("seed")
    return result


def unique_object(pairs: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_grid(path: Path) -> Dict[str, List[Any]]:
    raw = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if not isinstance(raw, dict):
        raise ValueError("The config must be a flat JSON object of nonempty lists")
    unknown, missing = set(raw) - set(DEFAULTS), REQUIRED_GRID_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            f"Unknown keys: {sorted(unknown)}; missing keys: {sorted(missing)}"
        )
    for key, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        expected = type(DEFAULTS[key])
        normalized = []
        for value in values:
            valid = type(value) is expected or (
                expected is float and type(value) is int
            )
            if not valid or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError(
                    f"Invalid type or nonfinite value for {key}: {value!r}"
                )
            value = float(value) if expected is float else value
            if value in normalized:
                raise ValueError(f"Duplicate value in {key}: {value!r}")
            normalized.append(value)
        raw[key] = normalized
    return dict(raw)


def expand_grid(grid: Mapping[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = sorted(grid)
    return [
        {**DEFAULTS, **dict(zip(keys, values))}
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def configuration_id(
    params: Mapping[str, Any], *, include_training_seed: bool = False
) -> str:
    selected = {
        k: v for k, v in params.items() if include_training_seed or k != "training_seed"
    }
    encoded = json.dumps(selected, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
