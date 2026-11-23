"""Multi-seed grid: the single source of truth for configurations and seeds."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.records import metadata_path, records_path
from utils.experiment_config import (
    DEFAULTS,
    RunConfig,
    configuration_id,
    expand_grid,
    parse_grid,
    read_unique_json,
)

EVAL_SEED_OFFSET = 10_000
FIRST_TRAIN_SEED = 1
# Owned by the evaluation protocol below, so they cannot appear in "parameters"
PROTOCOL_KEYS = {"training_seed", "eval_seed", "eval_episodes"}
REQUIRED_PARAMETERS = {"model", "env", "num_steps"}
SPEC_KEYS = {
    "parameters",
    "train_seeds",
    "eval_episodes",
    "eval_interval",
    "comparisons",
}


def derive_eval_seed(train_seed: int) -> int:
    """Each run gets its own evaluation episodes, disjoint from training seeds."""
    return EVAL_SEED_OFFSET + train_seed


@dataclass(frozen=True)
class GridSpec:
    parameters: Dict[str, List[Any]]
    train_seed_count: int
    eval_episodes: int
    eval_interval: int
    # (candidate model, baseline model) pairs for probability of improvement
    comparisons: List[Tuple[str, str]]

    @property
    def train_seeds(self) -> List[int]:
        return list(range(FIRST_TRAIN_SEED, FIRST_TRAIN_SEED + self.train_seed_count))


@dataclass(frozen=True)
class RunJob:
    config: RunConfig
    eval_interval: int

    @property
    def run_id(self) -> str:
        params = {**self.config.to_dict(), "eval_interval": self.eval_interval}
        digest = configuration_id(params, include_training_seed=True)
        config = self.config
        return (
            f"{config.model}-{config.env}-L{config.corridor_length}"
            f"-ctx{config.context}-layers{config.layers}"
            f"-seed{config.training_seed}-{digest}"
        )


def _positive_int(raw: Dict[str, Any], key: str) -> int:
    value = raw[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def _comparisons(raw: Any, models: List[Any]) -> List[Tuple[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("comparisons must be a list of [candidate, baseline] pairs")
    pairs = []
    for pair in raw:
        if not (isinstance(pair, list) and len(pair) == 2):
            raise ValueError(
                f"Invalid comparison {pair!r}; expected [candidate, baseline]"
            )
        unknown = [model for model in pair if model not in models]
        if unknown:
            raise ValueError(f"Comparison uses models outside the grid: {unknown}")
        pairs.append((str(pair[0]), str(pair[1])))
    return pairs


def load_grid_spec(path: Path, train_seed_count: Optional[int] = None) -> GridSpec:
    raw = read_unique_json(path)
    if not isinstance(raw, dict) or set(raw) != SPEC_KEYS:
        raise ValueError(
            f"{path} must be an object with exactly the keys {sorted(SPEC_KEYS)}"
        )
    owned = PROTOCOL_KEYS & set(raw["parameters"])
    if owned:
        raise ValueError(
            f"Set {sorted(owned)} through the protocol keys, not parameters"
        )
    parameters = parse_grid(raw["parameters"], REQUIRED_PARAMETERS)
    spec = GridSpec(
        parameters=parameters,
        train_seed_count=_positive_int(raw, "train_seeds"),
        eval_episodes=_positive_int(raw, "eval_episodes"),
        eval_interval=_positive_int(raw, "eval_interval"),
        comparisons=_comparisons(raw["comparisons"], parameters["model"]),
    )
    if train_seed_count is not None:
        if train_seed_count <= 0:
            raise ValueError("The number of training seeds must be positive")
        spec = replace(spec, train_seed_count=train_seed_count)
    return spec


def expand_jobs(spec: GridSpec) -> List[RunJob]:
    jobs = []
    for params in expand_grid(spec.parameters):
        for train_seed in spec.train_seeds:
            config = RunConfig(
                **{
                    **params,
                    "training_seed": train_seed,
                    "eval_seed": derive_eval_seed(train_seed),
                    "eval_episodes": spec.eval_episodes,
                }
            )
            config.validate()
            jobs.append(RunJob(config, spec.eval_interval))
    return jobs


def parse_filters(items: List[str]) -> Dict[str, Any]:
    """Turn KEY=VALUE strings into typed RunConfig values."""
    filters: Dict[str, Any] = {}
    for item in items:
        key, separator, text = item.partition("=")
        if not separator or key not in DEFAULTS:
            raise ValueError(f"Invalid filter {item!r}; expected RunConfig KEY=VALUE")
        kind = type(DEFAULTS[key])
        filters[key] = text.lower() == "true" if kind is bool else kind(text)
    return filters


def select_jobs(jobs: List[RunJob], filters: Dict[str, Any]) -> List[RunJob]:
    """Keep jobs whose configuration matches every filter, e.g. to split machines."""
    return [
        job
        for job in jobs
        if all(getattr(job.config, key) == value for key, value in filters.items())
    ]


def is_complete(job: RunJob, raw_dir: Path) -> bool:
    """Metadata is written last, so its presence marks a finished run."""
    return (
        records_path(raw_dir, job.run_id).is_file()
        and metadata_path(raw_dir, job.run_id).is_file()
    )
