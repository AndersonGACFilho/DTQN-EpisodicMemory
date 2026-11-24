"""Train one grid run with periodic greedy evaluation and save its tidy records."""

import logging
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

from evaluation.grid import RunJob
from evaluation.records import (
    EpisodeRecord,
    metadata_path,
    records_path,
    write_records,
)
from evaluation.runner import EpisodeOutcome, run_evaluation
from evaluation.sides import side_reader_for
from utils.experiment import make_environment, train_run
from utils.experiment_config import RunConfig
from utils.experiment_io import write_json
from utils.random import RNG

logger = logging.getLogger(__name__)

ATTEMPT_PREFIX = "attempt-"
SAVED_TRAINING_FIELDS = (
    "code",
    "versions",
    "parameter_count",
    "training_steps",
    "gradient_updates",
    "wall_clock_seconds",
    "device",
)


class AgentPolicy:
    """Greedy view of any agent built by utils.agent_utils.get_agent."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def reset(self, observation: Any) -> None:
        self.agent.context_reset(observation)

    def act(self) -> int:
        return int(self.agent.get_action(epsilon=0.0))

    def observe(self, observation: Any, action: int, reward: float, done: bool) -> None:
        self.agent.observe(observation, action, reward, done)


@contextmanager
def preserved_random_state() -> Iterator[None]:
    """Evaluating mid-training must leave every training random stream untouched."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    generator_state = RNG.rng.bit_generator.state
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        RNG.rng.bit_generator.state = generator_state


class RecordCollector:
    """Tags runner outcomes with the run identity and the current training step."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.step = 0
        self.records: List[EpisodeRecord] = []

    def write(self, episode: int, outcome: EpisodeOutcome) -> None:
        self.records.append(
            EpisodeRecord(
                model=self.config.model,
                env=self.config.env,
                context=self.config.context,
                layers=self.config.layers,
                corridor_length=self.config.corridor_length,
                train_seed=self.config.training_seed,
                eval_seed=self.config.eval_seed,
                step=self.step,
                episode=episode,
                success=outcome.success,
                episode_return=outcome.episode_return,
                episode_length=outcome.episode_length,
                goal_side=outcome.goal_side,
                chosen_side=outcome.chosen_side,
            )
        )


class PeriodicEvaluator:
    """Training step hook: evaluate every `interval` steps and at the final step."""

    def __init__(
        self, env: Any, config: RunConfig, interval: int, collector: RecordCollector
    ) -> None:
        self.env = env
        self.config = config
        self.interval = interval
        self.collector = collector

    def __call__(self, agent: Any, step: int) -> None:
        if step % self.interval == 0 or step == self.config.num_steps:
            self.evaluate(agent, step)

    def evaluate(self, agent: Any, step: int) -> None:
        self.collector.step = step
        # The same evaluation episodes at every step make the curve comparable
        self.env.seed(self.config.eval_seed)
        with preserved_random_state(), torch.no_grad():
            agent.eval_on()
            try:
                run_evaluation(
                    self.env,
                    AgentPolicy(agent),
                    self.collector,
                    self.config.eval_episodes,
                    side_reader_for(self.config.env),
                )
            finally:
                agent.eval_off()


def next_attempt_dir(run_dir: Path) -> Path:
    """Never reuse the folder of an interrupted attempt."""
    attempts = list(run_dir.glob(ATTEMPT_PREFIX + "*")) if run_dir.is_dir() else []
    return run_dir / f"{ATTEMPT_PREFIX}{len(attempts) + 1:03d}"


def run_job(
    job: RunJob,
    raw_dir: Path,
    artifacts_dir: Path,
    device: str,
    execution: Optional[Dict[str, Any]] = None,
) -> None:
    config = job.config
    artifacts = next_attempt_dir(artifacts_dir / job.run_id)
    logger.info("Training %s in %s", job.run_id, artifacts)
    collector = RecordCollector(config)
    eval_env, _ = make_environment(config)
    try:
        training = train_run(
            config,
            artifacts,
            device,
            step_hook=PeriodicEvaluator(eval_env, config, job.eval_interval, collector),
        )
    finally:
        eval_env.close()
    write_records(records_path(raw_dir, job.run_id), collector.records)
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": job.run_id,
        "config": config.to_dict(),
        "eval_interval": job.eval_interval,
        "eval_steps": sorted({record.step for record in collector.records}),
        "artifacts": str(artifacts.resolve()),
        "execution": execution or {},
        **{key: training[key] for key in SAVED_TRAINING_FIELDS},
    }
    write_json(metadata_path(raw_dir, job.run_id), metadata)
    logger.info("Saved %d records for %s", len(collector.records), job.run_id)
