"""Shared records, grid files and scripted T-Maze policies for rliable tests."""

import json
from pathlib import Path
from typing import Any, Dict, List

from envs.passive_tmaze import PassiveTMaze
from evaluation.job import RecordCollector
from evaluation.records import EpisodeRecord
from evaluation.runner import run_evaluation
from evaluation.sides import UP, read_tmaze_sides
from utils.experiment_config import RunConfig

LEGACY_EVAL_SEED = 10_000
LEGACY_EVAL_EPISODES = 50
TURN_UP, TURN_DOWN, MOVE_RIGHT = 1, 3, 0


class AlwaysDownPolicy:
    """Walks the corridor and always turns down, ignoring the cue."""

    def __init__(self, corridor_length: int) -> None:
        self.corridor_length = corridor_length
        self.steps = 0

    def reset(self, observation: Any) -> None:
        self.steps = 0

    def act(self) -> int:
        self.steps += 1
        return MOVE_RIGHT if self.steps <= self.corridor_length else TURN_DOWN

    def observe(self, observation: Any, action: int, reward: float, done: bool) -> None:
        pass


class CueFollowingPolicy(AlwaysDownPolicy):
    def reset(self, observation: Any) -> None:
        super().reset(observation)
        self.turn = TURN_UP if observation[1] > 0 else TURN_DOWN

    def act(self) -> int:
        return self.turn if super().act() == TURN_DOWN else MOVE_RIGHT


def record(**overrides: Any) -> EpisodeRecord:
    values: Dict[str, Any] = {
        "model": "DTQN",
        "env": "PassiveTMaze-v0",
        "context": 6,
        "layers": 2,
        "corridor_length": 10,
        "train_seed": 1,
        "eval_seed": 10_001,
        "step": 2000,
        "episode": 0,
        "success": True,
        "episode_return": 1.0,
        "episode_length": 11,
        "goal_side": UP,
        "chosen_side": UP,
    }
    values.update(overrides)
    return EpisodeRecord(**values)


def grid_file(directory: Path, **overrides: Any) -> Path:
    spec: Dict[str, Any] = {
        "parameters": {
            "model": ["DTQN", "DTQN-kNN"],
            "env": ["PassiveTMaze-v0"],
            "context": [2],
            "corridor_length": [3],
            "num_steps": [7],
        },
        "train_seeds": 3,
        "eval_episodes": 4,
        "eval_interval": 3,
        "comparisons": [["DTQN-kNN", "DTQN"]],
    }
    spec.update(overrides)
    path = directory / "grid.json"
    path.write_text(json.dumps(spec))
    return path


def evaluate_fixed_policy(policy: Any, corridor_length: int) -> List[EpisodeRecord]:
    config = RunConfig(
        env="PassiveTMaze-v0",
        corridor_length=corridor_length,
        eval_seed=LEGACY_EVAL_SEED,
        eval_episodes=LEGACY_EVAL_EPISODES,
    )
    env = PassiveTMaze(corridor_length)
    env.seed(config.eval_seed)
    collector = RecordCollector(config)
    run_evaluation(env, policy, collector, config.eval_episodes, read_tmaze_sides)
    return collector.records
