"""Greedy evaluation episodes against any policy; knows nothing about training."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from evaluation.sides import SideReader
from utils.env_processing import get_episode_success


class Policy(Protocol):
    def reset(self, observation: Any) -> None: ...

    def act(self) -> int: ...

    def observe(
        self, observation: Any, action: int, reward: float, done: bool
    ) -> None: ...


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    episode_return: float
    episode_length: int
    goal_side: Optional[str]
    chosen_side: Optional[str]


class EpisodeWriter(Protocol):
    def write(self, episode: int, outcome: EpisodeOutcome) -> None: ...


def run_episode(env: Any, policy: Policy, read_sides: SideReader) -> EpisodeOutcome:
    observation = env.reset()
    policy.reset(observation)
    episode_return, episode_length, done = 0.0, 0, False
    info: Dict[str, Any] = {}
    while not done:
        action = policy.act()
        observation, reward, done, info = env.step(action)
        truncated = bool(info.get("TimeLimit.truncated", False))
        policy.observe(observation, action, reward, bool(done and not truncated))
        episode_return += float(reward)
        episode_length += 1
    goal_side, chosen_side = read_sides(env)
    success = get_episode_success(
        info,
        episode_return,
        getattr(env.unwrapped, "requires_explicit_success", False),
    )
    return EpisodeOutcome(
        success=bool(success),
        episode_return=episode_return,
        episode_length=episode_length,
        goal_side=goal_side,
        chosen_side=chosen_side,
    )


def run_evaluation(
    env: Any,
    policy: Policy,
    writer: EpisodeWriter,
    episode_count: int,
    read_sides: SideReader,
) -> None:
    for episode in range(episode_count):
        writer.write(episode, run_episode(env, policy, read_sides))
