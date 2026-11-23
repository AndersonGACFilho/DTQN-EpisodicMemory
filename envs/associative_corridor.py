"""Four independent class-direction associations, recalled later in a corridor."""

from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from numpy.typing import NDArray


class AssociativeCorridor(gym.Env):  # type: ignore[type-arg]
    """Write and use each of four instructions once, with interleaving allowed.

    Observations are [A, B, C, D, is_query, direction]. Directions are +/-1
    only when an instruction is presented; queries and blanks contain zero.
    Right advances outside queries. Up/down answers a query and advances.
    Any incorrect action ends the episode, even after earlier correct answers.
    """

    requires_explicit_success = True

    def __init__(self, corridor_length: int = 10) -> None:
        if not isinstance(corridor_length, int) or corridor_length < 8:
            raise ValueError("corridor_length must be an integer of at least 8")
        self.corridor_length = corridor_length
        self.max_episode_steps = corridor_length
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1], dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self.instructions: Dict[int, Tuple[int, int]] = {}
        self.queries: Dict[int, int] = {}
        self.directions = np.zeros(4, dtype=np.int64)
        self.position = 0
        self.time_step = 0
        self.correct_answers = 0
        self._done = True

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(self) -> NDArray[np.float32]:  # type: ignore[override]
        # The final position is included, so the last action is always a query.
        positions = np.append(
            self._rng.choice(self.corridor_length - 1, size=7, replace=False),
            self.corridor_length - 1,
        )
        self._rng.shuffle(positions)
        self.directions = self._rng.choice([-1, 1], size=4)
        self.instructions = {}
        self.queries = {}
        for label in range(4):
            write, query = sorted(int(p) for p in positions[2 * label : 2 * label + 2])
            self.instructions[write] = (label, int(self.directions[label]))
            self.queries[query] = label
        self.position = self.time_step = self.correct_answers = 0
        self._done = False
        return self._observation()

    def _observation(self) -> NDArray[np.float32]:
        obs = np.zeros(6, dtype=np.float32)
        if self._done:
            return obs
        if self.position in self.instructions:
            label, direction = self.instructions[self.position]
            obs[label] = 1
            obs[5] = direction
        elif self.position in self.queries:
            obs[self.queries[self.position]] = 1
            obs[4] = 1
        return obs

    def step(
        self, action: int
    ) -> Tuple[NDArray[np.float32], float, bool, Dict[str, bool]]:
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        is_query = self.position in self.queries
        expected = 0
        if is_query:
            direction = self.directions[self.queries[self.position]]
            expected = 1 if direction > 0 else 3
        correct = action == expected
        self.time_step += 1
        reward = float(is_query and correct)
        if correct:
            self.position += 1
            self.correct_answers += int(is_query)
        self._done = not correct or self.time_step >= self.max_episode_steps
        success = (
            self._done
            and self.position == self.corridor_length
            and self.correct_answers == 4
        )
        return self._observation(), reward, self._done, {"is_success": success}

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("AssociativeCorridor has no renderer")
