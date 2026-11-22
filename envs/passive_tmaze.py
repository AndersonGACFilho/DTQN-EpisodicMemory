"""Passive T-Maze adapted from twni2016/Memory-RL (Ni et al., 2023).

Source: https://github.com/twni2016/Memory-RL/blob/main/envs/tmaze.py
Rewards follow configs/envs/tmaze_passive.py: goal reward 1, delay penalty -1/L.

MIT License

Copyright (c) 2023 Tianwei Ni, Michel Ma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from numpy.typing import NDArray


class PassiveTMaze(gym.Env):  # type: ignore[type-arg]
    """Remember an initial cue and navigate to its branch within L + 1 steps.

    Actions are right, up, left, down (0, 1, 2, 3). Observations are [0, cue]
    on reset, [0, 0] along the corridor and [1, y] at the junction/branches.
    The cue is shown only once, including if the agent returns to the start.
    """

    def __init__(self, corridor_length: int = 10) -> None:
        if not isinstance(corridor_length, int) or corridor_length < 1:
            raise ValueError("corridor_length must be a positive integer")
        self.corridor_length = corridor_length
        self.max_episode_steps = corridor_length + 1
        self.penalty = -1.0 / corridor_length
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._rng = np.random.default_rng()
        self.x = 0
        self.y = 0
        self.time_step = 0
        self.goal_y = 1
        self._done = True

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        self._rng = np.random.default_rng(seed)
        return [seed]

    # Keep the legacy reset API used by this runner and Gym 0.18.
    def reset(self) -> NDArray[np.float32]:  # type: ignore[override]
        self.x = self.y = self.time_step = 0
        self.goal_y = int(self._rng.choice([-1, 1]))
        self._done = False
        return np.array([0, self.goal_y], dtype=np.float32)

    def step(
        self, action: int
    ) -> Tuple[NDArray[np.float32], float, bool, Dict[str, bool]]:
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.time_step += 1
        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[action]
        next_x, next_y = self.x + dx, self.y + dy
        on_corridor = next_y == 0 and 0 <= next_x <= self.corridor_length
        on_branch = next_x == self.corridor_length and abs(next_y) == 1
        if on_corridor or on_branch:
            self.x, self.y = next_x, next_y

        self._done = self.time_step >= self.max_episode_steps
        success = self._done and self.y == self.goal_y
        if self._done:
            reward = float(success)
        else:
            reward = self.penalty if self.x < self.time_step else 0.0
        obs = np.array(
            [1, self.y] if self.x == self.corridor_length else [0, 0],
            dtype=np.float32,
        )
        return obs, reward, self._done, {"is_success": success}

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("PassiveTMaze has no renderer")
