"""Passive T-Maze with an independent mixture of Gaussian corridor distractors."""

import math
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from numpy.typing import NDArray

from envs.passive_tmaze import PassiveTMaze


class NoisyTMaze(PassiveTMaze):
    """Replace the corridor's zero value with a Gaussian distractor with probability p.

    Distractors use equally likely means -1 and +1 and standard deviation sigma.
    The initial cue, junction and branches, movement, rewards and horizon stay
    unchanged. At p=0 the observation space and trajectories match PassiveTMaze.
    """

    def __init__(
        self,
        corridor_length: int = 10,
        noise_level: float = 0.5,
        noise_std: float = 0.1,
    ) -> None:
        if not math.isfinite(noise_level) or not 0 <= noise_level <= 1:
            raise ValueError("noise_level must be finite and between 0 and 1")
        if not math.isfinite(noise_std) or noise_std <= 0:
            raise ValueError("noise_std must be finite and positive")
        super().__init__(corridor_length)
        self.noise_level = float(noise_level)
        self.noise_std = float(noise_std)
        self._noise_rng = np.random.default_rng()
        if self.noise_level > 0:
            self.observation_space = gym.spaces.Box(
                low=np.array([-1, -np.inf], dtype=np.float32),
                high=np.array([1, np.inf], dtype=np.float32),
                dtype=np.float32,
            )

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        seeds = super().seed(seed)
        # A child stream leaves the baseline goal RNG untouched and avoids
        # correlating distractors with the goal through an identical RNG stream.
        self._noise_rng = np.random.default_rng(
            np.random.SeedSequence(seed).spawn(1)[0]
        )
        return seeds

    def step(
        self, action: int
    ) -> Tuple[NDArray[np.float32], float, bool, Dict[str, bool]]:
        obs, reward, done, info = super().step(action)
        if (
            self.noise_level > 0
            and self.x < self.corridor_length
            and self._noise_rng.random() < self.noise_level
        ):
            mean = -1.0 if self._noise_rng.integers(2) == 0 else 1.0
            obs[1] = self._noise_rng.normal(mean, self.noise_std)
        return obs, reward, done, info
