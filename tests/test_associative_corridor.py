import unittest
from typing import Dict

import gym
import numpy as np
from numpy.typing import NDArray

from envs.associative_corridor import AssociativeCorridor
from utils.env_processing import get_env_max_steps, get_episode_success, make_env


def oracle_action(obs: NDArray[np.float32], memory: Dict[int, int]) -> int:
    """Read instructions and answer queries using observations only."""
    if obs[:4].any():
        label = int(np.argmax(obs[:4]))
        if obs[4]:
            return 1 if memory[label] > 0 else 3
        memory[label] = int(obs[5])
    return 0


class TestAssociativeCorridor(unittest.TestCase):
    def test_observation_driven_oracle_succeeds_in_1000_episodes(self) -> None:
        env = AssociativeCorridor(corridor_length=20)
        env.seed(7)
        for _ in range(1000):
            memory: Dict[int, int] = {}
            obs = env.reset()
            total_reward = 0.0
            for step in range(20):
                self.assertEqual(obs.dtype, np.float32)
                self.assertTrue(env.observation_space.contains(obs))
                is_query = bool(obs[4])
                if is_query:
                    self.assertEqual(obs[:4].sum(), 1)
                    self.assertEqual(obs[5], 0)
                obs, reward, done, info = env.step(oracle_action(obs, memory))
                self.assertEqual(reward, float(is_query))
                self.assertEqual(done, step == 19)
                self.assertEqual(info["is_success"], done)
                total_reward += reward
            self.assertEqual(len(memory), 4)
            self.assertEqual(total_reward, 4)
            self.assertEqual(env.correct_answers, 4)
            np.testing.assert_array_equal(obs, np.zeros(6))

    def test_schedule_has_four_pairs_can_interleave_and_changes(self) -> None:
        env = AssociativeCorridor(corridor_length=20)
        env.seed(42)
        schedules = set()
        mappings = set()
        last_classes = set()
        interleaved = False
        for _ in range(100):
            env.reset()
            self.assertEqual(len(env.instructions), 4)
            self.assertEqual(len(env.queries), 4)
            self.assertTrue(set(env.instructions).isdisjoint(env.queries))
            self.assertEqual({c for c, _ in env.instructions.values()}, set(range(4)))
            self.assertEqual(set(env.queries.values()), set(range(4)))
            self.assertIn(19, env.queries)
            for write, (label, direction) in env.instructions.items():
                query = next(p for p, c in env.queries.items() if c == label)
                self.assertLess(write, query)
                self.assertIn(direction, (-1, 1))
            interleaved |= min(env.queries) < max(env.instructions)
            schedules.add(tuple(sorted(env.queries.items())))
            mappings.add(tuple(env.directions))
            last_classes.add(env.queries[19])
        self.assertTrue(interleaved)
        self.assertGreater(len(schedules), 1)
        # Independent directions allow equal answers as well as different ones.
        self.assertEqual(len(mappings), 16)
        self.assertEqual(last_classes, set(range(4)))

    def test_seed_reproduces_complete_episodes(self) -> None:
        env = AssociativeCorridor(corridor_length=12)
        runs = []
        for _ in range(2):
            env.seed(123)
            observations = []
            for _ in range(10):
                memory: Dict[int, int] = {}
                obs = env.reset()
                done = False
                while not done:
                    observations.append(obs.copy())
                    obs, _, done, _ = env.step(oracle_action(obs, memory))
            runs.append(np.stack(observations))
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_wrong_corridor_action_ends_immediately(self) -> None:
        env = AssociativeCorridor()
        for action in (1, 2, 3):
            env.seed(1)
            obs = env.reset()
            self.assertEqual(obs[4], 0)
            _, reward, done, info = env.step(action)
            self.assertEqual(reward, 0)
            self.assertTrue(done)
            self.assertFalse(info["is_success"])
            self.assertEqual(env.time_step, 1)
            with self.assertRaises(RuntimeError):
                env.step(0)

    def test_wrong_answer_after_partial_reward_is_failure(self) -> None:
        env = AssociativeCorridor(corridor_length=12)
        env.seed(8)
        memory: Dict[int, int] = {}
        obs = env.reset()
        total_reward = 0.0
        while True:
            expected = oracle_action(obs, memory)
            if obs[4] and total_reward > 0:
                _, reward, done, info = env.step(3 if expected == 1 else 1)
                self.assertEqual(reward, 0)
                self.assertTrue(done)
                self.assertFalse(info["is_success"])
                self.assertGreater(total_reward, 0)
                self.assertTrue(env.requires_explicit_success)
                self.assertFalse(get_episode_success(info, total_reward, True))
                break
            obs, reward, done, _ = env.step(expected)
            total_reward += reward
            self.assertFalse(done)

    def test_right_and_left_do_not_answer_queries(self) -> None:
        for action in (0, 2):
            env = AssociativeCorridor(corridor_length=8)
            env.seed(1)
            memory: Dict[int, int] = {}
            obs = env.reset()
            while not obs[4]:
                obs, _, _, _ = env.step(oracle_action(obs, memory))
            _, reward, done, info = env.step(action)
            self.assertEqual(reward, 0)
            self.assertTrue(done)
            self.assertFalse(info["is_success"])

    def test_registration_horizons_and_unchanged_factories(self) -> None:
        env = gym.make("AssociativeCorridor-v0")
        self.assertEqual(get_env_max_steps(env), 10)
        self.assertEqual(env.observation_space.shape, (6,))
        env.close()
        for length in (8, 300):
            env = make_env("AssociativeCorridor-v0", corridor_length=length)
            memory: Dict[int, int] = {}
            obs = env.reset()
            for step in range(length):
                obs, _, done, info = env.step(oracle_action(obs, memory))
                self.assertEqual(done, step == length - 1)
            self.assertTrue(info["is_success"])
            self.assertEqual(get_env_max_steps(env), length)
            env.close()
        for env_id, horizon in (("PassiveTMaze-v0", 301), ("Memory-5-v0", 50)):
            env = make_env(env_id, corridor_length=300)
            self.assertEqual(get_env_max_steps(env), horizon)
            env.close()

    def test_invalid_length_action_and_lifecycle(self) -> None:
        for length in (-1, 0, 7, 8.5):
            with self.assertRaises(ValueError):
                AssociativeCorridor(corridor_length=length)  # type: ignore[arg-type]
        env = AssociativeCorridor()
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(4)


class TestEpisodeSuccess(unittest.TestCase):
    def test_legacy_success_rule_is_preserved(self) -> None:
        for info, episode_return, expected in (
            ({}, 1.0, True),
            ({}, 0.0, False),
            ({"is_success": False}, 3.0, True),
            ({"is_success": True}, -1.0, True),
            ({"is_success": False}, -1.0, False),
        ):
            with self.subTest(info=info, episode_return=episode_return):
                self.assertEqual(get_episode_success(info, episode_return), expected)

    def test_explicit_success_overrides_partial_return_only_when_opted_in(self) -> None:
        self.assertFalse(get_episode_success({"is_success": False}, 3, True))
        self.assertTrue(get_episode_success({"is_success": True}, 4, True))
        self.assertFalse(get_episode_success({}, 3, True))
