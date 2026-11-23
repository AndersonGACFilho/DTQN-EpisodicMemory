import unittest

import gym
import numpy as np

from envs.passive_tmaze import PassiveTMaze
from utils.env_processing import get_env_max_steps, make_env


class TestPassiveTMaze(unittest.TestCase):
    def test_oracle_and_episode_length(self) -> None:
        for length in (1, 10, 300):
            env = PassiveTMaze(corridor_length=length)
            env.seed(1)
            for _ in range(10):
                initial = env.reset()
                cue = initial[1]
                self.assertTrue(env.observation_space.contains(initial))
                for t in range(length):
                    obs, reward, done, info = env.step(0)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(reward, 0.0)
                    self.assertFalse(done)
                    self.assertFalse(info["is_success"])
                    np.testing.assert_array_equal(
                        obs, [1, 0] if t == length - 1 else [0, 0]
                    )
                obs, reward, done, info = env.step(1 if cue > 0 else 3)
                np.testing.assert_array_equal(obs, [1, cue])
                self.assertEqual(reward, 1.0)
                self.assertTrue(done)
                self.assertTrue(info["is_success"])
                self.assertEqual(env.time_step, length + 1)

    def test_wrong_branch_fails(self) -> None:
        env = PassiveTMaze(corridor_length=5)
        cue = env.reset()[1]
        for _ in range(5):
            env.step(0)
        _, reward, done, info = env.step(3 if cue > 0 else 1)
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertFalse(info["is_success"])

    def test_cue_is_not_repeated_and_delays_are_penalized(self) -> None:
        env = PassiveTMaze(corridor_length=5)
        env.reset()
        for action, expected_reward in [(1, -0.2), (0, -0.2), (2, -0.2)]:
            obs, reward, done, _ = env.step(action)
            np.testing.assert_array_equal(obs, [0, 0])
            self.assertAlmostEqual(reward, expected_reward)
            self.assertFalse(done)
        # Remaining in the corridor until the deadline must not count as success.
        for _ in range(3):
            _, reward, done, info = env.step(1)
        self.assertTrue(done)
        self.assertEqual(reward, 0.0)
        self.assertFalse(info["is_success"])

    def test_seed_reproduces_episode_cues(self) -> None:
        env = PassiveTMaze()
        env.seed(42)
        cues = [env.reset()[1] for _ in range(100)]
        env.seed(42)
        self.assertEqual(cues, [env.reset()[1] for _ in range(100)])
        self.assertEqual(set(cues), {-1, 1})

    def test_random_branch_and_oracle_sanity(self) -> None:
        env = PassiveTMaze(corridor_length=10)
        env.seed(1)
        rng = np.random.default_rng(2)
        for oracle in (False, True):
            successes = 0
            for _ in range(1000):
                cue = env.reset()[1]
                for _ in range(10):
                    env.step(0)
                action = (1 if cue > 0 else 3) if oracle else int(rng.choice([1, 3]))
                _, _, _, info = env.step(action)
                successes += info["is_success"]
            if oracle:
                self.assertEqual(successes, 1000)
            else:
                self.assertTrue(450 <= successes <= 550, successes)

    def test_registration_and_factory_set_the_horizon(self) -> None:
        env = gym.make("PassiveTMaze-v0")
        self.assertEqual(get_env_max_steps(env), 11)
        env.close()
        env = make_env("PassiveTMaze-v0", corridor_length=300)
        self.assertEqual(get_env_max_steps(env), 301)
        self.assertEqual(env.action_space.n, 4)
        self.assertEqual(env.observation_space.shape, (2,))
        env.close()
        # T-Maze options must not be passed to other environments.
        env = make_env("Memory-5-v0", corridor_length=300)
        self.assertEqual(get_env_max_steps(env), 50)
        env.close()

    def test_invalid_length_and_action(self) -> None:
        for length in (0, -1):
            with self.assertRaises(ValueError):
                PassiveTMaze(corridor_length=length)
        env = PassiveTMaze()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(4)


if __name__ == "__main__":
    unittest.main()
