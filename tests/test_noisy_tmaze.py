import unittest

import gym
import numpy as np

from envs.noisy_tmaze import NoisyTMaze
from envs.passive_tmaze import PassiveTMaze
from utils.env_processing import get_env_max_steps, make_env


class TestNoisyTMaze(unittest.TestCase):
    def test_zero_noise_matches_baseline_across_seeded_episodes(self) -> None:
        for length in (1, 10, 300):
            baseline = PassiveTMaze(length)
            noisy = NoisyTMaze(length, noise_level=0)
            baseline.seed(42)
            noisy.seed(42)
            assert isinstance(baseline.observation_space, gym.spaces.Box)
            assert isinstance(noisy.observation_space, gym.spaces.Box)
            np.testing.assert_array_equal(
                noisy.observation_space.low, baseline.observation_space.low
            )
            np.testing.assert_array_equal(
                noisy.observation_space.high, baseline.observation_space.high
            )
            self.assertEqual(
                noisy.observation_space.dtype, baseline.observation_space.dtype
            )
            self.assertEqual(noisy.max_episode_steps, baseline.max_episode_steps)
            rng = np.random.default_rng(19)
            for episode in range(30):
                cue = baseline.reset()
                np.testing.assert_array_equal(cue, noisy.reset())
                actions = [0] * length + [1 if cue[1] > 0 else 3]
                if episode % 3 == 1:
                    actions[-1] = 3 if cue[1] > 0 else 1
                elif episode % 3 == 2:
                    actions = list(rng.integers(4, size=length + 1))
                for action in actions:
                    obs, reward, done, info = baseline.step(action)
                    other_obs, other_reward, other_done, other_info = noisy.step(action)
                    np.testing.assert_array_equal(obs, other_obs)
                    self.assertEqual(
                        (reward, done, info), (other_reward, other_done, other_info)
                    )

    def test_noise_does_not_change_cues_dynamics_rewards_or_clean_locations(
        self,
    ) -> None:
        for level in (0.25, 1.0):
            baseline = PassiveTMaze(10)
            noisy = NoisyTMaze(10, noise_level=level, noise_std=2)
            baseline.seed(5)
            noisy.seed(5)
            for episode in range(30):
                initial = baseline.reset()
                np.testing.assert_array_equal(initial, noisy.reset())
                actions = [0] * 10 + [1 if initial[1] > 0 else 3]
                if episode % 2:
                    actions[:3] = [1, 0, 2]
                for action in actions:
                    obs, reward, done, info = baseline.step(action)
                    noisy_obs, other_reward, other_done, other_info = noisy.step(action)
                    self.assertEqual(
                        (reward, done, info), (other_reward, other_done, other_info)
                    )
                    self.assertEqual((baseline.x, baseline.y), (noisy.x, noisy.y))
                    self.assertEqual(obs[0], noisy_obs[0])
                    self.assertTrue(noisy.observation_space.contains(noisy_obs))
                    if baseline.x == 10:
                        np.testing.assert_array_equal(obs, noisy_obs)

    def test_oracle_succeeds_in_1000_episodes_at_each_noise_level(self) -> None:
        for level in (0.0, 0.25, 0.5, 1.0):
            for std in (0.1, 1.0):
                with self.subTest(noise_level=level, noise_std=std):
                    env = NoisyTMaze(10, noise_level=level, noise_std=std)
                    env.seed(1)
                    successes = 0
                    for _ in range(1000):
                        cue = env.reset()[1]
                        for t in range(10):
                            obs, reward, done, _ = env.step(0)
                            self.assertEqual(reward, 0)
                            self.assertFalse(done)
                            if t == 9:
                                np.testing.assert_array_equal(obs, [1, 0])
                        obs, reward, done, info = env.step(1 if cue > 0 else 3)
                        np.testing.assert_array_equal(obs, [1, cue])
                        self.assertEqual(reward, 1)
                        self.assertTrue(done)
                        successes += info["is_success"]
                    self.assertEqual(successes, 1000)

    def test_oracle_for_short_and_long_corridors(self) -> None:
        for length in (1, 300):
            env = NoisyTMaze(length, noise_level=1, noise_std=1)
            cue = env.reset()[1]
            for _ in range(length):
                env.step(0)
            _, reward, done, info = env.step(1 if cue > 0 else 3)
            self.assertEqual(reward, 1)
            self.assertTrue(done)
            self.assertTrue(info["is_success"])

    def test_gaussian_modes_frequency_and_unclipped_values(self) -> None:
        env = NoisyTMaze(20001, noise_level=0.4, noise_std=0.1)
        env.seed(123)
        env.reset()
        samples = np.array([env.step(0)[0][1] for _ in range(20000)])
        noisy = samples[samples != 0]
        self.assertAlmostEqual(len(noisy) / len(samples), 0.4, delta=0.015)
        self.assertAlmostEqual(float(np.mean(noisy > 0)), 0.5, delta=0.025)
        for mean in (-1, 1):
            mode = noisy[noisy * mean > 0]
            self.assertAlmostEqual(float(mode.mean()), mean, delta=0.01)
            self.assertAlmostEqual(float(mode.std()), 0.1, delta=0.01)
        self.assertTrue(np.any(noisy > 1))
        self.assertTrue(np.any(noisy < -1))
        for value in samples:
            self.assertTrue(
                env.observation_space.contains(np.array([0, value], dtype=np.float32))
            )

    def test_seed_reproduces_noise_and_episode_cues(self) -> None:
        env = NoisyTMaze(10, noise_level=0.5)
        runs = []
        for _ in range(2):
            env.seed(42)
            observations = []
            for _ in range(10):
                observations.append(env.reset())
                for _ in range(11):
                    observations.append(env.step(0)[0])
            runs.append(np.stack(observations))
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_registration_factory_and_options_do_not_affect_other_envs(self) -> None:
        env = gym.make("NoisyTMaze-v0")
        self.assertEqual(get_env_max_steps(env), 11)
        self.assertEqual(env.observation_space.shape, (2,))
        env.close()
        env = make_env("NoisyTMaze-v0", 300, noise_level=0.75, noise_std=0.3)
        self.assertEqual(get_env_max_steps(env), 301)
        self.assertEqual(env.unwrapped.noise_level, 0.75)
        self.assertEqual(env.unwrapped.noise_std, 0.3)
        env.close()
        for env_id, horizon in (("PassiveTMaze-v0", 301), ("Memory-5-v0", 50)):
            env = make_env(env_id, 300, noise_level=1, noise_std=2)
            self.assertEqual(get_env_max_steps(env), horizon)
            self.assertFalse(hasattr(env.unwrapped, "noise_level"))
            env.close()

    def test_invalid_parameters_actions_and_lifecycle(self) -> None:
        for level in (-0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                NoisyTMaze(noise_level=level)
        for std in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                NoisyTMaze(noise_std=std)
        for length in (0, -1):
            with self.assertRaises(ValueError):
                NoisyTMaze(corridor_length=length)
        env = NoisyTMaze(1)
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(4)
        env.step(0)
        env.step(1)
        with self.assertRaises(RuntimeError):
            env.step(0)
