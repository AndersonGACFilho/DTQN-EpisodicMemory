"""Collection-time cache immutability, causal replay, training and persistence."""

import argparse
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Tuple, cast
from unittest.mock import patch

import numpy as np
import torch
from numpy.typing import NDArray

from dtqn.agents.dtqn import DtqnAgent
from dtqn.agents.dtqn_knn_cached import CachedKnnDtqnAgent
from dtqn.buffers.cached_replay_buffer import CachedReplayBuffer
from dtqn.networks.dtqn_knn import DTQNKNN
from envs.passive_tmaze import PassiveTMaze
from tests.test_knn_integration import config
from utils.epsilon_anneal import LinearAnneal
from utils.experiment import (
    build_agent,
    effective_settings,
    eval_run,
    load_policy,
    train_run,
)
from utils.experiment_config import add_config_arguments, config_cli, config_from_args
from utils.logging_utils import RunningAverage
from utils.random import set_global_seed


def make_agent(**kwargs: Any) -> CachedKnnDtqnAgent:
    settings = config(model="DTQN-kNN-cached", **kwargs)
    settings.validate()
    set_global_seed(settings.training_seed)
    return cast(
        CachedKnnDtqnAgent,
        build_agent(
            settings,
            PassiveTMaze(settings.corridor_length),
            settings.corridor_length + 1,
            "cpu",
        ),
    )


def collect(agent: CachedKnnDtqnAgent, count: int) -> None:
    env = PassiveTMaze(8)
    for _ in range(count):
        agent.context_reset(env.reset())
        for step in range(9):
            # Skip get_action, as the harness does during random prefill.
            action = 0 if step < 8 else 1
            obs, reward, done, _ = env.step(action)
            agent.observe(obs, action, reward, done)
        agent.replay_buffer.flush()


class CachedKnnTests(unittest.TestCase):
    def test_capture_is_once_per_step_and_uses_each_source_network(self) -> None:
        for identity in (False, True):
            for a_embed in (0, 4):
                with self.subTest(identity=identity, a_embed=a_embed):
                    agent = make_agent(identity=identity, a_embed=a_embed)
                    target = cast(DTQNKNN, agent.target_network)
                    with torch.no_grad():
                        bias = target.knn_block.attention.attn.in_proj_bias
                        assert bias is not None
                        bias[target.knn_block.attention.attn.embed_dim :].add_(0.4)
                    agent.context_reset(np.array([0, 1], dtype=np.float32))
                    obs, actions = agent._acting_inputs()
                    with torch.no_grad():
                        expected_online = cast(DTQNKNN, agent.policy_network).newest_kv(
                            obs, actions
                        )
                        expected_target = target.newest_kv(obs, actions)
                    with patch.object(
                        target, "newest_kv", wraps=target.newest_kv
                    ) as capture:
                        agent.get_action(1.0)
                        buffer = agent.replay_buffer
                        before = {
                            name: getattr(buffer, name).copy() for name in buffer.BANKS
                        }
                        pending = agent.episode_memory.pending[0][0].clone()
                        with torch.no_grad():
                            for parameter in agent.policy_network.parameters():
                                parameter.add_(0.2)
                        agent.target_update()
                        agent.get_action(1.0)
                        self.assertEqual(capture.call_count, 1)
                    for name, previous in before.items():
                        np.testing.assert_array_equal(previous, getattr(buffer, name))
                    self.assertTrue(
                        torch.equal(pending, agent.episode_memory.pending[0][0])
                    )
                    self.assertEqual(len(agent.episode_memory.pending), 1)
                    for names, expected in (
                        (buffer.BANKS[:2], expected_online),
                        (buffer.BANKS[2:], expected_target),
                    ):
                        for name, tensor in zip(names, expected):
                            np.testing.assert_allclose(
                                getattr(buffer, name)[0, 0],
                                tensor[0].numpy(),
                                atol=1e-6,
                            )
                    self.assertFalse(
                        np.allclose(buffer.online_keys[0, 0], buffer.target_keys[0, 0])
                    )

    def test_prefill_capture_aligns_with_acting_fifo_and_actions(self) -> None:
        agent = make_agent(a_embed=4)
        agent.context_reset(np.array([0, 1], dtype=np.float32))
        for step in range(8):
            if step % 2 == 0:
                agent.get_action(1.0)
            agent.observe(np.array([0, 0], dtype=np.float32), step % 4, 0.0, False)
            start = max(0, step + 2 - agent.context_len)
            memory = agent.replay_memory(
                cast(DTQNKNN, agent.policy_network), np.array([0]), np.array([start])
            )
            acting = agent.episode_memory.external
            count = min(start, agent.memory_config.size)
            order = [
                (int(acting.ptr.item()) - count + i) % acting.size for i in range(count)
            ]
            self.assertTrue(torch.equal(memory.keys[:, :count], acting.keys[:, order]))
            self.assertTrue(
                torch.equal(memory.values[:, :count], acting.values[:, order])
            )
        self.assertEqual(int(agent.replay_buffer.valid[0].sum()), 8)
        with self.assertRaisesRegex(ValueError, "must not be overwritten"):
            buffer = agent.replay_buffer
            agent.replay_buffer.pos[1] = 0
            pair = (buffer.online_keys[0, 0], buffer.online_values[0, 0])
            buffer.cache_step(0, pair, pair)

    def test_prefix_boundaries_capacity_isolation_and_invalid_entries(self) -> None:
        agent = make_agent()
        collect(agent, 3)
        buffer = agent.replay_buffer
        # Distinct markers make selected episode/timestep/bank unambiguous.
        for row in range(3):
            for step in range(9):
                buffer.online_keys[row, step].fill(100 * row + step)
                buffer.target_keys[row, step].fill(1000 + 100 * row + step)
        episodes, starts = np.array([0, 1, 2]), np.array([0, 3, 8])
        keys, _, valid = buffer.prefix(episodes, starts, 4, False)
        np.testing.assert_array_equal(valid.sum(1), [0, 3, 4])
        np.testing.assert_array_equal(keys[1, :3, 0, 0], [100, 101, 102])
        np.testing.assert_array_equal(keys[2, :, 0, 0], [204, 205, 206, 207])
        self.assertFalse(keys[0].any())
        for row, start in zip(episodes, starts):
            buffer.online_keys[row, start:] = -5000
        after, _, _ = buffer.prefix(episodes, starts, 4, False)
        np.testing.assert_array_equal(keys, after)
        # Capacity can exceed the padded episode length without indexing padding.
        large, _, mask = buffer.prefix(np.array([1]), np.array([9]), 100, True)
        np.testing.assert_array_equal(large[0, :9, 0, 0], np.arange(1100, 1109))
        self.assertEqual(int(mask.sum()), 9)
        buffer.valid[2, 6] = False
        with self.assertRaisesRegex(ValueError, "Missing collection-time"):
            buffer.prefix(episodes, starts, 4, False)

    def test_replay_does_no_prefix_encoding_and_preserves_ddqn_and_gradients(
        self,
    ) -> None:
        agent = make_agent(tuf=1)
        # Diverse synthetic trajectories exercise the projection gradients;
        # an untrained clean corridor can have identically zero hidden inputs.
        rng = np.random.default_rng(5)
        for _ in range(4):
            agent.context_reset(rng.normal(size=2).astype(np.float32))
            for step in range(9):
                agent.observe(
                    rng.normal(size=2).astype(np.float32),
                    int(rng.integers(4)),
                    float(rng.normal()),
                    step == 8,
                )
            agent.replay_buffer.flush()
        network = cast(DTQNKNN, agent.policy_network)
        target = cast(DTQNKNN, agent.target_network)
        buffer = agent.replay_buffer
        before = {name: getattr(buffer, name).copy() for name in buffer.BANKS}
        calls: List[Tuple[DTQNKNN, NDArray[Any]]] = []
        original = agent.replay_memory

        def record(net: DTQNKNN, episodes: NDArray[Any], starts: NDArray[Any]) -> Any:
            calls.append((net, starts.copy()))
            return original(net, episodes, starts)

        # Choose nonempty prefixes to ensure the memory branch receives loss.
        original_sample = buffer.sample_with_metadata

        def sample(batch_size: int) -> Tuple[NDArray[Any], ...]:
            with patch("random.randint", return_value=4):
                return original_sample(batch_size)

        parameter = network.knn_block.attention.attn.in_proj_weight
        previous = parameter.detach().clone()
        with patch.object(
            network, "newest_kv", side_effect=AssertionError("re-encoded online prefix")
        ), patch.object(
            target, "newest_kv", side_effect=AssertionError("re-encoded target prefix")
        ), patch.object(agent, "replay_memory", side_effect=record), patch.object(
            buffer, "sample_with_metadata", side_effect=sample
        ):
            agent.train()
        self.assertEqual(len(calls), 3)
        self.assertIs(calls[0][0], network)
        self.assertIs(calls[1][0], network)
        self.assertIs(calls[2][0], target)
        np.testing.assert_array_equal(calls[1][1], calls[0][1] + 1)
        np.testing.assert_array_equal(calls[2][1], calls[0][1] + 1)
        self.assertFalse(torch.equal(previous, parameter))
        self.assertIsNotNone(parameter.grad)
        gate_grad = network.knn_block.attention.gate_bias.grad
        assert gate_grad is not None
        self.assertGreater(float(gate_grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertEqual(agent.num_train_steps, 1)
        for name, array in before.items():
            np.testing.assert_array_equal(array, getattr(buffer, name))
        for net, bank in ((network, "online"), (target, "target")):
            memory = agent.replay_memory(net, np.array([0]), np.array([5]))
            np.testing.assert_array_equal(
                memory.keys.numpy()[0], getattr(buffer, bank + "_keys")[0, 1:5]
            )
            self.assertFalse(memory.keys.requires_grad)

    def test_episode_overwrite_clears_all_banks(self) -> None:
        agent = make_agent(buf_size=45)
        collect(agent, 5)
        buffer = agent.replay_buffer
        self.assertTrue(buffer.valid[0, :9].all())
        previous = buffer.online_keys[1].copy()
        agent.context_reset(np.array([0, -1], dtype=np.float32))
        self.assertFalse(buffer.valid[0].any())
        for name in buffer.BANKS:
            self.assertFalse(getattr(buffer, name)[0].any())
        np.testing.assert_array_equal(previous, buffer.online_keys[1])

    def test_evaluation_is_separate_and_does_not_capture_target_or_replay(self) -> None:
        agent = make_agent()
        collect(agent, 1)
        buffer = agent.replay_buffer
        saved = buffer.online_keys.copy()
        train_state = agent.episode_memory
        agent.eval_on()
        agent.context_reset(np.array([0, -1], dtype=np.float32))
        with patch.object(
            agent.target_network,
            "newest_kv",
            side_effect=AssertionError("target used in eval"),
        ):
            for _ in range(6):
                agent.get_action()
                agent.observe(np.array([0, 0], dtype=np.float32), 0, 0.0, False)
        self.assertGreater(int(agent.episode_memory.external.count.item()), 0)
        self.assertIsNot(agent.episode_memory, train_state)
        np.testing.assert_array_equal(saved, buffer.online_keys)
        agent.eval_off()
        self.assertIs(agent.episode_memory, train_state)
        agent.context_reset(np.array([0, 1], dtype=np.float32))
        self.assertTrue(agent.episode_memory.external.is_empty)

    def test_disabled_is_exact_baseline_and_config_roundtrips(self) -> None:
        env = PassiveTMaze(8)
        set_global_seed(23)
        baseline = build_agent(config(model="DTQN"), env, 9, "cpu")
        rng = torch.get_rng_state()
        set_global_seed(23)
        settings = config(
            model="DTQN-kNN-cached", memory_enabled=False, memory_stop_grad=False
        )
        disabled = build_agent(settings, env, 9, "cpu")
        self.assertIs(type(disabled), DtqnAgent)
        self.assertNotIsInstance(disabled.replay_buffer, CachedReplayBuffer)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in baseline.policy_network.state_dict().items():
            self.assertTrue(
                torch.equal(value, disabled.policy_network.state_dict()[name])
            )
        obs = torch.randn(2, 3, 2)
        self.assertTrue(
            torch.equal(baseline.policy_network(obs), disabled.policy_network(obs))
        )
        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        restored = config_from_args(parser.parse_args(config_cli(settings.to_dict())))
        self.assertEqual(restored, settings)
        for values in (
            {"memory_stop_grad": False},
            {"bag_size": 1},
            {"memory_layer": 2},
        ):
            with self.assertRaises(ValueError):
                config(model="DTQN-kNN-cached", **values).validate()
        active = effective_settings(config(model="DTQN-kNN-cached"), 9)
        self.assertNotIn("memory_size", active["inactive_parameters"])

    def test_full_checkpoint_roundtrip_and_missing_bank_rejection(self) -> None:
        agent = make_agent()
        collect(agent, 4)
        agent.train()
        with tempfile.TemporaryDirectory() as temp:
            prefix = str(Path(temp) / "policy")
            average = RunningAverage(10)
            average.add(1.0)
            agent.save_checkpoint(
                prefix, "test-run", average, average, average, LinearAnneal(1, 0.1, 100)
            )
            restored = make_agent()
            result = restored.load_checkpoint(prefix)
            self.assertEqual(result[0], "test-run")
            self.assertEqual(restored.num_train_steps, agent.num_train_steps)
            for name in (*agent.replay_buffer.BANKS, "valid"):
                np.testing.assert_array_equal(
                    getattr(restored.replay_buffer, name),
                    getattr(agent.replay_buffer, name),
                )
            agent.train()
            restored.load_checkpoint(prefix)  # Restore RNG for the same sampled batch.
            restored.train()
            for name, value in agent.policy_network.state_dict().items():
                self.assertTrue(
                    torch.equal(value, restored.policy_network.state_dict()[name])
                )
            Path(prefix + "buffer_kv.npz").unlink()
            with self.assertRaisesRegex(ValueError, "missing its replay K/V banks"):
                restored.load_checkpoint(prefix)

    def test_cache_load_rejects_incomplete_or_incompatible_banks(self) -> None:
        agent = make_agent()
        collect(agent, 3)
        buffer = agent.replay_buffer
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.npz"
            buffer.valid[0, 0] = False
            buffer.save_cache(path)
            with self.assertRaisesRegex(ValueError, "missing K/V entries"):
                buffer.load_cache(path)
            buffer.valid[0, 0] = True
            buffer.save_cache(path)
            other = make_agent(heads=4)
            with self.assertRaisesRegex(ValueError, "shape or dtype"):
                other.replay_buffer.load_cache(path)

    def test_saved_model_evaluation_and_complete_trajectory_counts(self) -> None:
        settings = config(
            model="DTQN-kNN-cached", corridor_length=5, context=2, history=2
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = train_run(settings, root / "train")
            self.assertEqual(result["gradient_updates"], settings.num_steps)
            self.assertEqual(result["training_steps"], settings.num_steps)
            loaded, env, metadata = load_policy(root / "train/model.pt", "cpu")
            try:
                self.assertIsInstance(loaded, CachedKnnDtqnAgent)
                self.assertEqual(metadata["requested"]["model"], "DTQN-kNN-cached")
                self.assertFalse(loaded.replay_buffer.valid.any())
            finally:
                env.close()
            evaluated = eval_run(
                root / "train/model.pt", root / "eval", eval_episodes=3
            )
            self.assertEqual(evaluated["eval_episodes"], 3)


if __name__ == "__main__":
    unittest.main()
