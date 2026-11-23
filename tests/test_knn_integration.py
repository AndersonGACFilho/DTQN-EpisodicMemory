"""Causality, acting/replay agreement, DDQN isolation and saved-model round trips."""

import argparse
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import patch

import numpy as np
import torch
from numpy.typing import NDArray

from dtqn.agents.dqn import TrainMode
from dtqn.agents.dtqn import DtqnAgent
from dtqn.agents.dtqn_knn import KnnDtqnAgent
from dtqn.networks.dtqn_knn import DTQNKNN
from envs.passive_tmaze import PassiveTMaze
from utils.experiment import (
    build_agent,
    eval_run,
    load_policy,
    make_environment,
    train_run,
)
from utils.experiment_config import (
    RunConfig,
    add_config_arguments,
    config_cli,
    config_from_args,
)
from utils.random import set_global_seed


def config(**kwargs: Any) -> RunConfig:
    params: Dict[str, Any] = {
        "model": "DTQN-kNN",
        "env": "PassiveTMaze-v0",
        "corridor_length": 8,
        "context": 3,
        "history": 3,
        "in_embed": 16,
        "heads": 2,
        "layers": 2,
        "batch": 2,
        "buf_size": 300,
        "num_steps": 12,
        "prefill_steps": 30,
        "eval_episodes": 3,
        "memory_size": 4,
        "memory_top_k": 2,
    }
    params.update(kwargs)
    return RunConfig(**params)


def make_agent(**kwargs: Any) -> KnnDtqnAgent:
    settings = config(**kwargs)
    set_global_seed(settings.training_seed)
    env = PassiveTMaze(settings.corridor_length)
    return cast(
        KnnDtqnAgent, build_agent(settings, env, settings.corridor_length + 1, "cpu")
    )


class KnnIntegrationTests(unittest.TestCase):
    def test_external_read_changes_final_q_values_and_trains_gate(self) -> None:
        agent = make_agent()
        network = cast(DTQNKNN, agent.policy_network)
        observations = torch.randn(2, 3, 2)
        local_only = network(observations)
        memory = network.new_memory(2)
        shape = (2, 4, memory.num_heads, memory.head_dim)
        memory.add(torch.randn(shape), torch.randn(shape).mul(10.0))
        with_memory = network(observations, memory=memory)
        self.assertFalse(torch.equal(local_only, with_memory))
        with_memory.sum().backward()
        grad = network.knn_block.attention.gate_bias.grad
        self.assertIsNotNone(grad)
        assert grad is not None
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_existing_carflag_and_memorycards_start_with_new_variant(self) -> None:
        for name in ("DiscreteCarFlag-v0", "Memory-5-v0"):
            with self.subTest(env=name):
                settings = config(env=name, buf_size=100000)
                env, horizon = make_environment(settings)
                try:
                    agent = build_agent(settings, env, horizon, "cpu", evaluation=True)
                    agent.eval_on()
                    agent.context_reset(env.reset())
                    for _ in range(3):
                        action = agent.get_action(1.0)
                        observation, reward, done, _ = env.step(action)
                        agent.observe(observation, action, reward, done)
                        if done:
                            break
                finally:
                    env.close()

    def test_memory_flag_off_is_exact_baseline_including_rng(self) -> None:
        env = PassiveTMaze(8)
        for identity in (False, True):
            for a_embed in (0, 4):
                with self.subTest(identity=identity, a_embed=a_embed):
                    set_global_seed(31)
                    baseline = build_agent(
                        config(model="DTQN", identity=identity, a_embed=a_embed),
                        env,
                        9,
                        "cpu",
                    )
                    after_baseline = torch.get_rng_state()
                    set_global_seed(31)
                    disabled = build_agent(
                        config(
                            memory_enabled=False, identity=identity, a_embed=a_embed
                        ),
                        env,
                        9,
                        "cpu",
                    )
                    self.assertIs(type(disabled), DtqnAgent)
                    self.assertTrue(torch.equal(after_baseline, torch.get_rng_state()))
                    for name, tensor in baseline.policy_network.state_dict().items():
                        self.assertTrue(
                            torch.equal(
                                tensor, disabled.policy_network.state_dict()[name]
                            )
                        )
                    obs = torch.randn(2, 3, 2)
                    actions = torch.zeros(2, 3, 1, dtype=torch.long)
                    self.assertTrue(
                        torch.equal(
                            baseline.policy_network(obs, actions),
                            disabled.policy_network(obs, actions),
                        )
                    )

    def test_enabled_empty_memory_matches_baseline_and_preserves_initialization(
        self,
    ) -> None:
        env = PassiveTMaze(8)
        for identity in (False, True):
            for layer in (0, -1):
                with self.subTest(identity=identity, layer=layer):
                    set_global_seed(17)
                    baseline = build_agent(
                        config(model="DTQN", identity=identity), env, 9, "cpu"
                    )
                    after_baseline = torch.get_rng_state()
                    set_global_seed(17)
                    enabled = build_agent(
                        config(identity=identity, memory_layer=layer), env, 9, "cpu"
                    )
                    self.assertTrue(torch.equal(after_baseline, torch.get_rng_state()))
                    obs = torch.randn(2, 3, 2)
                    self.assertTrue(
                        torch.equal(
                            baseline.policy_network(obs), enabled.policy_network(obs)
                        )
                    )

    def test_acting_cache_matches_rebuilt_prefix_with_fifo_and_random_actions(
        self,
    ) -> None:
        for identity in (False, True):
            for a_embed in (0, 4):
                with self.subTest(identity=identity, a_embed=a_embed):
                    agent = make_agent(identity=identity, a_embed=a_embed)
                    network = cast(DTQNKNN, agent.policy_network)
                    env = PassiveTMaze(8)
                    agent.context_reset(env.reset())
                    for step in range(8):
                        if step % 2 == 0:
                            # Even random exploration must create an entry. A
                            # repeated call at the same timestep must not add one.
                            agent.get_action(1.0)
                            length = len(agent.episode_memory.pending)
                            agent.get_action(1.0)
                            self.assertEqual(len(agent.episode_memory.pending), length)
                        # On odd steps, observe must handle skipped get_action
                        # (the same path used by random prefill).
                        obs, reward, done, _ = env.step(0)
                        agent.observe(obs, 0, reward, done)
                        start = max(0, step + 2 - agent.context_len)
                        rebuilt = agent.replay_memory(
                            network, np.array([0]), np.array([start])
                        )
                        acting = agent.episode_memory.external
                        count = min(start, agent.memory_config.size)
                        self.assertEqual(acting.count.item(), count)
                        self.assertEqual(rebuilt.count.item(), count)
                        oldest = (int(acting.ptr.item()) - count) % acting.size
                        order = [(oldest + i) % acting.size for i in range(count)]
                        self.assertTrue(
                            torch.allclose(
                                acting.keys[:, order],
                                rebuilt.keys[:, :count],
                                atol=1e-6,
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                acting.values[:, order],
                                rebuilt.values[:, :count],
                                atol=1e-6,
                            )
                        )

    def test_prefix_cannot_read_sampled_window_future_or_another_episode(self) -> None:
        agent = make_agent()
        network = cast(DTQNKNN, agent.policy_network)
        buffer = agent.replay_buffer
        observations = cast(NDArray[np.float32], buffer.obss)
        rng = np.random.default_rng(12)
        buffer.obss[:3] = rng.normal(size=buffer.obss[:3].shape)
        episodes, starts = np.array([0, 1, 2]), np.array([3, 6, 0])
        before = agent.replay_memory(network, episodes, starts)
        for episode, start in zip(episodes, starts):
            observations[episode, start:] += 1000
        after_future_change = agent.replay_memory(network, episodes, starts)
        self.assertTrue(torch.equal(before.keys, after_future_change.keys))
        self.assertTrue(torch.equal(before.values, after_future_change.values))
        self.assertEqual(before.count.tolist(), [3, 4, 0])
        observations[0, :3] -= 200
        after_other_episode_change = agent.replay_memory(network, episodes, starts)
        self.assertFalse(
            torch.equal(before.keys[0], after_other_episode_change.keys[0])
        )
        self.assertTrue(
            torch.equal(before.keys[1:], after_other_episode_change.keys[1:])
        )

    def test_reset_and_train_eval_memories_are_independent(self) -> None:
        agent = make_agent()
        agent.context_reset(np.array([0, 1], dtype=np.float32))
        for _ in range(5):
            agent.observe(np.array([0, 0], dtype=np.float32), 0, 0.0, False)
        train_state = agent.episode_memory
        count = train_state.external.count.clone()
        self.assertGreater(int(count.item()), 0)
        agent.eval_on()
        agent.context_reset(np.array([0, -1], dtype=np.float32))
        self.assertTrue(agent.episode_memory.external.is_empty)
        self.assertEqual(len(agent.episode_memory.pending), 0)
        self.assertIsNot(agent.episode_memory, train_state)
        self.assertTrue(torch.equal(train_state.external.count, count))
        agent.eval_off()
        self.assertIs(agent.episode_memory, train_state)
        agent.context_reset(np.array([0, 1], dtype=np.float32))
        self.assertTrue(train_state.external.is_empty)
        self.assertEqual(len(train_state.pending), 0)

    def test_ddqn_uses_shifted_prefix_and_each_network_own_parameters(self) -> None:
        agent = make_agent()
        buffer = agent.replay_buffer
        rng = np.random.default_rng(8)
        buffer.obss[:3] = rng.normal(size=buffer.obss[:3].shape)
        buffer.episode_lengths[:3] = 9
        buffer.pos = [3, 0]
        calls: List[Tuple[torch.nn.Module, NDArray[np.int64]]] = []
        original = agent.replay_memory

        def capture(network: DTQNKNN, episodes: Any, starts: Any) -> Any:
            calls.append((network, starts.copy()))
            return original(network, episodes, starts)

        with patch.object(agent, "replay_memory", side_effect=capture):
            agent.train()
        self.assertEqual(len(calls), 3)
        self.assertIs(calls[0][0], agent.policy_network)
        self.assertIs(calls[1][0], agent.policy_network)
        self.assertIs(calls[2][0], agent.target_network)
        np.testing.assert_array_equal(calls[1][1], calls[0][1] + 1)
        np.testing.assert_array_equal(calls[2][1], calls[0][1] + 1)
        self.assertTrue(all(p.grad is None for p in agent.target_network.parameters()))
        self.assertEqual(agent.num_train_steps, 1)
        with torch.no_grad():
            target = cast(DTQNKNN, agent.target_network)
            target.knn_block.attention.attn.in_proj_weight.add_(1)
        policy_memory = original(
            cast(DTQNKNN, agent.policy_network), np.array([0]), np.array([4])
        )
        target_memory = original(target, np.array([0]), np.array([4]))
        self.assertFalse(torch.allclose(policy_memory.keys, target_memory.keys))

    def test_prefix_stop_gradient_and_differentiable_option(self) -> None:
        for stop_grad in (True, False):
            with self.subTest(stop_grad=stop_grad):
                agent = make_agent(memory_stop_grad=stop_grad)
                network = cast(DTQNKNN, agent.policy_network)
                memory = agent.replay_memory(network, np.array([0]), np.array([4]))
                self.assertEqual(memory.keys.requires_grad, not stop_grad)
                self.assertEqual(memory.values.requires_grad, not stop_grad)
                if not stop_grad:
                    (memory.keys.sum() + memory.values.sum()).backward()
                    self.assertIsNotNone(
                        network.knn_block.attention.attn.in_proj_weight.grad
                    )

    def test_cli_boolean_options_round_trip_and_validate_layer(self) -> None:
        requested = config(
            memory_enabled=False, memory_stop_grad=False, memory_gate=False
        )
        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        restored = config_from_args(parser.parse_args(config_cli(requested.to_dict())))
        self.assertEqual(restored.to_dict(), requested.to_dict())
        for invalid in (
            config(memory_layer=2),
            config(memory_size=0),
            config(bag_size=1),
        ):
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_full_training_and_saved_policy_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = config(corridor_length=5, context=2, history=2, layers=1)
            result = train_run(settings, root / "train")
            self.assertEqual(result["gradient_updates"], settings.num_steps)
            loaded, env, metadata = load_policy(root / "train/model.pt", "cpu")
            try:
                self.assertIsInstance(loaded, KnnDtqnAgent)
                self.assertEqual(metadata["requested"]["memory_size"], 4)
                self.assertEqual(loaded.train_mode, TrainMode.EVAL)
                loaded.context_reset(env.reset())
                self.assertTrue(loaded.episode_memory.external.is_empty)
            finally:
                env.close()
            evaluated = eval_run(
                root / "train/model.pt", root / "eval", eval_episodes=3
            )
            self.assertEqual(evaluated["eval_episodes"], 3)


if __name__ == "__main__":
    unittest.main()
