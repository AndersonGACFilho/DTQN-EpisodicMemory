import unittest
from typing import Any, Dict, List

import numpy as np
import torch

from dtqn.buffers.replay_buffer import ReplayBuffer
from dtqn.networks.dtqn import DTQN

OBS_DIM = 4
NUM_ACTIONS = 3
CONTEXT = 10


def make_dtqn(**kwargs: Any) -> DTQN:
    options: Dict[str, Any] = dict(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        embed_per_obs_dim=8,
        action_dim=0,
        inner_embed_size=32,
        num_heads=4,
        num_layers=2,
        history_len=CONTEXT,
        pos="learned",
    )
    options.update(kwargs)
    return DTQN(**options)


class TestDTQN(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_output_shape_for_each_variant(self) -> None:
        variants: List[Dict[str, Any]] = [
            {},
            {"pos": "sin"},
            {"pos": "none"},
            {"gate": "gru"},
            {"identity": True},
            {"action_dim": 8},
        ]
        obss = torch.randn(2, CONTEXT, OBS_DIM)
        actions = torch.randint(NUM_ACTIONS, (2, CONTEXT, 1))
        for variant in variants:
            with self.subTest(**variant):
                q_values = make_dtqn(**variant)(obss, actions)
                self.assertEqual(q_values.shape, (2, CONTEXT, NUM_ACTIONS))

    def test_shorter_history_than_context(self) -> None:
        q_values = make_dtqn()(torch.randn(1, 3, OBS_DIM))
        self.assertEqual(q_values.shape, (1, 3, NUM_ACTIONS))

    def test_bag(self) -> None:
        net = make_dtqn(bag_size=5, action_dim=8)
        q_values = net(
            torch.randn(2, CONTEXT, OBS_DIM),
            torch.randint(NUM_ACTIONS, (2, CONTEXT, 1)),
            torch.randn(2, 5, OBS_DIM),
            torch.randint(NUM_ACTIONS, (2, 5, 1)),
        )
        self.assertEqual(q_values.shape, (2, CONTEXT, NUM_ACTIONS))

    def test_q_values_do_not_depend_on_future_steps(self) -> None:
        net = make_dtqn().eval()
        obss = torch.randn(1, CONTEXT, OBS_DIM)
        changed = obss.clone()
        changed[:, 6:] = torch.randn(1, CONTEXT - 6, OBS_DIM)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(obss)[:, :6], net(changed)[:, :6]))


class TestReplayBuffer(unittest.TestCase):
    def test_sampled_windows_stay_inside_one_episode(self) -> None:
        max_steps = 20
        buffer = ReplayBuffer(
            buffer_size=10 * max_steps,
            env_obs_length=1,
            obs_mask=-1,
            max_episode_steps=max_steps,
            context_len=CONTEXT,
        )
        for episode in range(5):
            buffer.store_obs(np.array([episode * 100.0]))
            for t in range(1, max_steps + 1):
                obs = np.array([episode * 100.0 + t])
                buffer.store(
                    obs, np.array([0]), np.array([0.0]), np.array([t == max_steps]), t
                )
            buffer.flush()

        obss, actions, rewards, next_obss, *_ = buffer.sample(8)
        self.assertEqual(obss.shape, (8, CONTEXT, 1))
        steps = obss[..., 0] % 100
        episodes = obss[..., 0] // 100
        # Consecutive steps of a single episode, with next_obs shifted by one
        self.assertTrue((np.diff(steps, axis=1) == 1).all())
        self.assertTrue((episodes == episodes[:, :1]).all())
        self.assertTrue((next_obss[..., 0] == obss[..., 0] + 1).all())


if __name__ == "__main__":
    unittest.main()
