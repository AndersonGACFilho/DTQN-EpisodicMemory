"""Same seed, device and code give bit-identical training and evaluation."""

import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from evaluation.grid import RunJob, derive_eval_seed
from evaluation.job import run_job
from evaluation.records import EpisodeRecord, read_records, records_path
from utils.experiment_config import RunConfig
from utils.random import RUN_THREADS
from utils.torch_utils import select_action_values

MODELS = ("DQN", "DRQN", "DTQN", "DTQN-kNN")
ACTIONS = 4


def tiny_job(model: str) -> RunJob:
    config = RunConfig(
        model=model,
        env="PassiveTMaze-v0",
        corridor_length=3,
        context=2,
        history=2,
        in_embed=16,
        heads=2,
        layers=1,
        batch=2,
        buf_size=200,
        num_steps=12,
        prefill_steps=16,
        eval_episodes=4,
        eval_seed=derive_eval_seed(1),
    )
    return RunJob(config, eval_interval=6)


def train_once(
    job: RunJob, root: Path
) -> Tuple[Dict[str, torch.Tensor], List[EpisodeRecord]]:
    run_job(job, root / "raw", root / "runs", "cpu")
    (model_path,) = (root / "runs").glob("*/attempt-001/model.pt")
    weights = torch.load(model_path)["state_dict"]
    return weights, read_records(records_path(root / "raw", job.run_id))


class RepeatedRunTests(unittest.TestCase):
    def test_same_seed_gives_identical_weights_and_records(self) -> None:
        for model in MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                job = tiny_job(model)
                first = train_once(job, Path(directory) / "first")
                second = train_once(job, Path(directory) / "second")
                for key, tensor in first[0].items():
                    self.assertTrue(torch.equal(tensor, second[0][key]), key)
                self.assertEqual(first[1], second[1])

    def test_deterministic_mode_is_enabled_after_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_once(tiny_job("DTQN"), Path(directory))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(torch.get_num_threads(), RUN_THREADS)


class ActionSelectionTests(unittest.TestCase):
    def test_matches_gather_bit_for_bit(self) -> None:
        generator = torch.Generator().manual_seed(0)
        q_values = torch.randn(5, 3, ACTIONS, generator=generator)
        actions = torch.randint(ACTIONS, (5, 3, 1), generator=generator)
        expected_input = q_values.clone().requires_grad_()
        actual_input = q_values.clone().requires_grad_()
        expected = expected_input.gather(2, actions)
        actual = select_action_values(actual_input, actions)
        self.assertTrue(torch.equal(expected, actual))
        upstream = torch.randn(expected.shape, generator=generator)
        expected.backward(upstream)
        actual.backward(upstream)
        self.assertTrue(torch.equal(expected_input.grad, actual_input.grad))


if __name__ == "__main__":
    unittest.main()
