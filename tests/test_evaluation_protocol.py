"""Multi-seed evaluation protocol: seeds, grid control, records, periodic eval."""

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.grid import (
    EVAL_SEED_OFFSET,
    derive_eval_seed,
    expand_jobs,
    is_complete,
    load_grid_spec,
    parse_filters,
    select_jobs,
)
from evaluation.job import PeriodicEvaluator, RecordCollector
from evaluation.records import (
    metadata_path,
    read_records,
    records_path,
    write_records,
)
from tests.rliable_fixtures import grid_file, record
from utils.experiment import make_environment, train_run
from utils.experiment_config import RunConfig


class SeedTests(unittest.TestCase):
    def test_eval_seed_is_derived_from_each_train_seed(self) -> None:
        self.assertEqual(derive_eval_seed(3), EVAL_SEED_OFFSET + 3)
        seeds = [derive_eval_seed(seed) for seed in range(1, 6)]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_jobs_cover_every_seed_with_its_own_eval_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = load_grid_spec(grid_file(Path(directory)))
            jobs = expand_jobs(spec)
        self.assertEqual(len(jobs), 2 * 3)
        for job in jobs:
            self.assertEqual(
                job.config.eval_seed, derive_eval_seed(job.config.training_seed)
            )
            self.assertEqual(job.config.eval_episodes, 4)
        self.assertEqual(len({job.run_id for job in jobs}), len(jobs))

    def test_seed_count_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = load_grid_spec(grid_file(Path(directory)), train_seed_count=5)
        self.assertEqual(spec.train_seeds, [1, 2, 3, 4, 5])

    def test_grid_rejects_seeds_inside_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = grid_file(
                Path(directory),
                parameters={
                    "model": ["DTQN"],
                    "env": ["PassiveTMaze-v0"],
                    "num_steps": [7],
                    "eval_seed": [10_000],
                },
            )
            with self.assertRaisesRegex(ValueError, "eval_seed"):
                load_grid_spec(path)


class GridSelectionTests(unittest.TestCase):
    def test_filters_select_one_configuration_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            jobs = expand_jobs(load_grid_spec(grid_file(Path(directory))))
        filters = parse_filters(["model=DTQN-kNN", "training_seed=2"])
        self.assertEqual(filters, {"model": "DTQN-kNN", "training_seed": 2})
        (selected,) = select_jobs(jobs, filters)
        self.assertEqual(selected.config.model, "DTQN-kNN")
        self.assertEqual(selected.config.training_seed, 2)

    def test_unknown_filter_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "colour=red"):
            parse_filters(["colour=red"])

    def test_run_is_complete_only_with_records_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            job = expand_jobs(load_grid_spec(grid_file(raw_dir)))[0]
            self.assertFalse(is_complete(job, raw_dir))
            write_records(records_path(raw_dir, job.run_id), [record()])
            self.assertFalse(is_complete(job, raw_dir))
            metadata_path(raw_dir, job.run_id).write_text("{}")
            self.assertTrue(is_complete(job, raw_dir))


class RecordTests(unittest.TestCase):
    def test_records_round_trip(self) -> None:
        records = [
            record(),
            record(episode=1, success=False, episode_return=-0.3, chosen_side=None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            write_records(path, records)
            self.assertEqual(read_records(path), records)

    def test_unexpected_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            path.write_text("model,success\nDTQN,1\n")
            with self.assertRaisesRegex(ValueError, "expected"):
                read_records(path)


class PeriodicEvaluationTests(unittest.TestCase):
    def config(self, model: str) -> RunConfig:
        return RunConfig(
            model=model,
            env="PassiveTMaze-v0",
            corridor_length=3,
            context=2,
            in_embed=16,
            heads=2,
            layers=1,
            batch=2,
            buf_size=200,
            num_steps=7,
            prefill_steps=12,
            eval_episodes=3,
            eval_seed=derive_eval_seed(1),
        )

    def test_evaluation_does_not_change_training(self) -> None:
        for model in ("DTQN", "DTQN-kNN"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as root:
                config = self.config(model)
                train_run(config, Path(root) / "plain")
                collector = RecordCollector(config)
                env, _ = make_environment(config)
                evaluator = PeriodicEvaluator(env, config, 3, collector)
                train_run(config, Path(root) / "evaluated", step_hook=evaluator)
                env.close()
                plain, evaluated = (
                    torch.load(Path(root) / name / "model.pt")["state_dict"]
                    for name in ("plain", "evaluated")
                )
                for key in plain:
                    self.assertTrue(torch.equal(plain[key], evaluated[key]), key)
                self.assertEqual(
                    (Path(root) / "plain" / "train_episodes.csv").read_text(),
                    (Path(root) / "evaluated" / "train_episodes.csv").read_text(),
                )
                self.assertEqual(sorted({r.step for r in collector.records}), [3, 6, 7])
                self.assertEqual(len(collector.records), 3 * config.eval_episodes)


if __name__ == "__main__":
    unittest.main()
