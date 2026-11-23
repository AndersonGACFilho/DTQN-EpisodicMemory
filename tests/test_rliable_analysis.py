"""rliable analysis: score matrices, estimates, side bias and missing runs."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

from analysis.diagnostics import side_bias
from analysis.metrics import (
    IncompleteScoresError,
    aggregate_estimates,
    build_score_data,
    comparison_pairs,
    improvement_estimates,
)
from analysis.rliable_report import MissingRunsError, load_grid_records
from evaluation.grid import expand_jobs, load_grid_spec
from evaluation.records import EpisodeRecord
from evaluation.sides import DOWN
from tests.rliable_fixtures import (
    AlwaysDownPolicy,
    CueFollowingPolicy,
    evaluate_fixed_policy,
    grid_file,
    record,
)


class ScoreMatrixTests(unittest.TestCase):
    def records(self) -> List[EpisodeRecord]:
        records = []
        for model in ("DTQN", "DTQN-kNN"):
            for seed in (1, 2, 3):
                for corridor_length in (5, 10):
                    for episode in range(4):
                        # DTQN-kNN succeeds in `seed` of 4 episodes, DTQN in none
                        success = model == "DTQN-kNN" and episode < seed
                        records.append(
                            record(
                                model=model,
                                train_seed=seed,
                                corridor_length=corridor_length,
                                episode=episode,
                                success=success,
                            )
                        )
        return records

    def test_matrix_shape_is_seeds_by_tasks(self) -> None:
        data = build_score_data(self.records())
        scores = data.final_scores["DTQN-kNN context=6 layers=2"]
        self.assertEqual(scores.shape, (3, 2))
        np.testing.assert_allclose(scores[:, 0], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(data.final_scores["DTQN context=6 layers=2"], 0.0)

    def test_final_scores_use_the_last_step(self) -> None:
        late = [replace(r, step=4000, success=True) for r in self.records()]
        data = build_score_data(self.records() + late)
        self.assertEqual(data.steps, [2000, 4000])
        np.testing.assert_allclose(data.final_scores["DTQN context=6 layers=2"], 1.0)
        self.assertEqual(data.curves["DTQN context=6 layers=2"].shape, (3, 2, 2))

    def test_missing_evaluation_is_an_error(self) -> None:
        records = [r for r in self.records() if r.train_seed != 3 or r.model != "DTQN"]
        with self.assertRaisesRegex(IncompleteScoresError, "missing"):
            build_score_data(records)

    def test_estimates_and_improvement_pairs(self) -> None:
        data = build_score_data(self.records())
        pairs = comparison_pairs(data.algorithms, [("DTQN-kNN", "DTQN")])
        self.assertEqual(
            pairs, [("DTQN-kNN context=6 layers=2", "DTQN context=6 layers=2")]
        )
        points, intervals = aggregate_estimates(data.final_scores, reps=50)
        self.assertAlmostEqual(points["DTQN-kNN context=6 layers=2"][0], 0.5)
        self.assertEqual(intervals["DTQN context=6 layers=2"].shape, (2, 4))
        probabilities, _ = improvement_estimates(data.final_scores, pairs, reps=50)
        self.assertEqual(list(probabilities.values()), [1.0])


class SideBiasTests(unittest.TestCase):
    def test_fixed_side_policy_reproduces_the_legacy_62_percent(self) -> None:
        records = evaluate_fixed_policy(AlwaysDownPolicy(10), corridor_length=10)
        (result,) = side_bias(records)
        self.assertAlmostEqual(result.success_rate, 0.62)
        self.assertEqual(result.side_fractions[DOWN], 1.0)
        self.assertEqual(result.dominant_side, DOWN)
        self.assertTrue(result.flagged)

    def test_cue_following_policy_is_not_flagged(self) -> None:
        records = evaluate_fixed_policy(CueFollowingPolicy(10), corridor_length=10)
        (result,) = side_bias(records)
        self.assertEqual(result.success_rate, 1.0)
        self.assertFalse(result.flagged)
        self.assertTrue(all(r.goal_side == r.chosen_side for r in records))


class MissingRunsTests(unittest.TestCase):
    def test_report_names_runs_without_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            jobs = expand_jobs(load_grid_spec(grid_file(raw_dir)))
            with self.assertRaisesRegex(MissingRunsError, jobs[0].run_id):
                load_grid_records(jobs, raw_dir)


if __name__ == "__main__":
    unittest.main()
