"""Hypothesis criteria (issue #1), normalized returns and score spread."""

import unittest
from typing import Any, Dict, Tuple

import numpy as np

from analysis.criteria import (
    BEYOND_WINDOW,
    NON_INFERIORITY_MARGIN,
    SUCCESS_TARGET,
    WITHIN_WINDOW,
    evaluate_hypothesis,
    window_regime,
)
from analysis.metrics import (
    AGGREGATE_METRICS,
    NORMALIZED_RETURN,
    PAIR_SEPARATOR,
    Estimates,
    algorithm_label,
    build_score_data,
    normalize_per_task,
    score_spread,
)
from analysis.tables import aggregate_rows
from tests.rliable_fixtures import record

TASK = ("PassiveTMaze-v0", 10)
METRICS = list(AGGREGATE_METRICS)
IQM_INDEX = METRICS.index("IQM")


def algorithm(model: str, context: int) -> Tuple[Any, ...]:
    return (model, context, 2)


def estimates(
    candidate_iqm: float,
    baseline_iqm: float,
    interval: Tuple[float, float],
    context: int,
) -> Tuple[Estimates, Estimates]:
    candidate = algorithm_label(algorithm("DTQN-kNN", context))
    baseline = algorithm_label(algorithm("DTQN", context))
    points: Dict[str, Any] = {}
    for label, iqm in ((candidate, candidate_iqm), (baseline, baseline_iqm)):
        values = np.zeros(len(METRICS))
        values[IQM_INDEX] = iqm
        points[label] = values
    pair = f"{candidate}{PAIR_SEPARATOR}{baseline}"
    improvements: Estimates = (
        {pair: np.array(0.5)},
        {pair: np.array([[interval[0]], [interval[1]]], dtype=np.float64)},
    )
    return (points, {}), improvements


def verdict(
    candidate_iqm: float,
    baseline_iqm: float,
    interval: Tuple[float, float],
    context: int,
) -> Any:
    aggregates, improvements = estimates(candidate_iqm, baseline_iqm, interval, context)
    algorithms = [algorithm("DTQN-kNN", context), algorithm("DTQN", context)]
    (result,) = evaluate_hypothesis(
        algorithms, [TASK], aggregates, improvements, METRICS
    )
    return result


class WindowRegimeTests(unittest.TestCase):
    def test_context_shorter_than_the_episode_misses_the_cue(self) -> None:
        # Corridor 10: the cue is 11 observations before the decision
        self.assertEqual(window_regime(algorithm("DTQN", 6), [TASK]), BEYOND_WINDOW)
        self.assertEqual(window_regime(algorithm("DTQN", 10), [TASK]), BEYOND_WINDOW)
        self.assertEqual(window_regime(algorithm("DTQN", 11), [TASK]), WITHIN_WINDOW)

    def test_mixed_regimes_are_rejected(self) -> None:
        tasks = [("PassiveTMaze-v0", 5), ("PassiveTMaze-v0", 20)]
        with self.assertRaisesRegex(ValueError, "separately"):
            window_regime(algorithm("DTQN", 10), tasks)

    def test_unknown_environment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dependency length"):
            window_regime(algorithm("DTQN", 10), [("CarFlag-v0", 10)])


class BeyondWindowTests(unittest.TestCase):
    def test_target_and_improvement_pass(self) -> None:
        self.assertTrue(verdict(SUCCESS_TARGET, 0.5, (0.6, 0.9), context=6).met)

    def test_below_target_fails(self) -> None:
        self.assertFalse(verdict(SUCCESS_TARGET - 0.01, 0.5, (0.6, 0.9), 6).met)

    def test_interval_touching_no_difference_fails(self) -> None:
        self.assertFalse(verdict(1.0, 0.5, (0.5, 0.9), context=6).met)


class WithinWindowTests(unittest.TestCase):
    def test_tie_passes(self) -> None:
        self.assertTrue(verdict(0.98, 1.0, (0.3, 0.6), context=12).met)

    def test_iqm_beyond_margin_fails(self) -> None:
        drop = NON_INFERIORITY_MARGIN + 0.01
        self.assertFalse(verdict(1.0 - drop, 1.0, (0.3, 0.6), context=12).met)

    def test_interval_below_no_difference_fails(self) -> None:
        self.assertFalse(verdict(1.0, 1.0, (0.2, 0.45), context=12).met)

    def test_missing_comparison_is_explained(self) -> None:
        aggregates, _ = estimates(1.0, 1.0, (0.4, 0.6), context=12)
        algorithms = [algorithm("DTQN-kNN", 12), algorithm("DTQN", 12)]
        with self.assertRaisesRegex(ValueError, "comparisons"):
            evaluate_hypothesis(algorithms, [TASK], aggregates, ({}, {}), METRICS)


class ReturnAndSpreadTests(unittest.TestCase):
    def test_returns_are_scaled_per_task(self) -> None:
        means = {
            (("DTQN",), ("a",), 1, 1): -1.0,
            (("DTQN",), ("a",), 1, 2): 1.0,
            (("DTQN",), ("b",), 1, 1): 3.0,
            (("DTQN",), ("b",), 1, 2): 3.0,
        }
        scaled = normalize_per_task(means)
        self.assertEqual(scaled[(("DTQN",), ("a",), 1, 1)], 0.0)
        self.assertEqual(scaled[(("DTQN",), ("a",), 1, 2)], 1.0)
        self.assertEqual(scaled[(("DTQN",), ("b",), 1, 1)], 0.0)

    def test_normalized_return_score_matrix(self) -> None:
        records = [
            record(train_seed=1, episode_return=-0.5),
            record(train_seed=2, episode_return=1.0),
        ]
        data = build_score_data(records, NORMALIZED_RETURN)
        np.testing.assert_allclose(
            data.final_scores["DTQN context=6 layers=2"], [[0.0], [1.0]]
        )

    def test_spread_is_the_sample_std_over_runs(self) -> None:
        spreads = score_spread({"DTQN": np.array([[0.0], [1.0]])})
        self.assertAlmostEqual(spreads["DTQN"], float(np.std([0.0, 1.0], ddof=1)))

    def test_aggregate_rows_include_the_spread(self) -> None:
        points = {"DTQN": np.arange(len(METRICS), dtype=float)}
        intervals = {"DTQN": np.zeros((2, len(METRICS)))}
        (row,) = aggregate_rows((points, intervals), METRICS, {"DTQN": 0.25})
        self.assertEqual(row["Std"], 0.25)


if __name__ == "__main__":
    unittest.main()
