"""Score matrices from tidy records and rliable interval estimates.

Scores are success rates, already in [0, 1], so no normalization is needed.
Register a new aggregate metric by adding it to AGGREGATE_METRICS.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from rliable import library as rly
from rliable import metrics

from evaluation.records import EpisodeRecord

# Fields that identify an algorithm (one curve/bar) and a task (one matrix column)
ALGORITHM_FIELDS = ("model", "context", "layers")
TASK_FIELDS = ("env", "corridor_length")
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 0
PAIR_SEPARATOR = ","

Scores = NDArray[np.float64]
Estimates = Tuple[Dict[str, Scores], Dict[str, Scores]]

AGGREGATE_METRICS: Dict[str, Callable[[Scores], float]] = {
    "Mean": metrics.aggregate_mean,
    "IQM": metrics.aggregate_iqm,
    "Median": metrics.aggregate_median,
    "Optimality Gap": metrics.aggregate_optimality_gap,
}


class IncompleteScoresError(ValueError):
    pass


def algorithm_label(values: Sequence[Any]) -> str:
    model, *settings = values
    details = " ".join(
        f"{name}={value}" for name, value in zip(ALGORITHM_FIELDS[1:], settings)
    )
    return f"{model} {details}".strip()


def task_label(values: Sequence[Any]) -> str:
    return " ".join(f"{name}={value}" for name, value in zip(TASK_FIELDS, values))


def _key(record: EpisodeRecord, names: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(getattr(record, name) for name in names)


@dataclass(frozen=True)
class ScoreData:
    """Success rates indexed as [seed, task, step] for each algorithm."""

    algorithms: List[Tuple[Any, ...]]
    tasks: List[Tuple[Any, ...]]
    seeds: List[int]
    steps: List[int]
    curves: Dict[str, Scores]

    @property
    def final_scores(self) -> Dict[str, Scores]:
        """(n_seeds, n_tasks) matrices at the last evaluation step."""
        return {label: curve[:, :, -1] for label, curve in self.curves.items()}


def success_rates(
    records: Sequence[EpisodeRecord],
) -> Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...], int, int], float]:
    """Mean success per (algorithm, task, train seed, step) evaluation."""
    outcomes: Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...], int, int], List[bool]]
    outcomes = defaultdict(list)
    for record in records:
        key = (
            _key(record, ALGORITHM_FIELDS),
            _key(record, TASK_FIELDS),
            record.train_seed,
            record.step,
        )
        outcomes[key].append(record.success)
    return {key: float(np.mean(values)) for key, values in outcomes.items()}


def build_score_data(records: Sequence[EpisodeRecord]) -> ScoreData:
    rates = success_rates(records)
    if not rates:
        raise IncompleteScoresError("No evaluation records to analyse")
    algorithms = sorted({key[0] for key in rates}, key=lambda a: (a[1:], a[0]))
    tasks = sorted({key[1] for key in rates})
    seeds = sorted({key[2] for key in rates})
    steps = sorted({key[3] for key in rates})
    missing = [
        (algorithm_label(algorithm), task_label(task), seed, step)
        for algorithm in algorithms
        for task in tasks
        for seed in seeds
        for step in steps
        if (algorithm, task, seed, step) not in rates
    ]
    if missing:
        raise IncompleteScoresError(
            f"{len(missing)} evaluations are missing, e.g. "
            f"(algorithm, task, seed, step) = {missing[:3]}"
        )
    curves = {
        algorithm_label(algorithm): np.array(
            [
                [
                    [rates[(algorithm, task, seed, step)] for step in steps]
                    for task in tasks
                ]
                for seed in seeds
            ]
        )
        for algorithm in algorithms
    }
    return ScoreData(algorithms, tasks, seeds, steps, curves)


def _aggregate_all(scores: Scores) -> Scores:
    return np.array([metric(scores) for metric in AGGREGATE_METRICS.values()])


def _iqm_per_step(scores: Scores) -> Scores:
    return np.array(
        [metrics.aggregate_iqm(scores[..., step]) for step in range(scores.shape[-1])]
    )


def _interval_estimates(
    score_dict: Any, function: Callable[..., Any], reps: int
) -> Estimates:
    points, intervals = rly.get_interval_estimates(
        score_dict,
        function,
        reps=reps,
        confidence_interval_size=CONFIDENCE_LEVEL,
        random_state=np.random.RandomState(BOOTSTRAP_SEED),
    )
    return points, intervals


def aggregate_estimates(final_scores: Dict[str, Scores], reps: int) -> Estimates:
    """Point estimates and stratified bootstrap CIs, ordered as AGGREGATE_METRICS."""
    return _interval_estimates(final_scores, _aggregate_all, reps)


def iqm_curve_estimates(curves: Dict[str, Scores], reps: int) -> Estimates:
    return _interval_estimates(curves, _iqm_per_step, reps)


def comparison_pairs(
    algorithms: Sequence[Tuple[Any, ...]], comparisons: Sequence[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """(candidate, baseline) labels for models sharing every other setting."""
    available = set(algorithms)
    pairs = []
    for candidate_model, baseline_model in comparisons:
        for algorithm in algorithms:
            baseline = (baseline_model, *algorithm[1:])
            if algorithm[0] == candidate_model and baseline in available:
                pairs.append((algorithm_label(algorithm), algorithm_label(baseline)))
    return pairs


def improvement_estimates(
    final_scores: Dict[str, Scores], pairs: Sequence[Tuple[str, str]], reps: int
) -> Estimates:
    """P(candidate > baseline), keyed by 'candidate,baseline' as rliable plots expect."""
    pair_scores = {
        f"{candidate}{PAIR_SEPARATOR}{baseline}": (
            final_scores[candidate],
            final_scores[baseline],
        )
        for candidate, baseline in pairs
    }
    return _interval_estimates(pair_scores, metrics.probability_of_improvement, reps)


def performance_profiles(
    final_scores: Dict[str, Scores], thresholds: Scores, reps: int
) -> Estimates:
    """Fraction of runs scoring above each threshold (rliable score distributions).

    Same computation as rly.create_performance_profile, which cannot be seeded.
    """

    def profile(scores: Scores) -> Scores:
        return np.asarray(rly.score_distributions(scores, thresholds))

    return _interval_estimates(final_scores, profile, reps)
