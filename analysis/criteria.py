"""Pass/fail criteria for the hypothesis (issue #1), from rliable estimates.

Beyond the window (the cue is older than the context): the memory works when the
candidate's IQM final success rate reaches SUCCESS_TARGET and the whole 95% CI of
P(candidate > baseline) lies above NO_DIFFERENCE.

Within the window (the context still holds the cue): the candidate ties with the
baseline (non-inferiority) when the 95% CI of P(candidate > baseline) does not lie
entirely below NO_DIFFERENCE and its IQM is at most NON_INFERIORITY_MARGIN below
the baseline's. A miss is reported as "not met" and discussed in the paper.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from analysis.metrics import (
    ALGORITHM_FIELDS,
    PAIR_SEPARATOR,
    TASK_FIELDS,
    Estimates,
    algorithm_label,
    comparison_pairs,
)

SUCCESS_TARGET = 0.9
NON_INFERIORITY_MARGIN = 0.05
NO_DIFFERENCE = 0.5
HYPOTHESIS = ("DTQN-kNN", "DTQN")
IQM = "IQM"
BEYOND_WINDOW = "beyond window"
WITHIN_WINDOW = "within window"


def _tmaze_dependency(corridor_length: int) -> int:
    # The cue is observation 0 and the junction is observation corridor_length
    return corridor_length + 1


# Observations between the cue and the decision, per environment
DEPENDENCY_LENGTHS: Dict[str, Callable[[int], int]] = {
    "PassiveTMaze-v0": _tmaze_dependency,
    "NoisyTMaze-v0": _tmaze_dependency,
}


@dataclass(frozen=True)
class Verdict:
    candidate: str
    baseline: str
    regime: str
    candidate_iqm: float
    baseline_iqm: float
    improvement: float
    improvement_low: float
    improvement_high: float
    met: bool


def describe_criteria() -> str:
    return (
        f"Beyond the window: an IQM success rate of at least {SUCCESS_TARGET} and the "
        f"95% CI of P(candidate > baseline) above {NO_DIFFERENCE}. Within the window: "
        f"that CI not entirely below {NO_DIFFERENCE} and an IQM at most "
        f"{NON_INFERIORITY_MARGIN} below the baseline (non-inferiority)."
    )


def window_regime(algorithm: Tuple[Any, ...], tasks: Sequence[Tuple[Any, ...]]) -> str:
    """Whether the context of `algorithm` misses or holds the cue in every task."""
    context = algorithm[ALGORITHM_FIELDS.index("context")]
    beyond = set()
    for task in tasks:
        env = task[TASK_FIELDS.index("env")]
        if env not in DEPENDENCY_LENGTHS:
            raise ValueError(f"No dependency length registered for {env}")
        length = DEPENDENCY_LENGTHS[env](task[TASK_FIELDS.index("corridor_length")])
        beyond.add(length > context)
    if len(beyond) > 1:
        raise ValueError(
            f"{algorithm_label(algorithm)} is beyond the window in some tasks and "
            "within it in others; evaluate those tasks separately"
        )
    return BEYOND_WINDOW if beyond.pop() else WITHIN_WINDOW


def _is_met(
    regime: str, candidate_iqm: float, baseline_iqm: float, low: float, high: float
) -> bool:
    if regime == BEYOND_WINDOW:
        return candidate_iqm >= SUCCESS_TARGET and low > NO_DIFFERENCE
    return (
        high >= NO_DIFFERENCE and candidate_iqm >= baseline_iqm - NON_INFERIORITY_MARGIN
    )


def evaluate_hypothesis(
    algorithms: Sequence[Tuple[Any, ...]],
    tasks: Sequence[Tuple[Any, ...]],
    aggregates: Estimates,
    improvements: Estimates,
    metric_names: Sequence[str],
) -> List[Verdict]:
    """One verdict per candidate/baseline pair with equal non-model settings."""
    by_label = {algorithm_label(algorithm): algorithm for algorithm in algorithms}
    iqm_index = list(metric_names).index(IQM)
    points, _ = aggregates
    improvement_points, improvement_intervals = improvements
    verdicts = []
    for candidate, baseline in comparison_pairs(algorithms, [HYPOTHESIS]):
        key = f"{candidate}{PAIR_SEPARATOR}{baseline}"
        if key not in improvement_points:
            raise ValueError(f"Add {list(HYPOTHESIS)} to the grid's comparisons")
        low, high = (float(value) for value in improvement_intervals[key].reshape(-1))
        regime = window_regime(by_label[candidate], tasks)
        candidate_iqm = float(points[candidate][iqm_index])
        baseline_iqm = float(points[baseline][iqm_index])
        verdicts.append(
            Verdict(
                candidate=candidate,
                baseline=baseline,
                regime=regime,
                candidate_iqm=candidate_iqm,
                baseline_iqm=baseline_iqm,
                improvement=float(improvement_points[key]),
                improvement_low=low,
                improvement_high=high,
                met=_is_met(regime, candidate_iqm, baseline_iqm, low, high),
            )
        )
    return verdicts
