"""Detect policies that pick the same junction side regardless of the cue."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from analysis.metrics import ALGORITHM_FIELDS, TASK_FIELDS, algorithm_label, task_label
from evaluation.records import EpisodeRecord
from evaluation.sides import DOWN, NO_BRANCH, UP

SIDE_BIAS_THRESHOLD = 0.9
BRANCH_SIDES = (UP, DOWN)


@dataclass(frozen=True)
class SideBias:
    algorithm: str
    task: str
    train_seed: int
    step: int
    episodes: int
    success_rate: float
    side_fractions: Dict[str, float]
    flagged: bool

    @property
    def dominant_side(self) -> str:
        return max(BRANCH_SIDES, key=lambda side: self.side_fractions[side])


def _final_step_records(
    records: Sequence[EpisodeRecord],
) -> Dict[Tuple[Any, ...], List[EpisodeRecord]]:
    """Records of each run's last evaluation; runs without side data are skipped."""
    runs: Dict[Tuple[Any, ...], List[EpisodeRecord]] = defaultdict(list)
    for record in records:
        if record.chosen_side is not None:
            key = tuple(
                getattr(record, name) for name in ALGORITHM_FIELDS + TASK_FIELDS
            )
            runs[key + (record.train_seed,)].append(record)
    final_step = {key: max(r.step for r in run) for key, run in runs.items()}
    return {
        key: [r for r in run if r.step == final_step[key]] for key, run in runs.items()
    }


def side_bias(
    records: Sequence[EpisodeRecord], threshold: float = SIDE_BIAS_THRESHOLD
) -> List[SideBias]:
    results = []
    for key, run in sorted(_final_step_records(records).items()):
        episodes = len(run)
        fractions = {
            side: sum(r.chosen_side == side for r in run) / episodes
            for side in (*BRANCH_SIDES, NO_BRANCH)
        }
        algorithm_size = len(ALGORITHM_FIELDS)
        results.append(
            SideBias(
                algorithm=algorithm_label(key[:algorithm_size]),
                task=task_label(key[algorithm_size:-1]),
                train_seed=key[-1],
                step=run[0].step,
                episodes=episodes,
                success_rate=sum(r.success for r in run) / episodes,
                side_fractions=fractions,
                flagged=any(fractions[side] > threshold for side in BRANCH_SIDES),
            )
        )
    return results
