"""Execution plans and the executors that run grid jobs.

Register a new execution mode by adding its executor factory to EXECUTORS.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

from evaluation.grid import RunJob
from evaluation.job import run_job
from utils.random import RUN_THREADS

SEQUENTIAL = "sequential"
PARALLEL = "parallel"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

Task = Callable[[RunJob], None]
Outcome = Tuple[RunJob, Optional[BaseException]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPlan:
    device: str
    workers: int
    mode: str

    def to_dict(self) -> Dict[str, object]:
        return {**asdict(self), "threads_per_run": RUN_THREADS}


class Executor(Protocol):
    def outcomes(self, task: Task, jobs: Sequence[RunJob]) -> Iterator[Outcome]: ...


class InProcessExecutor:
    """Runs jobs one after another in this process."""

    def outcomes(self, task: Task, jobs: Sequence[RunJob]) -> Iterator[Outcome]:
        for job in jobs:
            try:
                task(job)
                yield job, None
            except Exception as error:
                yield job, error


def _initialize_worker() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ProcessExecutor:
    """Runs jobs in spawned worker processes.

    The parent never touches CUDA: once it holds a CUDA context, workers on the
    same GPU fail to initialize theirs ("CUDA error: out of memory").
    """

    def __init__(self, workers: int) -> None:
        self.workers = workers

    def outcomes(self, task: Task, jobs: Sequence[RunJob]) -> Iterator[Outcome]:
        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
        ) as pool:
            futures = {pool.submit(task, job): job for job in jobs}
            for future in as_completed(futures):
                yield futures[future], future.exception()


EXECUTORS: Dict[str, Callable[[ExecutionPlan], Executor]] = {
    SEQUENTIAL: lambda plan: InProcessExecutor(),
    PARALLEL: lambda plan: ProcessExecutor(plan.workers),
}


def run_tasks(task: Task, jobs: Sequence[RunJob], plan: ExecutionPlan) -> List[str]:
    """Run `task` on every job with the plan's executor; return the failed run ids."""
    if plan.mode not in EXECUTORS:
        raise ValueError(
            f"Unknown execution mode {plan.mode!r}; known: {sorted(EXECUTORS)}"
        )
    failed = []
    outcomes = EXECUTORS[plan.mode](plan).outcomes(task, jobs)
    for done, (job, error) in enumerate(outcomes, start=1):
        if error is None:
            logger.info("[%d/%d] finished %s", done, len(jobs), job.run_id)
            continue
        failed.append(job.run_id)
        logger.error(
            "[%d/%d] %s failed; continuing with the others",
            done,
            len(jobs),
            job.run_id,
            exc_info=error,
        )
    return failed


def run_plan(
    jobs: Sequence[RunJob], raw_dir: Path, artifacts_dir: Path, plan: ExecutionPlan
) -> List[str]:
    """Train and evaluate every job; return the ids of those that failed."""
    task = partial(
        run_job,
        raw_dir=raw_dir,
        artifacts_dir=artifacts_dir,
        device=plan.device,
        execution=plan.to_dict(),
    )
    return run_tasks(task, jobs, plan)
