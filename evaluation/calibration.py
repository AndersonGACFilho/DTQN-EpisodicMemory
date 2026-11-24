"""Choose the number of parallel workers from short probe runs on this machine."""

import logging
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

from evaluation.execution import PARALLEL, ExecutionPlan, run_plan
from evaluation.grid import RunJob
from utils.experiment import make_environment
from utils.experiment_config import RunConfig

PROBE_TRAINING_STEPS = 300
# Gradient updates start once the buffer holds more episodes than one batch
PROBE_EXTRA_EPISODES = 2
PROBE_EVAL_EPISODES = 10
WORKER_CANDIDATES = (1, 2, 4, 8)
# A larger worker count must beat the best so far by this factor to be chosen
MIN_SPEEDUP = 1.1
SECONDS_PER_MINUTE = 60

logger = logging.getLogger(__name__)


def probe_prefill_steps(config: RunConfig) -> int:
    env, horizon = make_environment(config)
    env.close()
    return (config.batch + PROBE_EXTRA_EPISODES) * horizon


def probe_jobs(jobs: Sequence[RunJob], count: int) -> List[RunJob]:
    """Short copies of the grid's jobs, cycling over its models in grid order."""
    by_model: Dict[str, RunJob] = {}
    for job in jobs:
        by_model.setdefault(job.config.model, job)
    representatives = list(by_model.values())
    probes = []
    for index in range(count):
        source = representatives[index % len(representatives)].config
        config = replace(
            source,
            num_steps=PROBE_TRAINING_STEPS,
            prefill_steps=probe_prefill_steps(source),
            eval_episodes=PROBE_EVAL_EPISODES,
            training_seed=index + 1,
        )
        probes.append(RunJob(config, eval_interval=PROBE_TRAINING_STEPS))
    return probes


def measure_throughput(jobs: Sequence[RunJob], device: str, workers: int) -> float:
    """Probe runs finished per minute with one probe per worker; 0 on failure.

    Probes always run in worker processes, even with one worker, so this process
    never opens a CUDA context that would block later workers.
    """
    probes = probe_jobs(jobs, workers)
    plan = ExecutionPlan(device, workers, PARALLEL)
    with tempfile.TemporaryDirectory(prefix="grid-calibration-") as directory:
        start = perf_counter()
        failed = run_plan(
            probes, Path(directory) / "raw", Path(directory) / "runs", plan
        )
        elapsed = perf_counter() - start
    if failed:
        return 0.0
    return len(probes) * SECONDS_PER_MINUTE / elapsed


def choose_workers(throughputs: Dict[int, float]) -> int:
    """Smallest worker count whose throughput no larger count beats by MIN_SPEEDUP."""
    best_workers, best_throughput = 1, 0.0
    for workers in sorted(throughputs):
        if throughputs[workers] >= best_throughput * MIN_SPEEDUP:
            best_workers, best_throughput = workers, throughputs[workers]
    return best_workers


def candidate_workers(job_count: int) -> List[int]:
    limit = min(job_count, os.cpu_count() or 1)
    return [workers for workers in WORKER_CANDIDATES if workers <= limit] or [1]


def calibrate_workers(jobs: Sequence[RunJob], device: str) -> int:
    """Measure increasing worker counts and stop once adding workers stops paying."""
    throughputs: Dict[int, float] = {}
    for workers in candidate_workers(len(jobs)):
        throughputs[workers] = measure_throughput(jobs, device, workers)
        logger.info(
            "Calibration: %d worker(s) on %s -> %.2f probe runs/min",
            workers,
            device,
            throughputs[workers],
        )
        if throughputs[workers] == 0.0:
            # On a GPU, each worker's CUDA context needs its own memory
            logger.info("%d worker(s) failed on %s; not trying more", workers, device)
            break
        if choose_workers(throughputs) != workers:
            break
    chosen = choose_workers(throughputs)
    logger.info("Using %d worker(s) on %s", chosen, device)
    return chosen
