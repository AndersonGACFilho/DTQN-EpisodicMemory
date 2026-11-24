"""Grid execution: routing, executors, worker calibration and parallel runs."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from analysis.rliable_report import run_devices
from evaluation.calibration import (
    PROBE_EXTRA_EPISODES,
    PROBE_TRAINING_STEPS,
    candidate_workers,
    choose_workers,
    probe_jobs,
)
from evaluation.execution import (
    EXECUTORS,
    PARALLEL,
    SEQUENTIAL,
    ExecutionPlan,
    run_plan,
    run_tasks,
)
from evaluation.grid import RunJob, expand_jobs, load_grid_spec
from evaluation.records import EpisodeRecord, metadata_path, read_records, records_path
from evaluation.router import mode_for, resolve_device, resolve_workers, route
from tests.rliable_fixtures import grid_file
from utils.experiment_io import read_json, write_json

TINY_PARAMETERS: Dict[str, List[Any]] = {
    "model": ["DTQN", "DTQN-kNN"],
    "env": ["PassiveTMaze-v0"],
    "corridor_length": [3],
    "context": [2],
    "in_embed": [16],
    "heads": [2],
    "layers": [1],
    "batch": [2],
    "buf_size": [200],
    "num_steps": [7],
    "prefill_steps": [12],
}


def tiny_jobs(directory: Path) -> List[RunJob]:
    path = grid_file(directory, parameters=TINY_PARAMETERS, train_seeds=2)
    return expand_jobs(load_grid_spec(path))


def fail_second_seed(job: RunJob) -> None:
    """Module-level, so spawned workers can unpickle it."""
    if job.config.training_seed == 2:
        raise RuntimeError("deliberate failure")


def run_records(jobs: List[RunJob], raw_dir: Path) -> Dict[str, List[EpisodeRecord]]:
    return {job.run_id: read_records(records_path(raw_dir, job.run_id)) for job in jobs}


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), "cuda")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_explicit_device_is_kept(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cpu"), "cpu")


class RouterTests(unittest.TestCase):
    def test_one_worker_runs_sequentially_more_run_in_parallel(self) -> None:
        self.assertEqual(mode_for(1), SEQUENTIAL)
        self.assertEqual(mode_for(3), PARALLEL)

    def test_route_builds_the_plan(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            plan = route([], "auto", "3")
        self.assertEqual(plan, ExecutionPlan("cpu", 3, PARALLEL))

    def test_auto_workers_calibrate(self) -> None:
        with patch("evaluation.router.calibrate_workers", return_value=4) as calibrate:
            self.assertEqual(resolve_workers("auto", [], "cuda"), 4)
        calibrate.assert_called_once_with([], "cuda")

    def test_invalid_worker_counts_are_rejected(self) -> None:
        for requested in ("0", "-2", "two"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                resolve_workers(requested, [], "cpu")

    def test_every_mode_has_an_executor(self) -> None:
        self.assertEqual(set(EXECUTORS), {SEQUENTIAL, PARALLEL})

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown execution mode"):
            run_tasks(fail_second_seed, [], ExecutionPlan("cpu", 1, "threads"))


class ExecutorTests(unittest.TestCase):
    def test_both_modes_report_failures_and_keep_going(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            jobs = tiny_jobs(Path(directory))
        failing = sorted(job.run_id for job in jobs if job.config.training_seed == 2)
        for plan in (
            ExecutionPlan("cpu", 1, SEQUENTIAL),
            ExecutionPlan("cpu", 2, PARALLEL),
        ):
            with self.subTest(mode=plan.mode):
                self.assertEqual(
                    sorted(run_tasks(fail_second_seed, jobs, plan)), failing
                )


class CalibrationTests(unittest.TestCase):
    def test_more_workers_must_pay_off(self) -> None:
        # 4 workers is only 5% faster than 2, below the required speedup
        self.assertEqual(choose_workers({1: 10.0, 2: 19.0, 4: 20.0}), 2)
        self.assertEqual(choose_workers({1: 10.0, 2: 12.0, 4: 30.0}), 4)

    def test_failed_worker_count_is_not_chosen(self) -> None:
        self.assertEqual(choose_workers({1: 10.0, 2: 0.0}), 1)

    def test_candidates_never_exceed_jobs_or_cores(self) -> None:
        with patch("os.cpu_count", return_value=12):
            self.assertEqual(candidate_workers(3), [1, 2])
            self.assertEqual(candidate_workers(100), [1, 2, 4, 8])
        with patch("os.cpu_count", return_value=1):
            self.assertEqual(candidate_workers(100), [1])

    def test_probes_cycle_models_and_fill_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probes = probe_jobs(tiny_jobs(Path(directory)), 3)
        self.assertEqual(
            [probe.config.model for probe in probes], ["DTQN", "DTQN-kNN", "DTQN"]
        )
        horizon = TINY_PARAMETERS["corridor_length"][0] + 1
        for probe in probes:
            self.assertEqual(probe.config.num_steps, PROBE_TRAINING_STEPS)
            self.assertEqual(
                probe.config.prefill_steps,
                (probe.config.batch + PROBE_EXTRA_EPISODES) * horizon,
            )


class ParallelRunTests(unittest.TestCase):
    def test_parallel_runs_match_sequential_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = tiny_jobs(root)
            sequential = ExecutionPlan("cpu", 1, SEQUENTIAL)
            parallel = ExecutionPlan("cpu", 2, PARALLEL)
            self.assertEqual(
                run_plan(jobs, root / "seq" / "raw", root / "seq" / "runs", sequential),
                [],
            )
            self.assertEqual(
                run_plan(jobs, root / "par" / "raw", root / "par" / "runs", parallel),
                [],
            )
            self.assertEqual(
                run_records(jobs, root / "seq" / "raw"),
                run_records(jobs, root / "par" / "raw"),
            )
            execution = read_json(metadata_path(root / "par" / "raw", jobs[0].run_id))[
                "execution"
            ]
            self.assertEqual(execution["workers"], 2)
            self.assertEqual(execution["device"], "cpu")
            self.assertEqual(execution["mode"], PARALLEL)


class ReportDeviceTests(unittest.TestCase):
    def test_report_lists_every_training_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            jobs = tiny_jobs(raw_dir)
            for index, job in enumerate(jobs):
                device = "cuda" if index % 2 else "cpu"
                write_json(metadata_path(raw_dir, job.run_id), {"device": device})
            self.assertEqual(run_devices(jobs, raw_dir), ["cpu", "cuda"])


if __name__ == "__main__":
    unittest.main()
