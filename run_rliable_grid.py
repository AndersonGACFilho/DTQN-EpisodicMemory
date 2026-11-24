"""Train and evaluate every (configuration, training seed) of a multi-seed grid.

Runs with saved results are skipped, so the command can be repeated to resume.
Use --only to run a subset, e.g. one configuration or seed per machine.
"""

import argparse
import logging
from pathlib import Path

from evaluation.execution import LOG_FORMAT, run_plan
from evaluation.grid import (
    expand_jobs,
    is_complete,
    load_grid_spec,
    parse_filters,
    select_jobs,
)
from evaluation.router import AUTO, route

DEFAULT_GRID = Path("experiments/rliable_tmaze.json")
DEFAULT_RAW_DIR = Path("results/raw")
DEFAULT_ARTIFACTS_DIR = Path("results/runs")

logger = logging.getLogger("run_rliable_grid")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument(
        "--seeds", type=int, help="Number of training seeds (default: from the grid)"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Run only matching jobs; repeat to combine, e.g. --only training_seed=3",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--device",
        default=AUTO,
        help="PyTorch device; 'auto' uses CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--workers",
        default="1",
        help="Parallel runs, or 'auto' to calibrate on short probes (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List the jobs and their status only"
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_arguments()
    spec = load_grid_spec(args.grid, args.seeds)
    jobs = select_jobs(expand_jobs(spec), parse_filters(args.only))
    pending = [job for job in jobs if not is_complete(job, args.raw_dir)]
    logger.info(
        "%d jobs selected, %d already complete", len(jobs), len(jobs) - len(pending)
    )
    if args.dry_run:
        for job in jobs:
            logger.info(
                "%s %s", "pending " if job in pending else "complete", job.run_id
            )
        return 0
    if not pending:
        return 0
    plan = route(pending, args.device, args.workers)
    logger.info(
        "Running %d jobs on %s, %s with %d worker(s)",
        len(pending),
        plan.device,
        plan.mode,
        plan.workers,
    )
    failed = run_plan(pending, args.raw_dir, args.artifacts_dir, plan)
    if failed:
        logger.error("%d of %d runs failed: %s", len(failed), len(pending), failed)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
