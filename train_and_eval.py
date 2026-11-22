"""Run a Cartesian JSON sweep, then compare saved final-policy evaluations."""

import argparse
from pathlib import Path

from utils.experiment_config import load_grid
from utils.experiment_io import read_json
from utils.experiment_sweep import (
    aggregate_batch,
    check_resume,
    execute_batch,
    make_cases,
    new_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Flat JSON grid; the only scientific parameter input",
    )
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device for all jobs (default: cpu)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Batch output root (default: results)",
    )
    parser.add_argument(
        "--project-name", help="Output subdirectory (default: config filename stem)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Extra train/eval output in process logs"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all combinations and validation errors; launch nothing",
    )
    modes.add_argument(
        "--resume",
        type=Path,
        help="Reuse completed stages from this batch; retry failures",
    )
    modes.add_argument(
        "--aggregate-only",
        type=Path,
        help="Regenerate runs.csv/summary.csv using saved results only",
    )
    args = parser.parse_args()
    try:
        if args.aggregate_only:
            summaries = aggregate_batch(args.aggregate_only.resolve())
            print(
                f"Aggregated {len(summaries)} configurations in {args.aggregate_only}"
            )
            return int(any(row["status"] == "incomplete" for row in summaries))
        config_path = args.config
        if config_path is None and args.resume:
            config_path = args.resume / "config.json"
        if config_path is None:
            parser.error("--config is required for a new batch")
        grid = load_grid(config_path)
        if args.dry_run:
            cases = make_cases(grid)
            for case in cases:
                print(
                    f"{case['id']}: {case['parameters']} "
                    f"status={case['status']} {case.get('error', '')}"
                )
            print(f"{len(cases)} train/evaluation pairs")
            return int(any(case["status"] == "invalid" for case in cases))
        if args.resume:
            batch = args.resume.resolve()
            check_resume(read_json(batch / "manifest.json"), grid)
        else:
            batch = new_batch(
                grid, args.output_root, args.project_name or config_path.stem
            )
        print(f"Batch: {batch}", flush=True)
        result = execute_batch(batch, args.device, args.verbose)
        print(f"Saved {batch / 'runs.csv'} and {batch / 'summary.csv'}")
        return result
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
