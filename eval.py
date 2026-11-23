"""Evaluate one final model greedily, using its saved environment settings."""

import argparse
from pathlib import Path

from utils.experiment import eval_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New/empty artifact directory"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Fixed number of greedy episodes (default: 100)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=10_000,
        help="Evaluation environment seed (default: 10000)",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        eval_run(
            args.model_path,
            args.output_dir,
            args.device,
            args.eval_episodes,
            args.eval_seed,
            args.verbose,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
