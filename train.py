"""Train one configuration, without intermediate evaluation."""

import argparse
from pathlib import Path

from utils.experiment import train_run
from utils.experiment_config import add_config_arguments, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New/empty artifact directory"
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        train_run(config_from_args(args), args.output_dir, args.device, args.verbose)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
