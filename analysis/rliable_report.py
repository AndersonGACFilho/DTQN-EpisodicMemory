"""rliable report for a multi-seed grid: aggregate metrics, probability of
improvement, performance profiles, IQM learning curves and side-bias checks.

Usage: python -m analysis.rliable_report --grid experiments/rliable_tmaze.json
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from analysis import plots, tables
from analysis.criteria import HYPOTHESIS, describe_criteria, evaluate_hypothesis
from analysis.diagnostics import SIDE_BIAS_THRESHOLD, side_bias
from analysis.metrics import (
    AGGREGATE_METRICS,
    NORMALIZED_RETURN,
    PAIR_SEPARATOR,
    aggregate_estimates,
    build_score_data,
    comparison_pairs,
    improvement_estimates,
    iqm_curve_estimates,
    performance_profiles,
    score_spread,
    task_label,
)
from evaluation.grid import RunJob, expand_jobs, is_complete, load_grid_spec
from evaluation.records import EpisodeRecord, metadata_path, read_records, records_path
from utils.experiment_io import read_json

DEVELOPMENT_REPS = 2_000
# Tables also written as LaTeX floats for the paper: name -> caption
LATEX_CAPTIONS = {
    "aggregate_metrics": (
        "Final success rate over training seeds: point estimate, 95% stratified "
        "bootstrap confidence interval and standard deviation over runs."
    ),
    "probability_of_improvement": (
        "Probability that a run of the candidate beats a run of the baseline on the "
        "same task, with 95% bootstrap confidence interval."
    ),
    "aggregate_metrics_return": (
        "Final return, min-max normalized per task over all runs: point estimate, "
        "95% stratified bootstrap confidence interval and standard deviation."
    ),
    "hypothesis_criteria": (
        "Hypothesis criteria per DTQN-kNN/DTQN pair (issue #1). " + describe_criteria()
    ),
    "side_bias_summary": (
        "Runs whose final evaluation ends in the same branch in more than 90% of "
        "episodes (fixed-side policies), and mean final success rate."
    ),
}
PROFILE_THRESHOLDS = np.linspace(0.0, 1.0, 101)

logger = logging.getLogger("rliable_report")


class MissingRunsError(RuntimeError):
    pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, help="Number of training seeds (default: from the grid)"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--reps",
        type=int,
        default=DEVELOPMENT_REPS,
        help="Bootstrap replications (default: 2000; use 50000 for final results)",
    )
    return parser.parse_args()


def load_grid_records(jobs: Sequence[RunJob], raw_dir: Path) -> List[EpisodeRecord]:
    missing = [job.run_id for job in jobs if not is_complete(job, raw_dir)]
    if missing:
        raise MissingRunsError(
            f"{len(missing)} of {len(jobs)} runs have no results in {raw_dir}: "
            + ", ".join(missing)
        )
    return [
        record
        for job in jobs
        for record in read_records(records_path(raw_dir, job.run_id))
    ]


def code_versions(jobs: Sequence[RunJob], raw_dir: Path) -> List[Dict[str, Any]]:
    versions = {
        (code["git_commit"], code["source_sha256"])
        for job in jobs
        for code in [read_json(metadata_path(raw_dir, job.run_id))["code"]]
    }
    return [
        {"git_commit": commit, "source_sha256": digest}
        for commit, digest in sorted(versions)
    ]


def run_devices(jobs: Sequence[RunJob], raw_dir: Path) -> List[str]:
    """Devices the runs trained on; mixing them adds hardware numerics to the comparison."""
    return sorted(
        {read_json(metadata_path(raw_dir, job.run_id))["device"] for job in jobs}
    )


def write_tables(
    summary_dir: Path, named_rows: Dict[str, List[tables.Row]], header: str
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    sections = [header]
    for name, rows in named_rows.items():
        tables.write_csv(summary_dir / f"{name}.csv", rows)
        if name in LATEX_CAPTIONS:
            label = "tab:" + name.replace("_", "-")
            latex = tables.latex_table(
                tables.with_intervals(rows), LATEX_CAPTIONS[name], label
            )
            (summary_dir / f"{name}.tex").write_text(latex)
        title = name.replace("_", " ").capitalize()
        sections.append(f"## {title}\n\n{tables.markdown_table(rows)}")
    (summary_dir / "summary.md").write_text("\n".join(sections))


def report_header(
    reps: int,
    seeds: Sequence[int],
    tasks: Sequence[Any],
    steps: Sequence[int],
    code: Any,
    devices: Sequence[str],
) -> str:
    return (
        "# rliable summary\n\n"
        f"- Training seeds: {list(seeds)}\n"
        f"- Tasks: {[task_label(task) for task in tasks]}\n"
        f"- Final step: {steps[-1]}; evaluation steps: {list(steps)}\n"
        f"- 95% stratified bootstrap CIs, {reps} replications\n"
        f"- Fixed-side policy: more than {SIDE_BIAS_THRESHOLD:.0%} of final "
        "evaluation episodes end in the same branch\n"
        f"- Code versions: {code}\n"
        f"- Devices: {list(devices)}\n"
        f"- Hypothesis criteria: {describe_criteria()}\n"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_arguments()
    spec = load_grid_spec(args.grid, args.seeds)
    jobs = expand_jobs(spec)
    records = load_grid_records(jobs, args.raw_dir)
    code = code_versions(jobs, args.raw_dir)
    if len(code) > 1:
        logger.warning("Runs come from %d different code versions", len(code))
    devices = run_devices(jobs, args.raw_dir)
    if len(devices) > 1:
        logger.warning("Runs trained on different devices: %s", devices)
    data = build_score_data(records)
    final_scores = data.final_scores
    metric_names = list(AGGREGATE_METRICS)
    pairs = comparison_pairs(data.algorithms, spec.comparisons)

    logger.info("Bootstrapping with %d replications", args.reps)
    aggregates = aggregate_estimates(final_scores, args.reps)
    profiles = performance_profiles(final_scores, PROFILE_THRESHOLDS, args.reps)
    curves = iqm_curve_estimates(data.curves, args.reps)
    diagnostics = side_bias(records)
    flagged = [result for result in diagnostics if result.flagged]
    if flagged:
        logger.warning(
            "%d of %d runs follow a fixed side", len(flagged), len(diagnostics)
        )

    colors = plots.algorithm_colors(list(final_scores))
    figures_dir = args.output_dir / "figures"
    plots.save_figure(
        plots.plot_aggregates(aggregates, metric_names, colors),
        figures_dir,
        "aggregate_metrics",
    )
    plots.save_figure(
        plots.plot_profiles(profiles, PROFILE_THRESHOLDS, colors),
        figures_dir,
        "performance_profiles",
    )
    plots.save_figure(
        plots.plot_learning_curves(curves, data.steps, colors),
        figures_dir,
        "iqm_learning_curves",
    )
    returns = build_score_data(records, NORMALIZED_RETURN).final_scores
    return_aggregates = aggregate_estimates(returns, args.reps)
    plots.save_figure(
        plots.plot_aggregates(return_aggregates, metric_names, colors),
        figures_dir,
        "aggregate_metrics_return",
    )
    named_rows = {
        "aggregate_metrics": tables.aggregate_rows(
            aggregates, metric_names, score_spread(final_scores)
        ),
        "aggregate_metrics_return": tables.aggregate_rows(
            return_aggregates, metric_names, score_spread(returns)
        ),
        "side_bias_summary": tables.side_bias_summary_rows(diagnostics),
        "side_bias": tables.side_bias_rows(diagnostics),
    }
    if pairs:
        improvements = improvement_estimates(final_scores, pairs, args.reps)
        plots.save_figure(
            plots.plot_improvement(improvements),
            figures_dir,
            "probability_of_improvement",
        )
        named_rows["probability_of_improvement"] = tables.improvement_rows(
            improvements, PAIR_SEPARATOR
        )
        if tuple(HYPOTHESIS) in spec.comparisons:
            verdicts = evaluate_hypothesis(
                data.algorithms, data.tasks, aggregates, improvements, metric_names
            )
            named_rows["hypothesis_criteria"] = tables.criteria_rows(verdicts)
    header = report_header(args.reps, data.seeds, data.tasks, data.steps, code, devices)
    write_tables(args.output_dir / "summary", named_rows, header)
    logger.info(
        "Wrote figures to %s and tables to %s", figures_dir, args.output_dir / "summary"
    )


if __name__ == "__main__":
    main()
