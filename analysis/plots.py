"""Figures from precomputed estimates; no metric is calculated here."""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from rliable import plot_utils  # noqa: E402

from analysis.metrics import PAIR_SEPARATOR, Estimates, Scores  # noqa: E402

# Paired palette: consecutive algorithms (e.g. DTQN, DTQN-kNN) share a hue
PALETTE = "tab20"
FIGURE_FORMATS = ("png", "pdf")
SCORE_LABEL = "Success rate"
COLUMN_WIDTH = 7.0
ROW_HEIGHT = 0.45

Colors = Dict[str, Tuple[float, float, float]]


def algorithm_colors(algorithms: Sequence[str]) -> Colors:
    return dict(zip(algorithms, sns.color_palette(PALETTE, n_colors=len(algorithms))))


def save_figure(figure: Figure, directory: Path, name: str) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.{extension}" for extension in FIGURE_FORMATS]
    for path in paths:
        figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_aggregates(
    estimates: Estimates, metric_names: Sequence[str], colors: Colors
) -> Figure:
    points, intervals = estimates
    # rliable places a shared x label at a fixed height that overlaps short figures
    figure, axes = plot_utils.plot_interval_estimates(
        points,
        intervals,
        metric_names=list(metric_names),
        algorithms=list(colors),
        colors=colors,
        xlabel="",
    )
    for axis in np.atleast_1d(axes):
        axis.set_xlabel(SCORE_LABEL, fontsize="large")
    return figure


def plot_improvement(estimates: Estimates) -> Figure:
    points, intervals = estimates
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 1 + ROW_HEIGHT * len(points)))
    plot_utils.plot_probability_of_improvement(
        points,
        {pair: interval.reshape(-1) for pair, interval in intervals.items()},
        pair_separator=PAIR_SEPARATOR,
        ax=axis,
        color_palette=PALETTE,
    )
    return figure


def plot_profiles(estimates: Estimates, thresholds: Scores, colors: Colors) -> Figure:
    profiles, intervals = estimates
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / 2))
    plot_utils.plot_performance_profiles(
        profiles,
        thresholds,
        performance_profile_cis=intervals,
        colors=colors,
        xlabel=r"Success rate ($\tau$)",
        ax=axis,
    )
    axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return figure


def plot_learning_curves(
    estimates: Estimates, steps: Sequence[int], colors: Colors
) -> Figure:
    points, intervals = estimates
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / 1.5))
    plot_utils.plot_sample_efficiency_curve(
        list(steps),
        points,
        intervals,
        algorithms=list(colors),
        colors=colors,
        xlabel="Training steps",
        ylabel=f"IQM {SCORE_LABEL.lower()}",
        ax=axis,
        labelsize="large",
        ticklabelsize="large",
    )
    axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return figure
