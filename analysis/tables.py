"""Summary rows and their CSV/Markdown files."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from analysis.diagnostics import SideBias
from analysis.metrics import Estimates

Row = Dict[str, Any]
DECIMALS = 3


def _format(value: Any) -> Any:
    return round(float(value), DECIMALS) if isinstance(value, float) else value


def aggregate_rows(estimates: Estimates, metric_names: Sequence[str]) -> List[Row]:
    points, intervals = estimates
    rows = []
    for algorithm, values in points.items():
        row: Row = {"algorithm": algorithm}
        for index, metric in enumerate(metric_names):
            row[metric] = float(values[index])
            row[f"{metric} CI low"] = float(intervals[algorithm][0, index])
            row[f"{metric} CI high"] = float(intervals[algorithm][1, index])
        rows.append(row)
    return rows


def improvement_rows(estimates: Estimates, separator: str) -> List[Row]:
    points, intervals = estimates
    rows = []
    for pair, probability in points.items():
        candidate, baseline = pair.split(separator)
        lower, upper = intervals[pair].reshape(-1)
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "P(candidate > baseline)": float(probability),
                "CI low": float(lower),
                "CI high": float(upper),
            }
        )
    return rows


def side_bias_rows(diagnostics: Sequence[SideBias]) -> List[Row]:
    return [
        {
            "algorithm": result.algorithm,
            "task": result.task,
            "train_seed": result.train_seed,
            "step": result.step,
            "episodes": result.episodes,
            "success_rate": result.success_rate,
            **{f"chose {side}": share for side, share in result.side_fractions.items()},
            "dominant_side": result.dominant_side,
            "fixed_side_policy": result.flagged,
        }
        for result in diagnostics
    ]


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Row]) -> str:
    if not rows:
        raise ValueError("No rows for the Markdown table")
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        *(
            "| " + " | ".join(str(_format(row[column])) for column in columns) + " |"
            for row in rows
        ),
    ]
    return "\n".join(lines) + "\n"
