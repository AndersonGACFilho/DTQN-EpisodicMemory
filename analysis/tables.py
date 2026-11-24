"""Summary rows and their CSV/Markdown files."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from analysis.criteria import Verdict
from analysis.diagnostics import SideBias
from analysis.metrics import Estimates

Row = Dict[str, Any]
DECIMALS = 3
INTERVAL_DECIMALS = 2
LOW_SUFFIX = " CI low"
HIGH_SUFFIX = " CI high"
IMPROVEMENT = "P(candidate > baseline)"
LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _format(value: Any) -> Any:
    return round(float(value), DECIMALS) if isinstance(value, float) else value


def aggregate_rows(
    estimates: Estimates, metric_names: Sequence[str], spreads: Dict[str, float]
) -> List[Row]:
    """Point estimates with CIs, plus the standard deviation over runs."""
    points, intervals = estimates
    rows = []
    for algorithm, values in points.items():
        row: Row = {"algorithm": algorithm}
        for index, metric in enumerate(metric_names):
            row[metric] = float(values[index])
            row[f"{metric} CI low"] = float(intervals[algorithm][0, index])
            row[f"{metric} CI high"] = float(intervals[algorithm][1, index])
        row["Std"] = spreads[algorithm]
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
                IMPROVEMENT: float(probability),
                IMPROVEMENT + LOW_SUFFIX: float(lower),
                IMPROVEMENT + HIGH_SUFFIX: float(upper),
            }
        )
    return rows


def criteria_rows(verdicts: Sequence[Verdict]) -> List[Row]:
    return [
        {
            "candidate": verdict.candidate,
            "baseline": verdict.baseline,
            "regime": verdict.regime,
            "candidate IQM": verdict.candidate_iqm,
            "baseline IQM": verdict.baseline_iqm,
            IMPROVEMENT: verdict.improvement,
            IMPROVEMENT + LOW_SUFFIX: verdict.improvement_low,
            IMPROVEMENT + HIGH_SUFFIX: verdict.improvement_high,
            "met": "yes" if verdict.met else "no",
        }
        for verdict in verdicts
    ]


def side_bias_summary_rows(diagnostics: Sequence[SideBias]) -> List[Row]:
    """Fixed-side runs per algorithm and task, for a compact table."""
    groups: Dict[Tuple[str, str], List[SideBias]] = {}
    for result in diagnostics:
        groups.setdefault((result.algorithm, result.task), []).append(result)
    return [
        {
            "algorithm": algorithm,
            "task": task,
            "fixed-side runs": f"{sum(r.flagged for r in results)}/{len(results)}",
            "mean success rate": sum(r.success_rate for r in results) / len(results),
        }
        for (algorithm, task), results in groups.items()
    ]


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


def with_intervals(rows: Sequence[Row]) -> List[Row]:
    """Merge 'X', 'X CI low' and 'X CI high' columns into one 'X [low, high]' column."""
    merged = []
    for row in rows:
        new_row: Row = {}
        for column, value in row.items():
            if column.endswith((LOW_SUFFIX, HIGH_SUFFIX)):
                continue
            low, high = row.get(column + LOW_SUFFIX), row.get(column + HIGH_SUFFIX)
            if low is None or high is None:
                new_row[column] = value
            else:
                digits = INTERVAL_DECIMALS
                new_row[column] = (
                    f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
                )
        merged.append(new_row)
    return merged


def _latex_text(value: Any) -> str:
    text = str(_format(value))
    return "".join(LATEX_ESCAPES.get(character, character) for character in text)


def latex_table(rows: Sequence[Row], caption: str, label: str) -> str:
    """A booktabs table float; the first column is left-aligned, the rest right."""
    if not rows:
        raise ValueError("No rows for the LaTeX table")
    columns = list(rows[0])
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{_latex_text(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_latex_text(column) for column in columns) + r" \\",
        r"\midrule",
        *(
            " & ".join(_latex_text(row[column]) for column in columns) + r" \\"
            for row in rows
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
