"""Tidy per-episode evaluation records and their CSV storage."""

import csv
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

NO_SIDE = ""


@dataclass(frozen=True)
class EpisodeRecord:
    model: str
    env: str
    context: int
    layers: int
    corridor_length: int
    train_seed: int
    eval_seed: int
    step: int
    episode: int
    success: bool
    episode_return: float
    episode_length: int
    goal_side: Optional[str]
    chosen_side: Optional[str]


COLUMNS = [field.name for field in fields(EpisodeRecord)]


def _parse_bool(text: str) -> bool:
    if text not in {"0", "1"}:
        raise ValueError(f"Expected 0 or 1, got {text!r}")
    return text == "1"


def _parse_side(text: str) -> Optional[str]:
    return None if text == NO_SIDE else text


_PARSERS: Dict[str, Callable[[str], object]] = {
    "model": str,
    "env": str,
    "context": int,
    "layers": int,
    "corridor_length": int,
    "train_seed": int,
    "eval_seed": int,
    "step": int,
    "episode": int,
    "success": _parse_bool,
    "episode_return": float,
    "episode_length": int,
    "goal_side": _parse_side,
    "chosen_side": _parse_side,
}


def _format(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    return NO_SIDE if value is None else value


def write_records(path: Path, records: Iterable[EpisodeRecord]) -> None:
    """Write atomically, so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        for record in records:
            writer.writerow([_format(value) for value in astuple(record)])
    temporary.replace(path)


def read_records(path: Path) -> List[EpisodeRecord]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != COLUMNS:
            raise ValueError(
                f"{path} has columns {reader.fieldnames}; expected {COLUMNS}"
            )
        return [
            EpisodeRecord(**{key: _PARSERS[key](row[key]) for key in COLUMNS})  # type: ignore[arg-type]
            for row in reader
        ]


def records_path(raw_dir: Path, run_id: str) -> Path:
    return raw_dir / f"{run_id}.csv"


def metadata_path(raw_dir: Path, run_id: str) -> Path:
    return raw_dir / f"{run_id}.json"
