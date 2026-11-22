"""Stream observations/actions and passive world snapshots for later plotting."""

import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def world_state(env: Any) -> Dict[str, Any]:
    """Read existing fields only: no reset, step, render, or random calls."""
    base = env.unwrapped
    name = type(base).__name__
    fields = {
        "PassiveTMaze": ("x", "y", "time_step", "goal_y", "corridor_length"),
        "NoisyTMaze": ("x", "y", "time_step", "goal_y", "corridor_length"),
        "AssociativeCorridor": (
            "position",
            "time_step",
            "correct_answers",
            "corridor_length",
            "instructions",
            "queries",
            "directions",
        ),
        "CarFlag": (
            "state",
            "heaven_position",
            "hell_position",
            "priest_position",
            "priest_delta",
            "min_position",
            "max_position",
        ),
        "Memory": (
            "state",
            "observation",
            "current_card",
            "num_pairs",
            "card_hidden",
            "card_removed",
        ),
    }.get(name, ())
    return {
        "adapter": name if fields else "observation_only",
        **{key: json_safe(getattr(base, key)) for key in fields},
    }


class TrajectoryWriter:
    def __init__(self, path: Path) -> None:
        self.file = gzip.open(path, "wt", encoding="utf-8")

    def write(self, event: str, **record: Any) -> None:
        self.file.write(
            json.dumps(
                json_safe({"schema_version": 1, "event": event, **record}),
                allow_nan=False,
            )
            + "\n"
        )
        if event == "episode_end":
            self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
