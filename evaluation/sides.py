"""Junction sides per environment, used to detect fixed-side policies.

Register a new environment by adding its reader to SIDE_READERS.
"""

from typing import Any, Callable, Dict, Optional, Tuple

UP = "up"
DOWN = "down"
NO_BRANCH = "none"

# (goal side, chosen side), read after the final step and before the next reset
Sides = Tuple[Optional[str], Optional[str]]
SideReader = Callable[[Any], Sides]


def _tmaze_side(y: int) -> str:
    return {1: UP, -1: DOWN}.get(y, NO_BRANCH)


def read_tmaze_sides(env: Any) -> Sides:
    maze = env.unwrapped
    return _tmaze_side(maze.goal_y), _tmaze_side(maze.y)


def read_no_sides(env: Any) -> Sides:
    return None, None


SIDE_READERS: Dict[str, SideReader] = {
    "PassiveTMaze-v0": read_tmaze_sides,
    "NoisyTMaze-v0": read_tmaze_sides,
}


def side_reader_for(env_name: str) -> SideReader:
    return SIDE_READERS.get(env_name, read_no_sides)
