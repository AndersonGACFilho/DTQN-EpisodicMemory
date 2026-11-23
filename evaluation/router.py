"""Route a grid to a device, a worker count and an execution mode.

The router only decides; it never creates CUDA tensors, so worker processes can
still open their own CUDA contexts afterwards.
"""

from typing import Sequence

import torch

from evaluation.calibration import calibrate_workers
from evaluation.execution import PARALLEL, SEQUENTIAL, ExecutionPlan
from evaluation.grid import RunJob

AUTO = "auto"


def resolve_device(requested: str) -> str:
    """'auto' picks CUDA when PyTorch can use it, otherwise the CPU."""
    if requested != AUTO:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_workers(requested: str, jobs: Sequence[RunJob], device: str) -> int:
    if requested == AUTO:
        return calibrate_workers(jobs, device)
    if not requested.isdigit() or int(requested) <= 0:
        raise ValueError(
            f"workers must be a positive integer or '{AUTO}', got {requested!r}"
        )
    return int(requested)


def mode_for(workers: int) -> str:
    return SEQUENTIAL if workers == 1 else PARALLEL


def route(jobs: Sequence[RunJob], device: str, workers: str) -> ExecutionPlan:
    resolved_device = resolve_device(device)
    resolved_workers = resolve_workers(workers, jobs, resolved_device)
    return ExecutionPlan(resolved_device, resolved_workers, mode_for(resolved_workers))
