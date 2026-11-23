"""Portable artifacts, code identity, and phase resource measurements."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import torch

from utils.trajectories import json_safe

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return dict(data)


def code_identity() -> Dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    paths = [
        p
        for folder in ("dtqn", "envs", "utils")
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}
    ]
    paths += [
        ROOT / name
        for name in (
            "train.py",
            "eval.py",
            "train_and_eval.py",
            "run_legacy.py",
            "pyproject.toml",
            "requirements.txt",
        )
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.exists():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return {"git_commit": commit, "source_sha256": digest.hexdigest()}


def runtime_versions() -> Dict[str, str]:
    import gym
    import numpy

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": numpy.__version__,
        "gym": str(getattr(gym, "__version__", "unknown")),
    }


class ResourceMeter:
    """Reset before model allocation; synchronize CUDA before stopping the clock."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.start = perf_counter()
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only cpu and cuda devices are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA requested but unavailable in this PyTorch installation"
                )
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def finish(self) -> Dict[str, Any]:
        peak = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
        return {
            "wall_clock_seconds": perf_counter() - self.start,
            "peak_gpu_memory_bytes": int(peak),
            "device": str(self.device),
        }
