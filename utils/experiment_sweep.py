"""Sequential process orchestration and aggregation of saved per-seed results."""

import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from scipy.stats import t

from utils.experiment import read_seed_result, write_seed_result
from utils.experiment_config import (
    RunConfig,
    config_cli,
    configuration_id,
    expand_grid,
    normalize_saved_parameters,
)
from utils.experiment_io import (
    ROOT,
    code_identity,
    read_json,
    runtime_versions,
    write_json,
)
from utils.logging_utils import RecordCSV

METRICS = [
    "success_rate",
    "mean_return",
    "mean_episode_length",
    "train_wall_clock_seconds",
    "eval_wall_clock_seconds",
    "total_wall_clock_seconds",
    "train_peak_gpu_memory_bytes",
    "eval_peak_gpu_memory_bytes",
    "peak_gpu_memory_bytes",
]


def describe(values: List[float]) -> Dict[str, Optional[float]]:
    mean = statistics.mean(values)
    if len(values) == 1:
        return {"mean": mean, "std": None, "ci95_low": None, "ci95_high": None}
    std = statistics.stdev(values)
    half = float(t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
    return {"mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def make_cases(grid: Mapping[str, List[Any]]) -> List[Dict[str, Any]]:
    cases = []
    for params in expand_grid(grid):
        case: Dict[str, Any] = {
            "id": configuration_id(params, include_training_seed=True),
            "config_id": configuration_id(params),
            "parameters": params,
            "status": "pending",
            "attempts": [],
        }
        try:
            RunConfig(**params).validate()
        except ValueError as error:
            case.update(status="invalid", error=str(error))
        cases.append(case)
    return cases


def new_batch(
    grid: Mapping[str, List[Any]], output_root: Path, project_name: str
) -> Path:
    if (
        not project_name
        or Path(project_name).name != project_name
        or project_name in {".", ".."}
    ):
        raise ValueError("project-name must be one directory name")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch = (output_root / project_name / f"{stamp}-{uuid4().hex[:8]}").resolve()
    batch.mkdir(parents=True)
    write_json(batch / "config.json", grid)
    write_json(
        batch / "manifest.json",
        {
            "schema_version": 1,
            "grid": grid,
            "code": code_identity(),
            "versions": runtime_versions(),
            "cases": make_cases(grid),
            "created_utc": stamp,
        },
    )
    return batch


def check_resume(manifest: Mapping[str, Any], grid: Mapping[str, List[Any]]) -> None:
    if manifest["grid"] != grid:
        raise ValueError("Resume config differs from the original grid")
    if manifest["code"] != code_identity():
        raise ValueError(
            "Resume requires the original code identity; use a new batch for changed code"
        )
    if manifest["versions"] != runtime_versions():
        raise ValueError("Resume requires the original Python/library versions")


def run_stage(
    batch: Path,
    manifest: Dict[str, Any],
    case: Dict[str, Any],
    phase: str,
    device: str,
    verbose: bool,
) -> bool:
    case_dir = batch / "runs" / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = 1 + sum(a["phase"] == phase for a in case["attempts"])
    destination = case_dir / f"{phase}-{attempt_number:03d}"
    log_path = destination.with_suffix(".log")
    params = case["parameters"]
    command = [
        sys.executable,
        str(ROOT / f"{phase}.py"),
        "--output-dir",
        str(destination),
        "--device",
        device,
    ]
    if verbose:
        command.append("--verbose")
    if phase == "train":
        command.extend(config_cli(params))
    else:
        command.extend(
            [
                "--model-path",
                str(batch / case["train_path"] / "model.pt"),
                "--eval-episodes",
                str(params["eval_episodes"]),
                "--eval-seed",
                str(params["eval_seed"]),
            ]
        )
    attempt: Dict[str, Any] = {
        "phase": phase,
        "path": str(destination.relative_to(batch)),
        "log": str(log_path.relative_to(batch)),
        "command": command,
        "status": "running",
        "device": device,
    }
    case["attempts"].append(attempt)
    case["status"] = f"{phase}_running"
    write_json(batch / "manifest.json", manifest)
    started = perf_counter()
    try:
        with log_path.open("w") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        attempt["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"{phase} exited {process.returncode}; see {log_path}")
        metadata_path = destination / f"{phase}_metadata.json"
        metadata = read_json(metadata_path)
        if (
            metadata["code"] != manifest["code"]
            or metadata["versions"] != manifest["versions"]
        ):
            raise ValueError(
                "Code or library versions changed during this batch; start a new batch"
            )
        if metadata["status"] != "complete" or (
            phase == "train" and not (destination / "model.pt").is_file()
        ):
            raise ValueError(f"Incomplete {phase} artifacts in {destination}")
        case[f"{phase}_path"] = str(destination.relative_to(batch))
        attempt["status"] = "complete"
        case["status"] = f"{phase}_complete"
        case.pop("error", None)
        success = True
    except Exception as error:
        attempt["status"] = "failed"
        attempt["error"] = str(error)
        case.update(status=f"{phase}_failed", error=str(error))
        success = False
    finally:
        attempt["process_wall_clock_seconds"] = perf_counter() - started
        write_json(batch / "manifest.json", manifest)
    return success


def execute_batch(batch: Path, device: str = "cpu", verbose: bool = False) -> int:
    manifest = read_json(batch / "manifest.json")
    # A process interrupted in a previous invocation is retained as an attempt.
    for case in manifest["cases"]:
        for attempt in case["attempts"]:
            if attempt["status"] == "running":
                attempt.update(
                    status="interrupted", error="Previous invocation did not finish"
                )
    for index, case in enumerate(manifest["cases"], start=1):
        if case["status"] == "invalid":
            continue
        # Reuse only complete, present stage artifacts. A lost file starts a new
        # attempt; existing attempt folders are never overwritten.
        for phase, required in (
            ("train", ("train_metadata.json", "model.pt")),
            (
                "eval",
                (
                    "eval_metadata.json",
                    "eval_episodes.csv",
                    "eval_trajectories.jsonl.gz",
                ),
            ),
        ):
            path = case.get(f"{phase}_path")
            if path and not all((batch / path / name).is_file() for name in required):
                case.pop(f"{phase}_path")
                if phase == "train":
                    case.pop("eval_path", None)
                case["status"] = f"{phase}_artifacts_missing"
        print(
            f"[{index}/{len(manifest['cases'])}] {case['id']} training_seed={case['parameters']['training_seed']}",
            flush=True,
        )
        if "train_path" not in case and not run_stage(
            batch, manifest, case, "train", device, verbose
        ):
            print(case["error"], flush=True)
            continue
        if "eval_path" not in case and not run_stage(
            batch, manifest, case, "eval", device, verbose
        ):
            print(case["error"], flush=True)
            continue
        try:
            write_seed_result(
                batch / case["train_path"],
                batch / case["eval_path"],
                batch / "runs" / case["id"] / "result.csv",
            )
            case["status"] = "complete"
            case.pop("error", None)
        except Exception as error:
            case.update(status="result_failed", error=str(error))
        write_json(batch / "manifest.json", manifest)
    summaries = aggregate_batch(batch)
    return int(any(row["status"] == "incomplete" for row in summaries))


def aggregate_batch(batch: Path) -> List[Dict[str, Any]]:
    """Read one canonical per-seed result; never load a policy or run an environment."""
    manifest = read_json(batch / "manifest.json")
    runs: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    seen = set()
    for case in manifest["cases"]:
        case["parameters"] = normalize_saved_parameters(case["parameters"])
        if case["id"] in seen:
            raise ValueError(f"Duplicate case in manifest: {case['id']}")
        seen.add(case["id"])
        row = {
            "run_id": case["id"],
            "config_id": case["config_id"],
            **case["parameters"],
            "status": case["status"],
            "error": case.get("error", ""),
            "attempts": len(case["attempts"]),
            "process_wall_clock_seconds": sum(
                a.get("process_wall_clock_seconds", 0) for a in case["attempts"]
            ),
            "result_path": str(batch / "runs" / case["id"] / "result.csv"),
        }
        if case["status"] == "complete":
            try:
                saved = read_seed_result(Path(row["result_path"]))
                for key, value in case["parameters"].items():
                    if saved.get(key) != str(value):
                        raise ValueError(f"Result parameters disagree: {key}")
                for metric in METRICS:
                    if not math.isfinite(float(saved[metric])):
                        raise ValueError(f"Nonfinite saved metric: {metric}")
                for key in ("source_sha256", "git_commit"):
                    if saved[key] != manifest["code"][key]:
                        raise ValueError(f"Result code identity disagrees: {key}")
                row.update(saved)
            except (OSError, ValueError, KeyError) as error:
                row.update(status="artifacts_invalid", error=str(error))
        runs.append(row)
        groups.setdefault(case["config_id"], []).append(row)
    columns = list(dict.fromkeys(key for row in runs for key in row))
    with RecordCSV(batch / "runs.csv", columns) as output:
        for row in runs:
            output.write(row)
    summaries = []
    for config_id, members in groups.items():
        complete = [row for row in members if row["status"] == "complete"]
        missing = [
            int(row["training_seed"]) for row in members if row["status"] != "complete"
        ]
        representative = next(
            c["parameters"] for c in manifest["cases"] if c["config_id"] == config_id
        )
        summary: Dict[str, Any] = {
            "config_id": config_id,
            **{
                key: value
                for key, value in representative.items()
                if key != "training_seed"
            },
            "expected_seeds": len(members),
            "completed_seeds": len(complete),
            "missing_seeds": json.dumps(sorted(missing)),
            "status": "incomplete"
            if missing
            else "single_seed"
            if len(complete) == 1
            else "complete",
        }
        for metric in METRICS:
            stats = (
                describe([float(row[metric]) for row in complete])
                if not missing
                else dict.fromkeys(["mean", "std", "ci95_low", "ci95_high"])
            )
            summary.update({f"{metric}_{key}": value for key, value in stats.items()})
        summaries.append(summary)
    with RecordCSV(
        batch / "summary.csv", list(summaries[0]) if summaries else []
    ) as output:
        for summary in summaries:
            output.write(summary)
    return summaries
