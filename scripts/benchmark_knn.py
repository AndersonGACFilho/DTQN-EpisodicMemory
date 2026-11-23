"""Compare gradient-update cost on identical synthetic replay, not learning quality.

Run from the repo root: python scripts/benchmark_knn.py --device cpu --output PATH
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.passive_tmaze import PassiveTMaze  # noqa: E402
from utils.experiment import build_agent  # noqa: E402
from utils.experiment_config import RunConfig  # noqa: E402
from utils.experiment_io import code_identity, runtime_versions, write_json  # noqa: E402
from utils.random import set_global_seed  # noqa: E402


def benchmark(device: str, updates: int, warmup: int) -> Dict[str, Any]:
    if updates < 1 or warmup < 0:
        raise ValueError("updates must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    rows: List[Dict[str, Any]] = []
    for model in ("DTQN", "DTQN-kNN"):
        config = RunConfig(
            model=model,
            env="PassiveTMaze-v0",
            corridor_length=15,
            context=4,
            history=4,
            in_embed=32,
            heads=4,
            layers=2,
            batch=8,
            buf_size=3000,
            memory_size=16,
            memory_top_k=4,
        )
        set_global_seed(1)
        env: Any = PassiveTMaze(15)
        try:
            agent = build_agent(config, env, 16, device)
            rng = np.random.default_rng(1)
            buffer = agent.replay_buffer
            buffer.obss[:12] = rng.normal(size=buffer.obss[:12].shape)
            buffer.actions[:12] = rng.integers(0, 4, size=buffer.actions[:12].shape)
            buffer.episode_lengths[:12] = 16
            buffer.dones[:12] = False
            buffer.pos = [12, 0]
            for _ in range(warmup):
                agent.train()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = perf_counter()
            for _ in range(updates):
                agent.train()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            rows.append(
                {
                    "parameters": config.to_dict(),
                    "updates": updates,
                    "seconds_per_update": (perf_counter() - start) / updates,
                }
            )
        finally:
            env.close()
    return {
        "device": device,
        "cpu_threads": 1,
        "warmup_updates": warmup,
        "versions": runtime_versions(),
        "code": code_identity(),
        "measurements": rows,
        "knn_over_baseline": rows[1]["seconds_per_update"]
        / rows[0]["seconds_per_update"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = benchmark(args.device, args.updates, args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(
        f"kNN/baseline update time: {result['knn_over_baseline']:.2f}x; {args.output}"
    )


if __name__ == "__main__":
    main()
