import os
import random
from typing import Tuple

import gym
import numpy as np
import torch

# cuBLAS needs a fixed workspace to be deterministic; it must be set before CUDA starts
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# CPU results change with the intra-op thread count, so every run uses the same one
RUN_THREADS = 1


class RNG:
    rng: np.random.Generator = None


def make_deterministic() -> None:
    """Same seed, device and code give the same result, or PyTorch raises an error."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch.set_num_threads(RUN_THREADS)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def set_global_seed(seed: int, *args: Tuple[gym.Env]) -> None:
    """Sets seed for PyTorch, NumPy, and random.

    Args:
        seed: The random seed to use.
        args: The gym environment(s) to seed.
    """
    random.seed(seed)
    tseed = random.randint(1, 1e6)
    npseed = random.randint(1, 1e6)
    ospyseed = random.randint(1, 1e6)
    torch.manual_seed(tseed)
    np.random.seed(npseed)
    for env in args:
        env.seed(seed=seed)
        env.observation_space.seed(seed=seed)
        env.action_space.seed(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)
    RNG.rng = np.random.Generator(np.random.PCG64(seed=seed))
