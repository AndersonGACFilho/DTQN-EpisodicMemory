# Experiment harness validation (#9)

Validated on Python 3.10.21, NumPy 1.22.4, Gym 0.23.1 and SciPy 1.8.1,
first with PyTorch 2.6.0+cpu and subsequently with PyTorch 2.7.1+cu128.
Real CUDA validation used an RTX 5060 Ti (16 GB) with NVIDIA driver 595.71.05.
SciPy was already declared in the runtime requirements.

| Criterion | Evidence | Result |
|---|---|---|
| AC1: one command runs N seeds and writes one CSV per seed | `experiments/smoke.json`: two corridor lengths x two training seeds, four complete train/eval jobs and four isolated `result.csv` files. The current `experiments/tmaze.json` expands to four single-seed pairs, completed on CPU and CUDA. | Passed |
| AC2: aggregation outputs mean/std/95% CI per configuration | Two smoke summary rows, each aggregating two seeds. Unit tests check sample std and Student-t endpoints for [1,2,3,4,5], incomplete groups, single seeds and retry accounting. | Passed |
| AC3: wall clock and peak allocated GPU memory | Per-phase and combined costs are present in seed/batch CSVs. CPU runs report zero GPU bytes. Real full CUDA runs report approximately 34.90/36.37 MiB peak allocated memory. Mocked tests also check CUDA synchronization/reset/API ordering. | Passed on CPU and CUDA |
| AC4: a configured fixed number of greedy evaluation episodes | Every smoke model is evaluated for exactly three episodes; every full CPU/CUDA model for exactly 100. Tests assert epsilon=0, unchanged policy tensors, no training call, empty evaluation replay, and identical seeded traces after loading the final bundle. | Passed |

Additional checks cover exact environment-step budgets, partial final episodes,
random prefill action logging/rounding, action-by-action replay of saved traces,
world snapshots that do not consume RNG, real TimeLimit truncation, explicit
associative success after positive partial returns, invalid grid rejection,
forbidden scientific CLI overrides, failure continuation, stage reuse on resume,
and saved-only aggregation without model loading.
Saved models/results written before the `seed` to `training_seed` rename remain
readable; old batch aggregation was checked without altering the original files.

Separate tiny DTQN train/eval jobs completed on all five local environments:
Car Flag, Memory Cards, Passive T-Maze, Noisy T-Maze and Associative Corridor.
The DQN observation regression is covered in both train and eval modes. These
small jobs validate execution and artifact integrity, not learning performance.

Checks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests
# 64 tests passed, including after the CUDA-enabled PyTorch upgrade

mypy
# 51 source files, no issues

# All pre-commit hooks passed on all tracked/new files, including new scripts.
pre-commit run --all-files

python train_and_eval.py --config experiments/tmaze.json --dry-run
# Four train/evaluation pairs in the current example, no execution

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_and_eval.py \
  --config experiments/smoke.json --project-name harness-validation --verbose
```

The local smoke artifacts are under:

```text
results/harness-validation/20261122T101716Z-95857b6c/
```

Completed full-run artifacts (ignored by Git) are under:

```text
results/tmaze/20261122T144108Z-0cba8505/          # CPU, PyTorch 2.6.0+cpu
results/t-maze-gpu/20261122T153526Z-1d231c47/   # CUDA, PyTorch 2.7.1+cu128
```

Both batches used the same source/configuration: 32,000 environment training
steps and 32,000 gradient updates per case, training seed 69, evaluation seed
10000 and 100 greedy evaluation episodes per model. Replay prefill recorded
50,000 steps for corridor length 15 and 50,031 for length 50 (finishing its last
episode). All final models, episode/loss CSVs and compressed trajectory files
were present. The completed CPU trajectories were checked step-by-step against
episode counts/returns; both evaluation batches matched their reported success
counts. Full runs have only one training seed per configuration, so their std/CI
fields are correctly blank; multi-seed aggregation is covered by the smoke run
and known-value confidence-interval tests.

| Context | Corridor | CPU training seconds | CUDA training seconds | CPU/CUDA evaluation successes out of 100 |
|---:|---:|---:|---:|---:|
| 14 | 15 | 412.069 | 191.078 | 43 / 57 |
| 16 | 15 | 432.977 | 193.816 | 100 / 100 |
| 14 | 50 | 393.106 | 187.742 | 0 / 0 |
| 16 | 50 | 425.286 | 187.871 | 0 / 0 |

These are execution checks, not a controlled hardware benchmark or a multi-seed
learning result: the PyTorch versions differ. The two short-context policies
always selected opposite fixed branches, accounting for 43% versus 57% success
on the shared evaluation goals. Both length-50 policies continued Right at the
junction. A separate real CUDA smoke also completed 32 gradient updates and four
evaluation episodes, with verified prefill/training/evaluation trajectory counts.

The pre-commit invocation used an explicit list of all tracked/new files so it
also checked files that had not yet been staged. The command above is the normal
equivalent after adding the files to Git. Check results do not satisfy the
separate requirement for a reviewed PR and merge into `main`.
