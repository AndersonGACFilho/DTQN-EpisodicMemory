# Experiment workflow (#9)

`train.py` trains one CLI configuration without intermediate evaluations and
always saves the final policy. `eval.py` loads that policy and runs one fixed
greedy evaluation. `train_and_eval.py` alone reads a JSON grid, runs every
combination in separate sequential processes, then compares the saved results.
The previous combined runner is preserved as `run_legacy.py` with its original
periodic evaluations and checkpoint behavior.

## Start a sweep

Activate the existing Python 3.10 venv. SciPy is already a runtime requirement
(`scipy==1.8.1`); the harness uses it for Student-t confidence intervals. It adds
no environment dependencies and uses local files without W&B.

```bash
# Inspect combinations without creating outputs or launching training.
python train_and_eval.py --config experiments/tmaze.json --dry-run

# Run the four context/corridor combinations and evaluate each final model.
python train_and_eval.py --config experiments/tmaze.json

# Four tiny jobs to check the pipeline, not learning performance.
python train_and_eval.py --config experiments/smoke.json
```

Each JSON value must be a nonempty list. The grid is the full Cartesian product,
including settings that have no effect in a particular environment. For example,
`"context": [14, 50]`, `"corridor_length": [15, 100]` and five `training_seed` values
produce 20 training/evaluation pairs. A singleton list fixes a parameter. No
zip-pairing, per-environment overrides, deduplication, or CLI scientific overrides
are performed. Unknown/duplicate keys, duplicate values, incorrect types and
nonfinite numbers reject the file before execution. Combination-specific errors
(e.g. incompatible attention dimensions or unavailable environments) are reported
per case, and other cases continue.

Required keys: `model`, `env`, `training_seed`, `num_steps`, `eval_episodes`, `eval_seed`.
Omitted scientific parameters use the defaults below. Files in `experiments/`
are examples, not a prescribed research grid or training budget.
The current `tmaze.json` uses contexts 14/16, corridor lengths 15/50 and training
seed 69: four pairs. Its single-seed summaries have blank std/CI fields. To
estimate variation across training runs, list multiple `training_seed` values.
Use `training_seed` in JSON and `--training-seed` in the new training CLI.
Earlier saved models/results with a `seed` field remain readable. The legacy
runner keeps its original `--seed` flag.

The model list can include `DTQN-kNN-cached` to compare collection-time K/V
against `DTQN-kNN`'s fresh replay prefixes and ordinary `DTQN`. It uses the same
memory options, requires `memory_stop_grad=true`, and preserves the existing
saved statistics and trajectories. See [cached memory](knn-cached.md) for its
capture rules and additional CPU storage.

Supported local IDs include `DiscreteCarFlag-v0`, `Memory-5-v0`,
`PassiveTMaze-v0`, `NoisyTMaze-v0` and `AssociativeCorridor-v0`. To sweep noise,
use `"env": ["NoisyTMaze-v0"]` and list `noise_level` / `noise_std` values.
To sweep the associative task, set `env` accordingly (corridor length >= 8).
Optional external environments still require their own provider packages.

## Run training and evaluation separately

```bash
python train.py --model DTQN --env PassiveTMaze-v0 \
  --context 14 --corridor-length 15 --num-steps 32000 \
  --training-seed 1 --device cpu --output-dir results/manual-train --verbose

python eval.py --model-path results/manual-train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/manual-eval --verbose
```

Use new/empty output directories; existing files are never silently overwritten.
These two scripts do not read sweep JSON. Training accepts one concrete set of
CLI values. Evaluation reconstructs the architecture and the **same environment
settings** from the saved model; its only scientific CLI options are episode
count and evaluation seed. `eval_episodes` / `eval_seed` in training arguments
are protocol metadata; training itself does not evaluate.

Training stops exactly at `num_steps` environment interactions, independently of
the number of gradient updates. An unfinished last episode is recorded as
`complete=false`, `success=null`, `completion_reason="step_budget"`; this does
not turn it into an environment terminal transition. Prefill uses random actions
and finishes its last episode, so its actual step count can exceed
`prefill_steps`. Those interactions are recorded separately and are excluded
from the training-step budget. With too few complete replay episodes, a run can
have fewer updates than interactions, including zero. The existing epsilon
schedule, replay sampler and Double DQN updates are reused.

Every successful training run saves `model.pt`: the final policy's complete
`state_dict` plus primitive architecture/environment/configuration metadata,
effective history/context/horizon, source identity, versions and parameter count.
It is an inference bundle, not an optimizer/replay checkpoint. `eval.py` disables
dropout, uses epsilon=0, resets policy context/bag per episode, never calls
training, and never stores evaluation experience in replay. It reseeds after
model construction so each training seed starts evaluation from the configured
`eval_seed`. It measures exactly `eval_episodes` complete episodes. For policies
that terminate at different times, a shared seed does not guarantee identical
later random events in every environment; initial seeds and environment rules
are shared, not entire action-dependent trajectories.

An explicit `max_episode_steps` installs a real Gym `TimeLimit`. Otherwise the
native finite horizon is required. Environments without either are rejected to
avoid unbounded episodes. Time-limit truncations retain the existing replay
bootstrapping convention. Actions must be discrete, with at most 256 choices
(the existing replay storage uses `uint8`).

## Operational flags

| Script | Flag | Default | Effect |
|---|---|---|---|
| Outer | `--config PATH` | Required for a new batch | Flat scientific grid |
| Outer | `--project-name NAME` | JSON filename stem | Output subdirectory |
| Outer | `--output-root PATH` | `results` | Batch root |
| All | `--device DEVICE` | `cpu` | CPU or CUDA device; never selected implicitly |
| All | `--verbose` | Off | Extra single-stage output; outer redirects it to process logs |
| Outer | `--dry-run` | Off | Print all expanded combinations/errors; create no batch |
| Outer | `--resume BATCH` | Off | Reuse completed stages, retry failures with new attempt folders |
| Outer | `--aggregate-only BATCH` | Off | Recompute tables from saved per-seed metrics; no model load/training |
| Train/eval | `--output-dir PATH` | Required | New/empty stage output directory |
| Eval | `--model-path PATH` | Required | Final inference bundle from `train.py` |
| Eval | `--eval-episodes N` | `100` | Fixed greedy episode count |
| Eval | `--eval-seed N` | `10000` | Evaluation seed, independent of the training seed |

Scientific JSON keys below map to `train.py` flags by replacing underscores with
hyphens. They cannot be supplied as overrides to `train_and_eval.py`.

| JSON key | Default | Effect |
|---|---|---|
| `model` | `DTQN` | Agent/network name: DTQN, DTQN-bag, DTQN-kNN, DTQN-kNN-cached, DRQN, ADRQN, DARQN, DQN |
| `env` | `DiscreteCarFlag-v0` | One environment ID or supported GridVerse YAML filename |
| `corridor_length` | `10` | Passive/Noisy length (>=1) or associative decision positions (>=8) |
| `noise_level` | `0.5` | Noisy T-Maze distractor probability [0,1] |
| `noise_std` | `0.1` | Positive Gaussian distractor standard deviation |
| `num_steps` | `2000000` | Exact number of training interactions after prefill |
| `prefill_steps` | `50000` | Minimum random prefill interactions; finishes the last episode |
| `training_seed` | `1` | Independent training/environment/network seed |
| `eval_seed` | `10000` | Shared evaluation seed for comparisons |
| `eval_episodes` | `100` | Greedy episodes per trained final model |
| `tuf` | `10000` | Gradient updates between target-network copies |
| `lr` | `0.0003` | Adam learning rate |
| `batch` | `32` | Replay batch size |
| `buf_size` | `500000` | Replay allocation in padded transitions; must fit at least batch+2 episodes |
| `context` | `50` | Observation window; forced to 1 for DQN |
| `history` | `50` | Intermediate Q targets; clipped to context |
| `obs_embed` | `8` | Embedding width per discrete observation element |
| `a_embed` | `0` | Action embedding width (0 disables it) |
| `in_embed` | `128` | Network width; DTQN width must divide evenly among heads |
| `max_episode_steps` | `-1` | Infer horizon, or apply an explicit positive time limit |
| `heads` | `8` | Transformer attention heads |
| `layers` | `2` | Transformer blocks |
| `dropout` | `0.0` | Transformer dropout in [0,1) |
| `discount` | `0.99` | Q-learning discount in [0,1] |
| `gate` | `res` | Transformer combination: res or gru |
| `identity` | `false` | Identity map reordering; standalone CLI uses `--identity` |
| `pos` | `learned` | Position encoding: learned, sin, none |
| `bag_size` | `0` | Existing persistent observation bag; DTQN only, not kNN episodic memory |
| `memory_enabled` | `true` | Active for DTQN-kNN and DTQN-kNN-cached; false selects the original DTQN network and agent |
| `memory_size` | `256` | External FIFO capacity in timesteps per episode |
| `memory_top_k` | `8` | Retrieved neighbours per query/head |
| `memory_layer` | `-1` | Zero-based block index; -1 selects the last block |
| `memory_gate` | `true` | Mix local and memory attention; false uses the local result |
| `memory_gate_bias_init` | `0.0` | Initial per-head sigmoid gate bias |
| `memory_stop_grad` | `true` | Detach prefix K/V; false enables differentiable prefixes only for DTQN-kNN; cached requires true |

Use `--no-memory-enabled`, `--no-memory-gate` or `--no-memory-stop-grad` to
disable these booleans in `train.py`; the sweep passes JSON false values correctly.
See [the integration guide](knn-integration.md) for memory population and validation.
Both kNN variants require `bag_size=0`. For a comparison, list `DTQN`,
`DTQN-kNN` and `DTQN-kNN-cached` in your JSON grid's `model` list.

The factory's existing model implementations are reused. A name being accepted
does not guarantee every legacy model/environment/embedding combination works;
construction or training errors produce failed cases with full process logs.
DTQN startup is checked on the five local tasks. Requested and effective
parameters are both saved, including inactive environment/transformer settings.
The base DQN agent now advances its observation context on each transition;
previously its policy kept seeing the episode's initial observation. The
recurrent/transformer agents already advanced their own contexts. Their learning
updates and the shared Double DQN targets are unchanged.

## Artifacts and episode replay

```text
results/<project>/<timestamp>-<unique-id>/
  config.json                    # copied normalized input grid
  manifest.json                  # requested configs, code, stage status, attempts
  runs.csv                       # one row per requested config/training seed
  summary.csv                    # across-seed comparison per config
  runs/<run-id>/
    result.csv                   # canonical one-row result for this seed
    train-001.log
    eval-001.log
    train-001/
      model.pt
      train_metadata.json
      train_losses.csv
      train_episodes.csv
      train_trajectories.jsonl.gz
    eval-001/
      eval_metadata.json
      eval_episodes.csv
      eval_trajectories.jsonl.gz
```

The run hash includes every requested scientific setting and seed. The config
hash excludes **only the training seed**, including evaluation seed/count and
inactive settings. Different batches get unique folders. Each process writes
to its own stage directory. `result.csv` contains requested parameters, actual
prefill/training/update counts, greedy success rate, mean return, mean episode
length, phase/combined costs, code identity, device and artifact paths.

Episode CSV rows contain phase, episode index, cumulative phase steps, return,
length, success, completeness, termination/truncation flags and completion
reason. Loss rows record each actual update, environment training step, TD error,
gradient norm and Q/target min/mean/max from that update. These are training
diagnostics, not independent evaluation seed results.

Trajectory files are streamed gzip JSON Lines, schema version 1:

- `episode_start`: phase (`prefill`, `train`, `eval`), episode index, phase-step
  counter, initial observation and world state.
- `step`: phase/episode/episode-step/phase-step, input observation, chosen action,
  reward, resulting observation, epsilon, termination/truncation, info and the
  resulting world state. Episode steps are one-based; start observations precede
  the first action. Arrays and NumPy scalars become JSON lists/numbers.
- `episode_end`: the episode CSV fields. Partial training episodes have no
  success label; completed evaluation episodes always do.

Local state snapshots include x/y/goal/time for Passive and Noisy T-Maze,
associative event schedule/directions/progress, Car Flag position/velocity and
goal/priest locations, and Memory Cards deck/visible cards/current card. These
are inspection data and are **never inputs to the policy**. The recorder only
reads fields and consumes no RNG. Optional external tasks record observations,
actions/rewards and info, with `adapter="observation_only"`; arbitrary hidden
world state is not promised. Neural activations/attention/replay-buffer internals
are not dumped. Saving observations and state snapshots also preserves noisy
and stochastic outcomes that actions alone cannot reconstruct.

To inspect incrementally without loading the whole training history:

```python
import gzip, json

with gzip.open("PATH/train_trajectories.jsonl.gz", "rt") as stream:
    for line in stream:
        record = json.loads(line)
        if record["event"] == "step" and record["phase"] == "train":
            print(
                record["episode"],
                record["episode_step"],
                record["action"],
                record["world_state"],
            )
```

This supplies data for future plotting/rendering; it does not implement a renderer.

## Statistics and resource accounting

`summary.csv` groups independent training seeds, using each seed's final greedy
evaluation success rate as one observation. It also summarizes per-seed mean
return/length and costs. For N >= 2 complete seeds: arithmetic mean, sample
standard deviation (`ddof=1`), and the two-sided Student-t interval
`mean +/- t(0.975, N-1) * std / sqrt(N)`. Intervals are not clipped to [0,1].
Evaluation episodes and training checkpoints are not treated as independent
training seeds. Use at least five seeds for research comparisons.

If any requested seed fails or is missing, the group is `incomplete`, identifies
missing seeds, and withholds comparison statistics. A complete single-seed group
reports its mean but leaves std/CI blank (`single_seed`). Failure is never counted
as zero policy success. Associative Corridor requires explicit `is_success`, so a
positive partial return still counts as failure; other environments retain the
existing success flag or positive-return convention.

Train/eval wall-clock seconds include environment/model setup, prefill when
training, action/observation recording, execution and final model saving. They
exclude Python process startup/import time and post-measurement metadata writing.
Outer attempt logs also record process elapsed time, including failed attempts;
`runs.csv` exposes the sum of all attempts. Final-policy resource statistics refer
to the successful training/evaluation attempts only.

For CUDA, each phase synchronizes and resets peak memory statistics before model
allocation, synchronizes at the end, then records
`torch.cuda.max_memory_allocated(device)` **in bytes**. This measures PyTorch
allocated memory, not total GPU usage or reserved cache. Combined time is the
sum of successful phase times; combined peak is the maximum phase peak. CPU runs
are labeled `cpu` and record GPU bytes as zero. GPU execution requires a
CUDA-enabled PyTorch installation; this workflow does not change the venv's
existing CPU/GPU installation.

## Failures, resume and reaggregation

The outer script completes other combinations after a failure and exits nonzero
if any requested seed is incomplete. Inspect `runs.csv`, `manifest.json` and the
case's process logs. By default, rerunning creates a fresh batch.

```bash
python train_and_eval.py --resume results/tmaze/BATCH
python train_and_eval.py --aggregate-only results/tmaze/BATCH
```

Resume requires the same grid, source identity and Python/library versions.
Completed stages are skipped. A failed evaluation reuses its trained model;
failed/interrupted training starts from scratch. Previous attempts and logs are
retained, but a seed is counted once through its canonical `result.csv`. Missing
stage artifacts cause a new attempt. This is stage reuse, not continuation of an
optimizer/replay/environment state. Aggregate-only reads the saved manifest and
per-seed results, without running environments or loading weights, and can be
used after code changes.

## Acceptance checks

| #9 criterion | Implementation / validation |
|---|---|
| AC1: one command, N seeds, CSV per seed | JSON Cartesian runner; isolated `result.csv`; multi-job CPU smoke and process integration tests |
| AC2: mean, std, 95% CI per configuration | Saved-only aggregation; tested sample std and known Student-t interval, single/missing seeds and attempt counting |
| AC3: wall clock and CUDA peak | Separate synchronized phase meters and combined columns; real CPU/CUDA runs and mocked CUDA API/order checks |
| AC4: configured fixed greedy episodes | Final bundle loaded by evaluation-only process; episode counts, epsilon=0, deterministic replay and unchanged weights/replay tested |

Run `python -m unittest discover -s tests`, `mypy`, and `pre-commit run --all-files`.
Reviewed PR and merge into `main` remain separate requirements of the issue.
See [the validation record](experiment-harness-validation.md) for local checks
and smoke artifact locations.
