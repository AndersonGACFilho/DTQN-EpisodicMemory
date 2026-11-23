# Pluggable kNN Episodic Memory for Deep Q-Learning Agents in Partially Observable Environments

Pluggable kNN episodic memory (from Memorizing Transformers) for DTQN and DRQN agents in partially observable RL. The project investigates access to information beyond the context window; the [environment guide](docs/environments.md) describes the tasks, their limitations, and implementation availability.

> **Status:** UFRGS Deep Learning final project (Group 7), due **11/30/2026**. This repository is a fork of [kevslinger/DTQN](https://github.com/kevslinger/DTQN); the original DTQN documentation is kept [below](#deep-transformer-q-networks-for-partially-observable-reinforcement-learning).
>
> **Paper (Overleaf):** _link to be added_ ([#42](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/42))

## Overview

In partially observable problems (POMDPs), the current observation is not enough to act and the agent must remember something seen earlier in the episode. The Deep Transformer Q-Network (DTQN) applies a transformer decoder over the last K observations, so anything older than K steps falls out of context, and Cherepanov et al. (2024) show that DTQN then performs at chance.

We couple the external kNN memory from Memorizing Transformers (Wu et al., 2022) to DTQN: (key, value) pairs from earlier steps of the episode are stored and retrieved by similarity, giving access to information beyond the window.

## Research question

Can a pluggable kNN memory let Deep Q-Learning agents use information beyond the context window, without hurting performance when the information is inside it?

- **Beyond the window (dependency > K):** DTQN + memory reaches a quantitative success target where the original DTQN is at chance.
- **Within the window (dependency ≤ K):** DTQN + memory ties with the original DTQN.

The quantitative target and the tie criterion are defined in #1. Every configuration uses at least 5 seeds and reports mean ± std with 95% confidence intervals.

## Environments and what each tests

The [standardized environment guide](docs/environments.md) covers every configured
environment: **35 IDs or YAML configurations across 14 families**, including the
two new variants. Each entry has a paper-ready purpose and limitation, mechanics,
observations/actions, rewards/success, configuration, usage, rendering status,
and sources. Related sizes and observation variants are listed explicitly.

| Project environment | Experimental role | Guide |
|---|---|---|
| **Passive T-Maze** | Recall of one instruction across a controlled delay | [Description and limitations](docs/environments.md#passive-t-maze) |
| **Noisy T-Maze** | Recall with Gaussian corridor distractors | [Description and limitations](docs/environments.md#noisy-t-maze) |
| **Associative Corridor** | Four class-specific instruction/use pairs, with interleaving allowed | [Description and limitations](docs/environments.md#associative-corridor) |
| **Car Flag** | Preserve the baseline's performance within the context window | [Description and limitations](docs/environments.md#car-flag) |
| **Memory Cards** | Preserve the baseline's performance within the context window | [Description and limitations](docs/environments.md#memory-cards) |

**Car Flag and Memory Cards only check for no performance loss within the window
in this study, using the selected `K = 50` setup.** They are not evidence of
retrieval beyond that window. These are experimental roles, not newly measured
results.

The guide also covers [GridVerse](docs/environments.md#gridverse-memory),
[MiniHack](docs/environments.md#minihack-room),
[Hallway](docs/environments.md#hallway), and
[HeavenHell](docs/environments.md#heavenhell). All five project environments are
merged into `main`; the other families require optional provider packages.
The guide's examples use the current `train.py` / `eval.py` workflow. Paper reuse for
[#2](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/2) is
[tracked separately](docs/environments.md#acceptance-criteria-and-paper-handoff).

## Agents and baselines

| Agent | Role |
|---|---|
| DTQN (original) | Main baseline |
| DTQN with K covering the corridor | Answers "why not increase K?", with time and GPU-memory cost, where feasible |
| DTQN + kNN memory | Proposed method |
| DRQN (+ kNN memory) | **Optional**: only if the DTQN results are promising and time allows; needs its own coupling (what is key/value and how the read combines with the hidden state) and hypothesis |

## Design questions addressed

The proposal was graded 100/100; the professor's ten comments are mapped to issues in #4 and answered in the paper.

- **Memory during training:** DTQN trains on length-K subsequences sampled from the replay buffer, so the memory must hold the (key, value) pairs of the steps before the window. These are recomputed with the current network under `no_grad` at every training step, with stop-gradient memory as the default ([design note](docs/memory-population.md), #3).
- **Keys without gradient:** nothing forces the query to retrieve the cue step; the noisy T-Maze and the stop-gradient vs. differentiable ablation test this.
- **Retrieval diagnostic:** #30 specifies logging whether the cue step is in the top-k at decision time, to separate retrieval failure from usage failure; this diagnostic is not yet implemented.
- **Ablations:** gate, top-k, memory size, key layer, stop-gradient vs. differentiable memory.
- **Compute budget:** an early pilot estimates GPU-hours before the experiment grid is fixed; the desired but infeasible grid is discussed in the paper.

## kNN episodic memory

The memory lives in [`dtqn/networks/knn_memory.py`](dtqn/networks/knn_memory.py) (#5):

- `KNNMemory` is a per-episode FIFO store of per-head (key, value) pairs with exact top-k retrieval by cosine similarity. It holds one memory per batch element, so a training batch can carry the prefixes of different episodes.
- `KNNAttention` is causal multi-head attention that also attends to the retrieved neighbours and mixes both per head with a learned gate `g_h = sigmoid(b_h)`. With the gate off, or with an empty memory, its output is exactly that of the `nn.MultiheadAttention` it wraps.

Both are configured by `MemoryConfig`. `DTQN-kNN` now connects them to DTQN's
acting and replay-training paths (#7). The agent caches each newest timestep's
K/V, moves it into external memory on context eviction, and resets memory per
episode. Training reconstructs earlier keys from the same replay episode using
the current network; the Double DQN target reconstructs its own shifted prefix.

`DTQN-kNN-cached` reuses that architecture but saves detached online and target
K/V during collection and gathers them during replay without recomputation.
Entries are never refreshed after weight updates. See the
[cached variant](docs/knn-cached.md) for commands, storage costs and the staleness
tradeoff. Both variants work with the train/eval scripts and JSON sweeps.

Train and evaluate one memory-enabled policy with the current scripts:

```bash
python train.py --model DTQN-kNN --env PassiveTMaze-v0 \
  --context 14 --history 14 --corridor-length 15 --memory-size 32 \
  --memory-top-k 4 --batch 4 --prefill-steps 128 --num-steps 32000 \
  --training-seed 1 --device cpu --output-dir results/knn-example/train
python eval.py --model-path results/knn-example/train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/knn-example/eval
```

See [integration instructions and validation](docs/knn-integration.md) for
single-run commands, population rules, measured update cost and acceptance checks.
This retains DTQN's full-window local forward; it does not introduce incremental
local KV caching. A speed or accuracy advantage over increasing context is an
experimental question, not an assumption.
For a JSON comparison of all three models, use the
[example grid in the environment guide](docs/environments.md#shared-execution-and-evaluation-conventions).

| Option | Default | Effect |
|---|---|---|
| `size` | `256` | Steps stored per episode; the oldest step is evicted when full |
| `top_k` | `8` | Neighbours retrieved per query |
| `layer` | `-1` (last) | Transformer layer whose attention reads the memory |
| `gate` | `True` | `False` uses local attention only |
| `gate_bias_init` | `0.0` | Initial gate bias `b_h` |
| `stop_grad` | `True` | Stored keys and values receive no gradient |

The defaults are provisional until the pilot (#11) and the ablations (#22). Unit tests:

```bash
python -m unittest discover -s tests
```

The new `train.py` / JSON options are:

| Flag / JSON key | Default | Effect |
|---|---|---|
| `--model` / `model` | `DTQN` | Select `DTQN-kNN` for fresh replay K/V or `DTQN-kNN-cached` for collection-time K/V |
| `--memory-enabled` / `memory_enabled` | `true` for both kNN variants | `--no-memory-enabled` / `false` selects the original network and agent |
| `--memory-size` / `memory_size` | `256` | FIFO capacity in earlier timesteps |
| `--memory-top-k` / `memory_top_k` | `8` | Neighbours per query and head |
| `--memory-layer` / `memory_layer` | `-1` | Selected block, zero-based; -1 is last |
| `--memory-gate` / `memory_gate` | `true` | `--no-memory-gate` / `false` uses local attention only |
| `--memory-gate-bias-init` / `memory_gate_bias_init` | `0.0` | Initial per-head sigmoid gate bias |
| `--memory-stop-grad` / `memory_stop_grad` | `true` | `--no-memory-stop-grad` / `false` enables differentiable prefix K/V for the policy loss |

Memory options are inactive for other models. Both kNN variants require `bag_size=0`.
`DTQN-kNN-cached` additionally requires `memory_stop_grad=true`.
Model saving/loading, trajectories and the existing experiment statistics include
the new variant. The finite FIFO can evict the cue; memory capacity must cover
the distance beyond local context. Differentiable K/V do not make hard top-k
selection differentiable.

## Development

Every pull request and push to `main` runs [CI](.github/workflows/ci.yml): lint and format checks ([pre-commit](.pre-commit-config.yaml) with ruff), type checking (mypy) and unit tests, on Python 3.10 with torch 1.11 (CPU). Tool settings live in [`pyproject.toml`](pyproject.toml).

Set up the same checks locally:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

`pre-commit install` adds two git hooks: ruff and file hygiene run on every commit, and mypy and the unit tests run on every push. To run them by hand:

```bash
pre-commit run --all-files
python -m mypy
python -m unittest discover -s tests
```

mypy runs in strict mode. Modules inherited from the DTQN upstream are listed in `pyproject.toml` with their errors ignored; new modules must pass, and a legacy module leaves the list once it does.

## Running Passive T-Maze

Passive T-Maze is available as `PassiveTMaze-v0`. With the project environment
activated, run this from the repository root:

```bash
python train.py --model DTQN --env PassiveTMaze-v0 --corridor-length 10 \
  --context 50 --history 50 --prefill-steps 128 --num-steps 2000 --batch 4 \
  --training-seed 1 --device cpu --output-dir results/tmaze-smoke/train --verbose
python eval.py --model-path results/tmaze-smoke/train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/tmaze-smoke/eval --verbose
```

`--corridor-length` defaults to 10 and also configures Noisy T-Maze and
Associative Corridor. A Passive T-Maze episode lasts `corridor_length + 1` steps;
increase the length beyond `--context` to test
dependencies outside the observation window. This runs ordinary DTQN; use the
`DTQN-kNN` workflow above for episodic memory. No additional runtime packages are
required beyond Gym and NumPy already used by the runner.

The port follows [Memory-RL's Passive T-Maze](https://github.com/twni2016/Memory-RL/blob/main/envs/tmaze.py)
and its [experiment configuration](https://github.com/twni2016/Memory-RL/blob/main/configs/envs/tmaze_passive.py).
Actions are right/up/left/down (0/1/2/3). The initial observation is `[0, cue]`,
where cue is -1 or +1; intermediate observations are `[0, 0]`, and the junction
and branches use `[1, y]`. A correct terminal branch earns 1, an incorrect one
earns 0, and nonterminal steps that fall behind the shortest route incur `-1/L`.
The cue is not shown again when returning to the start. Episode seeds are local
to each environment. There is no graphical renderer in this port.

The command requests 128 random prefill interactions, rounded up to finish the
last episode, in addition to 2,000 training interactions. Training always saves
the final `model.pt`; metadata, episode/loss CSVs and step trajectories are
written beside it. Evaluation runs separately with the saved environment settings.
Use new/empty output directories. The default prefill remains 50,000 when omitted.

Run the environment sanity checks with:

```bash
python -m unittest tests.test_passive_tmaze -v
```

The oracle must succeed in every episode. The approximately 50% baseline uses
correct corridor traversal followed by a random up/down choice; fully random
navigation generally fails to reach the junction before the deadline. This
clarifies the random-policy criterion in #6 without changing upstream dynamics.

## Running Noisy T-Maze (#20)

`NoisyTMaze-v0` keeps Passive T-Maze's actions, movement, rewards, initial cue,
clean junction/branch observations, and `corridor_length + 1` horizon. Only the
second coordinate of subsequent corridor observations changes:

- `--noise-level p` is the probability of replacing zero with a distractor,
  from 0 to 1; default **0.5**.
- `--noise-std sigma` is the finite positive standard deviation of each
  Gaussian; default **0.1**.
- Each replacement independently selects a mean of -1 or +1 with equal
  probability, then samples `Normal(mean, sigma**2)`. Otherwise the value stays
  zero. Samples are not clipped, so the noisy observation space allows values
  outside [-1, 1].

At `--noise-level 0`, the observation space and seeded trajectories are identical
to Passive T-Maze. Noise has its own seeded RNG, leaving the goal/cue sequence
unchanged across noise levels. The oracle reads the clean initial instruction,
walks right and chooses the indicated branch; its solution is valid at every
supported noise level and standard deviation. Rewards are never modified.

```bash
python train.py --model DTQN --env NoisyTMaze-v0 --corridor-length 15 \
  --noise-level 0.5 --noise-std 0.1 --context 14 --history 14 --device cpu \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --output-dir results/noisy-tmaze-smoke/train --verbose
python eval.py --model-path results/noisy-tmaze-smoke/train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/noisy-tmaze-smoke/eval --verbose

python -m unittest tests.test_noisy_tmaze -v
```

Both training and evaluation use the saved noise settings. The final model,
metadata and trajectories live under the supplied output directories; corridor
and noise parameters are recorded in the metadata. Use fresh directories for
fresh runs. The flags are not forwarded to Car Flag, Memory
Cards or clean Passive T-Maze. No additional dependencies or renderer are added.
This command runs ordinary DTQN; select `DTQN-kNN` in the new workflow to use
episodic memory. The multi-seed noisy experiment is #21.

This variant supplies near-cue distractors, not exact duplicates: continuous
Gaussian samples almost surely differ from exactly +/-1. The initial-step
retrieval strategy remains possible, so this experiment alone does not establish
general content-based retrieval.

Tests cover #20's configurable selection (AC1), exact zero-noise equivalence
(AC2), and an observation-driven oracle succeeding in 1,000 episodes per setting
at noise levels 0, 0.25, 0.5 and 1 with standard deviations 0.1 and 1 (AC3).
The implementation is merged into `main`; these environment checks do not
establish learned-policy performance.

## Running Associative Corridor

`AssociativeCorridor-v0` is an additional recall task with four independent
class/direction associations. Each episode contains one instruction and one use
of each class. Instructions and queries can be interleaved, but an instruction
always precedes its own query. Directions and event positions change on reset.
The final position is always a query; unoccupied positions are blank corridor.

Observations are six `float32` values: `[A, B, C, D, is_query, direction]`.
The first four values are a one-hot class code, or all zeros in blank corridor.

| Situation | Example |
|---|---|
| Instruction: B means Down | `[0, 1, 0, 0, 0, -1]` |
| Blank corridor | `[0, 0, 0, 0, 0, 0]` |
| Query: which direction for B? | `[0, 1, 0, 0, 1, 0]` |

Direction is `-1`/`+1` only on instructions. Queries do not reveal the answer.
Use Right (0) at instructions and blanks, and Up (1) or Down (3) at queries.
A correct query gives +1 and advances; correct corridor actions give 0 and
advance. Any incorrect action, including Left (2), ends the episode with reward
0. A successful episode answers all four queries and returns 4. Earlier correct
answers still contribute to return when a later action fails, but that episode
is counted as a failure. Only this environment opts into that explicit success
rule; existing environments retain their evaluation behavior.

`--corridor-length` is the number of decision positions: minimum 8, default 10.
A successful episode takes exactly that many steps (unlike Passive T-Maze's
`L + 1`). Random positions allow different recall delays; a longer corridor does
not guarantee that every instruction/query pair exceeds the context window.
The `instructions` and `queries` dictionaries on `env.unwrapped` expose event
positions for inspecting a seeded schedule; the agent receives observations only.

```bash
python train.py --model DTQN --env AssociativeCorridor-v0 --corridor-length 100 \
  --context 14 --history 14 --device cpu --prefill-steps 128 --num-steps 2000 \
  --batch 4 --training-seed 1 --output-dir results/associative-smoke/train --verbose
python eval.py --model-path results/associative-smoke/train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/associative-smoke/eval --verbose

python -m unittest tests.test_associative_corridor -v
```

Run this separately from environments with two-dimensional observations.
The output directories contain final weights, settings, episode/loss CSVs and
trajectories. Use fresh directories for a fresh run. No new packages or graphical
renderer are included. This runs ordinary DTQN; select `DTQN-kNN` or
`DTQN-kNN-cached` for the integrated memory variants. The task tests
class-dependent recall, not the necessity of kNN over every other memory architecture.

## Multi-seed evaluation with rliable

Comparisons follow [rliable](https://github.com/google-research/rliable)
(Agarwal et al., NeurIPS 2021): several training seeds, aggregate metrics with
bootstrap confidence intervals, performance profiles, probability of improvement
and learning curves with uncertainty bands.

### Run the grid

[`experiments/rliable_tmaze.json`](experiments/rliable_tmaze.json) is the single
source of truth for the grid: `parameters` lists `RunConfig` values (full
Cartesian product), `train_seeds` the number of seeds (1..N), `eval_episodes`,
`eval_interval` (training steps between evaluations) and `comparisons`
(`[candidate, baseline]` models). Check that its hyperparameters match the runs you
want to reproduce before launching it.

```bash
# List the 60 jobs (2 models x 3 contexts x 2 layers x 5 seeds) and their status
python run_rliable_grid.py --grid experiments/rliable_tmaze.json --dry-run

# Run everything; repeat the same command to resume after an interruption
python run_rliable_grid.py --grid experiments/rliable_tmaze.json --device cuda

# Split the work between machines, e.g. one model or one seed each
python run_rliable_grid.py --grid experiments/rliable_tmaze.json --only model=DTQN-kNN
python run_rliable_grid.py --grid experiments/rliable_tmaze.json --only training_seed=3

# Use 10 seeds instead of the grid's 5
python run_rliable_grid.py --grid experiments/rliable_tmaze.json --seeds 10
```

Each run trains with the existing `train.py` code path and evaluates the greedy
policy every `eval_interval` steps and at the final step, always on the same
`eval_episodes` episodes. The evaluation seed is `10000 + training_seed`, so runs
do not share evaluation episodes. Evaluation saves and restores every random
stream: the trained weights are identical to a run without evaluation. A run is
complete when both files exist and is then skipped:

- `results/raw/<run_id>.csv`: one row per evaluation episode with `model, env,
  context, layers, corridor_length, train_seed, eval_seed, step, episode, success,
  episode_return, episode_length, goal_side, chosen_side`. Sides are `up`, `down`,
  or `none` (junction not reached), and empty for environments without a junction.
- `results/raw/<run_id>.json`: the full configuration, evaluation steps, git commit
  and source hash, library versions, wall-clock time and the artifact folder.

Training artifacts (final `model.pt`, CSV logs, trajectories) go to
`results/runs/<run_id>/attempt-NNN/`; an interrupted attempt is kept and a new one
starts. Add a junction reader to `SIDE_READERS` in
[`evaluation/sides.py`](evaluation/sides.py) for a new environment, and a model
name to the grid for a new model.

### Generate the report

```bash
python -m analysis.rliable_report --grid experiments/rliable_tmaze.json
# Final numbers: more bootstrap replications (slower)
python -m analysis.rliable_report --grid experiments/rliable_tmaze.json --reps 50000
```

The report reads only the grid's runs and stops with the run ids of any run that
has no results. It writes PNG and PDF figures to `results/figures/`
(`aggregate_metrics`, `probability_of_improvement`, `performance_profiles`,
`iqm_learning_curves`) and CSV/Markdown tables to `results/summary/`
(`summary.md`, `aggregate_metrics.csv`, `probability_of_improvement.csv`,
`side_bias.csv`). Scores are final success rates, already in [0, 1]. An algorithm
is a (model, context, layers) combination and a task an (environment, corridor
length) combination; score matrices have shape `(n_seeds, n_tasks)`.

### Interpret the results

- **IQM** (interquartile mean): the mean of the middle 50% of all run x task
  scores. It ignores the best and worst 25%, so it is less sensitive to outlier
  seeds than the mean and less noisy than the median. Prefer it as the headline.
- **95% CI**: stratified bootstrap over training seeds within each task. With few
  seeds the intervals are wide; overlapping intervals mean the data do not
  separate the algorithms, not that they are equal. Report the interval, not only
  the point estimate.
- **Optimality gap**: the mean shortfall below a perfect success rate of 1
  (lower is better).
- **Probability of improvement** P(X > Y): the chance that a random run of X
  beats a random run of Y on the same task (ties count one half). 0.5 means no
  difference. A CI entirely above 0.5 supports an improvement; the
  [rliable paper](https://arxiv.org/abs/2108.13264) treats a lower bound above 0.5
  together with an upper bound above 0.75 as statistically meaningful.
- **Performance profile**: the fraction of runs with a success rate above each
  threshold tau. A curve that stays higher dominates.
- **Side bias**: `side_bias.csv` marks runs whose final evaluation ends in the
  same branch in more than 90% of episodes. Such a policy ignores the cue; its
  success rate equals the share of goals on that side (for example, 31 of the 50
  episodes from seed 10000 have the goal down, so always turning down scores 62%).

## Project management

Work is tracked in the [GitHub Project](https://github.com/users/AndersonGACFilho/projects/19) (Kanban, Roadmap, Goals, Epics, Priorities and one board per person).

| Milestone | Due | Goal |
|---|---|---|
| M1: Feedback plan & design | 11/17 | Feedback mapped to tasks; quantitative criteria, environment roles and memory design fixed |
| M2: Memory module & environments | 11/20 | Passive T-Maze, evaluation harness and kNN memory integrated into DTQN |
| M3: Pilot & experiment grid | 11/22 | GPU-hour pilot, large-K cost and the final experiment grid |
| M4: Experiments | 11/25 | Main experiments, within-window check, noisy T-Maze and ablations |
| M5: Analysis & figures | 11/26 | Results table, retrieval analysis and figures |
| M6: Paper draft | 11/28 | All sections written and reviewed |
| M7: Submission | 11/30 | Paper submitted |

| Role | Responsibility | Paper sections |
|---|---|---|
| **P1** (Estevan Küster) | Passive T-Maze, evaluation harness, main experiment, results, figures | Results |
| **P2** (Anderson Gonçalves) | Memory design and module, DTQN integration, retrieval diagnostic, ablations | Method |
| **P3** (Esthevan Ferretti) | DTQN baseline, GPU-hour pilot, large-K cost, within-window check, DRQN (optional) | Related work, setup |
| **P4** (João Augusto Laner) | Feedback in the paper, targets, noisy T-Maze, experiment grid, coordination, submission | Introduction, discussion, conclusion, abstract |

**Conventions:** issue types (Epic, Feature, Task, Experiment, Writing, Research, Bug); form templates in [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE); epics group tasks as sub-issues; blocked by / blocking links follow the critical path; labels `P1`–`P4` mark the people involved and `optional` marks the DRQN work.

## References

- Esslinger, Platt and Amato (2022). *Deep Transformer Q-Networks for Partially Observable Reinforcement Learning.* [arXiv:2206.01078](https://arxiv.org/abs/2206.01078)
- Wu, Rabe, Hutchins and Szegedy (2022). *Memorizing Transformers.* ICLR 2022. [arXiv:2203.08913](https://arxiv.org/abs/2203.08913)
- Hausknecht and Stone (2015). *Deep Recurrent Q-Learning for Partially Observable MDPs.* [arXiv:1507.06527](https://arxiv.org/abs/1507.06527)
- Cherepanov et al. (2024). *Unraveling the Complexity of Memory in RL Agents.* [arXiv:2412.06531](https://arxiv.org/abs/2412.06531)
- Ni, Ma, Eysenbach and Bacon (2023). *When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment.* NeurIPS 2023. [arXiv:2307.03864](https://arxiv.org/abs/2307.03864)

---

# Deep Transformer Q-Networks for Partially Observable Reinforcement Learning

Deep Transformer Q-Network (DTQN) is an extension of [DQN](https://www.nature.com/articles/nature14236) and [DRQN](https://arxiv.org/abs/1507.06527) designed to encode an agent's history effectively for solving partially observable reinforcement learning tasks.
Our architecture is built from a Transformer Decoder (like [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)).
DTQN is a sequence-to-sequence model; that is, given a history of the agent's interactions (either observation or action-observation) with the environment, DTQN outputs a sequence of Q-values.
Each element in the output sequence represents the Q-values for each action, given the agent's history up to that point.
For instance, the 3rd vector of Q-values was generated based on only the first three interactions between the agent and the environment.
This method allows us to train much more efficiently than a Sequence-to-One method, since we get `Sequence` more data points to use for training.
Our results providence strong evidence indicating a transformer can solve partially observable domains faster than previous recurrent approaches.
Our paper is now publicly available on arXiv!
You can read it [here](https://arxiv.org/abs/2206.01078).

Please note that we are continuing to work on this repository to extend and improve DTQN.
As such, the code in this branch may not reflect the code submitted with the original paper.
We will keep the [paper](https://github.com/kevslinger/DTQN/tree/paper) branch frozen with the code from the original paper, and only update it as needed to fix bugs.

## Table of Contents
- [Deep Transformer Q-Networks for Partially Observable Reinforcement Learning](#deep-transformer-q-networks-for-partially-observable-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Creating Environment](#creating-environment)
    - [Installing gym-gridverse](#installing-gym-gridverse)
    - [Installing rl-parsers](#installing-rl-parsers)
    - [Installing gym-pomdps](#installing-gym-pomdps)
  - [Training, final evaluation, and parameter sweeps](#training-final-evaluation-and-parameter-sweeps)
  - [Legacy combined runner](#legacy-combined-runner)
    - [Experiment Argument Details](#experiment-argument-details)
    - [Ablations](#ablations)
      - [Transformer Decoder Structure](#transformer-decoder-structure)
      - [Positional Encodings](#positional-encodings)
      - [Intermediate Q-value prediction](#intermediate-q-value-prediction)
  - [Citing DTQN](#citing-dtqn)
  - [Contributing](#contributing)

## Installation

To run our code, you must first set up your environment.
We recommend using `virtualenv` with `pip` to set up a virtual environment and dependency manager.
For our experiments, we use `python3.8`; while other versions of python will probably work, they are untested and we cannot guarantee the same performance.

### Creating Environment

First, create a virtual environment with `python3.8`, and then install the dependencies. This can be done with:

```bash
virtualenv venv -p 3.8
source venv/bin/activate
pip install -r requirements.txt
```

This basic setup will allow you to run the `Car Flag` and `Memory Cards` environments from our paper.
If you only have interest in running those domains, skip to the [current training workflow](#training-final-evaluation-and-parameter-sweeps).
Otherwise, you need to install the `gym-gridverse` repo to run the gridverse experiments, and the `gym-pomdps` and `rl-parsers` repos to run the classic POMDPs experiments.

### Installing gym-gridverse

To install gym-gridverse, you need to clone the `gym-gridverse` [github repo](https://github.com/abaisero/gym-gridverse.git).
Finally, once you have the source code, pip install it into the virtual enviroment.
Install its dependencies first, then the package itself with `--no-deps`:

```bash
git clone git@github.com:abaisero/gym-gridverse.git
pip install imageio imageio-ffmpeg more_itertools schema
pip install --no-deps ./gym-gridverse
```

A plain `pip install .` fails on recent compilers. `gym==0.18` declares
`Pillow<=7.2.0`, so pip downgrades Pillow and builds 7.2.0 from source, which
modern GCC rejects (C23 `bool`, incompatible pointer types). `--no-deps` keeps
the Pillow already in the environment. The clone is not needed after
installation, and `gym-gridverse/` is git-ignored if you clone it into the
repository root.

Check the installation with:

```bash
python -c "from utils.env_processing import make_env; make_env('gv_memory.5x5.yaml').reset()"
```

### Installing rl-parsers

If you also wish to run the Hallway and HeavenHell experiments, you will need to install `rl-parsers` and `gym-pomdps`.
First, install `rl-parsers` by cloning the [github repo](https://github.com/abaisero/rl-parsers.git) and installing it into your python virtual environment.

```bash
git clone git@github.com:abaisero/rl-parsers.git
cd rl-parsers
pip install -e .
```

### Installing gym-pomdps

Next, install `gym-pomdps` by cloning the [github repo](https://github.com/abaisero/gym-pomdps.git) and installing it into your python virtual environment.

```bash
git clone git@github.com:abaisero/gym-pomdps.git
cd gym-pomdps
pip install .
```

Tada! Now, you should be ready to run experiments.

## Training, final evaluation, and parameter sweeps

The new workflow separates training from final evaluation and records every
training, prefill and evaluation action for later inspection:

```bash
# Inspect the requested pairs before running them (currently four).
python train_and_eval.py --config experiments/tmaze.json --dry-run
python train_and_eval.py --config experiments/tmaze.json

# Quick four-job pipeline check.
python train_and_eval.py --config experiments/smoke.json
```

Only `train_and_eval.py` reads the JSON. Each list forms part of a full Cartesian
product. It invokes `train.py` once per configuration/seed, then `eval.py` on that
run's saved final model with the same environment settings. Results are compared
only after all jobs finish. Defaults are CPU, local logging, and output under
`results/<config-filename>/<batch>/`; use `--device cuda` for a CUDA installation.
A fresh invocation creates a new batch. `--resume BATCH` reuses successful stages;
`--aggregate-only BATCH` regenerates tables without executing experiments.

Each seed gets `result.csv`, final `model.pt`, episode/loss CSVs, resource metadata
and compressed step-by-step trajectories. `runs.csv` and `summary.csv` collect
success/return/length, wall-clock seconds and peak allocated GPU bytes, with
across-seed mean, sample std and Student-t 95% confidence intervals. Missing seeds
are flagged; incomplete groups do not receive comparison statistics.

Train/evaluate one configuration without a sweep:

```bash
python train.py --model DTQN --env PassiveTMaze-v0 --context 14 \
  --corridor-length 15 --num-steps 32000 --training-seed 1 \
  --output-dir results/manual-train
python eval.py --model-path results/manual-train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --output-dir results/manual-eval
```

See [the harness guide](docs/experiment-harness.md) for every flag/default/effect,
JSON rules, trajectory schemas, statistics, resource accounting and resume.
SciPy (`scipy==1.8.1`, already in `requirements.txt`) supplies the t distribution.
The supplied `tmaze.json` uses one training seed, so its std/CI fields are blank.
List multiple `training_seed` values to compare independent training runs.

| New workflow flag | Default | Effect |
|---|---|---|
| `--config` (outer) | Required for a new batch | Read the flat JSON parameter grid |
| `--output-root` (outer) | `results` | Parent folder for batch output |
| `--project-name` (outer) | Config filename stem | Output subdirectory |
| `--device` (all) | `cpu` | Select CPU or CUDA explicitly |
| `--verbose` (all) | Off | Print stage details; outer saves them in process logs |
| `--dry-run` (outer) | Off | List combinations/errors without running them |
| `--resume` (outer) | None | Reuse completed stages and retry failures |
| `--aggregate-only` (outer) | None | Rebuild comparisons from saved results |
| `--env` (train) | `DiscreteCarFlag-v0` | Select one training environment |
| `--output-dir` (train/eval) | Required | Write to a new/empty stage folder |
| `--model-path` (eval) | Required | Load a final trained model bundle |
| `--training-seed` (train) | `1` | Seed initialization, exploration, replay and the training environment |
| `--prefill-steps` (train) | `50000` | Minimum random prefill interactions; finish the last episode |
| `--eval-episodes` (eval) | `100` | Fixed number of greedy evaluation episodes |
| `--eval-seed` (eval) | `10000` | Seed evaluation independently from training |

Scientific JSON keys use underscores, e.g. `training_seed` and `eval_episodes`.
The outer script accepts scientific settings only from JSON; `train.py` receives
one concrete configuration through CLI flags. All inherited hyperparameter
defaults and model-specific effects are listed in the harness guide.

The former `run.py` is preserved as `run_legacy.py` for the original combined
training/periodic-evaluation workflow described below. It keeps its old flags,
W&B support and checkpoint semantics. New scripts always save the final policy
and do not need `--save-policy`, `--disable-wandb`, or `--eval-frequency`.

## Legacy combined runner

Our experiment script is `run_legacy.py`.
The arguments are explained at the bottom of the file.
Using the command
```shell
python run_legacy.py
```

You will run the default settings, which will run DTQN on the Car Flag domain as tested in the paper.

Here we provide a mapping from domain name as used in the paper to domain name used in the code:

- Hallway: `POMDP-hallway-episodic-v0`
- HeavenHell: `POMDP-heavenhell_3-episodic-v0`
- Gridverse memory 5x5: `gv_memory.5x5.yaml`
- Gridverse memory 7x7: `gv_memory.7x7.yaml`
- Gridverse Memory 9x9: `gv_memory.9x9.yaml`
- Gridverse four rooms 7x7: `gv_memory_four_rooms.7x7.yaml`
- Car Flag: `DiscreteCarFlag-v0`
- Memory Cards: `Memory-5-v0`

If you want to change the environment, use the `--envs` flag.
You can add multiple `envs` as long as they share the same observation and action space.
In that case, the agent will collect data with policy rollouts by randomly sampling one of the provided envs.
During evaluation, it will evaluate the policy separately on each environment, and record the results for each domain separately.
To reproduce our results, you may need to change the `in-embed` flag as well.
For `POMDP-hallway-episodic-v0`, `POMDP-heavenhell_3-episodic-v0`, and `DiscreteCarFlag-v0` domains, we used `--in-embed 64`.
For all others tasks, we use `--in-embed 128`.
For instance, to reproduce our Gridverse memory 7x7 experiment, you can use the command:

```shell
python run_legacy.py --envs gv_memory.7x7.yaml --in-embed 128
```

Which will run for 2,000,000 timesteps.
In our paper, we run our experiments with random seeds 1, 2, 3, 4, 5.

### Experiment Argument Details

We use [weights and biases](https://wandb.ai) to log our results.
If you do not have a weights and biases account, we recommend you get one!
However, if you still do not want to use weights and biases, you can use the `--disable-wandb` flag.
Then your results will be stored to a CSV file in `policies/<project_name>/<env>/<config>.csv`.

If you do not have access to a gpu, set `--device cpu` to train on CPU.

If you want command line outputs to view training results, use `--verbose`.

| Argument | Description |
| -------- | ----------- |
| `--project-name` | The project name (for wandb) or directory name (for local logging) to store the results. |
| `--disable-wandb` | Use `--disable-wandb` to log locally. |
| `--time-limit` | Time limit allowed for job. Useful for some cluster jobs such as slurm. |
| `--model` | Network model to use. |
| `--envs` | Domain to use. You can supply multiple domains, but they must have the same observation and action space. With multiple environments, the agent will sample a new one on each episode reset for conducting policy rollouts and collection experience. During evaluation, it will perform the same evaluation for each domain (Note: this may significantly slow down your run! Consider increasing the eval-frequency or reducing the eval-episodes). |
| `--num-steps` | Number of steps to train the agent. |
| `--tuf` | How many steps between each (hard) target network update. |
| `--lr` | Learning rate for the optimizer. |
| `--batch` | Batch size. |
| `--buf-size` | Number of timesteps to store in replay buffer. Note that we store the max length episodes given by the environment, so episodes that take longer will be padded at the end. This does not affect training but may affect the number of real observations in the buffer. |
| `--eval-frequency` | How many training timesteps between agent evaluations. |
| `--eval-episodes` | Number of episodes for each evaluation period. |
| `--device` | Pytorch device to use. |
| `--context` | For DRQN and DTQN, the context length to use to train the network. |
| `--obs-embed` | For discrete observation domains only. The number of features to give each observation. |
| `--a-embed` | The number of features to give each action. A value of 0 will prevent the policy from using the previous action. |
| `--in-embed` | The dimensionality of the network. In the transformer, this is referred to as `d_model`. |
| `--max-episode-steps` | The maximum number of steps allowed in the environment. If `env` has a `max_episode_steps`, this will be inferred. Otherwise, this argument must be supplied. |
| `--seed` | The random seed to use. |
| `--save-policy` | Use this to save the policy so you can load it later for rendering. |
| `--verbose` | Print out evaluation results as they come in to the console. |
| `--render` | Enjoy mode (NOTE: must have a trained policy saved). |
| `--history` | This is how many (intermediate) Q-values we use to train for each context. To turn off intermediate Q-value prediction, set `--history 1`. To use the entire context, set history equal to the context length. |
| `--heads` | Number of heads to use for the transformer. |
| `--layers` | Number of transformer blocks to use for the transformer. |
| `--dropout` | Dropout probability. |
| `--discount` | Discount factor. |
| `--gate` | Combine step to use. |
| `--identity` | Whether or not to use identity map reordering. |
| `--pos` | The type of positional encodings to use. |
| `--bag-size` | The size of the persistent memory bag. |
| `--slurm-job-id` | The `$SLURM_JOB_ID` assigned to this job. |
| |


### Ablations

#### Transformer Decoder Structure
To run DTQN with the GRU-like gating mechanism, use `--gate gru`.

To run DTQN with identity map reordering, use `--identity`.

To use both GRU-like gating as well as identity map reordering, use `--gate gru --identity`.

#### Positional Encodings
To run DTQN with the sinusoidal positional encodings, use `--pos sin`.

To run DTQN without positional encodings, use `--pos none`.

#### Intermediate Q-value prediction
To run DTQN without intermediate Q-value prediction, use `--history`.

## Citing DTQN

To cite this paper/code in publications, please use the following bibtex:

```bibtex
@article{esslinger2022dtqn,
  title = {Deep Transformer Q-Networks for Partially Observable Reinforcement Learning},
  author = {Esslinger, Kevin and Platt, Robert and Amato, Christopher},
  journal= {arXiv preprint arXiv:2206.01078},
  year = {2022},
}

```

## Contributing

Feel free to open a pull request with updates, contributions, and modifications.
