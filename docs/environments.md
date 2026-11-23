# Environment guide

This guide documents the repository's complete configured environment inventory
and the current train/final-evaluation workflow. It records task definitions and
experimental roles; it does not report new learning results. The inventory and
interfaces were checked on **2026-11-23** against `main` at `ed83da1`, including
the merged Passive/Noisy T-Maze, Associative Corridor, experiment harness and
both DTQN kNN variants.

Every family uses the same seven fields: **Identity and availability**, **Paper-ready
description**, **Task mechanics**, **Interface**, **Configuration**, **Usage**, and
**References**. The two sentences labeled **What it tests** and **Limitation** are
the material intended for later paper reuse. Technical instructions remain separate
so they can be omitted from a paper without losing the experimental rationale.

## Inventory and availability

There are **35 selectable IDs or configuration filenames across 14 families**:
five project tasks, eight GridVerse configurations, twenty MiniHack variants, and
two classic POMDPs. Families share a description, while their tables enumerate
every size or observation variant. Arbitrary environments in a user's global Gym
registry are outside this inventory.

| Family                                                      | Choices | Code availability in the checked snapshot                   |
|-------------------------------------------------------------|--------:|-------------------------------------------------------------|
| [Passive T-Maze](#passive-t-maze)                           |       1 | `main`, local implementation                                |
| [Noisy T-Maze](#noisy-t-maze)                               |       1 | `main`, local implementation                                |
| [Associative Corridor](#associative-corridor)               |       1 | `main`, local implementation                                |
| [Car Flag](#car-flag)                                       |       1 | `main`, local implementation                                |
| [Memory Cards](#memory-cards)                               |       1 | `main`, local implementation                                |
| [GridVerse Memory](#gridverse-memory)                       |       6 | `main`, YAML configurations and optional provider           |
| [GridVerse Memory Four Rooms](#gridverse-memory-four-rooms) |       2 | `main`, YAML configurations and optional provider           |
| [MiniHack Room](#minihack-room)                             |       6 | `main`, registrations and optional provider                 |
| [MiniHack Dark Room](#minihack-dark-room)                   |       6 | `main`, registrations and optional provider                 |
| [MiniHack MazeWalk](#minihack-mazewalk)                     |       3 | `main`, registrations and optional provider                 |
| [MiniHack Mapped Maze](#minihack-mapped-maze)               |       3 | `main`, registrations and optional provider                 |
| [MiniHack Custom Maze](#minihack-custom-maze)               |       2 | `main`, level definition and optional provider              |
| [Hallway](#hallway)                                         |       1 | Optional `gym_pomdps` registration referenced by the README |
| [HeavenHell](#heavenhell)                                   |       1 | Optional `gym_pomdps` registration referenced by the README |

`main` in this table means the checked source snapshot, not a promise that optional
dependencies are installed. All five project tasks are available from one current
checkout; no environment feature-branch switch is required. Source links for the
three added implementations are pinned to the audited `main` commit.

## Shared execution and evaluation conventions

- Run from the **repository root**, with the project's virtual environment
  activated. The installed validation stack is Python 3.10.21, NumPy 1.22.4,
  Gym 0.23.1 and Torch 2.7.1+cu128; this audit uses CPU. The inherited runtime
  requirements instead pin Gym 0.18.0 and Torch 1.11.0. This is still the legacy
  Gym API: `reset()` returns an observation and `step()` returns
  `(observation, reward, done, info)`.
- `train.py` trains **one configuration**, without intermediate evaluations,
  and always saves its final `model.pt`. `eval.py` loads that model and its saved
  environment settings for greedy episodes without learning. Both accept CLI
  values; only `train_and_eval.py` reads a JSON grid and loops over its combinations.
- Current flags are `--env` and `--training-seed`. `run.py` was renamed to
  `run_legacy.py`; its plural `--envs`, `--seed`, periodic evaluation, W&B and
  checkpoint flags belong to that older workflow. The new scripts use local
  logging and need no `--disable-wandb`, `--save-policy` or `--eval-frequency`.
- `--context K` sets the local observation-window capacity. `--history` selects
  how many final token predictions contribute directly to TD loss; it does not
  limit attention. Values above K are clipped. `DTQN` is the baseline;
  `DTQN-kNN` rebuilds external replay K/V, and `DTQN-kNN-cached` reuses detached
  collection-time K/V. See the [integration](knn-integration.md) and
  [cached-memory](knn-cached.md) guides for options and costs. Both require
  `bag_size=0`; the cached variant requires `memory_stop_grad=true`.
- `--prefill-steps` defaults to **50,000 additional random interactions**, finishing
  the final prefill episode. `--num-steps` then counts training interactions
  exactly; an unfinished final episode is recorded as partial. The examples use
  a smaller explicit prefill and batch for execution checks, not learning claims.
  Updates start only once enough completed replay episodes exist.
- `--max-episode-steps N` on `train.py` installs a real outer **TimeLimit** and
  sets the replay horizon; it can shorten the native limit but cannot extend a
  provider's internal deadline. Otherwise a finite native horizon is required.
  Replay episode lengths are `int32`; the old 255-step storage limit is fixed.
- Reported success is terminal `info['is_success']` **or** positive episode
  return, except Associative Corridor requires **explicit terminal success**.
  Positive partial return followed by failure is not success there. Forwarded
  provider info does not establish a task-specific scientific success metric;
  verify the fallback before interpreting optional-provider success rates.

Each family below gives one current training example. Evaluate its final policy
by replacing the paths with that example's output directory:

```bash
python eval.py --model-path results/passive-example/train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/passive-example/eval --verbose
```

Use new/empty output directories. Training writes `model.pt`,
`train_metadata.json`, `train_episodes.csv`, `train_losses.csv` and
`train_trajectories.jsonl.gz`; evaluation writes `eval_metadata.json`,
`eval_episodes.csv` and `eval_trajectories.jsonl.gz`. Traces record every prefill,
training and evaluation action with observations, rewards, termination/truncation
and available world-state snapshots. Metadata records settings, timing and GPU
memory. A saved final policy does not reproduce the changing policies used
earlier in training; the saved traces preserve those interactions.

For a comparison, save this example as `experiments/environment-comparison.json`:

```json
{
  "model": ["DTQN", "DTQN-kNN", "DTQN-kNN-cached"],
  "env": ["PassiveTMaze-v0"],
  "context": [14],
  "history": [14],
  "corridor_length": [15],
  "memory_size": [32],
  "memory_top_k": [4],
  "batch": [4],
  "prefill_steps": [128],
  "num_steps": [32000],
  "training_seed": [1, 2, 3, 4, 5],
  "eval_episodes": [100],
  "eval_seed": [10000]
}
```

```bash
python train_and_eval.py --config experiments/environment-comparison.json \
  --project-name environment-comparison --device cpu --dry-run
python train_and_eval.py --config experiments/environment-comparison.json \
  --project-name environment-comparison --device cpu
```

Every list participates in the Cartesian product: this grid gives 15 train/eval
pairs. Batch outputs are under `results/<project-name>/<batch>/`, with per-run
artifacts, `runs.csv` and `summary.csv` containing across-training-seed statistics.
Environment and memory settings unused by a model are recorded as inactive.
The shared evaluation seed fixes initial evaluation randomness across training
seeds; it is separate from initialization/exploration/replay randomness.

Neither new CLI renders episodes, and `model.pt` is not the legacy runner's
checkpoint format. Use saved trajectories for subsequent plotting or a provider's
renderer in separate code. The per-family rendering status below describes the
environment API, not an implemented playback command.

GridVerse, MiniHack and `gym_pomdps` were **not installed locally** during this
audit. Their commands are checked against the current CLI and source inventory,
but execution remains conditional on the stated provider and compatibility
requirements. The pinned MiniHack sources and unpinned POMDP provider references
describe the versions inspected for the original guide; no provider upgrade is
implied by refreshing this workflow.

## Passive T-Maze

### Identity and availability

ID: `PassiveTMaze-v0`. Source: [Passive T-Maze implementation][passive-code],
on the audited `main` snapshot. Requires the runner's existing Gym and
NumPy dependencies; optional GridVerse/MiniHack/POMDP packages are unnecessary.

### Paper-ready description

**What it tests:** Passive T-Maze measures whether an agent can use an initial
branch instruction after a controlled delay.

**Limitation:** Its single distinctive cue can be solved with one retained bit
and does not establish the necessity of content-based retrieval.

### Task mechanics

Each reset samples the correct branch, Up or Down, with equal probability. The
agent sees the instruction once, traverses a corridor, and selects a branch.
The cue is not repeated when revisiting the start. This port follows the passive
task and reward configuration of [Ni et al. (2023)][ni2023].

### Interface

| Situation | Two-value `float32` observation |
|---|---|
| Reset; desired branch Up / Down | `[0, +1]` / `[0, -1]` |
| Subsequent corridor observation | `[0, 0]` |
| Junction | `[1, 0]` |
| Entered Up / Down branch | `[1, +1]` / `[1, -1]` |

The first coordinate indicates the junction column, including its branches;
there is no separate instruction-validity flag. The second coordinate means the
desired direction only at reset and the actual vertical position at the branches.
Actions are Right `0`, Up `1`, Left `2`, Down `3`; wall collisions leave the agent
in place. Nonterminal steps behind the shortest route incur `-1/L`. At the fixed
deadline, the correct branch gives 1 and other positions give 0; `is_success`
indicates whether the final branch matches the initial cue.

### Configuration

`--corridor-length L`: integer at least 1, default 10. The horizon is **L + 1**
actions. At the junction, the initial observation remains in a K-observation
window only when `K >= L + 1`; choosing `L >= K` excludes it. Corridor length
changes delay, not the number of bits to retain.

Correct traversal followed by a random Up/Down decision succeeds with probability
1/2. Uniformly random actions throughout the episode instead succeed with
probability `4**(-(L + 1))`, because the deadline permits only the shortest route.

### Usage

Run from the current repository root:

```bash
python train.py --model DTQN --env PassiveTMaze-v0 \
  --corridor-length 100 --context 50 --history 50 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/passive-example/train --verbose
```

Output: `results/passive-example/train/model.pt` and the shared training artifacts.
**Rendering:** not implemented in this port.

### References

[Ni et al. (2023), Section 3.2][ni2023]; [upstream environment][memory-rl-code];
[upstream passive configuration][memory-rl-config]; [local port][passive-code].

## Noisy T-Maze

### Identity and availability

ID: `NoisyTMaze-v0`. Source: [Noisy T-Maze implementation][noisy-code], on
`main`, building on the Passive T-Maze port. Requires
only the same runtime dependencies as that port.

### Paper-ready description

**What it tests:** Noisy T-Maze assesses cue recall in the presence of corridor
observations drawn from distributions centered on the valid cue values.

**Limitation:** The initial instruction remains identifiable by its timing, so
success does not establish retrieval among arbitrary competing instructions.

### Task mechanics

The initial cue, junction and branches remain clean. On each subsequent corridor
observation, zero is replaced with probability `p` by an independent sample from
an equally weighted mixture of `Normal(-1, sigma**2)` and `Normal(+1, sigma**2)`.
Distractors are independent of the goal and use a separate seeded RNG. Gaussian
samples almost surely differ from exactly +/-1; these are near-cue distractors,
not deliberately identical instruction tokens.

### Interface

Observations are `[junction_flag, value]` as in Passive T-Maze, with only the
corridor's second coordinate potentially replaced. Positive noise uses unbounded
second-coordinate Box limits; samples are not clipped. Actions, movement,
rewards, terminal `is_success` and the **L + 1** horizon are unchanged. The
scripted oracle can ignore all corridor values after reading the initial cue.

### Configuration

| Parameter | Default | Effect |
|---|---:|---|
| `--corridor-length` | 10 | Positive integer corridor length, as in Passive T-Maze |
| `--noise-level` | 0.5 | Finite replacement probability `0 <= p <= 1` |
| `--noise-std` | 0.1 | Finite positive Gaussian standard deviation |

At `p=0`, spaces and seeded observations/rewards/termination match the clean port.
Increasing `p` changes distractor frequency; increasing `sigma` changes spread,
not the two means. Neither parameter changes the task's reward structure.

### Usage

Run from the current repository root:

```bash
python train.py --model DTQN --env NoisyTMaze-v0 \
  --corridor-length 15 --noise-level 0.5 --noise-std 0.1 --context 14 --history 14 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/noisy-example/train --verbose
```

Output: `results/noisy-example/train/model.pt`; metadata stores corridor and
noise settings. **Rendering:** no renderer is added. This variant is selected
by its own ID; noise flags do not
modify Car Flag, Memory Cards or clean Passive T-Maze.

### References

[Implementation and parameter validation][noisy-code];
[recorded tests and CPU checks][noisy-validation];
[issue #20][issue20]; [Ni et al. (2023)][ni2023] for the underlying clean task.
The Gaussian extension is this project's variant, not a result from that paper.

## Associative Corridor

### Identity and availability

ID: `AssociativeCorridor-v0`. Source: [Associative Corridor implementation][assoc-code],
on `main`. This is the four-decision
variant, not the single-junction task from Memory-RL. No extra runtime package is
required beyond Gym and NumPy used by the runner.

### Paper-ready description

**What it tests:** Associative Corridor measures recall of four independently
randomized class-direction associations when each class is queried later in the episode.

**Limitation:** Random event placement permits short recall delays and contextual
compression, so success alone does not demonstrate beyond-window retrieval or the necessity of kNN.

### Task mechanics

Classes A-D each receive one Up/Down instruction and one later use. Eight distinct
positions are sampled, including the final corridor position, and assigned in
pairs to classes; each pair's earlier event supplies its instruction and its later
event queries it. Instructions and queries can interleave. Directions are sampled
independently, so all classes may happen to share a direction. Unoccupied
positions are blank, and the final position is always a query.

### Interface

The six `float32` values are `[A, B, C, D, is_query, direction]`:

| Situation | Example observation |
|---|---|
| Instruction: B means Down | `[0, 1, 0, 0, 0, -1]` |
| Blank corridor | `[0, 0, 0, 0, 0, 0]` |
| Query: which direction for B? | `[0, 1, 0, 0, 1, 0]` |
| Terminal observation | `[0, 0, 0, 0, 0, 0]` |

The class is one-hot; direction is +/-1 only when an instruction is given and is
zero otherwise. Right `0` advances at instructions and blanks. Up `1` or Down `3`
answers a query and advances when correct; a correct query gives +1. Every valid
but incorrect action, including Left `2`, terminates immediately with reward 0.
Success requires all four correct answers; its return is 4. A failed episode can
retain rewards from earlier answers, so the evaluator explicitly uses
`is_success`, rather than positive return, for this environment alone.

### Configuration

`--corridor-length L`: integer at least **8**, default 10; here it counts decision
positions, and a successful episode lasts **L** steps. It is not the clean
T-Maze's `L + 1` horizon. Class count is fixed at four. The seeded `instructions`
and `queries` dictionaries expose the schedule on `env.unwrapped`; only the
observation is provided to the policy.

A longer corridor does not guarantee every instruction/use gap exceeds K.
Always recalling the first instruction is insufficient for the four independent
answers, but a later contextual value may encode several instructions. With
correct corridor actions and independent random Up/Down answers, whole-episode
success is `1/16`, not 1/2; per-answer accuracy is a different metric.

### Usage

Run as its own training configuration; its six-value observations differ from
the two-value T-Maze observations:

```bash
python train.py --model DTQN --env AssociativeCorridor-v0 \
  --corridor-length 100 --context 14 --history 14 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/associative-example/train --verbose
```

Output: `results/associative-example/train/model.pt`; metadata stores corridor
length. **Rendering:** not implemented.

### References

[Implementation and event scheduling][assoc-code];
[recorded oracle, evaluation and CPU checks][assoc-validation]. The scientific
role above is the proposed use of this new task, not a published empirical result.

## Car Flag

### Identity and availability

ID: `DiscreteCarFlag-v0`. Source: [local Car Flag implementation][car-code] and
[registration][registration] on `main`. Pyglet is imported by the local class even
for headless use; it is included in the inherited runtime requirements.

### Paper-ready description

**What it tests:** In this study, Car Flag checks only for no performance loss
within the context window when memory is added to the baseline at K = 50.

**Limitation:** Performance on this selected control configuration cannot
establish that an agent retrieves information beyond its context window.

### Task mechanics

A car starts near the center of a one-dimensional track, visits an oracle region
to learn which endpoint is rewarding, then drives to that endpoint. The goal
side is randomized each reset. This is a within-window control role for the
present study, not a statement that all possible Car Flag variants have short
memory requirements or that a new baseline experiment has been completed.

### Interface

The declared observation is a three-value Box: `[position, velocity, direction]`,
with position in [-1.1, 1.1], velocity in [-0.07, 0.07], and direction in [-1, 1].
Direction is +/-1 in the oracle region `[0.3, 0.7]` and zero elsewhere. Discrete
actions `0/1/2` apply negative/zero/positive acceleration; they do not mean the
T-Maze's Right/Up/Left. Reaching the goal endpoint gives +1, the other gives -1,
and intermediate steps give 0. Terminal `is_success` records a positive goal reward.

### Configuration

The registered discrete task has a **200-step TimeLimit**. Track, acceleration,
oracle location and reward parameters are class constants; there are no
environment-specific CLI flags. The class also supports a continuous constructor
option, but that is not the registered task and the runner assumes discrete actions.
The examples use the upstream recommendation `--in-embed 64` and `K=50`.

### Usage

```bash
python train.py --model DTQN --env DiscreteCarFlag-v0 \
  --in-embed 64 --context 50 --history 50 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/car-flag-example/train --verbose
```

**Rendering:** the class has an inherited Gym/Pyglet graphical renderer. It uses
`gym.envs.classic_control.rendering`, which is absent in the tested Gym 0.23.1
stack; successful headless training is not a verification of rendering there.

### References

[Implementation][car-code]; [Esslinger et al. (2022), Appendix B.4][dtqn2022];
[issue #2][issue2] for the selected within-window control role.

## Memory Cards

### Identity and availability

ID: `Memory-5-v0`. Source: [local Memory Cards implementation][cards-code] and
[registration][registration] on `main`. Requires Gym and NumPy.

### Paper-ready description

**What it tests:** In this study, Memory Cards checks only for no performance
loss within the context window when memory is added to the baseline at K = 50.

**Limitation:** Its selected five-pair, 50-step configuration cannot establish
retrieval of observations older than that window.

### Task mechanics

Five pairs are shuffled among ten positions. One unremoved card is revealed at
each step, and the agent selects the position of its matching partner. Correct
pairs are removed until the board is empty. Useful history associates card values
with positions, rather than remembering a single initial instruction.

### Interface

Observation space is `MultiDiscrete([7] * 10)`: ten positions with hidden marker
**0**, card values **1-5**, and removed marker **6**. The action space is
`Discrete(10)`, indexing the guessed partner position. The current implementation
gives **0 for a correct match** and **-1 for an incorrect guess**, including choosing
the currently revealed card. All pairs removed sets `is_success=True`; timeouts
are failures. These values follow the executable code, not the older illustrative
docstring's removed marker -1 or its +1 match rewards.

### Configuration

The registration fixes `num_pairs=5` and a **50-step TimeLimit**. There is no
pair-count CLI flag; changing pair count requires another registration or direct
construction. With `K=50`, the selected episode's observations fit within the
window, including its initial observation at the final action decision.

### Usage

```bash
python train.py --model DTQN --env Memory-5-v0 \
  --context 50 --history 50 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/cards-example/train --verbose
```

**Rendering:** no renderer is implemented in the local class.

### References

[Implementation][cards-code]; [Esslinger et al. (2022), Appendix B.3][dtqn2022];
[issue #2][issue2] for the selected within-window control role.

## GridVerse Memory

### Identity and availability

These YAML files are present on `main`; execution additionally requires
[gym-gridverse][gv-package], which was not installed for this audit.

| Configuration filename | Grid shape | Repository TimeLimit |
|---|---|---:|
| `gv_memory.5x5.yaml` | 5 x 5 | 250 |
| `gv_memory.7x7.yaml` | 7 x 7 | 250 |
| `gv_memory.9x9.yaml` | 9 x 9 | 250 |
| `gv_memory.11x11.yaml` | 11 x 11 | 250 |
| `gv_memory.13x13.yaml` | 13 x 13 | 250 |
| `gv_memory.21x21.yaml` | 21 x 21 | 250 |

### Paper-ready description

**What it tests:** GridVerse Memory tests retaining a beacon's color while
localizing and navigating to a matching exit under partial visibility.

**Limitation:** Exploration and navigation contribute to difficulty, so grid
size alone does not specify an exact required recall delay.

### Task mechanics

The provider's `memory` reset function creates a grid with colored beacons and
exits, using RED, GREEN, BLUE and YELLOW. A matching exit is rewarding; the agent
must acquire the color information and navigate with a restricted field of view.

### Interface

Actions, in YAML order, are MOVE_FORWARD `0`, MOVE_BACKWARD `1`, MOVE_LEFT `2`,
MOVE_RIGHT `3`, TURN_LEFT `4`, TURN_RIGHT `5`. The configured partially occluded
view covers 2 x 3 cells. The local wrapper sums grid channels 0 and 2
(object/color) and flattens the result into a categorical vector, discarding the
remaining channel. Reward components are +5 for a matching exit, -5 for a wrong
exit, and a living reward of -0.05. Reaching an exit terminates; otherwise the
repository caps the episode at 250. Provider info is forwarded unchanged, so
reported success follows the shared evaluator rule.

### Configuration

Select a filename from the table through `--env`; the factory resolves it under
`envs/gridverse/`. Shape differs across these files; colors, view area, actions
and reward components remain the same. `--corridor-length` does not configure
these tasks. Their presence on disk does not install the provider.

### Usage

After installing the optional provider using the README's instructions (not
executed locally during this audit):

```bash
python train.py --model DTQN --env gv_memory.7x7.yaml \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/gv-memory-example/train --verbose
```

**Rendering:** delegated to the optional provider through the wrappers; not
verified with the local dependency stack.

### References

[YAML configurations][gv-configs]; [local wrapper][gv-wrapper];
[factory and TimeLimit][factory]; [provider][gv-package];
[Esslinger et al. (2022), Appendix B.2][dtqn2022].

## GridVerse Memory Four Rooms

### Identity and availability

These files are present on `main` and require the optional [GridVerse provider][gv-package].

| Configuration filename          | Grid shape | Room layout | Repository TimeLimit |
|---------------------------------|------------|-------------|---------------------:|
| `gv_memory_four_rooms.7x7.yaml` | 7 x 7      | 2 x 2       |                  250 |
| `gv_memory_four_rooms.9x9.yaml` | 9 x 9      | 2 x 2       |                  250 |

### Paper-ready description

**What it tests:** GridVerse Memory Four Rooms tests recalling a beacon's target
color while exploring a partitioned environment to find a matching exit.

**Limitation:** Procedural layout and exploration introduce demands beyond
memory retention, preventing a clean attribution of performance to recall distance.

### Task mechanics

The `memory_rooms` reset function uses a 2 x 2 room layout, one beacon, two exits,
and the same four-color palette. The agent locates the information and exits
through the one associated with the beacon color.

### Interface

The six action indices, categorical 2 x 3 observation processing, reward
components (+5/-5 exit rewards and -0.05 living reward), and forwarded info are
the same as [GridVerse Memory](#gridverse-memory). An exit ends the episode;
the repository also applies a 250-step TimeLimit. The reported-success
rule must be distinguished from the task's matching-exit definition.

### Configuration

Choose either YAML filename; the grid shape changes while the listed room,
beacon and exit counts remain fixed. No T-Maze flags apply. The selected file
and K must both be recorded when discussing whether recall exceeds the window.

### Usage

With the optional provider installed; this example was not executed locally:

```bash
python train.py --model DTQN --env gv_memory_four_rooms.7x7.yaml \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/gv-rooms-example/train --verbose
```

**Rendering:** provider-dependent, delegated through wrappers and not verified here.

### References

[Configurations][gv-configs]; [wrapper][gv-wrapper]; [factory][factory];
[provider][gv-package]; [Esslinger et al. (2022), Appendix B.2][dtqn2022].

## MiniHack provider conventions

The next five families use [local registrations][registration] and
[MiniHackWrapper][mh-wrapper]. The inherited requirements pin MiniHack **0.1.3**
and NLE **0.8.1**. Their providers were source-inspected, not installed or run for
this audit. The sources below use the matching MiniHack version.

- The navigation action set contains the eight compass directions, indexed in
  the provider's action tuple; it is not the four-action T-Maze convention.
- `v0` usually selects a 9 x 9 **cell** glyph crop, flattened to 81 categorical
  entries; `v1` usually selects 3 x 3 cells, flattened to 9 entries. Explicit
  per-family tables take precedence, especially for the custom maze.
- Pixel variants select a 9 x 9 **cell** crop rendered into a larger RGB image;
  9 x 9 cells does not mean nine by nine pixels. The wrapper declares a
  channel-first `uint8` Box and uses `reshape`, not `transpose`, when converting
  the provider image; documentation preserves that existing implementation.
- Default task rewards have a win component +1 and loss/abort component 0, plus
  the provider's time penalty, including its default frozen-step penalty. These
  are not a guarantee that total return equals the win component.
- Reaching the down staircase is the default navigation objective. Provider
  termination can also occur before the repository's outer TimeLimit. The
  wrapper forwards info rather than translating `end_status` into `is_success`;
  the runner's reported success therefore uses the shared fallback.
- Registration suffixes and sizes choose fixed wrapper options. There are no
  CLI flags for crop size, lighting, or changing the task's action set.

Sources: [navigation defaults][mh-navigation], [rewards and crop handling][mh-base],
[local wrapper][mh-wrapper], and [registrations][registration].

## MiniHack Room

### Identity and availability

Registered on `main`; requires the optional MiniHack/NLE stack above.

| Repository ID   | Provider ID              | Observation         | Outer cap / provider cap |
|-----------------|--------------------------|---------------------|--------------------------|
| `MH-Room-5-v0`  | `MiniHack-Room-5x5-v0`   | Glyphs, 9 x 9 cells | 100 / 100                |
| `MH-Room-5-v1`  | `MiniHack-Room-5x5-v0`   | Glyphs, 3 x 3 cells | 100 / 100                |
| `MH-Room-5-v2`  | `MiniHack-Room-5x5-v0`   | Pixels, 9 x 9 cells | 100 / 100                |
| `MH-Room-15-v0` | `MiniHack-Room-15x15-v0` | Glyphs, 9 x 9 cells | 300 / 300                |
| `MH-Room-15-v1` | `MiniHack-Room-15x15-v0` | Glyphs, 3 x 3 cells | 300 / 300                |
| `MH-Room-15-v2` | `MiniHack-Room-15x15-v0` | Pixels, 9 x 9 cells | 300 / 300                |

### Paper-ready description

**What it tests:** MiniHack Room tests navigation to a staircase under the
repository's restricted glyph or pixel observations.

**Limitation:** A fixed empty-room route can be learned without demonstrating
selective retrieval of an earlier instruction.

### Task mechanics

The selected provider IDs use lit, empty rooms with fixed start and goal positions;
they are not the separate `Room-Random` variants. The agent moves from the start
to the staircase in a 5 x 5 or 15 x 15 room.

### Interface

Use the [MiniHack conventions](#minihack-provider-conventions): eight compass
actions, the table's crop encoding, navigation reward components plus time
penalties, and provider termination. The table shows both time caps; the first
one reached ends the episode. Success reporting follows the shared rule.

### Configuration

Select the exact ID to choose size and crop type. Default provider room limits
are `size * 20`, matching these outer registrations. Replay lengths are now
`int32`, so the former 255-step storage limit does not apply.

### Usage

Conditional on the optional dependencies; not executed locally:

```bash
python train.py --model DTQN --env MH-Room-5-v0 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/mh-room-example/train --verbose
```

**Rendering:** delegated to MiniHack/NLE; no custom renderer is added by the local
wrapper, and compatibility with the pinned stack was not exercised here.

### References

[Local registrations][registration]; [wrapper][mh-wrapper];
[MiniHack 0.1.3 room definitions][mh-room]; [reward implementation][mh-base].

## MiniHack Dark Room

### Identity and availability

Registered on `main`; requires the optional MiniHack/NLE stack.

| Repository ID       | Provider ID                   | Observation         | Outer cap / provider cap |
|---------------------|-------------------------------|---------------------|--------------------------|
| `MH-DarkRoom-5-v0`  | `MiniHack-Room-Dark-5x5-v0`   | Glyphs, 9 x 9 cells | 100 / 100                |
| `MH-DarkRoom-5-v1`  | `MiniHack-Room-Dark-5x5-v0`   | Glyphs, 3 x 3 cells | 100 / 100                |
| `MH-DarkRoom-5-v2`  | `MiniHack-Room-Dark-5x5-v0`   | Pixels, 9 x 9 cells | 100 / 100                |
| `MH-DarkRoom-15-v0` | `MiniHack-Room-Dark-15x15-v0` | Glyphs, 9 x 9 cells | 300 / 300                |
| `MH-DarkRoom-15-v1` | `MiniHack-Room-Dark-15x15-v0` | Glyphs, 3 x 3 cells | 300 / 300                |
| `MH-DarkRoom-15-v2` | `MiniHack-Room-Dark-15x15-v0` | Pixels, 9 x 9 cells | 300 / 300                |

### Paper-ready description

**What it tests:** MiniHack Dark Room tests navigation and retention of useful
spatial information during exploration with reduced visibility.

**Limitation:** Randomized placement and exploration are entangled with memory,
so the task does not isolate a fixed instruction-to-use delay.

### Task mechanics

These provider variants use unlit rooms and randomized start/goal placement.
The objective is to find the staircase from local observations; this is not a
single instruction followed by an otherwise blank corridor.

### Interface

The [MiniHack conventions](#minihack-provider-conventions) apply: eight compass
actions, the listed glyph/pixel crops, default navigation rewards plus time
penalties, provider termination and the table's outer cap. Reported success is
the flag-or-positive-return rule rather than a new local classifier.

### Configuration

Choose 5 x 5 or 15 x 15 through the ID and use its suffix for observation type.
Lighting and randomization are provider-constructor settings, not runner flags.
The 300-step cases use the current `int32` replay-length storage.

### Usage

Conditional on the optional dependencies; not executed locally:

```bash
python train.py --model DTQN --env MH-DarkRoom-5-v1 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/mh-dark-example/train --verbose
```

**Rendering:** delegated to MiniHack/NLE; not verified with the local stack.

### References

[Local registrations][registration]; [wrapper][mh-wrapper];
[MiniHack room/dark-room constructors][mh-room]; [rewards][mh-base].

## MiniHack MazeWalk

### Identity and availability

Registered on `main`; requires optional MiniHack/NLE.

| Repository ID | Provider ID | Observation | Outer cap / provider cap |
|---|---|---|---|
| `MH-Maze-9-v0` | `MiniHack-MazeWalk-9x9-v0` | Glyphs, 9 x 9 cells | 180 / 200 |
| `MH-Maze-9-v1` | `MiniHack-MazeWalk-9x9-v0` | Glyphs, 3 x 3 cells | 180 / 200 |
| `MH-Maze-9-v2` | `MiniHack-MazeWalk-9x9-v0` | Pixels, 9 x 9 cells | 180 / 200 |

### Paper-ready description

**What it tests:** MiniHack MazeWalk tests spatial exploration and navigation
through a generated maze using local observations.

**Limitation:** Maze layout, exploration and provider visibility mechanisms
affect difficulty alongside memory, without a controlled recall distance.

### Task mechanics

The provider generates a 9 x 9 maze and places a staircase; the selected variants
are not premapped. The agent explores passages to reach that staircase.

### Interface

The [MiniHack conventions](#minihack-provider-conventions) specify actions,
observation processing, navigation rewards and success-reporting behavior.
The repository's 180-step outer cap precedes the provider's 200-step cap unless
the task terminates earlier.

### Configuration

The ID suffix chooses glyph crop size or pixels; all three use the same provider
task size. Neither maze dimensions nor observation crop are set by T-Maze flags.

### Usage

Conditional on the optional dependencies; not executed locally:

```bash
python train.py --model DTQN --env MH-Maze-9-v0 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/mh-maze-example/train --verbose
```

**Rendering:** delegated to MiniHack/NLE; not verified locally.

### References

[Registrations][registration]; [wrapper][mh-wrapper];
[MiniHack 0.1.3 MazeWalk constructors][mh-maze]; [reward defaults][mh-base].

## MiniHack Mapped Maze

### Identity and availability

Registered on `main`; requires optional MiniHack/NLE. The pixel registration has
an existing provider mismatch, recorded explicitly below rather than corrected.

| Repository ID     | Actual provider ID                | Observation         | Outer cap / provider cap |
|-------------------|-----------------------------------|---------------------|--------------------------|
| `MH-MazeMap-9-v0` | `MiniHack-MazeWalk-Mapped-9x9-v0` | Glyphs, 9 x 9 cells | 180 / 200                |
| `MH-MazeMap-9-v1` | `MiniHack-MazeWalk-Mapped-9x9-v0` | Glyphs, 3 x 3 cells | 180 / 200                |
| `MH-MazeMap-9-v2` | `MiniHack-MazeWalk-9x9-v0`        | Pixels, 9 x 9 cells | 180 / 200                |

### Paper-ready description

**What it tests:** MiniHack Mapped Maze registrations test navigation from cropped
observations, with premapping enabled in the glyph variants.

**Limitation:** The pixel variant currently selects the ordinary unmapped task,
so these registrations cannot be treated as a clean mapped-versus-unmapped comparison.

### Task mechanics

The mapped provider adds its `premapped` level flag. The agent still receives
the local wrapper's selected crop and must reach the staircase. The v2 ID lacks
that mapped provider setting despite its repository name.

### Interface

The [MiniHack conventions](#minihack-provider-conventions) apply to compass
actions, glyph/pixel processing, reward components and shared success reporting.
The effective numeric cap is 180 steps, bounded by the outer registration before
the provider's 200-step limit.

### Configuration

Use the actual provider ID in the table when describing experiments. The mapped
name alone does not guarantee all observation variants have matching task semantics.
No code or registration is changed by this documentation.

### Usage

Conditional on the optional dependencies; not executed locally:

```bash
python train.py --model DTQN --env MH-MazeMap-9-v1 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/mh-mapped-example/train --verbose
```

**Rendering:** delegated to MiniHack/NLE; not verified locally.

### References

[Actual local registrations][registration]; [wrapper][mh-wrapper];
[Mapped and unmapped provider constructors][mh-maze]; [rewards][mh-base].

## MiniHack Custom Maze

### Identity and availability

Registered on `main`; requires optional MiniHack/NLE. Case matters: the IDs use
lowercase `maze`.

| Repository ID | Actual provider                                       | Observation         | Outer cap / provider cap |
|---------------|-------------------------------------------------------|---------------------|--------------------------|
| `MH-maze-v1`  | `MiniHack-Navigation-Custom-v0` with local level text | Glyphs, 3 x 3 cells | 180 / 100                |
| `MH-maze-v2`  | `MiniHack-Navigation-Custom-v0` with local level text | Pixels, 9 x 9 cells | 180 / 100                |

### Paper-ready description

**What it tests:** MiniHack Custom Maze tests navigation through the repository's
fixed level using cropped glyph or pixel observations.

**Limitation:** Its fixed premapped layout permits route-specific solutions and
does not independently establish general memory retrieval.

### Task mechanics

The `des_maze_v0` string in the registration file specifies walls, a starting
branch region, a down staircase and a `premapped` flag. The wrapper passes this
description to the custom navigation provider instead of choosing a named room
or MazeWalk generator.

### Interface

Compass actions, crops, reward components and forwarded info follow the
[MiniHack conventions](#minihack-provider-conventions). For the pinned provider,
custom navigation defaults to **100** steps. This can terminate an episode
before the repository's outer **180**-step TimeLimit.

### Configuration

The two IDs select the same level text with different observation encodings.
There is no CLI level-file or episode-limit override for the nested provider;
the 180-step registration is not evidence that 180 actions are actually available.

### Usage

Conditional on the optional dependencies; not executed locally:

```bash
python train.py --model DTQN --env MH-maze-v1 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/mh-custom-example/train --verbose
```

**Rendering:** delegated to MiniHack/NLE; not verified locally.

### References

[Local level and registrations][registration]; [wrapper][mh-wrapper];
[Custom navigation registration and limit][mh-navigation]; [rewards][mh-base].

## Classic POMDP provider conventions

The final two IDs are named in the inherited README and are registered by the
optional [gym-pomdps provider][pomdp-package], not by local environment classes.
That provider requires [rl-parsers][rl-parsers]. Its inspected source is version
1.0.0 on `master`; the repository does not pin a provider revision.

The provider returns a categorical observation and adds one extra initial
observation symbol. Episodic termination follows model `reset` transitions.
Its registrations do not impose a numeric TimeLimit. The current `train.py`
requires a finite horizon and installs a real outer TimeLimit when given
`--max-episode-steps`. The examples explicitly choose **250 steps** as an
execution-check cap, not a paper-recommended horizon or a property of the models.
The saved policy retains that cap for evaluation. A provider's own termination
can end an episode earlier.

These optional-provider commands were parsed but not executed locally. Provider
installation/API compatibility is still required; the selected cap does not
establish that the task has been solved or its success metric validated.

## Hallway

### Identity and availability

ID: `POMDP-hallway-episodic-v0`. Referenced by the README on `main`; requires the
optional classic POMDP provider above, which was not installed locally.

### Paper-ready description

**What it tests:** Hallway tests localization and navigation from aliased
observations in a stochastic partially observable model.

**Limitation:** Its localization and control demands do not specify a single
fixed instruction-to-decision distance.

### Task mechanics

The model samples an initial hidden pose in a corridor world. Observation and
transition uncertainty require the agent to use its interaction history when
navigating toward rewarding goal states.

### Interface

The inspected model has 60 hidden states, five numbered actions (`0` through `4`)
and 21 model observation symbols. The provider exposes `Discrete(22)` observations
after adding its initial symbol; these are category IDs, not coordinates. The
model rewards transitions into its four goal states with +1, otherwise zero by
default. Episodic reset transitions end the episode. Provider info contains
`reward_cat`, not a newly supplied `is_success` flag; reported success uses the
runner's return fallback.

### Configuration

There is no native numeric horizon in the inspected provider registration.
Use `train.py --max-episode-steps N` to install a bounded rollout; N=250 below is
only an example cap. The upstream README recommends `--in-embed 64`.
Model-specific dynamics are defined in the `.pomdp` file, not through T-Maze flags.

### Usage

After installing the optional provider and parser; not executed locally:

```bash
python train.py --model DTQN --env POMDP-hallway-episodic-v0 \
  --max-episode-steps 250 --in-embed 64 --context 50 --history 50 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/hallway-example/train --verbose
```

**Rendering:** the inspected provider does not register a Hallway renderer.

### References

[Hallway model][hallway-model]; [provider API][pomdp-api];
[provider registration][pomdp-registration]; [renderer selection][pomdp-renderers];
[Esslinger et al. (2022), Section 5.1 and Appendix B.1][dtqn2022].

## HeavenHell

### Identity and availability

ID: `POMDP-heavenhell_3-episodic-v0`. Referenced by the README on `main`; requires
the optional classic POMDP provider and parser, not installed locally.

### Paper-ready description

**What it tests:** HeavenHell tests gathering an informative priest observation
and using that information to choose the rewarding destination.

**Limitation:** Information gathering, navigation and recall are coupled, so
success does not isolate the effect of external retrieval.

### Task mechanics

The agent navigates to an informative priest region to disambiguate which side
contains heaven, then travels to the corresponding endpoint. The model encodes
alternative hidden goal assignments and observation/transition probabilities.

### Interface

The inspected `_3` model has 28 hidden states, four actions in model order:
North `0`, South `1`, East `2`, West `3`, and 15 model observations. The provider
adds an initial symbol, yielding `Discrete(16)` observations. Heaven yields a
+1 reward component and hell -1; other rewards default to zero. Episodic reset
transitions terminate. The provider forwards `reward_cat` info and the runner
uses its reported-success rule.

### Configuration

`_3` identifies a provider model file, not a corridor-length CLI setting. No
numeric horizon is supplied by the inspected provider registration. Set
`train.py --max-episode-steps N` to install a real outer TimeLimit and bound replay
allocation; the example uses N=250 solely as an execution-check cap. The upstream
README recommends `--in-embed 64`.

### Usage

Conditional on the optional provider and parser; not executed locally:

```bash
python train.py --model DTQN --env POMDP-heavenhell_3-episodic-v0 \
  --max-episode-steps 250 --in-embed 64 --context 50 --history 50 \
  --prefill-steps 128 --num-steps 2000 --batch 4 --training-seed 1 \
  --device cpu --output-dir results/heavenhell-example/train --verbose
```

**Rendering:** the provider supplies a matplotlib-backed renderer for
`heavenhell_3` (`human` and `rgb_image` modes); it was not verified locally.

### References

[HeavenHell model][heavenhell-model]; [provider API][pomdp-api];
[provider registration][pomdp-registration]; [renderer selection][pomdp-renderers];
[Esslinger et al. (2022), Appendix B.1][dtqn2022].

## Documentation validation

The 2026-11-23 refresh checked all **35 choices**, all **14 seven-field family
templates**, **36 documented command invocations** across this guide, the README
and both kNN guides, and **74 local links/anchors**. Commands were parsed with
the actual entrypoint parsers; the example JSON grid dry-run expanded to 15 valid
train/eval pairs. Optional providers were not instantiated.

Separate CPU CLI runs on Passive T-Maze, Noisy T-Maze, Associative Corridor,
Car Flag and Memory Cards each completed **16 training interactions, 16 gradient
updates and two final-policy evaluation episodes**. These checks retained the
guide's environment/context settings, reduced the batch to two, and requested
prefill of at least three native horizons so completed replay episodes were
available. Model saving/loading and the listed CSV, metadata and trajectory files
were checked, including one recorded action per training interaction. These are
execution checks, not convergence or performance-preservation results.

## Acceptance criteria and paper handoff

This guide is the linked documentation deliverable for [issue #2][issue2].
Coverage extends beyond its original four tasks to the complete configured
inventory and the new Associative Corridor variant.

| Criterion                                                               | Documentation treatment                                                                                                           |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| AC1: each environment's purpose and limitation                          | Every family has the same two explicitly labeled sentences; all configured variants are enumerated under their applicable family. |
| AC2: Car Flag and Memory Cards only check for no loss within the window | Both entries and the README explicitly state this role for the selected K = 50 configuration.                                     |
| AC3: reuse in the experimental setup section (#15)                      | **Deferred at the user's request.** Paper-ready wording is provided, but no Overleaf or paper setup section has been changed.     |

For later paper reuse, copy each applicable **What it tests** and **Limitation**
sentence together and retain its reference. The implementation details and usage
blocks are supporting reproducibility material, not claims of experimental
results. Listing an inherited or newly implemented task here does not imply
that it was included in the project's final experiment grid.

### Literature

- Ni, T., Ma, M., Eysenbach, B., and Bacon, P.-L. (2023). *When Do Transformers
  Shine in RL? Decoupling Memory from Credit Assignment.* NeurIPS.
  [Paper][ni2023].
- Esslinger, K., Platt, R., and Amato, C. (2022). *Deep Transformer Q-Networks for
  Partially Observable Reinforcement Learning.* Foundation Models for Decision
  Making Workshop at NeurIPS. [Paper][dtqn2022].

[issue2]: https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/2
[issue20]: https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/20
[ni2023]: https://arxiv.org/html/2307.03864v4#S3.SS2
[dtqn2022]: https://arxiv.org/pdf/2206.01078
[memory-rl-code]: https://github.com/twni2016/Memory-RL/blob/main/envs/tmaze.py
[memory-rl-config]: https://github.com/twni2016/Memory-RL/blob/main/configs/envs/tmaze_passive.py
[passive-code]: https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/blob/ed83da1d6df52031c305b82a39560f8a8d5d1404/envs/passive_tmaze.py
[noisy-code]: https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/blob/ed83da1d6df52031c305b82a39560f8a8d5d1404/envs/noisy_tmaze.py
[noisy-validation]: https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/blob/ed83da1d6df52031c305b82a39560f8a8d5d1404/docs/noisy-tmaze-validation.md
[assoc-code]: https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/blob/ed83da1d6df52031c305b82a39560f8a8d5d1404/envs/associative_corridor.py
[assoc-validation]: https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/blob/ed83da1d6df52031c305b82a39560f8a8d5d1404/docs/associative-corridor-validation.md
[registration]: ../envs/__init__.py
[factory]: ../utils/env_processing.py
[car-code]: ../envs/car_flag.py
[cards-code]: ../envs/memory_cards.py
[gv-configs]: ../envs/gridverse/
[gv-wrapper]: ../envs/gv_wrapper.py
[gv-package]: https://github.com/abaisero/gym-gridverse
[mh-wrapper]: ../envs/mini_hack.py
[mh-navigation]: https://github.com/facebookresearch/minihack/blob/v0.1.3/minihack/navigation.py
[mh-base]: https://github.com/facebookresearch/minihack/blob/v0.1.3/minihack/base.py
[mh-room]: https://github.com/facebookresearch/minihack/blob/v0.1.3/minihack/envs/room.py
[mh-maze]: https://github.com/facebookresearch/minihack/blob/v0.1.3/minihack/envs/mazewalk.py
[pomdp-package]: https://github.com/abaisero/gym-pomdps
[rl-parsers]: https://github.com/abaisero/rl-parsers
[pomdp-api]: https://github.com/abaisero/gym-pomdps/blob/master/gym_pomdps/envs/pomdp.py
[pomdp-registration]: https://github.com/abaisero/gym-pomdps/blob/master/gym_pomdps/__init__.py
[pomdp-renderers]: https://github.com/abaisero/gym-pomdps/blob/master/gym_pomdps/rendering/__init__.py
[hallway-model]: https://github.com/abaisero/gym-pomdps/blob/master/gym_pomdps/pomdps/hallway.pomdp
[heavenhell-model]: https://github.com/abaisero/gym-pomdps/blob/master/gym_pomdps/pomdps/heavenhell_3.pomdp
