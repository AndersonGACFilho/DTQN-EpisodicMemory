# DTQN-kNN: training and acting (#7)

Select `DTQN-kNN` to use the existing `KNNAttention` in one DTQN block.
The observation/action embeddings, remaining blocks, residual connections,
normalization and Q head are reused. Ordinary `DTQN` retains its original path.
No additional runtime packages are needed.

For collection-time K/V reused without replay-prefix encoding, select the
separate [DTQN-kNN-cached variant](knn-cached.md). The population rules below
describe the original fresh-prefix `DTQN-kNN` variant.

## Run

From the repo root with the existing venv activated:

```bash
# Two small train/final-eval pairs. This checks execution, not convergence.
python train_and_eval.py --config experiments/knn-smoke.json \
  --project-name knn-smoke --device cpu

# One longer run. Prefill is additional to the training-step budget.
python train.py --model DTQN-kNN --env PassiveTMaze-v0 \
  --context 14 --history 14 --corridor-length 15 \
  --memory-size 64 --memory-top-k 8 \
  --num-steps 32000 --training-seed 1 --device cpu \
  --output-dir results/knn-L15-K14-train --verbose

python eval.py --model-path results/knn-L15-K14-train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/knn-L15-K14-eval --verbose
```

Use new output directories. Replace `cpu` with `cuda` on an appropriately
configured GPU installation. The final bundle includes the memory configuration
and learned gate parameters. Evaluation reconstructs the variant from that
bundle and starts every episode with empty memory. Memory contents are runtime
state, not persistent model weights. Existing action/observation trajectories,
episode statistics, losses, timing and GPU-memory accounting remain available.

## Options

All options below are saved scientific parameters. In a sweep JSON, replace
hyphens with underscores and put values in lists. They affect `DTQN-kNN`; other
models record them as inactive. Boolean false values are passed correctly from
the sweep to its subprocesses.

| Training flag | Default | Effect |
|---|---|---|
| `--memory-enabled` / `--no-memory-enabled` | Enabled for `DTQN-kNN` | Disabled selects the original DTQN network and agent, preserving fixed-seed outputs and initialization |
| `--memory-size` | 256 | Maximum earlier timesteps retained per episode, with one K/V vector per head per timestep |
| `--memory-top-k` | 8 | Neighbours per query/head; capped at storage capacity; unfilled slots are masked |
| `--memory-layer` | -1 | Python-style block index, zero-based; -1 selects the last block |
| `--memory-gate` / `--no-memory-gate` | Enabled | Mix local and retrieved attention; disabled uses local attention while still exercising population |
| `--memory-gate-bias-init` | 0.0 | Initial per-head bias; sigmoid(0) gives equal local/retrieved weights where memory exists |
| `--memory-stop-grad` / `--no-memory-stop-grad` | Enabled | Detach prefix K/V by default; disabled preserves gradients through selected prefix entries during training |

Memory size and top-k must be positive. The selected layer must exist.
`DTQN-kNN` requires `bag_size=0`; the older observation bag is a separate method.
The differentiable option does not make discrete top-k selection differentiable.
The gate is a learned scalar per head, not a query-dependent write decision.

## Population and causality

The implementation follows [the population design](memory-population.md):

- At acting time, the selected layer returns K/V for the current context. Keep
  only the newest timestep's K/V in a FIFO while that timestep remains local.
  On context eviction, move that entry into external memory. Repeated action
  requests at the same timestep replace the pending entry, never duplicate it.
- Compute K/V even for epsilon-random actions and when the random-prefill loop
  calls `observe` without `get_action`. No relevance filter or learned write gate
  discards timesteps. Episode reset clears the local FIFO and external store.
  Training and evaluation episodes have separate runtime stores.
- During replay, sampling returns episode rows and local-window starts. For
  start `s`, reconstruct only `[max(0, s-M), s)` of that episode, where `M` is
  capacity. Each retained step is encoded in the length-at-most-K window ending
  at that step, using the current parameters through the chosen layer's input
  projections. Earlier tokens from that window are needed to reproduce the
  contextual key, even if they are older than the retained external prefix.
- For Double DQN's next-state window, start is `s+1` and memory ends before
  `s+1`. The online network chooses the action with its own recomputed keys;
  the target network evaluates it with its own keys. Neither uses cached keys
  produced by replay collection or by the other network.
- The default prefix pass is under `no_grad`. Differentiable prefixes retain
  their graph only for the current policy-loss pass; bootstrap calls remain
  under `no_grad`. Reconstruction is batched by window length and chunked into
  at most 256 windows to bound temporary activation size.

The memory receives pre-attention K/V projections, not the mixed attention output.
Action embeddings, when enabled, use the previous-action convention from replay;
the new agent converts the context's incoming-action bookkeeping accordingly.
The original agent's action handling is unchanged.

Local attention still processes the whole current window on each forward, as
DTQN already does. The pending K/V FIFO is for later external insertion; it is
not an incremental local-attention cache. A timestep still inside the context
when an episode ends need not enter external memory.

## Comparison with extending context

An extended context recomputes representations and allows attention over all
included tokens in each block. This memory variant keeps ordinary attention at
K tokens and supplies selected older K/V at one block. Storage is capped at M
older steps. For Passive T-Maze, the initial cue is available at the junction
only if it is still local or has not been FIFO-evicted from external memory.
At junction index L, cue retention requires `K+M >= L+1`; this is necessary,
not sufficient for successful retrieval and use.

The tradeoff is selective access rather than full joint processing of the older
context. Retrieval can miss the needed entry, detached cached representations
cannot receive credit from the later read, and large contexts may perform better
when affordable. This implementation uses exact cosine search over all M keys,
not an approximate index. Replay-prefix reconstruction adds substantial work.
No speed, peak-memory or success advantage over a larger context is presumed;
the pilot and experiments must measure those comparisons.

## Validation

```bash
python -m unittest tests.test_knn_memory tests.test_knn_integration -v
python train_and_eval.py --config experiments/knn-short-check.json \
  --project-name knn-short-check --device cpu
python scripts/benchmark_knn.py --device cpu --output results/knn-update-cost.json
```

The benchmark uses identical synthetic completed replay, one CPU thread,
5 warm-up updates and 50 measured updates per model. It includes prefix work
and the optimizer, excludes action collection/prefill, and synchronizes CUDA
before and after timing. It records configuration, versions and source identity.
Its samples are for execution cost, not evidence of learning.

CPU validation on 2026-11-22, Python 3.10 / torch 2.7.1+cu128, one CPU thread:

| Quick short-maze check | DTQN | DTQN-kNN |
|---|---|---|
| L=3, K=8, 500 training steps | 43/100 evaluation successes | 43/100 |
| L=3, K=8, 1500 training steps | 100/100 evaluation successes | 100/100 |
| Synthetic replay: L=15, K=4, M=16, width=32, 2 layers, 4 heads, batch=8 | 3.77 ms/update | 10.21 ms/update |

The short checks use training seed 1, evaluation seed 10000, 128 prefill steps,
width 32, 2 layers, 4 heads, batch 8, history 8, top-k 4 and M=16.
These are independent runs at two budgets, not intermediate evaluations in
`train.py`. They show no regression at those two budgets in the requested quick
check; they are not a multi-seed non-inferiority result or proof of identical
learning curves. Training wall time is higher with memory. The synthetic cost
ratio here is approximately 2.71x and is not a large-context comparison.

On the RTX 5060 Ti with the same synthetic cost-check configuration, CUDA
measured 4.58 ms/update for DTQN and 13.48 ms/update for DTQN-kNN (2.94x).
This tiny model is not a GPU-throughput benchmark. Separate CUDA smoke runs
completed 32 gradient updates plus 3 saved-policy evaluation episodes in each
gradient mode, also exercising identity ordering and action embeddings.
The full local suite has 75 passing tests; strict mypy and pre-commit checks pass.

| Issue #7 criterion | Evidence |
|---|---|
| AC1: memory off preserves DTQN | Fixed-seed parameter, RNG and output equality, including identity ordering and action embeddings |
| AC2: full Passive T-Maze episode | Training plus saved-model evaluation run with L greater than K |
| AC3: replay memory precedes the sampled window | Changing sampled/future observations cannot alter the prefix; batches isolate episode rows; acting and reconstructed keys match through FIFO wrapping |
| AC4: reset every episode | Reset tests plus independent train/eval stores and repeated evaluation episodes |
| AC5: short-maze learning sanity check | Equal success at 500 and 1500 steps in the single-seed quick comparison above |

Additional tests cover random exploration/prefill, repeated forward calls,
shifted DDQN prefixes, target-network isolation, both gradient modes, and saving
and reloading the complete variant. Review and merge remain separate from local
implementation/testing. Cue-step indices and hit-rate reporting belong to #30;
this change does not claim to implement that diagnostic or the research grid.
