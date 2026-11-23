# DTQN-kNN-cached: collection-time external memory

`DTQN-kNN-cached` has the same architecture as [DTQN-kNN](knn-integration.md):
one existing transformer block uses kNN attention, with the same projections,
retrieval and learned per-head gate. The difference is how replay gets its
external K/V. This implements option B from the
[population design](memory-population.md), as a separate experimental variant.

## Run

With the existing virtual environment activated, from the repository root:

```bash
python train.py --model DTQN-kNN-cached --env PassiveTMaze-v0 \
  --context 14 --history 14 --corridor-length 15 \
  --memory-size 32 --memory-top-k 4 \
  --prefill-steps 128 --num-steps 32000 --training-seed 1 \
  --device cpu --output-dir results/knn-cached-train --verbose

python eval.py --model-path results/knn-cached-train/model.pt \
  --eval-episodes 100 --eval-seed 10000 --device cpu \
  --output-dir results/knn-cached-eval --verbose
```

Use fresh output directories. Use `--device cuda` for a configured CUDA installation.
To compare models with the outer script, use this model list in your JSON grid:

```json
"model": ["DTQN", "DTQN-kNN", "DTQN-kNN-cached"]
```

All existing trajectory, episode, loss, timing and GPU-memory records remain
available. The cached variant takes the same memory settings as DTQN-kNN,
requires `bag_size=0` and `memory_stop_grad=true`, and rejects the differentiable
memory option. With `memory_enabled=false`, it uses the exact ordinary DTQN
network and agent, without allocating replay K/V banks.

## Collection and replay

For each observation, save the newest token's K/V the first time that token is
processed. The online representation comes from its normal acting forward.
The target representation comes from a separate `no_grad` forward through the
target's preceding layers and selected projections. Both captures happen before
the subsequent optimizer update. Exploration and random prefill also capture
K/V. Repeated action requests may recompute local attention, but never replace
the first saved K/V or its pending entry.

Two detached float32 K/V banks live on CPU, alongside replay episodes. Every
collected decision timestep has an entry, including timesteps that have not yet
left local context: future replay windows can start anywhere in that episode.
The terminal successor observation does not need a cached entry because it
cannot precede a sampled training window. Padding has no valid K/V.

For each sampled window, gather a prefix of at most `memory_size` earlier
entries and transfer it to the network's device. Populate the existing batched
`KNNMemory` from those tensors; do not encode the prefix again.

| Forward | Bank | Eligible prefix, exclusive end |
|---|---|---|
| Online current window starting at `s` | Online | `[max(0, s-M), s)` |
| Online next window starting at `s+1` | Online | `[max(0, s+1-M), s+1)` |
| Target next window starting at `s+1` | Target | `[max(0, s+1-M), s+1)` |

The online network still chooses the bootstrap action, and the target network
evaluates that action. Memory cannot contain another episode or current/future
window entries. Missing eligible cache entries are errors, never a reason to
recompute them silently. Reusing an episode row clears all its banks and masks.

During interaction, the newest online K/V remains pending while local and moves
to external memory on context eviction. Train/eval episode stores are separate
and reset at episode boundaries. Evaluation uses only the loaded online policy;
it neither captures target K/V nor writes evaluation data into training replay.

## Staleness and cost

Saved K/V never refresh after online optimizer steps or target synchronization.
Each bank therefore contains representations produced by historical versions of
its respective network. Even immediately after target synchronization, older
target K/V may not match the copied network's current embedding space. This is
an intentional experimental tradeoff, not equivalent to fresh-prefix training.

| Property                       | DTQN-kNN                             | DTQN-kNN-cached                                     |
|--------------------------------|--------------------------------------|-----------------------------------------------------|
| External K/V during replay     | Re-encoded with each current network | Gathered from collection-time banks                 |
| Prefix encoder work per update | Three reconstruction calls           | None                                                |
| Target collection work         | No separate target capture           | One prefix/projection forward per decision timestep |
| Extra CPU storage for K/V      | No replay banks                      | Four float32 arrays: online K/V and target K/V      |
| Gradient through external K/V  | Optional differentiable ablation     | Always detached                                     |
| Local context forward/backward | Normal DTQN computation              | Normal DTQN computation                             |

Local attention still recomputes the current context on every forward; this is
not incremental local KV caching. Exact kNN search, tensor transfers, temporary
memory construction and the three ordinary Double DQN forwards still cost time.

Approximate K/V storage is `16 * padded_replay_timesteps * in_embed` bytes.
At 500,000 timesteps and width 128 this is about 1.0 GB (0.95 GiB), plus the
per-episode extra slot and validity mask. `memory_size` controls gathered memory,
not total cached replay storage. Reducing `buf_size` reduces the stored cache.

`model.pt` remains an inference bundle: it stores weights and configuration,
not replay. Evaluation builds fresh episode memory with the loaded weights.
The older full-training checkpoint API additionally writes `buffer_kv.npz`;
resuming this variant requires its compatible, complete banks. Original
DTQN/DTQN-kNN checkpoints and inference bundles remain readable as before.

## Validation

```bash
python -m unittest tests.test_knn_cached tests.test_knn_integration -v
python scripts/benchmark_knn.py --device cpu \
  --output results/knn-cached-update-cost.json
```

The benchmark uses identical synthetic trajectories for all three models,
captures caches before timing, and reports collection time separately from
gradient-update time. It also records allocated replay-cache bytes, configuration
and source identity. It measures execution cost, not learning quality; compare
saved final evaluations across training seeds to assess representation staleness.

CPU validation on 2026-11-23 used Python 3.10.21, torch 2.7.1+cu128 and one
CPU thread. All 85 tests passed, as did strict mypy, ruff and pre-commit checks.
Runtime execution on torch 1.11 was unavailable locally because its shared
library could not load under the host's executable-stack restrictions.

For the synthetic update benchmark (L=15, K=4, M=16, top-k=4, width=32,
two blocks, four heads, batch=8), after 10 warm-up and 300 measured updates:

| Model           | Time per update | Collection of 192 synthetic transitions | Allocated replay K/V and mask |
|-----------------|-----------------|-----------------------------------------|-------------------------------|
| DTQN            | 3.58 ms         | 0.005 s                                 | 0 bytes                       |
| DTQN-kNN        | 9.96 ms         | 0.150 s                                 | 0 bytes                       |
| DTQN-kNN-cached | 5.99 ms         | 0.205 s                                 | 1,630,827 bytes               |

Cached updates took about 40% less time than fresh-prefix updates in this
CPU check; collection was more expensive. This is not a GPU speedup claim.
The raw record is in
`results/knn-cached-validation/20261123T034434Z-5dc8d023/cpu-update-cost.json`.

The same directory holds a six-job JSON sweep: all three models on Passive and
Noisy T-Maze, L=8, K=6, M=16, width=32, batch=4, training seed 1, 512 training
steps and 20 evaluation episodes. Every job completed train/save/load/evaluate
and aggregation. Every trace contained 512 training, 135 prefill and 180
evaluation steps. Training wall times were 5.39/5.43 s for fresh kNN and
4.17/4.33 s for cached kNN on Passive/Noisy respectively. Evaluation success
was 9/20 for every case except cached Noisy T-Maze, which was 0/20. These short
smoke budgets verify execution, not learning equivalence or convergence.
