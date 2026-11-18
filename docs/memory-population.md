# Populating the kNN memory from replay subsequences

Design decision for [#3](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/3) (epic [#37](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/37)). It fixes how the memory gets the (key, value) pairs of the steps before the training window, at acting time and at training time. The module itself is [#5](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/5) and its integration into DTQN is [#7](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/7).

## Summary

| Question | Decision |
|---|---|
| What the memory holds | Every step of the current episode before the start of the local window |
| How it is filled at training time | Keys and values of the prefix are **recomputed with the current network under `no_grad`** at every training step (option A) |
| How it is filled at acting time | A step enters the memory when it is evicted from the context window, with the key and value computed when it was the newest token |
| Where keys come from | Key/value projections of the attention in transformer layer `l` (default: last layer; ablated in [#22](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/22)) |
| Gradient | **Stop-gradient by default**; the differentiable memory is an ablation |
| Replay buffer | No new stored arrays; `sample()` also returns the episode index and window start; `episode_lengths` becomes `int32` |

## Context

DTQN trains on length-K windows sampled from the replay buffer (`ReplayBuffer.sample` in [`dtqn/buffers/replay_buffer.py`](../dtqn/buffers/replay_buffer.py)): it picks an episode, draws a start uniformly in `[0, episode_length - K]` and returns steps `start … start + K - 1`, plus the same window shifted by one for the Double DQN target. The network only ever sees those K steps, so a memory that is not filled explicitly would be empty during training and the agent could never learn to read it.

Two facts about the current buffer shape the options:

- It already stores **whole episodes**, one per row (`obss[max_size, max_episode_steps + 1, ...]`, `actions`, `rewards`, `dones`), so the steps before the window are available without changing what is stored.
- `episode_lengths` is `uint8`, so any episode longer than 255 steps overflows. A Passive T-Maze with a long corridor reaches this, whatever option is chosen.

Memorizing Transformers (Wu et al., 2022) stores non-differentiable keys and values per document and clears the memory between documents. Here **the episode plays the role of the document**: the memory is reset at every episode boundary, and a sampled window reads only steps of its own episode.

## What the memory holds

For a window that starts at step `s`, the memory holds steps `[0, s)` of the same episode (the last `M` of them, `M` = memory size), and the window holds `[s, s + K)`. Local attention and memory therefore cover the whole past with no overlap and no gap. Every position of the window reads the same memory: position `i` sees `[0, s)` through the memory and `[s, s + i]` through causal local attention.

The same rule holds when acting: the context keeps the last K steps, and a step enters the memory when `Context.add_transition` ([`utils/context.py`](../utils/context.py)) evicts it. This is the same hook the persistent-memory bag already uses in `DtqnAgent.observe`. Episodes no longer than K never fill the memory, in training or in acting.

## Where the keys come from

The key and value of step `t` are the key/value projections of the attention in layer `l`, computed in the window that ends at `t` (that is, when `t` was the newest token). At acting time this costs nothing extra: `get_action` already runs that forward pass under `no_grad`, so the agent keeps the newest token's key and value in a FIFO of length K and moves them into the memory on eviction. At training time the same quantity is recomputed for the prefix (below), so the keys the agent learns to read are computed the same way as the ones it reads when acting.

Only layers `1 … l` are needed to produce the keys, so the recomputation stops at layer `l`.

## Options compared

### A. Recompute prefix keys with the current network under `no_grad`

At every training step, for each sampled window, gather the prefix `[max(0, s - M), s)` from the buffer, run the sliding windows that end at each prefix step through layers `1 … l` under `no_grad`, and take their key/value projections.

- **Cost:** about `B · min(s, M) · K` extra tokens through `l` layers per training step, without backward. With the defaults (`B = 32`, `K = 50`) and `M = 100`, that is up to 160k tokens, against 1.6k tokens for one forward of the batch: the extra forward dominates the training step. The model is small (`d_model = 128`, 2 layers), so this is milliseconds on a GPU, but the pilot ([#11](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/11)) must measure it. The target network needs its own pass (see below), so the total is about twice that.
- **Staleness:** none. Keys and queries come from the same parameters at every step.
- **Replay buffer:** nothing new is stored. `sample()` must also return the episode index and the window start (or the padded prefix with a mask, like `sample_with_bag` returns the bag), and `episode_lengths` must become `int32`.
- **Ablations:** changing the key layer or switching to a differentiable memory needs no new data.

### B. Store keys and values at collection time

Write the key and value computed while acting into the buffer, next to each observation, and read them back with the window.

- **Cost:** no extra compute at training time. Storage of `buf-size × 2 × d_model × 4` bytes per key layer, about 0.5 GB for the default 500k-step buffer with `d_model = 128`.
- **Staleness:** a key is as old as its episode, up to the whole buffer (hundreds of thousands of updates with the default buffer). The query is computed by the current network, so query and key spaces drift apart as training goes on. Memorizing Transformers tolerates staleness within a document, which is short-lived; replayed episodes are not.
- **Replay buffer:** new arrays `keys` and `values` of shape `[max_size, max_episode_steps, d_model]` per key layer, filled in `store()` and cleared in `cleanse_episode()`; `episode_lengths` must become `int32`.
- **Ablations:** the differentiable memory is impossible (stored tensors have no graph), and every key-layer setting needs its own data collection.

| | A: recompute under `no_grad` | B: store at collection |
|---|---|---|
| Extra compute per training step | about `B · min(s, M) · K` tokens through `l` layers, ×2 with the target network | none |
| Extra storage | none | about `buf-size × 2 × d_model × 4` B per key layer (≈0.5 GB by default) |
| Staleness of keys | none | up to the age of the oldest episode in the buffer |
| Replay buffer changes | return episode index and start; `episode_lengths` → `int32` | new `keys`/`values` arrays; `episode_lengths` → `int32` |
| Differentiable ablation | possible | impossible |
| Key-layer ablation | free | needs new data per setting |

## Decision

**Option A.** Stale keys are the main threat to the research question: if the stored key of the cue step no longer matches the query, retrieval fails for a reason that has nothing to do with the memory idea, and the retrieval diagnostic ([#30](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/30)) would measure buffer age instead of the method. Option A removes staleness, keeps the buffer format, and is the only option that allows two of the ablations the professor asked for (key layer and stop-gradient vs. differentiable). Its cost is compute, which the pilot measures before the experiment grid is fixed.

**Target network.** The Double DQN target reads the window shifted by one, which starts at `s + 1`, so its memory is `[0, s + 1)`. The online network picks the argmax with keys it computed itself, and the target network evaluates it with keys recomputed by the target network, so that each network's queries match its own key space. Both use the same prefix gather; the online pass can compute `[0, s + 1)` once and mask step `s` for the online window.

**Fallbacks if the pilot shows the cost is too high**, in order:

1. *Per-episode key cache refreshed at every target update.* Store keys in the buffer as in B, but recompute an episode's keys whenever the target network is updated (or lazily, the first time the episode is sampled after an update). Staleness is bounded by `--tuf` steps instead of by the buffer age.
2. *Token-local keys.* Take the key of step `t` from the window in which `t` is evicted, where it sits at position 0 and attends only to itself. Recomputing it needs one length-1 sequence per prefix step (`B · M` tokens, about K times cheaper), and acting and training still match exactly, but the key carries no context.

## Stop-gradient vs. differentiable memory

**Default: stop-gradient.** Keys and values are detached, as in Memorizing Transformers: the loss trains the queries, the gate and the local path, but not the stored keys. This is what makes option A affordable (the prefix pass runs under `no_grad`) and matches the reference method.

**Ablation: differentiable memory** ([#22](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/22)). The prefix pass runs with gradients, so the loss can shape the keys of the cue step. Activation memory grows with `B · M · K`, so the prefix is capped at `M` steps. This ablation, together with the noisy T-Maze, tests whether keys without gradient are a bottleneck.

## What #5 and #7 inherit

- **Module ([#5](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/5)):** a per-episode store that accepts a whole batch of prefixes at once (training) as well as one insertion per step (acting); options memory size `M`, top-k, key layer `l`, gate on/off, stop-gradient on/off (default on).
- **Integration ([#7](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/7)):** `ReplayBuffer.sample()` returns the episode index and window start; `episode_lengths` is `int32`; `DtqnAgent.train()` recomputes prefix keys for the online and target networks; `DtqnAgent.observe()` moves evicted steps into the memory; `context_reset()` resets it. With the memory flag off, none of this runs.

## Method section text

Draft paragraph for the method section ([#8](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/8)):

> DTQN is trained on windows of K consecutive steps sampled from a replay buffer that stores whole episodes. As in Memorizing Transformers, where the memory holds the keys and values of earlier segments of the same document, our memory holds the keys and values of earlier steps of the same episode and is cleared at every episode boundary. For a window starting at step s, the memory contains the last M steps of the interval [0, s), so that the memory and the causal local attention together cover the whole history without overlap. The key and value of step t are the key and value projections of layer l computed in the window that ends at t. When acting, they are produced by the forward pass that selects the action and enter the memory when step t leaves the context window. During training, they are recomputed for the prefix of each sampled window with the current parameters and without gradient, so that keys are never stale with respect to the queries; the target network recomputes its own keys for the shifted window. By default, gradients do not flow into the memory (stop-gradient); a differentiable variant is evaluated as an ablation.
