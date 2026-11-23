"""Episode-local memory while acting and fresh, causal prefixes during replay."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import Module

from dtqn.agents.dqn import TrainMode
from dtqn.agents.dtqn import DtqnAgent
from dtqn.networks.dtqn_knn import DTQNKNN, KV
from dtqn.networks.knn_memory import KNNMemory, MemoryConfig
from utils.random import RNG


@dataclass
class EpisodeMemory:
    external: KNNMemory
    pending: Deque[KV] = field(default_factory=deque)
    last_timestep: int = -1


class KnnDtqnAgent(DtqnAgent):
    # Bound the temporary attention activations used to reconstruct B*M windows.
    PREFIX_CHUNK_SIZE = 256

    def __init__(self, *args: Any, memory_config: MemoryConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory_config = memory_config
        assert isinstance(self.policy_network, DTQNKNN)
        self.episode_memories = {
            mode: EpisodeMemory(self.policy_network.new_memory())
            for mode in (TrainMode.TRAIN, TrainMode.EVAL)
        }
        self.sample_episode_indices = np.empty(0, dtype=np.int64)
        self.sample_starts = np.empty(0, dtype=np.int64)

    @property
    def episode_memory(self) -> EpisodeMemory:
        return self.episode_memories[self.train_mode]

    def context_reset(self, obs: NDArray[Any]) -> None:
        super().context_reset(obs)
        state = self.episode_memory
        state.external.reset()
        state.pending.clear()
        state.last_timestep = -1

    @torch.no_grad()
    def _acting_forward(self) -> torch.Tensor:
        assert isinstance(self.policy_network, DTQNKNN)
        length = min(self.context_len, self.context.timestep + 1)
        obss = torch.as_tensor(
            self.context.obs[:length], dtype=self.obs_tensor_type, device=self.device
        ).unsqueeze(0)
        # Context stores the incoming action beside each new observation;
        # DTQN/replay expect outgoing actions and shift them right internally.
        actions = np.zeros((length, 1), dtype=np.int64)
        if length > 1:
            actions[:-1] = self.context.action[1:length]
        action_tensor = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        state = self.episode_memory
        q_values, (keys, values) = self.policy_network.forward_with_memory(
            obss, action_tensor, state.external
        )
        newest = (
            keys[:, -1:].detach().clone(),
            values[:, -1:].detach().clone(),
        )
        if state.last_timestep == self.context.timestep:
            state.pending[-1] = newest
        else:
            state.pending.append(newest)
            state.last_timestep = self.context.timestep
        return q_values

    @torch.no_grad()
    def get_action(self, epsilon: float = 0.0) -> int:
        # Preserve the baseline exploration RNG sequence, but compute K/V even
        # when exploration chooses the action so the episode cache has no holes.
        exploring = RNG.rng.random() < epsilon
        random_action = int(RNG.rng.integers(self.num_actions)) if exploring else 0
        q_values = self._acting_forward()
        if exploring:
            return random_action
        return int(torch.argmax(q_values[:, -1], dim=-1).item())

    def observe(
        self, obs: NDArray[Any], action: int, reward: float, done: bool
    ) -> None:
        state = self.episode_memory
        # The harness's random prefill (and external callers) can skip get_action.
        if state.last_timestep != self.context.timestep:
            self._acting_forward()
        super().observe(obs, action, reward, done)
        if self.context.timestep >= self.context_len:
            keys, values = state.pending.popleft()
            state.external.add(keys, values)

    def _sample_replay(self) -> Tuple[NDArray[Any], ...]:
        sample = self.replay_buffer.sample_with_metadata(self.batch_size)
        self.sample_episode_indices, self.sample_starts = sample[7:]
        return sample[:7]

    def replay_memory(
        self, network: DTQNKNN, episodes: NDArray[Any], starts: NDArray[Any]
    ) -> KNNMemory:
        """Encode each retained prefix step in the local window ending there.

        No window can read another episode, the sampled context, or future steps.
        The source is raw replay, never embeddings cached by an older policy.
        """
        memory = network.new_memory(len(episodes))
        capacity = self.memory_config.size
        # Group equally sized windows to avoid padding changing token positions
        # or the previous-action convention. Most windows have context_len tokens.
        groups: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for batch, (episode, start) in enumerate(zip(episodes, starts)):
            begin = max(0, int(start) - capacity)
            for slot, step in enumerate(range(begin, int(start))):
                lo = max(0, step - self.context_len + 1)
                groups.setdefault(step - lo + 1, []).append(
                    (batch * capacity + slot, int(episode), lo, step + 1)
                )
        if not groups:
            return memory

        keys_parts: List[torch.Tensor] = []
        values_parts: List[torch.Tensor] = []
        positions: List[int] = []
        differentiable = torch.is_grad_enabled() and not self.memory_config.stop_grad
        with torch.set_grad_enabled(differentiable):
            for windows in groups.values():
                for offset in range(0, len(windows), self.PREFIX_CHUNK_SIZE):
                    chunk = windows[offset : offset + self.PREFIX_CHUNK_SIZE]
                    obss = np.stack(
                        [self.replay_buffer.obss[ep, lo:hi] for _, ep, lo, hi in chunk]
                    )
                    actions = np.stack(
                        [
                            self.replay_buffer.actions[ep, lo:hi]
                            for _, ep, lo, hi in chunk
                        ]
                    )
                    keys, values = network.newest_kv(
                        torch.as_tensor(
                            obss, dtype=self.obs_tensor_type, device=self.device
                        ),
                        torch.as_tensor(actions, dtype=torch.long, device=self.device),
                    )
                    keys_parts.append(keys)
                    values_parts.append(values)
                    positions.extend(row[0] for row in chunk)
            shape = (
                len(episodes) * capacity,
                memory.num_heads,
                memory.head_dim,
            )
            index = torch.tensor(positions, dtype=torch.long, device=self.device)
            keys = (
                keys_parts[0]
                .new_zeros(shape)
                .index_copy(0, index, torch.cat(keys_parts))
            )
            values = (
                values_parts[0]
                .new_zeros(shape)
                .index_copy(0, index, torch.cat(values_parts))
            )
            lengths = torch.as_tensor(np.minimum(starts, capacity), device=self.device)
            valid = (
                torch.arange(capacity, device=self.device)[None, :] < lengths[:, None]
            )
            memory.add(
                keys.reshape(
                    len(episodes), capacity, memory.num_heads, memory.head_dim
                ),
                values.reshape(
                    len(episodes), capacity, memory.num_heads, memory.head_dim
                ),
                valid,
            )
        return memory

    def _training_forward(
        self,
        network: Module,
        obss: torch.Tensor,
        actions: torch.Tensor,
        bag_obss: Optional[torch.Tensor],
        bag_actions: Optional[torch.Tensor],
        next_window: bool = False,
    ) -> torch.Tensor:
        assert isinstance(network, DTQNKNN)
        starts = self.sample_starts + int(next_window)
        memory = self.replay_memory(network, self.sample_episode_indices, starts)
        return cast(torch.Tensor, network(obss, actions, memory=memory))
