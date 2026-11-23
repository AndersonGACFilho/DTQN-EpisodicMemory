"""DTQN-kNN with collection-time K/V reused unchanged by replay updates."""

from pathlib import Path
from typing import Any, Tuple, cast

import torch
from numpy.typing import NDArray

from dtqn.agents.dqn import TrainMode
from dtqn.agents.dtqn_knn import KnnDtqnAgent
from dtqn.buffers.cached_replay_buffer import CachedReplayBuffer
from dtqn.networks.dtqn_knn import DTQNKNN
from dtqn.networks.knn_memory import KNNMemory, MemoryConfig
from utils.epsilon_anneal import LinearAnneal
from utils.logging_utils import RunningAverage


class CachedKnnDtqnAgent(KnnDtqnAgent):
    replay_buffer: CachedReplayBuffer

    def __init__(self, *args: Any, memory_config: MemoryConfig, **kwargs: Any) -> None:
        if not memory_config.stop_grad:
            raise ValueError("DTQN-kNN-cached requires memory_stop_grad = true")
        super().__init__(*args, memory_config=memory_config, **kwargs)
        network = cast(DTQNKNN, self.policy_network)
        attention = network.knn_block.attention
        self.replay_buffer = CachedReplayBuffer(
            self.replay_buffer, attention.num_heads, attention.head_dim
        )

    def _remember_newest(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        obss: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        state = self.episode_memory
        if state.last_timestep == self.context.timestep:
            return  # Keep the first representation, even if weights have changed.
        if self.train_mode == TrainMode.TRAIN:
            target = cast(DTQNKNN, self.target_network)
            with torch.no_grad():
                target_keys, target_values = target.newest_kv(obss, actions)
            self.replay_buffer.cache_step(
                self.context.timestep,
                (
                    keys[0, -1].detach().cpu().numpy(),
                    values[0, -1].detach().cpu().numpy(),
                ),
                (
                    target_keys[0].detach().cpu().numpy(),
                    target_values[0].detach().cpu().numpy(),
                ),
            )
        super()._remember_newest(keys, values, obss, actions)

    def replay_memory(
        self, network: DTQNKNN, episodes: NDArray[Any], starts: NDArray[Any]
    ) -> KNNMemory:
        if network is not self.policy_network and network is not self.target_network:
            raise ValueError("Cached memory must use the online or target network")
        keys, values, valid = self.replay_buffer.prefix(
            episodes,
            starts,
            self.memory_config.size,
            target=network is self.target_network,
        )
        memory = network.new_memory(len(episodes))
        memory.add(
            torch.as_tensor(keys, device=self.device),
            torch.as_tensor(values, device=self.device),
            torch.as_tensor(valid, device=self.device),
        )
        return memory

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        wandb_id: str,
        episode_successes: RunningAverage,
        episode_rewards: RunningAverage,
        episode_lengths: RunningAverage,
        eps: LinearAnneal,
    ) -> None:
        super().save_checkpoint(
            checkpoint_dir,
            wandb_id,
            episode_successes,
            episode_rewards,
            episode_lengths,
            eps,
        )
        self.replay_buffer.save_cache(Path(checkpoint_dir + "buffer_kv.npz"))

    def load_checkpoint(
        self, checkpoint_dir: str
    ) -> Tuple[str, RunningAverage, RunningAverage, RunningAverage, float]:
        path = Path(checkpoint_dir + "buffer_kv.npz")
        if not path.is_file():
            raise ValueError("Cached-agent checkpoint is missing its replay K/V banks")
        result = super().load_checkpoint(checkpoint_dir)
        self.replay_buffer.load_cache(path)
        return result
