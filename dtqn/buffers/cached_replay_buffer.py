"""Replay-aligned, immutable collection-time online and target representations."""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from dtqn.buffers.replay_buffer import ReplayBuffer


class CachedReplayBuffer(ReplayBuffer):
    BANKS = ("online_keys", "online_values", "target_keys", "target_values")

    def __init__(self, source: ReplayBuffer, heads: int, head_dim: int) -> None:
        # Adopt the existing raw arrays without reallocating or changing sampling.
        self.__dict__.update(vars(source))
        shape = (self.max_size, self.max_episode_steps + 1, heads, head_dim)
        self.online_keys = np.zeros(shape, dtype=np.float32)
        self.online_values = np.zeros(shape, dtype=np.float32)
        self.target_keys = np.zeros(shape, dtype=np.float32)
        self.target_values = np.zeros(shape, dtype=np.float32)
        self.valid = np.zeros(shape[:2], dtype=np.bool_)

    @property
    def cache_bytes(self) -> int:
        return int(
            sum(getattr(self, name).nbytes for name in self.BANKS) + self.valid.nbytes
        )

    def cleanse_episode(self, episode_idx: int) -> None:
        super().cleanse_episode(episode_idx)
        for name in self.BANKS:
            getattr(self, name)[episode_idx].fill(0)
        self.valid[episode_idx].fill(False)

    def cache_step(
        self,
        step: int,
        online: Tuple[NDArray[Any], NDArray[Any]],
        target: Tuple[NDArray[Any], NDArray[Any]],
    ) -> None:
        row = self.pos[0] % self.max_size
        if step != self.pos[1]:
            raise ValueError("Cached K/V must align with the current replay timestep")
        if self.valid[row, step]:
            raise ValueError("Collection-time K/V entries must not be overwritten")
        for name, value in zip(self.BANKS, (*online, *target)):
            bank = getattr(self, name)
            if value.shape != bank.shape[2:]:
                raise ValueError("Cached K/V shape does not match the memory layer")
        for name, value in zip(self.BANKS, (*online, *target)):
            getattr(self, name)[row, step] = value
        self.valid[row, step] = True

    def prefix(
        self, episodes: NDArray[Any], starts: NDArray[Any], capacity: int, target: bool
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[np.bool_]]:
        if np.any(starts < 0) or np.any(starts > self.episode_lengths[episodes]):
            raise ValueError("Memory prefix must end within its replay episode")
        lengths = np.minimum(starts, capacity)
        slots = np.arange(capacity)[None, :]
        valid = slots < lengths[:, None]
        indices = np.where(valid, np.maximum(0, starts - capacity)[:, None] + slots, 0)
        rows = episodes[:, None]
        if np.any(valid & ~self.valid[rows, indices]):
            raise ValueError("Missing collection-time K/V in sampled replay prefix")
        bank = "target" if target else "online"
        keys = getattr(self, bank + "_keys")[rows, indices]
        values = getattr(self, bank + "_values")[rows, indices]
        keys[~valid] = 0
        values[~valid] = 0
        return keys, values, valid

    def save_cache(self, path: Path) -> None:
        np.savez(
            path,
            version=np.array(1),
            valid=self.valid,
            **{name: getattr(self, name) for name in self.BANKS},
        )

    def load_cache(self, path: Path) -> None:
        if not path.is_file():
            raise ValueError("Cached-agent checkpoint is missing its replay K/V banks")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {*self.BANKS, "valid", "version"}:
                raise ValueError("Invalid cached replay checkpoint fields")
            if archive["version"].item() != 1:
                raise ValueError("Unsupported cached replay checkpoint version")
            restored = {name: archive[name] for name in (*self.BANKS, "valid")}
        for name, array in restored.items():
            expected = getattr(self, name)
            if array.shape != expected.shape or array.dtype != expected.dtype:
                raise ValueError("Cached replay checkpoint shape or dtype mismatch")
        for row in range(min(self.pos[0], self.max_size)):
            if row != self.pos[0] % self.max_size:
                length = int(self.episode_lengths[row])
                if not restored["valid"][row, :length].all():
                    raise ValueError("Cached replay checkpoint has missing K/V entries")
        for name, array in restored.items():
            setattr(self, name, array)
