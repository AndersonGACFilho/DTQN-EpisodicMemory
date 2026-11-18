import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryConfig:
    """Options of the kNN episodic memory (Memorizing Transformers, Wu et al., 2022).

    Args:
        size:           Number of steps stored per episode. When full, the oldest step is evicted.
        top_k:          Number of neighbours retrieved for each query.
        layer:          Index of the transformer layer whose attention reads the memory. Default: last layer.
        gate:           Whether to mix retrieved and local attention. `False` returns the local attention only.
        gate_bias_init: Initial value of the per-head gate bias b_h, with g_h = sigmoid(b_h).
        stop_grad:      Whether stored keys and values are detached from the graph.
    """

    size: int = 256
    top_k: int = 8
    layer: int = -1
    gate: bool = True
    gate_bias_init: float = 0.0
    stop_grad: bool = True


class KNNMemory:
    """Per-episode store of (key, value) pairs with exact top-k retrieval.

    Holds one FIFO memory per batch element, so a training batch can carry the prefixes of
    different episodes. Keys and values are stored per attention head.

    Args:
        size:       Number of steps stored per batch element.
        num_heads:  Number of attention heads.
        head_dim:   Dimensionality of each head.
        stop_grad:  Whether stored keys and values are detached from the graph. Default: `True`
        device:     Device of the stored tensors.
    """

    def __init__(
        self,
        size: int,
        num_heads: int,
        head_dim: int,
        stop_grad: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.size = size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.stop_grad = stop_grad
        self.device = device
        self.reset(batch_size=1)

    @classmethod
    def from_config(
        cls,
        config: MemoryConfig,
        num_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
    ) -> "KNNMemory":
        return cls(config.size, num_heads, head_dim, config.stop_grad, device)

    def reset(self, batch_size: Optional[int] = None) -> None:
        """Empties the memory. Call at the start of every episode."""
        if batch_size is None:
            batch_size = self.keys.size(0)
        shape = (batch_size, self.size, self.num_heads, self.head_dim)
        self.keys = torch.zeros(shape, device=self.device)
        self.values = torch.zeros(shape, device=self.device)
        # Next slot to write (always the oldest once full) and number of stored steps
        self.ptr = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.count = torch.zeros(batch_size, dtype=torch.long, device=self.device)

    @property
    def batch_size(self) -> int:
        return self.keys.size(0)

    @property
    def is_empty(self) -> bool:
        return bool((self.count == 0).all())

    def add(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> None:
        """Appends steps in time order, evicting the oldest ones when full.

        keys, values are batch x steps x heads x head_dim
        valid        is batch x steps (bool), marks the real steps of padded prefixes
        """
        batch_size, steps = keys.shape[:2]
        assert (
            batch_size == self.batch_size
        ), f"Memory holds {self.batch_size} episodes, got {batch_size}"
        if valid is None:
            valid = torch.ones(
                batch_size, steps, dtype=torch.bool, device=keys.device
            )
        if self.stop_grad:
            keys = keys.detach()
            values = values.detach()

        num_new = valid.sum(1)
        # Same slots as inserting the steps one at a time; only the last `size` valid
        # steps survive, so no slot is written twice
        position = valid.long().cumsum(1) - 1
        keep = valid & (position >= (num_new - self.size).unsqueeze(1))
        slots = (self.ptr.unsqueeze(1) + position) % self.size

        batch_idx, step_idx = keep.nonzero(as_tuple=True)
        slot_idx = slots[batch_idx, step_idx]
        self.keys = self.keys.index_put((batch_idx, slot_idx), keys[batch_idx, step_idx])
        self.values = self.values.index_put(
            (batch_idx, slot_idx), values[batch_idx, step_idx]
        )

        self.ptr = (self.ptr + num_new) % self.size
        self.count = (self.count + num_new).clamp(max=self.size)

    def search(
        self, queries: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact top-k retrieval by cosine similarity, per head.

        queries is batch x seq_len x heads x head_dim
        returns keys, values as batch x seq_len x heads x k x head_dim and
        a mask as batch x seq_len x heads x k that is `False` for empty slots
        """
        k = min(k, self.size)
        # [batch x seq_len x heads x size]
        similarity = torch.einsum(
            "bthd,bmhd->bthm",
            F.normalize(queries, dim=-1),
            F.normalize(self.keys, dim=-1),
        )
        filled = torch.arange(self.size, device=queries.device) < self.count.unsqueeze(1)
        similarity = similarity.masked_fill(
            ~filled[:, None, None, :], -float("inf")
        )
        top_similarity, top_idx = similarity.topk(k, dim=-1)

        batch_size, seq_len = queries.shape[:2]
        batch_idx = torch.arange(batch_size, device=queries.device)[:, None, None, None]
        head_idx = torch.arange(self.num_heads, device=queries.device)[None, None, :, None]
        return (
            self.keys[batch_idx, top_idx, head_idx],
            self.values[batch_idx, top_idx, head_idx],
            torch.isfinite(top_similarity),
        )


class KNNAttention(nn.Module):
    """Causal multi-head attention that also reads the top-k neighbours from a `KNNMemory`.

    The local path is an `nn.MultiheadAttention`, with the same parameters and initialisation as
    the attention in `TransformerLayer`. Retrieved and local attention are mixed per head by a
    learned gate g_h = sigmoid(b_h), as in Memorizing Transformers.

    Args:
        embed_size: The dimensionality of the layer.
        num_heads:  Number of heads to use for MultiHeadAttention.
        dropout:    Dropout percentage on the attention weights.
        config:     Memory options (`top_k`, `gate`, `gate_bias_init`).
    """

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        dropout: float = 0.0,
        config: Optional[MemoryConfig] = None,
    ):
        super().__init__()
        self.config = config if config is not None else MemoryConfig()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_bias = nn.Parameter(
            torch.full((num_heads,), self.config.gate_bias_init)
        )
        # Just storage for attention weights for visualization
        self.alpha = None

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-head query, key and value: batch x seq_len x embed -> batch x seq_len x heads x head_dim"""
        q, k, v = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(
            3, dim=-1
        )
        shape = (*x.shape[:2], self.num_heads, self.head_dim)
        return q.reshape(shape), k.reshape(shape), v.reshape(shape)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        memory: Optional[KNNMemory] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x         is batch x seq_len x embed
        attn_mask is seq_len x seq_len, additive (-inf where attention is not allowed)
        returns the attention output (batch x seq_len x embed) and the per-head keys and values
        of `x` (batch x seq_len x heads x head_dim), to be inserted into the memory by the caller
        """
        q, k, v = self.project(x)

        if not self.config.gate or memory is None or memory.is_empty:
            out, self.alpha = self.attn(
                x, x, x, attn_mask=attn_mask, average_attn_weights=True
            )
            return out, (k, v)

        scale = 1.0 / math.sqrt(self.head_dim)
        dropout = self.attn.dropout if self.training else 0.0

        # Local causal attention: [batch x heads x seq_len x seq_len]
        scores = torch.einsum("bthd,bshd->bhts", q * scale, k) + attn_mask
        weights = F.softmax(scores, dim=-1)
        self.alpha = weights.mean(1)
        local = torch.einsum(
            "bhts,bshd->bthd", F.dropout(weights, dropout, self.training), v
        )

        # Attention over the retrieved neighbours: [batch x seq_len x heads x k]
        mem_keys, mem_values, mem_mask = memory.search(q, self.config.top_k)
        mem_scores = torch.einsum("bthd,bthkd->bthk", q * scale, mem_keys)
        mem_scores = mem_scores.masked_fill(
            ~mem_mask, torch.finfo(mem_scores.dtype).min
        )
        mem_weights = F.dropout(F.softmax(mem_scores, dim=-1), dropout, self.training)
        retrieved = torch.einsum("bthk,bthkd->bthd", mem_weights, mem_values)

        # Queries with nothing to retrieve (empty memory) keep the local output unscaled
        gate = torch.sigmoid(self.gate_bias) * mem_mask.any(-1)
        gate = gate.unsqueeze(-1)
        mixed = gate * retrieved + (1.0 - gate) * local

        out = self.attn.out_proj(mixed.reshape(*x.shape[:2], -1))
        return out, (k, v)
