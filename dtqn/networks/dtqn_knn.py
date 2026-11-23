"""DTQN with one memory-reading attention block; runtime memory belongs to the agent."""

from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from dtqn.networks.dtqn import DTQN
from dtqn.networks.knn_memory import KNNAttention, KNNMemory, MemoryConfig
from dtqn.networks.transformer import TransformerIdentityLayer, TransformerLayer

KV = Tuple[torch.Tensor, torch.Tensor]


class MemoryTransformerLayer(nn.Module):
    """Reuse a DTQN block's parameters and residual/normalization ordering."""

    def __init__(
        self, original: TransformerLayer, config: MemoryConfig, identity: bool
    ) -> None:
        super().__init__()
        self.identity = identity
        # Constructing the adapter must not change the initial weights or RNG
        # sequence relative to baseline DTQN. Reuse its initialized attention.
        with torch.random.fork_rng(devices=[]):
            self.attention = KNNAttention(
                original.attention.embed_dim,
                original.attention.num_heads,
                original.attention.dropout,
                config,
            )
        self.attention.attn = original.attention
        self.layernorm1 = original.layernorm1
        self.layernorm2 = original.layernorm2
        self.ffn = original.ffn
        self.attn_gate = original.attn_gate
        self.mlp_gate = original.mlp_gate
        self.attn_mask = original.attn_mask
        self.alpha = None

    def project(self, x: torch.Tensor) -> KV:
        if self.identity:
            x = self.layernorm1(x)
        _, keys, values = self.attention.project(x)
        return keys, values

    def forward(
        self, x: torch.Tensor, memory: Optional[KNNMemory] = None
    ) -> Tuple[torch.Tensor, KV]:
        attention_input = self.layernorm1(x) if self.identity else x
        attended, kv = self.attention(
            attention_input, self.attn_mask[: x.size(1), : x.size(1)], memory
        )
        self.alpha = self.attention.alpha
        x = self.attn_gate(x, F.relu(attended))
        if self.identity:
            ffn = self.ffn(self.layernorm2(x))
            x = self.mlp_gate(x, F.relu(ffn))
        else:
            x = self.layernorm1(x)
            x = self.layernorm2(self.mlp_gate(x, F.relu(self.ffn(x))))
        return x, kv


class DTQNKNN(DTQN):
    def __init__(self, *args: Any, memory_config: MemoryConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.bag_size:
            raise ValueError("DTQN-kNN uses episodic memory; bag_size must be zero")
        self.memory_config = memory_config
        count = len(self.transformer_layers)
        index = memory_config.layer
        if not -count <= index < count:
            raise ValueError("memory_layer must select an existing transformer layer")
        self.memory_layer = index % count
        original = self.transformer_layers[self.memory_layer]
        self.transformer_layers[self.memory_layer] = MemoryTransformerLayer(
            original, memory_config, isinstance(original, TransformerIdentityLayer)
        )

    @property
    def knn_block(self) -> MemoryTransformerLayer:
        return cast(MemoryTransformerLayer, self.transformer_layers[self.memory_layer])

    def new_memory(self, batch_size: int = 1) -> KNNMemory:
        attention = self.knn_block.attention
        memory = KNNMemory.from_config(
            self.memory_config,
            attention.num_heads,
            attention.head_dim,
            next(self.parameters()).device,
        )
        memory.reset(batch_size)
        return memory

    def newest_kv(
        self, obss: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> KV:
        """Prefix reconstruction stops before the chosen layer's attention read."""
        x = self.embed_context(obss, actions)
        for index in range(self.memory_layer):
            x = self.transformer_layers[index](x)
        keys, values = self.knn_block.project(x)
        return keys[:, -1], values[:, -1]

    def forward_with_memory(
        self,
        obss: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        memory: Optional[KNNMemory] = None,
    ) -> Tuple[torch.Tensor, KV]:
        x = self.embed_context(obss, actions)
        kv: Optional[KV] = None
        for index, block in enumerate(self.transformer_layers):
            if index == self.memory_layer:
                x, kv = block(x, memory)
            else:
                x = block(x)
        assert kv is not None
        return self.ffn(x), kv

    def forward(
        self,
        obss: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        bag_obss: Optional[torch.Tensor] = None,
        bag_actions: Optional[torch.Tensor] = None,
        memory: Optional[KNNMemory] = None,
    ) -> torch.Tensor:
        return self.forward_with_memory(obss, actions, memory)[0]
