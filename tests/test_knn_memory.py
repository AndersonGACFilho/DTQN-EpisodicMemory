import unittest
from typing import Optional, Tuple

import torch
import torch.nn as nn

from dtqn.networks.knn_memory import KNNAttention, KNNMemory, MemoryConfig

HEADS = 2
HEAD_DIM = 4
EMBED = HEADS * HEAD_DIM


def causal_mask(seq_len: int) -> torch.Tensor:
    # Same mask as TransformerLayer
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask[mask.bool()] = -float("inf")
    return mask


def random_kv(batch: int, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch, steps, HEADS, HEAD_DIM),
        torch.randn(batch, steps, HEADS, HEAD_DIM),
    )


class TestKNNMemory(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_query_equal_to_stored_key_is_top1(self) -> None:
        memory = KNNMemory(size=16, num_heads=HEADS, head_dim=HEAD_DIM)
        keys, values = random_kv(1, 10)
        memory.add(keys, values)

        for t in range(10):
            found_keys, found_values, mask = memory.search(keys[:, t : t + 1], k=3)
            self.assertTrue(mask[..., 0].all())
            self.assertTrue(torch.equal(found_keys[0, 0, :, 0], keys[0, t]))
            self.assertTrue(torch.equal(found_values[0, 0, :, 0], values[0, t]))

    def test_full_memory_evicts_oldest(self) -> None:
        size = 8
        memory = KNNMemory(size=size, num_heads=HEADS, head_dim=HEAD_DIM)
        keys, values = random_kv(1, size + 3)
        # One step at a time, as when acting
        for t in range(size + 3):
            memory.add(keys[:, t : t + 1], values[:, t : t + 1])

        self.assertEqual(memory.count.item(), size)
        self.assertEqual(memory.keys.shape[1], size)
        stored = memory.keys[0].reshape(size, -1)
        for t in range(size + 3):
            present = (stored == keys[0, t].reshape(1, -1)).all(-1).any().item()
            self.assertEqual(present, t >= 3, f"step {t}")

    def test_batched_insert_matches_stepwise_insert(self) -> None:
        size = 8
        first = random_kv(1, 3)
        keys, values = random_kv(1, size + 5)
        batched = KNNMemory(size=size, num_heads=HEADS, head_dim=HEAD_DIM)
        batched.add(*first)
        batched.add(keys, values)
        stepwise = KNNMemory(size=size, num_heads=HEADS, head_dim=HEAD_DIM)
        stepwise.add(*first)
        for t in range(size + 5):
            stepwise.add(keys[:, t : t + 1], values[:, t : t + 1])

        self.assertTrue(torch.equal(batched.count, stepwise.count))
        self.assertTrue(torch.equal(batched.ptr, stepwise.ptr))
        self.assertTrue(torch.equal(batched.keys, stepwise.keys))

    def test_padded_prefixes_of_different_lengths(self) -> None:
        size = 4
        memory = KNNMemory(size=size, num_heads=HEADS, head_dim=HEAD_DIM)
        memory.reset(batch_size=3)
        keys, values = random_kv(3, 6)
        lengths = torch.tensor([0, 2, 6])
        valid = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
        memory.add(keys, values, valid)

        self.assertEqual(memory.count.tolist(), [0, 2, 4])
        # Element 1 keeps its 2 steps, element 2 keeps its last 4
        self.assertTrue(torch.equal(memory.keys[1, :2], keys[1, :2]))
        stored = memory.keys[2].reshape(size, -1)
        for t in range(6):
            present = (stored == keys[2, t].reshape(1, -1)).all(-1).any().item()
            self.assertEqual(present, t >= 2, f"step {t}")

        _, _, mask = memory.search(torch.randn(3, 1, HEADS, HEAD_DIM), k=3)
        self.assertEqual(mask[:, 0, 0].sum(-1).tolist(), [0, 2, 3])

    def test_reset_empties_memory(self) -> None:
        memory = KNNMemory(size=8, num_heads=HEADS, head_dim=HEAD_DIM)
        memory.add(*random_kv(1, 5))
        self.assertFalse(memory.is_empty)

        memory.reset()
        self.assertTrue(memory.is_empty)
        _, _, mask = memory.search(torch.randn(1, 1, HEADS, HEAD_DIM), k=3)
        self.assertFalse(mask.any())


class TestKNNAttention(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, EMBED)
        self.mask = causal_mask(5)

    def reference_attention(self, layer: KNNAttention) -> torch.Tensor:
        mha = nn.MultiheadAttention(EMBED, HEADS, batch_first=True)
        mha.load_state_dict(layer.attn.state_dict())
        out: torch.Tensor = mha(self.x, self.x, self.x, attn_mask=self.mask)[0]
        return out

    def full_memory(self) -> KNNMemory:
        memory = KNNMemory(size=8, num_heads=HEADS, head_dim=HEAD_DIM)
        memory.reset(batch_size=2)
        memory.add(*random_kv(2, 8))
        return memory

    def test_gate_disabled_returns_local_attention(self) -> None:
        layer = KNNAttention(EMBED, HEADS, config=MemoryConfig(gate=False))
        out, _ = layer(self.x, self.mask, self.full_memory())
        self.assertTrue(torch.equal(out, self.reference_attention(layer)))

    def test_empty_memory_returns_local_attention(self) -> None:
        layer = KNNAttention(EMBED, HEADS)
        memory = KNNMemory(size=8, num_heads=HEADS, head_dim=HEAD_DIM)
        memory.reset(batch_size=2)
        reference = self.reference_attention(layer)

        out, _ = layer(self.x, self.mask, memory)
        self.assertTrue(torch.equal(out, reference))
        out, _ = layer(self.x, self.mask, None)
        self.assertTrue(torch.equal(out, reference))

    def test_partly_empty_batch_keeps_local_attention_where_empty(self) -> None:
        layer = KNNAttention(EMBED, HEADS)
        memory = KNNMemory(size=8, num_heads=HEADS, head_dim=HEAD_DIM)
        memory.reset(batch_size=2)
        valid = torch.tensor([[False] * 4, [True] * 4])
        memory.add(*random_kv(2, 4), valid)

        out, _ = layer(self.x, self.mask, memory)
        reference = self.reference_attention(layer)
        self.assertTrue(torch.allclose(out[0], reference[0], atol=1e-6))
        self.assertFalse(torch.allclose(out[1], reference[1], atol=1e-6))

    def test_gate_enabled_reads_memory_and_trains_gate(self) -> None:
        layer = KNNAttention(EMBED, HEADS)
        out, (k, v) = layer(self.x, self.mask, self.full_memory())

        self.assertEqual(out.shape, self.x.shape)
        self.assertEqual(k.shape, (2, 5, HEADS, HEAD_DIM))
        self.assertEqual(v.shape, (2, 5, HEADS, HEAD_DIM))
        self.assertFalse(torch.allclose(out, self.reference_attention(layer)))
        out.sum().backward()
        self.assertIsNotNone(layer.gate_bias.grad)
        self.assertTrue(layer.gate_bias.grad.abs().sum() > 0)

    def test_returned_keys_match_attention_projections(self) -> None:
        layer = KNNAttention(EMBED, HEADS)
        _, (k, v) = layer(self.x, self.mask)
        w_k, w_v = layer.attn.in_proj_weight.chunk(3)[1:]
        b_k, b_v = layer.attn.in_proj_bias.chunk(3)[1:]
        self.assertTrue(
            torch.allclose(k.reshape(2, 5, EMBED), self.x @ w_k.T + b_k, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(v.reshape(2, 5, EMBED), self.x @ w_v.T + b_v, atol=1e-6)
        )

    def run_with_stored_source(self, stop_grad: bool) -> Optional[torch.Tensor]:
        layer = KNNAttention(EMBED, HEADS, config=MemoryConfig(stop_grad=stop_grad))
        memory = KNNMemory(
            size=8, num_heads=HEADS, head_dim=HEAD_DIM, stop_grad=stop_grad
        )
        memory.reset(batch_size=2)
        source = torch.randn(2, 8, HEADS, HEAD_DIM, requires_grad=True)
        memory.add(source.mul(2.0), source.mul(3.0))
        out, _ = layer(self.x, self.mask, memory)
        out.sum().backward()
        grad: Optional[torch.Tensor] = source.grad
        return grad

    def test_stop_grad_blocks_gradient_into_memory(self) -> None:
        self.assertIsNone(self.run_with_stored_source(stop_grad=True))

    def test_differentiable_memory_passes_gradient(self) -> None:
        grad = self.run_with_stored_source(stop_grad=False)
        assert grad is not None
        self.assertTrue(grad.abs().sum() > 0)


if __name__ == "__main__":
    unittest.main()
