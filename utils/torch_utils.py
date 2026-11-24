import torch
import torch.nn as nn
import torch.nn.functional as F


def select_action_values(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Same values and gradients as q_values.gather(-1, actions), bit for bit.

    gather's CUDA backward uses scatter_add, which PyTorch's deterministic mode
    rejects. Multiplying by a one-hot mask and summing adds only exact zeros.
    """
    mask = F.one_hot(actions.squeeze(-1), q_values.shape[-1]).to(q_values.dtype)
    return (q_values * mask).sum(dim=-1, keepdim=True)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.in_proj_bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
