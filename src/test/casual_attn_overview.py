# python3
# Author: Scc_hy
# Create Date: 2024-09-11
# Func: Casual attention
# =========================================================================

import torch 
from torch import nn
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import netron


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


cfg = GPT2Config()
print(cfg.add_cross_attention)
# tf = GPT2Model(config=cfg)
blk = GPT2Block(cfg, layer_idx=0)
hidden_states = torch.randn(10, 1024, 768)
torch.onnx.export(blk, hidden_states, 'GPT2Block.onnx')
netron.start('GPT2Block.onnx')

# GPT2Attention
gpt2_att = blk.attn

split_size = gpt2_att.split_size
# [batch len emb] 
hidden_states = torch.randn(10, 1024, 768)
print(gpt2_att.c_attn.weight.shape)
# 1- hidden -> q k v
query, key, value = gpt2_att.c_attn(hidden_states).split(split_size, dim=2)
print(f'{query.shape=}') # [batch len emb] 
# 2- split head: q k v -> multi head q k v
query = gpt2_att._split_heads(query, gpt2_att.num_heads, gpt2_att.head_dim)
key = gpt2_att._split_heads(key, gpt2_att.num_heads, gpt2_att.head_dim)
value = gpt2_att._split_heads(value, gpt2_att.num_heads, gpt2_att.head_dim)
print(f'{query.shape=}') # [batch, n_head, len, head_emb] 

# 3- attention 
#  3.1 A = QK^T
attn_weights = torch.matmul(query, key.transpose(-1, -2)) / torch.full([], value.size(-1) ** 0.5)
# batch=0, head=0
org_b0h0_att = attn_weights[0, 0, ...].detach().cpu().numpy()
#  3.2 mask 
max_positions = 1024
causal_mask = torch.tril(
    torch.ones((max_positions, max_positions), dtype=torch.bool)
).view(1, 1, max_positions, max_positions)
mask_value = torch.finfo(attn_weights.dtype).min
mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
# where mask
attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
# batch=0, head=0
masked_b0h0_att = attn_weights[0, 0, ...].detach().cpu().numpy()
#  3.3 A = softmax(A)
attn_weights = nn.functional.softmax(attn_weights, dim=-1) # [batch, n_head, len, len] 
#  3.4  O = AV
attn_output = torch.matmul(attn_weights, value)            # [batch, n_head, len, head_emb] 
# 4- q k v -> merge head -> attn_out # [batch, len, head_emb*n_head] 
attn_output = gpt2_att._merge_heads(attn_output, gpt2_att.num_heads, gpt2_att.head_dim)


# plot
import numpy as np 
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(org_b0h0_att[:, :, np.newaxis])
axes[0].set_title('org attention')

axes[1].imshow(masked_b0h0_att[:, :, np.newaxis])
axes[1].set_title('masked attention')
plt.show()
 

# 用于在使用混合精度训练时对注意力机制进行优化。当启用这个选项时，它会在计算注意力权重之前对键（K）进行缩放，并在执行注意力权重的点积和softmax操作时将其转换为float32精度，以提高训练的稳定性
# reorder_and_upcast_attn=True,
# scale_attn_by_inverse_layer_idx=True  8460

# MLP 
blk.mlp

import math 

def act(ipt):
    return 0.5 * ipt * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (ipt + 0.044715 * np.power(ipt, 3.0))))
# $0.5x [1+ tanh(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3))]$
a = np.linspace(-4, 4, 100000)
plt.plot(a, act(a))
plt.show()