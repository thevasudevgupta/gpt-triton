# TRITON_INTERPRET=1 python3 test.py

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.activations import ACT2FN

from ffn import ffn_kernel
from layer_norm import fused_layer_norm
from mlp import FusedMLP
from utils import get_inputs

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)
x = torch.rand((249, 121), device=device)
mlp = FusedMLP(x.shape[1]).to(device)
z = mlp(x)
print(z.shape, z)

M = 249
K = 123
N = 123
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 16

add_residual = True
apply_layer_norm = True

x_torch, w_torch, b_torch = get_inputs(M, K, N, device=device)
x, w, b = get_inputs(M, K, N, device=device)

if add_residual:
    assert K == N
    r = x
    r_torch = x_torch
else:
    r = r_torch = None

if apply_layer_norm:
    layer_norm = nn.LayerNorm(K).to(device)
    x_torch = layer_norm(x_torch)
    x = fused_layer_norm(x, layer_norm.weight.data, layer_norm.bias.data)
    print("layer norm diff:", (x - x_torch).abs().max())

z_torch = x_torch.to(torch.float16) @ w_torch.to(torch.float16)
z_torch = ACT2FN["gelu_new"](z_torch.to(torch.float32) + b_torch)

grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
z = torch.empty((M, N), device=device)
ffn_kernel[grid](
    x,
    w,
    z,
    M,
    N,
    K,
    apply_gelu=True,
    dropout_prob=0.0,
    b_ptr=b,
    r_ptr=r,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
)

if add_residual:
    z_torch += r_torch

print("diff:", (z - z_torch).abs().max())
print(z)
print(z_torch)

# implementation is inspired from flash-attn-v1 algo
# TODO: read about flash-2 and see if we can switch to that
# TODO: then read about flash-3 and see if we can switch to that instead

# TODO: can we do score computation for only unmasked positions?
# pytorch flex-attention does something like that - it would make computation 50% efficient
