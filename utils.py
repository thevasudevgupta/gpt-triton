import math

import torch
import triton
import triton.language as tl


# tl.math.tanh doesn't exist in CPU version of triton
@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu_new(x):
    pi = math.pi
    a = tl.math.sqrt(2.0 / pi)
    b = x + 0.044715 * x * x * x
    return 0.5 * x * (1.0 + tanh(a * b))


@triton.jit
def dropout(x, p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random > p, x / (1 - p), 0.0)


def get_inputs(M, K, N, device):
    torch.manual_seed(1337)
    x = torch.rand((M, K), device=device)
    w = torch.rand((K, N), device=device)
    b = torch.rand((N,), device=device)
    return x, w, b
