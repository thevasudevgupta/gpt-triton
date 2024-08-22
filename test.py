# TRITON_INTERPRET=1 pytest -sv test.py

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from kernels import (flash_attention_v1, fused_embeddings, fused_ffn,
                     fused_layer_norm)


def _get_inputs(M, K, N, device):
    torch.manual_seed(1337)
    x = torch.rand((M, K), device=device)
    w = torch.rand((K, N), device=device)
    b = torch.rand((N,), device=device)
    r = torch.rand_like(x)
    if K != N:
        r = r_torch = None
    return x, w, b, r


def test_fused_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.tensor(
        [
            [7, 10, 2, 0, 5, 3, 7, 10, 2, 0, 5, 3, 7, 10, 2, 0, 5, 3],
            [14, 3, 1, 10, 2, 0, 14, 3, 1, 10, 14, 3, 1, 10, 14, 3, 1, 10],
        ],
        dtype=torch.long,
        device=device,
    )
    wte = torch.rand((16, 8), device=device)
    wpe = torch.rand((20, 8), device=device)

    z_torch = wte[x] + wpe[torch.arange(x.shape[1], device=device)][None]
    z = fused_embeddings(x, wte, wpe)

    assert torch.allclose(z, z_torch, atol=1e-5), (z - z_torch).abs().max()


def test_fused_layer_norm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = 249
    K = 123
    N = 123
    x, *_ = _get_inputs(M, K, N, device)
    x_torch, *_ = _get_inputs(M, K, N, device)

    layer_norm = nn.LayerNorm(K).to(device)
    x_torch = layer_norm(x_torch)
    x = fused_layer_norm(x, layer_norm.weight.data, layer_norm.bias.data)

    assert torch.allclose(x, x_torch, atol=1e-5), (x - x_torch).abs().max()


def test_fused_ffn():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    M = 199
    K = 129
    N = 129
    x_torch, w_torch, b_torch, r_torch = _get_inputs(M, K, N, device)
    x, w, b, r = _get_inputs(M, K, N, device)

    z_torch = x_torch @ w_torch
    z_torch = ACT2FN["gelu_new"](z_torch + b_torch)
    if r_torch is not None:
        z_torch += r_torch

    z = fused_ffn(x, w, bias=b, residual=r, add_gelu=True)

    # TODO: how can we do better precision?
    assert torch.allclose(z, z_torch, atol=9e-2), (z - z_torch).abs().max()
