# TRITON_INTERPRET=1 pytest -sv test.py::test_flash_attention_v1

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from gpt import FusedGPT, convert_hf_and_load_model
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


def _get_attn_inputs(B, N, L, H, device):
    torch.manual_seed(1337)
    q = torch.rand((B, N, L, H), device=device)
    k = torch.rand_like(q)
    v = torch.rand_like(q)
    return q, k, v


def torch_attention(q, k, v):
    import math

    assert q.shape == k.shape == v.shape
    B, N, L, H = q.shape
    q, k, v = map(lambda x: x.view(B * N, L, H), (q, k, v))
    z = (q @ k.transpose(1, 2)) / math.sqrt(H)
    attn_mask = torch.tril(torch.ones((L, L), dtype=torch.bool))
    z = torch.where(attn_mask, z, float("-inf"))
    z = z.softmax(-1) @ v
    return z.view(B, N, L, H)


def test_flash_attention_v1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 3
    N = 9
    L = 199
    H = 256
    # TODO: why do we get more error with H == 128?
    # other heads gives good results
    q, k, v = _get_attn_inputs(B, N, L, H, device)
    z_torch = torch_attention(q, k, v)
    z = flash_attention_v1(q, k, v)
    print((z - z_torch).abs().max())
    print(z - z_torch)
    assert torch.allclose(z, z_torch, atol=1e-4), (z - z_torch).abs().max()


def test_gpt2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"
    model, hf_model = convert_hf_and_load_model(model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    with torch.no_grad():
        string = "I am vasudev gupta. I like AI."
        inputs = tokenizer(string, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hf_out = hf_model(**inputs).last_hidden_state
        out = model(inputs["input_ids"])
        print((out - hf_out).abs().max())
        print((out - hf_out).abs())
