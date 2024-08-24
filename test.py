# TRITON_INTERPRET=1 pytest -sv test.py

import math

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN

from gpt import (FusedGPT, GPTConfig, convert_hf_and_load_model, estimate_days,
                 get_num_parameters)
from kernels import (flash_attention_v1, fused_embeddings, fused_ffn,
                     fused_layer_norm)


def _get_inputs(M, K, N, device):
    torch.manual_seed(1337)
    x = torch.rand((M, K), device=device, dtype=torch.float32)
    w = torch.rand((K, N), device=device, dtype=torch.float32)
    b = torch.rand((N,), device=device, dtype=torch.float32)
    r = torch.rand_like(x, dtype=torch.float32)
    if K != N:
        r = r_torch = None
    return x, w, b, r


@pytest.mark.parametrize("vocab_size", [2, 32])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("hidden_size", [32, 128, 256])
@pytest.mark.parametrize("seqlen, block_size", [(10, 20), (20, 20)])
def test_fused_embeddings(batch_size, seqlen, vocab_size, block_size, hidden_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randint(
        0, vocab_size, size=(batch_size, seqlen), dtype=torch.long, device=device
    )
    wte = torch.rand((vocab_size, hidden_size), device=device)
    wpe = torch.rand((block_size, hidden_size), device=device)

    z_torch = wte[x] + wpe[torch.arange(x.shape[1], device=device)][None]
    z = fused_embeddings(x, wte, wpe)

    assert torch.allclose(z, z_torch, atol=1e-5), (z - z_torch).abs().max()


@pytest.mark.parametrize("M", [249, 32])
@pytest.mark.parametrize("K", [123, 128, 64])
def test_fused_layer_norm(M, K):
    N = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, *_ = _get_inputs(M, K, N, device)
    x_torch, *_ = _get_inputs(M, K, N, device)

    layer_norm = nn.LayerNorm(K).to(device)
    x_torch = layer_norm(x_torch)
    x = fused_layer_norm(x, layer_norm.weight.data, layer_norm.bias.data)

    assert torch.allclose(x, x_torch, atol=1e-5), (x - x_torch).abs().max()


def torch_ffn(x, w, b=None, r=None):
    z = x @ w
    if b is not None:
        z += b
    z = ACT2FN["gelu_new"](z)
    if r is not None:
        z += r
    return z


@pytest.mark.parametrize("M,N,K", [(128, 128, 256), (199, 129, 129), (61, 31, 23)])
@pytest.mark.parametrize("add_gelu", [True, False])
@pytest.mark.parametrize("add_bias", [True, False])
def test_fused_ffn(M, N, K, add_gelu, add_bias):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_torch, w_torch, b_torch, r_torch = _get_inputs(M, K, N, device)
    x, w, b, r = _get_inputs(M, K, N, device)

    if not add_bias:
        b_torch = None
        b = None

    z_torch = torch_ffn(x_torch, w_torch, b=b_torch, r=r_torch)

    z = fused_ffn(x, w, bias=b, residual=r, add_gelu=True)
    assert torch.allclose(z, z_torch, atol=1e-5), (z - z_torch).abs().max()


def _get_attn_inputs(B, N, L, H, device):
    torch.manual_seed(1337)
    q = torch.rand((B, N, L, H), device=device)
    k = torch.rand_like(q)
    v = torch.rand_like(q)
    return q, k, v


def torch_attention(q, k, v):
    assert q.shape == k.shape == v.shape
    B, N, L, H = q.shape
    q, k, v = map(lambda x: x.view(B * N, L, H), (q, k, v))
    z = (q @ k.transpose(1, 2)) / math.sqrt(H)
    attn_mask = torch.tril(torch.ones((L, L), dtype=torch.bool))
    z = torch.where(attn_mask, z, float("-inf"))
    z = z.softmax(-1) @ v
    return z.view(B, N, L, H)


@pytest.mark.parametrize("B,N", [(3, 9), (2, 7)])
@pytest.mark.parametrize("L", [199, 128, 63])
@pytest.mark.parametrize("H", [64, 128, 256])
def test_flash_attention_v1(B, N, L, H):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = _get_attn_inputs(B, N, L, H, device)
    z_torch = torch_attention(q, k, v)
    z = flash_attention_v1(q, k, v)
    assert torch.allclose(z, z_torch, atol=1e-5), (z - z_torch).abs().max()


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
        print((out - hf_out).abs())
        # TODO: need to look at why we can't do low precision
        assert torch.allclose(out, hf_out, atol=1e-1), (out - hf_out).abs().max()


def test_flops():
    config = GPTConfig()
    model = FusedGPT(config).eval()
    num_tokens = 1024
    fwd_flops = model.get_fwd_flops(num_tokens)
    total_flops = fwd_flops * 3
    num_parameters = get_num_parameters(model)
    r = (fwd_flops * 3) / (6 * num_parameters * num_tokens)
    assert r >= 0.9995, r


def test_estimate_days():
    # llama-3.1 paper reports 54 days for pre-training 405B parameter model
    # its very close to what we get from following equation
    flops = 6 * (405 * 10**9) * (15 * 10**12)
    t = estimate_days(flops, mfu=0.45, gpu="h100", num_gpus=16_000)
    assert t == 59.24544994944388, t
