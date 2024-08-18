import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# basically, we wanna make 1 block with just 2 triton kernels
# 1st kernel for attention computation
# 2nd kernel for mlp computation
# and fuse as many operations as possible


# https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L177
# fuse wte, wpe, add, dropout into 1 kernel
# TODO: can we do lookup in triton?

# https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L104
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.n_embd, bias=config.bias)

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y + residual


# https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L105C9-L105C39

# TODO: we really wanna fuse all those into one kernel and not just put code one after another LOL
# => lets solve the math equation on-paper first and see if we can do all ops together with minimal loading?


class MLP(nn.Module):
    def __init__(self, config, add_final_layer_norm=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.ln_f = (
            nn.LayerNorm(config.n_embd, bias=config.bias)
            if add_final_layer_norm
            else None
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        x = x + residual

        if self.ln_f is not None:
            x = self.ln_f(x)
        return x


# mlp = DROPOUT(GELU(LN(X) @ A + a) @ B + b) + X


import math
from typing import Optional

import triton
import triton.language as tl


@triton.jit
def _layer_norm():
    return


@triton.jit
def _matmul():
    return


@triton.jit
def _gelu_new(x):
    pi = math.pi
    a = tl.math.sqrt(2.0 / pi)
    b = x + 0.044715 * (x * x * x)
    return 0.5 * x * (1.0 + tl.math.tanh(a * b))


@triton.jit
def _dropout(x, p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random > p, x / (1 - p), 0.0)


@triton.jit
def _mlp(x_ptr, p=0.0, seed=1337):
    gelu_new(x)
    dropout(x, p, seed, offset)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 2
    n_head: int = 12
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True


class TritonMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout
        self.ln_f = (
            nn.LayerNorm(config.n_embd, bias=config.bias)
            if layer_idx == (config.n_layer - 1)
            else None
        )

    def forward(self, x: torch.Tensor):
        # TODO: can we do this in-place instead?
        z = torch.empty_like(x)
        grid = lambda META: (META[""])
        _mlp[grid](
            x,
            z,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.c_fc.weight,
            self.c_fc.bias,
            self.c_proj.weight,
            self.c_proj.bias,
            dropout=(self.dropout if self.training else 0.0),
            ln_f_weight=self.ln_f.weight if self.ln_f is not None else None,
            ln_f_bias=self.lnf_f.bias if self.ln_f is not None else None,
        )
        return z


if __name__ == "__main__":
    torch.manual_seed(1337)

    config = GPTConfig()
    model = TritonMLP(config).train()
    x = torch.rand((2, 5, 4), dtype=torch.float32)
    z = model(x)

    import ipdb

    ipdb.set_trace()
