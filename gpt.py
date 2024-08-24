# TRITON_INTERPRET=1 python3 gpt.py

from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import GPT2Model as HFGPT2

from kernels import (flash_attention_v1, fused_embeddings, fused_ffn,
                     fused_layer_norm, matmul_and_split_qkv)

GPU_TO_FLOPS = {
    "v100": 130 * 10**12,
    "a100": 312 * 10**12,
    "h100": 989 * 10**12,
}


class FusedAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, dropout_prob=0.0, cast_dtype_for_dot=True
    ):
        super().__init__()
        self.cast_dtype_for_dot = cast_dtype_for_dot
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads

        self.hidden_size = hidden_size

        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.layer_norm_bias = nn.Parameter(torch.zeros(hidden_size))

        self.c_attn_weight = nn.Parameter(torch.rand(hidden_size, 3 * hidden_size))
        self.c_attn_bias = nn.Parameter(torch.rand(3 * hidden_size))

        self.c_proj_weight = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.c_proj_bias = nn.Parameter(torch.rand(hidden_size))

    def forward(self, x):
        residual = x
        x = fused_layer_norm(x, self.layer_norm_weight.data, self.layer_norm_bias.data)
        q, k, v = matmul_and_split_qkv(
            x, self.c_attn_weight.data, self.c_attn_bias.data, self.num_heads
        )
        dropout_prob = self.dropout_prob if self.training else 0.0
        x = flash_attention_v1(
            q,
            k,
            v,
            dropout_prob=dropout_prob,
            cast_dtype_for_dot=self.cast_dtype_for_dot,
        )
        x = x.transpose(1, 2).contiguous().view(residual.shape)
        x = fused_ffn(
            x,
            self.c_proj_weight.data,
            bias=self.c_proj_bias.data,
            residual=residual,
            add_gelu=False,
            dropout_prob=dropout_prob,
            cast_dtype_for_dot=self.cast_dtype_for_dot,
        )
        return x

    def get_fwd_flops(self, num_tokens):
        h = self.hidden_size
        layer_norm = num_tokens * h + num_tokens * h
        c_attn = num_tokens * (3 * h) * (2 * h) + num_tokens * (3 * h)
        c_proj = num_tokens * h * (2 * h) + num_tokens * h
        return layer_norm + c_attn + c_proj


class FusedMLP(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.0, cast_dtype_for_dot=True):
        super().__init__()

        self.cast_dtype_for_dot = cast_dtype_for_dot
        self.dropout_prob = dropout_prob

        self.layer_norm_weight = nn.Parameter(torch.ones((hidden_size,)))
        self.layer_norm_bias = nn.Parameter(torch.zeros((hidden_size,)))

        intermediate_size = 4 * hidden_size

        self.ffn1_weight = nn.Parameter(torch.rand(hidden_size, intermediate_size))
        self.ffn1_bias = nn.Parameter(torch.rand(intermediate_size))

        self.ffn2_weight = nn.Parameter(torch.rand(intermediate_size, hidden_size))
        self.ffn2_bias = nn.Parameter(torch.rand(hidden_size))

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def forward(self, x):
        # mlp = DROPOUT(GELU(LN(X) @ A + a) @ B + b) + X
        dropout_prob = self.dropout_prob if self.training else 0.0
        residual = x
        x = fused_layer_norm(x, self.layer_norm_weight.data, self.layer_norm_bias.data)
        x = fused_ffn(
            x,
            self.ffn1_weight.data,
            bias=self.ffn1_bias.data,
            residual=None,
            add_gelu=True,
            dropout_prob=dropout_prob,
            cast_dtype_for_dot=self.cast_dtype_for_dot,
        )
        x = fused_ffn(
            x,
            self.ffn2_weight.data,
            bias=self.ffn2_bias.data,
            residual=residual,
            add_gelu=False,
            dropout_prob=dropout_prob,
            cast_dtype_for_dot=self.cast_dtype_for_dot,
        )
        return x

    def get_fwd_flops(self, num_tokens):
        h = self.hidden_size
        mid = self.intermediate_size
        layer_norm = num_tokens * h + num_tokens * h
        ffn1 = num_tokens * mid * (2 * h) + num_tokens * mid
        ffn2 = num_tokens * h * (2 * mid) + num_tokens * h
        return layer_norm + ffn1 + ffn2


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class FusedGPT(nn.Module):
    def __init__(self, config, cast_dtype_for_dot=True):
        super().__init__()
        self.cast_dtype_for_dot = cast_dtype_for_dot
        self.config = config

        self.wte_weight = nn.Parameter(torch.rand(config.vocab_size, config.n_embd))
        self.wpe_weight = nn.Parameter(torch.rand(config.block_size, config.n_embd))

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    FusedAttention(
                        config.n_embd,
                        config.n_head,
                        dropout_prob=config.dropout,
                        cast_dtype_for_dot=cast_dtype_for_dot,
                    ),
                    FusedMLP(
                        config.n_embd,
                        dropout_prob=config.dropout,
                        cast_dtype_for_dot=cast_dtype_for_dot,
                    ),
                )
                for _ in range(config.n_layer)
            ]
        )
        self.layer_norm_weight = nn.Parameter(torch.ones((config.n_embd,)))
        self.layer_norm_bias = nn.Parameter(torch.zeros((config.n_embd,)))

        # TODO: we don't wanna consume consume 2x memory here because of transpose and contiguous
        # instead implement transposed matmul in triton kernel
        # self.lm_head_weight = self.wte.weight.data.T.contiguous()

    def forward(self, x):
        # it does causal automatically, no need of separate attention/padding mask
        dropout_prob = self.config.dropout_prob if self.training else 0.0
        x = fused_embeddings(
            x, self.wte_weight.data, self.wpe_weight.data, dropout_prob=dropout_prob
        )
        for block in self.blocks:
            x = block(x)
        x = fused_layer_norm(x, self.layer_norm_weight, self.layer_norm_bias)
        # x = fused_ffn(
        #     x,
        #     self.lm_head_weight,
        #     bias=None,
        #     residual=None,
        #     add_gelu=False,
        #     dropout_prob=0.0,
        #     cast_dtype_for_dot=self.cast_dtype_for_dot,
        # )
        return x

    def get_fwd_flops(self, num_tokens):
        h = self.config.n_embd
        v = self.config.vocab_size
        p = self.config.block_size
        wte = num_tokens * h * (2 * v)
        wpe = num_tokens * h * (2 * p)
        blocks = sum(
            [
                module.get_fwd_flops(num_tokens)
                for block in self.blocks
                for module in block
            ]
        )
        layer_norm = num_tokens * h + num_tokens * h
        return blocks + layer_norm + wte + wpe


def convert_huggingface_to_triton(hf_sd, hf_config):
    config = GPTConfig(
        vocab_size=hf_config.vocab_size,
        block_size=hf_config.n_ctx,
        n_layer=hf_config.n_layer,
        n_head=hf_config.n_head,
        n_embd=hf_config.n_embd,
        dropout=0.1,
    )
    mapping = {
        "wte.weight": "wte_weight",
        "wpe.weight": "wpe_weight",
        "ln_f.weight": "layer_norm_weight",
        "ln_f.bias": "layer_norm_bias",
    }
    block = {
        "h.{i}.ln_1.weight": "blocks.{i}.0.layer_norm_weight",
        "h.{i}.ln_1.bias": "blocks.{i}.0.layer_norm_bias",
        "h.{i}.attn.bias": None,
        "h.{i}.attn.c_attn.weight": "blocks.{i}.0.c_attn_weight",
        "h.{i}.attn.c_attn.bias": "blocks.{i}.0.c_attn_bias",
        "h.{i}.attn.c_proj.weight": "blocks.{i}.0.c_proj_weight",
        "h.{i}.attn.c_proj.bias": "blocks.{i}.0.c_proj_bias",
        "h.{i}.ln_2.weight": "blocks.{i}.1.layer_norm_weight",
        "h.{i}.ln_2.bias": "blocks.{i}.1.layer_norm_bias",
        "h.{i}.mlp.c_fc.weight": "blocks.{i}.1.ffn1_weight",
        "h.{i}.mlp.c_fc.bias": "blocks.{i}.1.ffn1_bias",
        "h.{i}.mlp.c_proj.weight": "blocks.{i}.1.ffn2_weight",
        "h.{i}.mlp.c_proj.bias": "blocks.{i}.1.ffn2_bias",
    }
    for k, v in block.items():
        if v is None:
            continue
        for i in range(config.n_layer):
            mapping[k.format(i=i)] = v.format(i=i)
    sd = {}
    for k, v in tqdm(hf_sd.items()):
        sd[mapping[k]] = v
    return sd, config


def convert_hf_and_load_model(model_id, device, cast_dtype_for_dot=True):
    hf_model = HFGPT2.from_pretrained(model_id)
    state_dict, config = convert_huggingface_to_triton(
        hf_model.state_dict(), hf_model.config
    )
    model = FusedGPT(config, cast_dtype_for_dot=cast_dtype_for_dot)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), hf_model.to(device).eval()


def estimate_days(flops, mfu=0.45, gpu="h100", num_gpus=1):
    # its probably very hard to achieve 0.45 mfu - LOL
    # but thats kinda SOTA in papers from top labs
    assert gpu in GPU_TO_FLOPS
    return flops / (mfu * GPU_TO_FLOPS[gpu] * 3600 * 24 * num_gpus)


def get_num_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def compute_mfu(flops_per_second, gpu="h100"):
    assert gpu in GPU_TO_FLOPS
    return flops_per_second / GPU_TO_FLOPS[gpu]
