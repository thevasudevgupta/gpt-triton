# TRITON_INTERPRET=1 python3 gpt.py

from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import GPT2Model as HFGPT2

from kernels import (flash_attention_v1, fused_embeddings, fused_ffn,
                     fused_layer_norm, matmul_and_split_qkv)


class FusedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads

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
        x = flash_attention_v1(q, k, v, dropout_prob=dropout_prob)
        x = x.transpose(1, 2).contiguous().view(residual.shape)
        x = fused_ffn(
            x,
            self.c_proj_weight.data,
            bias=self.c_proj_bias.data,
            residual=residual,
            add_gelu=False,
            dropout_prob=dropout_prob,
        )
        return x


class FusedMLP(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob

        self.layer_norm_weight = nn.Parameter(torch.ones((hidden_size,)))
        self.layer_norm_bias = nn.Parameter(torch.zeros((hidden_size,)))

        intermediate_size = 4 * hidden_size

        self.ffn1_weight = nn.Parameter(torch.rand(hidden_size, intermediate_size))
        self.ffn1_bias = nn.Parameter(torch.rand(intermediate_size))

        self.ffn2_weight = nn.Parameter(torch.rand(intermediate_size, hidden_size))
        self.ffn2_bias = nn.Parameter(torch.rand(hidden_size))

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
        )
        x = fused_ffn(
            x,
            self.ffn2_weight.data,
            bias=self.ffn2_bias.data,
            residual=residual,
            add_gelu=False,
            dropout_prob=dropout_prob,
        )
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class FusedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte_weight = nn.Parameter(torch.rand(config.vocab_size, config.n_embd))
        self.wpe_weight = nn.Parameter(torch.rand(config.block_size, config.n_embd))

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    FusedAttention(
                        config.n_embd, config.n_head, dropout_prob=config.dropout
                    ),
                    FusedMLP(config.n_embd, dropout_prob=config.dropout),
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
        # )
        return x


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


def convert_hf_and_load_model(model_id):
    hf_model = HFGPT2.from_pretrained(model_id).eval()
    state_dict, config = convert_huggingface_to_triton(
        hf_model.state_dict(), hf_model.config
    )
    model = FusedGPT(config).eval()
    model.load_state_dict(state_dict)
    return model, hf_model


model_id = "gpt2"
model, hf_model = convert_hf_and_load_model(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
hf_model.to(device)

with torch.no_grad():
    string = "I am vasudev gupta. I like AI."
    inputs = tokenizer(string, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hf_out = hf_model(**inputs).last_hidden_state
    out = model(inputs["input_ids"])
    print((out - hf_out).abs().max())
    print((out - hf_out).abs())
    import ipdb

    ipdb.set_trace()
