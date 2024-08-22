import torch
import torch.nn as nn

from kernels import flash_attention_v1, fused_ffn, fused_layer_norm


class FusedMLP(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.0):
        super().__init__()

        self.dropout_prob = dropout_prob

        self.layer_norm_weight = nn.Parameter(torch.ones((hidden_size,)))
        self.layer_norm_bias = nn.Parameter(torch.zeros((hidden_size,)))

        self.ffn1_weight = nn.Parameter(torch.rand(hidden_size, 4 * hidden_size))
        self.ffn1_bias = nn.Parameter(
            torch.rand(
                4 * hidden_size,
            )
        )

        self.ffn2_weight = nn.Parameter(torch.rand(4 * hidden_size, hidden_size))
        self.ffn2_bias = nn.Parameter(
            torch.rand(
                hidden_size,
            )
        )

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


class FusedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads
        self.layer_norm_weight = nn.Parameter(torch.ones((hidden_size,)))
        self.layer_norm_bias = nn.Parameter(torch.zeros((hidden_size,)))
        self.c_proj_weight = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.c_proj_bias = nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, x):
        residual = x
        x = fused_layer_norm(x, self.layer_norm_weight.data, self.layer_norm_bias.data)
        q, k, v = matmul_and_split_qkv(
            x, self.c_attn_weight.data, self.c_attn_bias.data, self.num_heads
        )
        dropout_prob = self.dropout_prob if self.training else 0.0
        x = fused_attention(q, k, v, dropout_prob=dropout_prob)
        x = x.transpose(1, 2).view(residual.shape)
        x = fused_ffn2(
            x,
            self.c_proj_weight.data,
            self.c_proj_bias.data,
            residual,
            dropout_prob=dropout_prob,
        )
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 2
    n_head: int = 12
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True


class FusedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.nembd)

        self.blocks = nn.Sequential(
            FusedLayerNorm(config.nembd),
            FusedAttention(config.n_embd, dropout_prob=config.dropout),
            FusedMLP(config.nembd, dropout_prob=config.dropout),
        )
        self.layer_norm_weight = nn.Parameter(torch.ones((config.nembd)))
        self.layer_norm_bias = nn.Parameter(torch.zeros((config.nembd)))

        # TODO: we don't wanna consume consume 2x memory here because of transpose and contiguous
        # instead implement transposed matmul in triton kernel
        self.lm_head_weight = self.wte.weight.data.T.contiguous()

    def forward(self, x):
        # it does causal automatically, no need of separate attention/padding mask
        dropout_prob = self.config.dropout_prob if self.training else 0.0
        x = fused_embeddings(
            x, self.wte.weight.data, self.wpe.weight.data, dropout_prob=dropout_prob
        )
        for block in self.blocks:
            x = block(x)
        x = fused_layer_norm(x, self.layer_norm_weight, self.layer_norm_bias)
        x = ffn_kernel(
            x,
            self.lm_head_weight,
            bias=None,
            residual=None,
            add_gelu=False,
            dropout_prob=0.0,
        )
        return x


config = GPTConfig()
model = GPT(config)


def convert_huggingface_to_triton(state_dict, config):
    return state_dict, config


from transformers import AutoTokenizer
from transformers import GPT2Model as HFGPT2

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = HFGPT2.from_pretrained(model_id)
state_dict, config = convert_huggingface_to_triton(
    hf_model.state_dict(), hf_model.config
)
model = FusedGPT(config)

with torch.no_grad():
    string = "I am vasudev gupta. I like AI."
    hf_out = hf_model(**tokenizer(string))
    out = model(tokenizer(string)["input_ids"])
    print((out - hf_out).abs().max())
