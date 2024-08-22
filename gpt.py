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
