import torch
import torch.nn as nn

from ffn import fused_ffn
from layer_norm import fused_layer_norm


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
