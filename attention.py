import torch
import triton.language as tl

from ffn import fused_ffn


# TODO: what if we just write separate kernel for this?
@torch.no_grad()
def matmul_and_split_qkv(x, weight, bias):
    # x: (batch_size, seqlen, hidden_size)
    x = fused_ffn1(x, weight, bias, add_gelu=False)
    q, k, v = x.split(self.n_embd, dim=2)
    # (batch_size, seqlen, num_heads, head_size)
    # TODO: following is unecessary read & write - memory bound operation
    q, k, v = map(lambda x: x.transpose(1, 2).contiguous(), (q, k, v))
    # (batch_size, num_heads, seqlen, head_size)
    # Splits the tensor into chunks. Each chunk is a view of the original tensor.
    return q, k, v


@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # x: (N0, T)
    # out: (N0, T)
    pid_0 = tl.program_id(0)
    x_ptr += pid_0 * T
    z_ptr += pid_0 * T

    max_value, denominator = 0.0, 0.0
    for i in range(0, T, B1):
        offset = tl.arange(i, i + B1)
        x = tl.load(x_ptr + offset, mask=offset < T, other=0)

        block_max_value = tl.max(x, keep_dims=True)
        new_max_value = tl.where(
            block_max_value > max_value, block_max_value, max_value
        )

        x = tl.exp(x - new_max_value)
        denominator = denominator / tl.exp(new_max_value - max_value)

        denominator += tl.sum(x)
        max_value = new_max_value

    for i in range(0, T, B1):
        offset = tl.arange(i, i + B1)
        x = tl.load(x_ptr + offset, mask=offset < T, other=0)
        z = tl.exp(x - max_value)
        z = z / denominator
        tl.store(z_ptr + offset, z, mask=offset < T)

    return


@triton.jit
def fused_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    z_ptr,
    Lq: int,
    Lk: int,
    H: int,
    dropout_prob: float = 0.0,
    seed: int = 1337,
    BLOCK_SIZE_Lq: tl.constexpr = 16,
    BLOCK_SIZE_Lk: tl.constexpr = 16,
    BLOCK_SIZE_MID: tl.constexpr = 16,
):
    # launch conditions
    # 1st dim: batch_size * num_heads
    # 2nd dim: seqlen
    # 3rd dim: head_size

    # shift q by (Lq * H) and v, k by (Lk * H)
    # and the do regular matrix multiplication to obtain (L)
    # apply softmax and compute value vector on-the-fly

    # q: (1, seqlen, head_size)
    # k: (1, seqlen, head_size)
    # v: (1, seqlen, head_size)
    # out: (1, seqlen, head_size)

    # TODO: eventually support batch_size > 1 as well

    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # intuition is this: In normal math, we basically take 1 row of X & 1 column of W
    # and just multiply element wise and add stuff
    # but here we add multiple consecutive rows of X & multiple consecutive rows of W
    # and do dot product basically
    # we basically move over output matrix and computes each block in each kernel

    # Lq: number of query tokens in sequence
    # Lk: number of key tokens in sequence
    # H: head size

    # pid_0: vertical
    # pid_1: horizontal
    # q: (Lq, H)
    # k: (H, Lk)
    # v: (Lk, H)

    # a = q @ k => (Lq, Lk)
    # s = softmax(a) => (Lq, Lk)
    # s @ v => (Lq, H)

    # q block size: (BLOCK_SIZE_Lq, BLOCK_SIZE_MID)
    # k block size: (BLOCK_SIZE_MID, BLOCK_SIZE_Lk)
    # z block size: (BLOCK_SIZE_Lq, BLOCK_SIZE_Lk)

    # these are the pointer of 1st element for each block in output matrix

    # we basically add row-block-shift here
    row_id = pid_0 * BLOCK_SIZE_Lq

    # we basically add column-block-shift here
    column_id = pid_1 * BLOCK_SIZE_Lk

    # print(f"row_id={row_id}, column_id={column_id}")

    # each block in s would be of shape-(BLOCK_SIZE_Lq, BLOCK_SIZE_Lk)
    # block of size: BLOCK_SIZE_Lq x BLOCK_SIZE_MID would move in horizontal direction
    # block of size: BLOCK_SIZE_MID x BLOCK_SIZE_Lk would move in vertical direction

    # we need this loop because we might not be able to fit full row of X & full column of W in-memory
    s = tl.zeros((BLOCK_SIZE_Lq, BLOCK_SIZE_Lk), dtype=tl.float32)
    for mid in range(0, H, BLOCK_SIZE_MID):
        q_v = row_id + tl.arange(0, BLOCK_SIZE_Lq)[:, None]
        q_h = tl.arange(0, BLOCK_SIZE_MID)[None, :] + mid
        # print(f"x_v={x_v} {x_v < M} {M}, x_h={x_h} {x_h < K} {K}")
        q = tl.load(q_ptr + q_v * H + q_h, mask=(q_v < Lq) & (q_h < H), other=0)
        # (BLOCK_SIZE_Lq, BLOCK_SIZE_MID)

        k_v = tl.arange(0, BLOCK_SIZE_MID)[:, None] + mid
        k_h = column_id + tl.arange(0, BLOCK_SIZE_Lk)[None, :]
        # print(f"w_v={w_v} {w_v < K} {K}, w_h={w_h} {w_h < N} {N}")
        w = tl.load(k_ptr + k_v * Lk + k_h, mask=(k_v < H) & (k_h < Lk), other=0)
        # (BLOCK_SIZE_MID, BLOCK_SIZE_Lk)

        s = tl.dot(x, w, acc=z)
        # (BLOCK_SIZE_Lq, BLOCK_SIZE_Lk)

    z_v = row_id + tl.arange(0, BLOCK_SIZE_M)[:, None]
    z_h = column_id + tl.arange(0, BLOCK_SIZE_N)[None, :]
    z_offset = z_v * N + z_h
    z_mask = (z_v < M) & (z_h < N)

    # print(f"z_v={z_v}, z_h={z_h}")
    tl.store(z_ptr + z_offset, z, mask=z_mask)


@torch.no_grad()
def fused_attention(q, k, v):

    # B, T, C = x.size()
    # # batch size, sequence length, embedding dimensionality (n_embd)

    # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # # (B, nh, T, hs)
    # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # # (B, nh, T, hs)
    # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # # (B, nh, T, hs)

    # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

    # # manual implementation of attention
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
    # att = F.softmax(att, dim=-1)
    # att = self.attn_dropout(att)
    # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    # y = y.transpose(1, 2).contiguous().view(B, T, C)

    grid = ()
    launch_trition(
        fused_attention_kernel,
        grid,
    )

    return


class FusedAttention(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.layer_norm_weight = nn.Parameter(torch.ones((hidden_size,)))
        self.layer_norm_bias = nn.Parameter(torch.zeros((hidden_size,)))
        self.c_proj_weight = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.c_proj_bias = nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, x):
        residual = x
        x = fused_layer_norm(x, self.layer_norm_weight.data, self.layer_norm_bias.data)
        q, k, v = matmul_and_split_qkv(
            x, self.c_attn_weight.data, self.c_attn_bias.data
        )
        dropout_prob = self.dropout_prob if self.training else 0.0
        x = fused_attention(q, k, v, dropout_prob=dropout_prob)
        x = fused_ffn2(
            x,
            self.c_proj_weight.data,
            self.c_proj_bias.data,
            residual,
            dropout_prob=dropout_prob,
        )
        return x
