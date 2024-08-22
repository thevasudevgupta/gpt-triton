import math

import torch
import triton
import triton.language as tl

# TODO: shift to `make_block_ptr`?


# tl.math.tanh doesn't exist in CPU version of triton
@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu_new(x):
    pi = math.pi
    a = tl.math.sqrt(2.0 / pi)
    b = x + 0.044715 * x * x * x
    return 0.5 * x * (1.0 + tanh(a * b))


@triton.jit
def dropout(x, p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random > p, x / (1 - p), 0.0)


@triton.jit
def fused_embeddings_kernel(
    x_ptr,
    wte_ptr,
    wpe_ptr,
    z_ptr,
    B,
    L,
    V,
    P,
    N,
    dropout_prob=0.0,
    seed=1337,
    BLOCK_SIZE: tl.constexpr = 16,
):
    # f = dropout(wte(x) + wpe(x))

    # x: (B*S,)
    # wte: (V, N)
    # wpe: (P, N)
    # z: (B*S, N)

    pid = tl.program_id(0)
    wte_ptr += tl.load(x_ptr + pid) * N
    wpe_ptr += (pid % L) * N
    z_ptr += pid * N

    for k in range(0, N, BLOCK_SIZE):
        offset = k + tl.arange(0, BLOCK_SIZE)
        mask = offset < N

        z = tl.load(wte_ptr + offset, mask=mask, other=0.0)
        z += tl.load(wpe_ptr + offset, mask=mask, other=0.0)
        z = dropout(z, dropout_prob, seed, offset)

        tl.store(z_ptr + offset, z, mask=mask)


@torch.no_grad()
def fused_embeddings(x, wte, wpe, dropout_prob=0.0):
    B, L = x.shape
    V, N = wte.shape
    P = wpe.shape[0]
    assert wte.shape[1] == wpe.shape[1]
    assert x.is_contiguous()
    assert wte.is_contiguous()
    assert wpe.is_contiguous()

    z = torch.empty((B * L, N), device=x.device)
    grid = (z.shape[0],)
    fused_embeddings_kernel[grid](
        x.view(-1), wte, wpe, z, B, L, V, P, N, dropout_prob=dropout_prob
    )

    return z.view((B, L, N))


@triton.jit
def fused_layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, z_ptr, N, eps=1e-5, BLOCK_SIZE: tl.constexpr = 16
):
    # x: (M, N)
    # launch with 1D grid along M direction

    row_id = tl.program_id(0)
    x_ptr += row_id * N
    z_ptr += row_id * N

    x_mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offset, mask=(offset < N), other=0.0)
        x_mean += x.to(tl.float32)
    x_mean = tl.sum(x_mean) / N

    x_var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offset, mask=(offset < N), other=x_mean)
        x = x.to(tl.float32)
        x_var += (x - x_mean) * (x - x_mean)
    x_var = tl.sum(x_var) / N
    rstd = 1 / tl.sqrt(x_var + eps)

    # TODO: we could prevent this extra loop if we fuse it in ffn block?
    # but thats quite hacky - so, lets move with extra loop for now
    for i in range(0, N, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < N

        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        w = tl.load(w_ptr + offset, mask=mask, other=0.0)
        b = tl.load(b_ptr + offset, mask=mask, other=0.0)

        z = (x - x_mean) * rstd
        z = z * w + b

        tl.store(z_ptr + offset, z, mask=mask)


@torch.no_grad()
def fused_layer_norm(x, weight, bias):
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()
    assert weight.shape == bias.shape
    assert x.shape[-1] == weight.shape[0]
    M, N = x.shape
    BLOCK_SIZE = 128
    z = torch.empty(x.shape, device=x.device)
    grid = (M,)
    fused_layer_norm_kernel[grid](x, weight, bias, z, N, BLOCK_SIZE=BLOCK_SIZE)
    return z


@triton.jit
def fused_ffn_kernel(
    x_ptr,
    w_ptr,
    z_ptr,
    M,
    N,
    K,
    b_ptr=None,
    r_ptr=None,
    apply_gelu=False,
    dropout_prob=0.0,
    seed=1337,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
):
    # f = dropout(gelu(x @ w + b)) + residual
    # launch with 2D grid of blocks along M & N directions

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # intuition is this: In normal math, we basically take 1 row of X & 1 column of W
    # and just multiply element wise and add stuff
    # but here we add multiple consecutive rows of X & multiple consecutive rows of W
    # and do dot product basically

    # pid_m: vertical
    # pid_n: horizontal

    # we basically move over output matrix and computes each block in each kernel

    # x: (M, K)
    # w: (K, N)
    # b: (N,)
    # z: (M, N)

    # x block size: (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # w block size: (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # z block size: (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # these are the pointer of 1st element for each block in output matrix

    # we basically add row-block-shift here
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]

    # we basically add column-block-shift here
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]

    # each block in z would be of shape-(M, N)
    # block of size: BLOCK_SIZE_M x BLOCK_SIZE_K would move in horizontal direction
    # block of size: BLOCK_SIZE_K x BLOCK_SIZE_N would move in vertical direction

    # we need this loop because we might not be able to fit full row of X & full column of W in-memory
    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        x = tl.load(
            x_ptr + (offs_m % M) * K + x_k, mask=(offs_m < M) & (x_k < K), other=0.0
        )
        # TODO: need to read why casting to fp16 is important here
        x = x.to(tl.float16)
        # (BLOCK_SIZE_M, BLOCK_SIZE_K)

        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        w = tl.load(
            w_ptr + w_k * N + (offs_n % N), mask=(w_k < K) & (offs_n < N), other=0.0
        )
        w = w.to(tl.float16)
        # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        z = tl.dot(x, w, acc=z)
        # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0)
        z += b.to(tl.float32)
    # (1, BLOCK_SIZE_N)

    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)

    if apply_gelu:
        z = gelu_new(z)
    if dropout_prob > 0:
        z = dropout(z, dropout_prob, seed, z_offset)

    if r_ptr is not None:
        r = tl.load(r_ptr + z_offset, mask=z_mask)
        z += r.to(tl.float32)

    tl.store(z_ptr + z_offset, z, mask=z_mask)


@torch.no_grad()
def fused_ffn(x, weight, bias=None, residual=None, add_gelu=False, dropout_prob=0.0):
    # f = dropout(gelu(x @ w + b)) + residual

    M, K = x.shape
    N = weight.shape[1]
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)

    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert x.shape[1] == weight.shape[0]
    if bias is not None:
        assert bias.is_contiguous()
        assert weight.shape[1] == bias.shape[0]
    if residual is not None:
        assert residual.is_contiguous()
        assert residual.shape == z.shape

    BLOCK_SIZE_M = min(16, M)
    BLOCK_SIZE_N = min(16, N)
    BLOCK_SIZE_K = min(16, K)
    assert BLOCK_SIZE_K >= 16, "triton doesn't support block size < 16"

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    fused_ffn_kernel[grid](
        x,
        weight,
        z,
        M,
        N,
        K,
        apply_gelu=add_gelu,
        dropout_prob=dropout_prob,
        b_ptr=bias,
        r_ptr=residual,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return z


# @triton.jit
# def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
#     # x: (N0, T)
#     # out: (N0, T)
#     pid_0 = tl.program_id(0)
#     x_ptr += pid_0 * T
#     z_ptr += pid_0 * T

#     max_value, denominator = 0.0, 0.0
#     for i in range(0, T, B1):
#         offset = tl.arange(i, i + B1)
#         x = tl.load(x_ptr + offset, mask=offset < T, other=0)

#         block_max_value = tl.max(x, keep_dims=True)
#         new_max_value = tl.where(
#             block_max_value > max_value, block_max_value, max_value
#         )

#         x = tl.exp(x - new_max_value)
#         denominator = denominator / tl.exp(new_max_value - max_value)

#         denominator += tl.sum(x)
#         max_value = new_max_value

#     for i in range(0, T, B1):
#         offset = tl.arange(i, i + B1)
#         x = tl.load(x_ptr + offset, mask=offset < T, other=0)
#         z = tl.exp(x - max_value)
#         z = z / denominator
#         tl.store(z_ptr + offset, z, mask=offset < T)


# # TODO: what if we just write separate kernel for this?
# @torch.no_grad()
# def matmul_and_split_qkv(x, weight, bias):
#     # x: (batch_size, seqlen, hidden_size)
#     x = fused_ffn1(x, weight, bias, add_gelu=False)
#     q, k, v = x.split(self.n_embd, dim=2)
#     # (batch_size, seqlen, num_heads, head_size)
#     # TODO: following is unecessary read & write - memory bound operation
#     q, k, v = map(lambda x: x.transpose(1, 2).contiguous(), (q, k, v))
#     # (batch_size, num_heads, seqlen, head_size)
#     # Splits the tensor into chunks. Each chunk is a view of the original tensor.
#     return q, k, v


# TODO: read about flash-2 and see if we can switch to that
# TODO: then read about flash-3 and see if we can switch to that instead
# TODO: can we do score computation for only unmasked positions?
# pytorch flex-attention does something like that - it would make computation 50% efficient
@triton.jit
def flash_attention_v1_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0: tl.constexpr, T: tl.constexpr, B0: tl.constexpr
):
    # q: (N0, T)
    # k: (N0, T)
    # v: (N0, T)
    # z: (N0, T)
    pid_0 = tl.program_id(0)

    # assuming that `T` can stay sram fully and doesn't require blocking
    # this assumptions was made for original implementation of flash attention as well
    # its reasonable as most of LLMs use head size <= 256

    offs_b = pid_0 * B0 + tl.arange(0, B0)[:, None]
    offs_t = tl.arange(0, T)[None, :]

    q_mask = offs_b < N0
    # this remains in sram throughtout computation
    q_offs = offs_b * T + offs_t

    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    # (B0, T)

    # loop over k, v and compute attention & weighted v
    z = tl.zeros((B0, T), dtype=tl.float32)
    max_value = tl.zeros((B0, 1), dtype=tl.float32) + float("-inf")
    denominator = tl.zeros((B0, 1), dtype=tl.float32)
    for i in range(0, N0, B0):

        mask = (i + tl.arange(0, B0)[:, None]) < N0
        offs = (i + tl.arange(0, B0)[:, None]) * T + tl.arange(0, T)[None, :]

        k = tl.load(k_ptr + offs, mask=mask, other=0.0)
        # (B0, T)

        qk = tl.dot(q, k.trans(1, 0))
        # (B0, B0)

        block_max_value = tl.max(qk, axis=1, keep_dims=True)
        # (B0, 1)
        new_max_value = tl.where(
            block_max_value > max_value, block_max_value, max_value
        )
        # (B0, 1)

        qk = tl.exp(qk - new_max_value)
        # (B0, B0)

        multiplier = tl.exp(max_value - new_max_value)
        denominator *= multiplier
        z *= multiplier

        denominator += tl.sum(qk, axis=1, keep_dims=True)
        max_value = new_max_value
        # (B0, 1)

        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        # (B0, T)

        z = tl.dot(qk, v, acc=z)
        # (B0, T)

    z /= denominator
    tl.store(z_ptr + q_offs, z, mask=q_mask)


@torch.no_grad()
def flash_attention_v1(q, k, v):

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
