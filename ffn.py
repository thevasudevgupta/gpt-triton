import torch
import triton
import triton.language as tl

from utils import dropout, gelu_new


@triton.jit
def ffn_kernel(
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
    ffn_kernel[grid](
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
