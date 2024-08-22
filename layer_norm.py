import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
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
    M, N = x.shape
    BLOCK_SIZE = 128
    z = torch.empty(x.shape, device=x.device)
    grid = (M,)
    layer_norm_kernel[grid](x, weight, bias, z, N, BLOCK_SIZE=BLOCK_SIZE)
    return z
