# TRITON_INTERPRET=1 python3 embeddings.py

import torch
import triton
import triton.language as tl

from utils import dropout


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

    z = torch.empty((B * L, N), device=x.device)
    grid = (z.shape[0],)
    fused_embeddings_kernel[grid](
        x.view(-1), wte, wpe, z, B, L, V, P, N, dropout_prob=dropout_prob
    )

    return z.view((B, L, N))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.tensor(
        [
            [7, 10, 2, 0, 5, 3, 7, 10, 2, 0, 5, 3, 7, 10, 2, 0, 5, 3],
            [14, 3, 1, 10, 2, 0, 14, 3, 1, 10, 14, 3, 1, 10, 14, 3, 1, 10],
        ],
        dtype=torch.long,
        device=device,
    )
    wte = torch.rand((16, 8), device=device)
    wpe = torch.rand((20, 8), device=device)

    z_torch = wte[x] + wpe[torch.arange(x.shape[1], device=device)][None]
    z = fused_embeddings(x, wte, wpe)

    print("diff:", (z - z_torch).abs().max())
