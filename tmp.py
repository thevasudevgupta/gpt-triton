# TRITON_INTERPRET=1 python3 tmp.py

import torch

from gpt import FusedGPT, GPTConfig, estimate_days, get_num_parameters

print("training time (in hours):", t)

import ipdb

ipdb.set_trace()

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

print(z.shape, z)

print("diff:", (z - z_torch).abs().max())
print(z)
print(z_torch)
