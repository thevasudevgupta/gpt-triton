# TRITON_INTERPRET=1 python3 test.py

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)
x = torch.rand((249, 121), device=device)
mlp = FusedMLP(x.shape[1]).to(device)
z = mlp(x)
print(z.shape, z)

print("diff:", (z - z_torch).abs().max())
print(z)
print(z_torch)
