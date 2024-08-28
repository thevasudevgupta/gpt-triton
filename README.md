Triton implementation of GPT/LLAMA models. Objective of this project is to understand how much performance can be squeezed out if we implement full-GPT-block in one triton kernel.

**Performance**

triton implementation is more fast & memory efficient compared to HuggingFace Transformers implementation.

```bash
python3 bench.py
```

time taken to process batch size - 512x300 on 1 A100 40 GB

| precision              | HuggingFace GPT | Triton GPT |
|------------------------|-----------------|------------|
| fp32                   | 1800 ms         | -          |
| tf32                   | 623.32 ms       | 462 ms     |
| mixed precision (fp16) | 510.80 ms       | 273 ms     |
| fp16                   | 301.92 ms       | -          |

```python
from gpt import compute_mfu
# fwd MFU

# HuggingFace GPT (fp16)
compute_mfu(2 * 124 * 10**6 * 512*512 / 0.302, gpu="h100")
# 21.76%

# HuggingFace GPT (mixed precision)
compute_mfu(2 * 124 * 10**6 * 512*512 / 0.510, gpu="h100")
# 12.88%

# triton (mixed precision)
compute_mfu(2 * 124 * 10**6 * 512*512 / 0.273, gpu="h100")
# 24.07%
```

**Supported Features**
* [x] fused implementation of several components of GPT block (for eg: `dropout(wte(x) + wpe(x))`, `dropout(wx + b)`, `gelu(wx + b)`)
* [x] flash attention v1 algorithm
* [x] GPT2 implementation in triton
* [x] support for loading pre-trained weights of huggingface-gpt2
* [ ] support KV cache & sampling for inference loop
* [ ] implement back-propogation of GPT block in triton (i.e. solving the math problem)
* [ ] implement paged-attention from vLLM project in triton
* [ ] implement flash attention v2 & v3
* [ ] add kernels for LLAMA-3.1
* [ ] implement adamw in triton (with FSDP-stage2 support)

**Installation**

```bash
pip3 install -r requirements.txt
# `numpy<2` is hard-requirement for running on CPU
# else triton gives garbage - likely some bug in triton
```

**Running tests**

```python
# you can run following command on CPU
TRITON_INTERPRET=1 pytest -sv test.py

# you can run following command on GPU
pytest -sv test.py
```
