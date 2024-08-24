Triton implementation of GPT/LLAMA models. Objective of this project is to understand how much performance can be squeezed out if we implement full-GPT-block in one triton kernel.

**Performance**

triton implementation is 2x faster and 4x memory efficient compared to HuggingFace Transformers implementation.

```bash
python3 bench.py

# 1 A100 40 GB
# torch: batch_size = 512 && t = 1801.32
# triton: batch_size = 512 && t = 789.14
# torch: batch_size = 1024 && OOM
# triton: batch_size = 2048 && t = 3153.70
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
