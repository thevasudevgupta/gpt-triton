# TRITON_INTERPRET=1 python3 bench.py

import torch
import triton
from transformers import AutoTokenizer

from gpt import convert_hf_and_load_model

STRING = """\
Large language models have been shown to achieve remarkable performance across a variety of natural\
language tasks using few-shot learning, which drastically reduces the number of task-specific training\
examples needed to adapt the model to a particular application. To further our understanding of the\
impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer\
language model, which we call Pathways Language Model (PaLM).\
We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient\
training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-ofthe-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a\
number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned stateof-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the\
recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous\
improvements from model scale, meaning that performance steeply increased as we scaled to our largest\
model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we\
demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias\
and toxicity, and study the extent of training data memorization with respect to model scale. Finally,\
we discuss the ethical considerations related to large language models and discuss potential mitigation\
strategies.\
"""


def run_benchmark(provider, warmup=25, rep=100):
    assert torch.cuda.is_available()
    device = "cuda"
    model_id = "gpt2"
    model, hf_model = convert_hf_and_load_model(model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # triton is slow for batch_size = 1 with current settings but much faster with batch > 1
    inputs = tokenizer(
        [STRING] * 512, return_tensors="pt", max_length=512, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # z_torch = hf_model(**inputs).last_hidden_state
        # z = model(inputs["input_ids"])
        # print("diff:", z - z_torch)
        if provider == "torch":
            fn = lambda: hf_model(**inputs).last_hidden_state
            return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if provider == "triton":
            fn = lambda: model(inputs["input_ids"])
            return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


# 1 A100 40 GB
# torch: batch_size = 512 && t = 1801.32
# triton: batch_size = 512 && t = 789.14
# torch: batch_size = 1024 && OOM
# triton: batch_size = 2048 && t = 3153.70

print("triton:", run_benchmark("triton"))
print("torch:", run_benchmark("torch"))
