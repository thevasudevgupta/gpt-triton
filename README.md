triton implementation of gpt/llama models

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

Objective of this repo is following:
* find out how much performance improvement we can squeeze out if we implement full GPT block in single triton kernel
* eventually, wanna find out if we can do the back-propogation of GPT block manually and optimise the memory requirements (i.e. solving the math problem)
