tiny implementation of gpt/llama models in triton.

```bash
pip3 install -r requirements.txt
# `numpy<2` is hard-requirement for running on CPU
# else triton gives garbage - likely some bug in triton
```

```python
# you can run following command on CPU
TRITON_INTERPRET=1 python3 test.py

# you can run following command on GPU
python3 test.py
```
