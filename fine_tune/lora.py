# #!/usr/bin/env python3
# """
# lora.py

# LoRA adapter and batching utilities for MLX, including adapter save/load helpers.
# """

# import json
# import random
# import mlx.core as mx
# import mlx.nn   as nn

# class LoRALinear(nn.Module):
#     """
#     LoRA wrapper around an existing nn.Linear weight matrix W.
#     Implements __call__ so it seamlessly replaces nn.Linear.
#     """
#     def __init__(self, W, r=8, alpha=None):
#         super().__init__()
#         self.W     = W
#         self.r     = r
#         self.alpha = alpha or (2 * r)
#         self.scale = self.alpha / r
#         self.A = mx.random.normal((r, W.shape[1])) * 0.01
#         self.B = mx.random.normal((W.shape[0], r)) * 0.01
#         if hasattr(W, "bias"):
#             self.bias = W.bias

#     def forward(self, x):
#         return x @ self.W.T + (self.scale * (x @ self.A.T) @ self.B.T)
#     __call__ = forward

#     @classmethod
#     def from_linear(cls, lin, r=8, alpha=None):
#         return cls(lin.weight, r=r, alpha=alpha)

# def load_pairs(jsonl_path):
#     """Yield (question, answer) tuples from a JSONL file."""
#     with open(jsonl_path) as f:
#         for line in f:
#             o = json.loads(line)
#             yield o["question"].strip(), o["answer"].strip()

# def batch_iter(pairs, tokenizer, batch_size, pad_to, shuffle=False):
#     """
#     Yield (inputs, targets, lengths) batches indefinitely until caller stops.
#     inputs  = tokenizer.encode(question)
#     targets = tokenizer.encode(question+answer)
#     Pads/truncates to pad_to using pad_id=0.
#     """
#     data = list(pairs)
#     idxs = list(range(len(data)))
#     while True:
#         if shuffle:
#             random.shuffle(idxs)
#         for i in range(0, len(idxs), batch_size):
#             chunk = [data[j] for j in idxs[i : i + batch_size]]
#             qs, ans = zip(*chunk)
#             enc_q  = [tokenizer.encode(q) for q in qs]
#             enc_qa = [tokenizer.encode(q + a) for q, a in zip(qs, ans)]
#             inp = [(seq + [0]*pad_to)[:pad_to] for seq in enc_q]
#             tgt = [(seq + [0]*pad_to)[:pad_to] for seq in enc_qa]
#             L   = [len(seq) for seq in enc_q]
#             yield mx.array(inp), mx.array(tgt), mx.array(L)

# def save_lora_adapters(model, filepath):
#     """
#     Save only the LoRA adapter parameters (A and B) in .npz.
#     """
#     import numpy as np
#     state = {}
#     for name, mod in model.named_modules():
#         if isinstance(mod, LoRALinear):
#             state[f"{name}.A"] = np.array(mod.A.tolist())
#             state[f"{name}.B"] = np.array(mod.B.tolist())
#     np.savez(filepath, **state)

# def load_lora_adapters(model, filepath):
#     """
#     Load LoRA adapter parameters (A and B) from .npz into the model.
#     """
#     import numpy as np
#     data = np.load(filepath)
#     for key, arr in data.items():
#         module_name, param = key.rsplit(".", 1)
#         mod = model
#         for part in module_name.split("."):
#             mod = getattr(mod, part)
#         setattr(mod, param, mx.array(arr))

from __future__ import annotations
import json, os, random
import mlx.core as mx
import mlx.nn   as nn

class LoRALinear(nn.Module):
    def __init__(self, W: mx.array, *, r:int=8, alpha:int|None=None, dropout:float=0.1):
        super().__init__()
        self.W = W                        # frozen base
        self.r = r
        self.alpha = alpha or 2*r
        self.scale = self.alpha / self.r
        self.p   = dropout
        self.A = mx.random.normal((r, W.shape[1]))*0.01
        self.B = mx.random.normal((W.shape[0], r))*0.01
        self.bias = getattr(W, "bias", None)

    def __call__(self, x:mx.array, *, training:bool|None=None) -> mx.array:
        base  = x @ self.W.T
        delta = (x @ self.A.T) @ self.B.T * self.scale
        train_flag = training if training is not None else self.training
        if train_flag and self.p>0.0:
            keep = 1.0 - self.p
            mask = mx.random.bernoulli(p=keep, shape=delta.shape) / keep
            delta *= mask
        out = base + delta
        return out + (self.bias if self.bias is not None else 0.0)

    @classmethod
    def from_linear(cls, lin:nn.Linear, *, r:int=8, alpha:int|None=None, dropout:float=0.1):
        m = cls(lin.weight, r=r, alpha=alpha, dropout=dropout)
        if lin.bias is not None: m.bias = lin.bias
        return m

def load_pairs(path:str):
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                o = json.loads(ln); yield o["question"].strip(), o["answer"].strip()

def save_lora_adapters(model:nn.Module, fp:str):
    import numpy as np, pathlib, os
    state={}
    for n,m in model.named_modules():
        if isinstance(m,LoRALinear):
            state[f"{n}.A"]=np.array(m.A.tolist())
            state[f"{n}.B"]=np.array(m.B.tolist())
    pathlib.Path(fp).parent.mkdir(parents=True,exist_ok=True)
    np.savez(fp,**state)

def load_lora_adapters(model:nn.Module, fp:str):
    import numpy as np, pathlib
    if not pathlib.Path(fp).is_file(): return
    data=np.load(fp)
    for key, arr in data.items():
        mod_name, param = key.rsplit(".", 1)
        mod = model
        for part in mod_name.split("."):
            if part.isdigit():           # ← NEW: list index
                mod = mod[int(part)]
            else:                        # normal attribute
                mod = getattr(mod, part)
        setattr(mod, param, mx.array(arr))