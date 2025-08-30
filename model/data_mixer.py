# data_mixer.py
from __future__ import annotations
import os, time, random
from typing import Iterator, Dict, Any
import numpy as np
import sentencepiece as spm
from datasets import load_dataset, DownloadConfig
from huggingface_hub.utils import HfHubHTTPError

import mlx.core as mx

def resilient_dataset_iter(ds, rank: int, *, backoff_max: float = 60.0):
    backoff = 2.0
    while True:
        it = iter(ds)
        try:
            for ex in it:
                yield ex
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                sleep_for = min(backoff, backoff_max) * (1.0 + random.random())
                print(f"[{time.strftime('%H:%M:%S')}] [Rank {rank}] HF 429 → sleep {sleep_for:.1f}s", flush=True)
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, backoff_max)
                continue
            if code in {500, 502, 503, 504, None}:
                time.sleep(5.0 * (1.0 + 0.5 * random.random()))
                continue
            raise
        except Exception:
            time.sleep(5.0 * (1.0 + 0.5 * random.random()))
            continue

def encode_sp(example: Dict[str, Any], *, sp: spm.SentencePieceProcessor, key: str):
    ids = sp.encode(example[key], out_type=int, add_bos=True, add_eos=True)
    return {"ids": ids}

def sample_generator(dataset_iter: Iterator[Dict[str, Any]], ctx: int, bs: int) -> Iterator[mx.array]:
    window, buf = ctx + 1, []
    while True:
        for ex in dataset_iter:
            buf.extend(ex["ids"])
            while len(buf) >= window * bs:
                chunk = np.asarray(buf[: window * bs], dtype=np.int32)
                del buf[: window * bs]
                yield mx.array(chunk).reshape(bs, window)

def interleave_batches(gens: Dict[str, Iterator[mx.array]], weights: Dict[str, float]):
    # deterministic cycling by integer batch counts per "epoch" of K batches
    keys = list(gens.keys())
    tot = sum(weights.values())
    counts = {k: max(1, int(round(1000 * (weights[k] / tot)))) for k in keys}
    order = []
    for k in keys:
        order += [k] * counts[k]
    # simple round-robin over this fixed order
    i = 0
    while True:
        k = order[i]
        i = (i + 1) % len(order)
        yield next(gens[k])

def build_50_50_stream(
    *,
    rank: int,
    size: int,
    ctx_len: int,
    bs: int,
    sp_model_path: str,
    text_ds_id: str,
    code_ds_id: str,
    text_split: str = "train",
    code_split: str = "train",
    text_cfg: str | None = None,
    code_cfg: str | None = None,
    shuffle_buffer: int = 20000,
    allow_code_licenses: set[str] | None = None,
):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    dlcfg = DownloadConfig(max_retries=5)

    # TEXT stream
    text = load_dataset(text_ds_id, text_cfg, split=text_split, streaming=True, download_config=dlcfg, trust_remote_code=True)
    text = text.map(lambda ex: encode_sp(ex, sp=sp, key="text"))
    text = text.shard(num_shards=size, index=rank, contiguous=True)
    text = text.shuffle(seed=42 + rank, buffer_size=shuffle_buffer)
    text_iter = resilient_dataset_iter(text, rank)
    gen_text = sample_generator(text_iter, ctx_len, bs)

    # CODE stream
    code = load_dataset(code_ds_id, code_cfg, split=code_split, streaming=True, download_config=dlcfg, trust_remote_code=True)
    # (Optional) license filter if the code dataset exposes a license field.
    if allow_code_licenses is not None:
        def _ok(ex):
            lic = (ex.get("license") or ex.get("licenses") or "").lower()
            return any(x in lic for x in allow_code_licenses)
        code = code.filter(_ok)
    # choose the text field; many code sets use "content" or "text"
    text_key = "content" if "content" in code.features else "text"
    code = code.map(lambda ex: encode_sp(ex, sp=sp, key=text_key))
    code = code.shard(num_shards=size, index=rank, contiguous=True)
    code = code.shuffle(seed=1337 + rank, buffer_size=shuffle_buffer)
    code_iter = resilient_dataset_iter(code, rank)
    gen_code = sample_generator(code_iter, ctx_len, bs)

    # 50/50 by batches → ≈50/50 by tokens (since batches are same size)
    mix = interleave_batches({"text": gen_text, "code": gen_code}, {"text": 0.5, "code": 0.5})
    return mix