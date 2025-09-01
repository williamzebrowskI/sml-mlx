# model/data_mixer.py
# Robust 50/50 text+code streaming mixer for HF streaming datasets
# - No reliance on .features (which can be None in streaming)
# - Auto-detects text/code field names per example
# - Per-rank shard + shuffle, optional license filter
# - Alternates TEXT batch, then CODE batch (≈50/50 by batches)

from __future__ import annotations
import itertools, random, time
from typing import Iterator, Optional, Set, Dict, Any

import numpy as np
import mlx.core as mx
import sentencepiece as spm
from datasets import load_dataset, DownloadConfig


def _resilient_iter(ds, rank: int, backoff_max: float = 60.0):
    """Yield examples from a streaming dataset, retrying on transient errors."""
    backoff = 2.0
    while True:
        it = iter(ds)
        try:
            for ex in it:
                yield ex
            # If the stream ends, restart it
        except Exception as e:
            sleep_for = min(backoff, backoff_max) * (1.0 + 0.5 * random.random())
            print(f"[data_mixer][Rank {rank}] stream hiccup {type(e).__name__}: {e} → sleep {sleep_for:.1f}s", flush=True)
            time.sleep(sleep_for)
            backoff = min(backoff * 2.0, backoff_max)
            continue


def _extract_text(ex: Dict[str, Any], prefer_code: bool) -> str:
    """Pick the best string field to tokenize without relying on .features."""
    if not isinstance(ex, dict):
        return str(ex)

    # Heuristics: prioritize typical fields for each domain, then fall back to any string
    code_first = ["content", "code", "clean_code", "completion", "target", "docstring", "text"]
    text_first = ["text", "content", "document", "body", "paragraph", "content_text"]
    primary = code_first if prefer_code else text_first
    secondary = text_first if prefer_code else code_first

    for k in primary + secondary:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # Some datasets nest strings under other names; fall back to the first non-empty string value
    for v in ex.values():
        if isinstance(v, str) and v.strip():
            return v

    return ""


def _pack_stream(
    ds_iter: Iterator[Dict[str, Any]],
    sp: spm.SentencePieceProcessor,
    ctx: int,
    bs: int,
    *,
    prefer_code: bool,
) -> Iterator[mx.array]:
    """Tokenize + pack into fixed (bs, ctx+1) windows."""
    window = ctx + 1
    buf: list[int] = []
    for ex in ds_iter:
        txt = _extract_text(ex, prefer_code=prefer_code)
        if not txt:
            continue
        ids = sp.encode(txt, out_type=int, add_bos=True, add_eos=True)
        if not ids:
            continue
        buf.extend(ids)
        while len(buf) >= window * bs:
            arr = np.asarray(buf[: window * bs], dtype=np.int32)
            del buf[: window * bs]
            yield mx.array(arr).reshape(bs, window)


def build_50_50_stream(
    *,
    rank: int,
    size: int,
    ctx_len: int,
    bs: int,
    sp_model_path: str,
    text_ds_id: str,
    code_ds_id: str,
    text_cfg: Optional[str] = None,
    code_cfg: Optional[str] = None,
    text_split: str = "train",
    code_split: str = "train",
    shuffle_buffer: int = 20_000,
    allow_code_licenses: Optional[Set[str]] = None,
) -> Iterator[mx.array]:
    """
    Infinite generator of (bs, ctx_len+1) batches, alternating one TEXT batch
    then one CODE batch. Works with streaming datasets that have features=None.
    """
    dlcfg = DownloadConfig(max_retries=5)

    text = load_dataset(
        text_ds_id, text_cfg, split=text_split, streaming=True,
        download_config=dlcfg, trust_remote_code=True
    )
    code = load_dataset(
        code_ds_id, code_cfg, split=code_split, streaming=True,
        download_config=dlcfg, trust_remote_code=True
    )

    # Per-rank shard + shuffle (contiguous to keep order chunks intact)
    text = text.shard(num_shards=size, index=rank, contiguous=True).shuffle(seed=1234 + rank, buffer_size=shuffle_buffer)
    code = code.shard(num_shards=size, index=rank, contiguous=True).shuffle(seed=5678 + rank, buffer_size=shuffle_buffer)

    # Optional license filter for code (only if allow list is provided and field exists)
    if allow_code_licenses:
        allowed = {s.lower() for s in allow_code_licenses}
        def keep(ex):
            for k in ("license", "licenses", "licence", "license_name"):
                v = ex.get(k)
                if isinstance(v, str) and any(a in v.lower() for a in allowed):
                    return True
            # If no license field, drop it (conservative) — comment next line to allow all
            return False
        code = code.filter(keep)

    it_text = _resilient_iter(text, rank)
    it_code = _resilient_iter(code, rank)

    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    gen_text = _pack_stream(it_text, sp, ctx_len, bs, prefer_code=False)
    gen_code = _pack_stream(it_code, sp, ctx_len, bs, prefer_code=True)

    # Alternate forever; if one side is temporarily empty, yield from the other
    for tbatch, cbatch in itertools.zip_longest(gen_text, gen_code):
        if tbatch is not None:
            yield tbatch
        if cbatch is not None:
            yield cbatch