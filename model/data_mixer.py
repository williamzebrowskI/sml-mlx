# model/data_mixer.py
# Robust 50/50 text+code streaming mixer for HF streaming datasets
# - No reliance on .features (which can be None in streaming)
# - Auto-detects text/code field names per *example* (works across datasets)
# - Optional language filter for code (language/path/extension)
# - Per-rank shard + shuffle, optional license allowlist (non-strict)
# - Alternates TEXT batch, then CODE batch (≈50/50 by batches)

from __future__ import annotations
import itertools, random, time, os
from typing import Iterator, Optional, Set, Dict, Any, Tuple

import numpy as np
import mlx.core as mx
import sentencepiece as spm
from datasets import load_dataset, DownloadConfig


# Heuristics for extracting the text field from arbitrary dataset rows
CODE_LIKE_KEYS = ["content", "code", "clean_code", "completion", "target", "docstring", "text"]
TEXT_LIKE_KEYS = ["text", "content", "document", "body", "paragraph", "content_text", "article"]

LICENSE_KEYS = ("license", "licenses", "licence", "license_name", "license_label")
LANG_KEYS = ("language", "lang")
PATH_KEYS = ("path", "filepath", "file_path", "repo_path", "relative_path")

DEFAULT_CODE_EXTS = (".py",)  # only Python by default


def _resilient_iter(ds, rank: int, backoff_max: float = 60.0):
    """Yield examples from a streaming dataset, retrying on transient errors."""
    backoff = 2.0
    while True:
        it = iter(ds)
        try:
            for ex in it:
                yield ex
            # If the stream ends (rare in streaming), just loop to reopen.
        except Exception as e:
            sleep_for = min(backoff, backoff_max) * (1.0 + 0.5 * random.random())
            print(f"[data_mixer][Rank {rank}] stream hiccup {type(e).__name__}: {e} → sleep {sleep_for:.1f}s", flush=True)
            time.sleep(sleep_for)
            backoff = min(backoff * 2.0, backoff_max)
            continue


def _first_nonempty_str(ex: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _infer_language(ex: Dict[str, Any]) -> Optional[str]:
    # Try explicit language keys
    lang = _first_nonempty_str(ex, LANG_KEYS)
    if lang:
        return lang.strip().lower()
    # Guess from file extension
    path = _first_nonempty_str(ex, PATH_KEYS)
    if path and isinstance(path, str):
        _, dot, ext = path.rpartition(".")
        if dot and ext:
            return ext.lower()
    return None


def _has_allowed_license(ex: Dict[str, Any], allowed: Set[str]) -> bool:
    """Return True if license is acceptable; if no license field, allow (non-strict)."""
    for k in LICENSE_KEYS:
        v = ex.get(k)
        if isinstance(v, str):
            low = v.lower()
            if any(a in low for a in allowed):
                return True
            # license present but not in allowlist → reject
            return False
    # No license information → allow (flip to strict by returning False here)
    return True


def _extract_text(ex: Dict[str, Any], prefer_code: bool) -> str:
    """Pick the best string field to tokenize without relying on .features."""
    if not isinstance(ex, dict):
        return str(ex)

    primary = CODE_LIKE_KEYS if prefer_code else TEXT_LIKE_KEYS
    secondary = TEXT_LIKE_KEYS if prefer_code else CODE_LIKE_KEYS

    for k in list(primary) + list(secondary):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # Fall back to any non-empty string value
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
    min_chars: int = 1,
) -> Iterator[mx.array]:
    """Tokenize + pack into fixed (bs, ctx+1) windows."""
    window = ctx + 1
    buf: list[int] = []
    for ex in ds_iter:
        txt = _extract_text(ex, prefer_code=prefer_code)
        if not txt or len(txt) < min_chars:
            continue
        ids = sp.encode(txt, out_type=int, add_bos=True, add_eos=True)
        if not ids:
            continue
        buf.extend(ids)
        while len(buf) >= window * bs:
            arr = np.asarray(buf[: window * bs], dtype=np.int32)
            del buf[: window * bs]
            yield mx.array(arr).reshape(bs, window)


def _wrap_with_filters(
    ds,
    *,
    rank: int,
    shuffle_buffer: int,
    seed: int,
    size: int,
    prefer_code: bool,
    allow_code_licenses: Optional[Set[str]] = None,
    code_lang: Optional[str] = "python",
    code_exts: Tuple[str, ...] = DEFAULT_CODE_EXTS,
):
    """
    Apply per-rank sharding, shuffling, and (for code) optional filters:
    - license allowlist (non-strict; allows missing license fields),
    - language filter via 'language' field or filename extension.
    """
    ds = ds.shard(num_shards=size, index=rank, contiguous=True).shuffle(
        seed=seed + rank, buffer_size=shuffle_buffer
    )

    if prefer_code:
        # License allowlist (if provided)
        if allow_code_licenses:
            allowed = {s.lower() for s in allow_code_licenses}
            ds = ds.filter(lambda ex: _has_allowed_license(ex, allowed))

        # Language filter (default python)
        if code_lang or code_exts:
            lang_norm = (code_lang or "").strip().lower()
            exts = tuple(e.lower() for e in (code_exts or ()))
            def _keep_lang(ex):
                lang = _infer_language(ex)
                if lang_norm:
                    # Treat 'py' and 'python' as the same
                    if lang in (lang_norm, "py") or (lang_norm in ("python",) and lang == "py"):
                        return True
                # If not from explicit language, try path extension
                path = _first_nonempty_str(ex, PATH_KEYS) or ""
                path_low = path.lower()
                if exts and any(path_low.endswith(ext) for ext in exts):
                    return True
                # If nothing matches, reject
                return False
            ds = ds.filter(_keep_lang)

    return ds


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
    code_lang: Optional[str] = "python",
    code_exts: Tuple[str, ...] = DEFAULT_CODE_EXTS,
    min_chars: int = 1,
) -> Iterator[mx.array]:
    """
    Infinite generator of (bs, ctx_len+1) batches, alternating one TEXT batch
    then one CODE batch. Works with streaming datasets that have features=None.

    Notes:
    - `allow_code_licenses`: if provided, we *accept* rows with no license field.
      Rows with a license string that doesn't match are dropped.
    - `code_lang`/`code_exts`: default keeps Python (language='python' or '.py' path).
      Set `code_lang=None` and `code_exts=()` to allow all languages.
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

    # Per-rank shard + shuffle + optional filters
    text = _wrap_with_filters(
        text, rank=rank, shuffle_buffer=shuffle_buffer, seed=1234, size=size,
        prefer_code=False, allow_code_licenses=None, code_lang=None, code_exts=()
    )
    code = _wrap_with_filters(
        code, rank=rank, shuffle_buffer=shuffle_buffer, seed=5678, size=size,
        prefer_code=True, allow_code_licenses=allow_code_licenses,
        code_lang=code_lang, code_exts=code_exts
    )

    # Build resilient iterators
    it_text = _resilient_iter(text, rank)
    it_code = _resilient_iter(code, rank)

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    # Generators (tokenize + pack)
    gen_text = _pack_stream(it_text, sp, ctx_len, bs, prefer_code=False, min_chars=min_chars)
    gen_code = _pack_stream(it_code, sp, ctx_len, bs, prefer_code=True, min_chars=min_chars)

    # Friendly one-time logs (rank 0 only)
    if rank == 0:
        al = sorted(list({a.lower() for a in (allow_code_licenses or set())})) if allow_code_licenses else "none"
        print(
            f"[Mixer] text_ds='{text_ds_id}' code_ds='{code_ds_id}' "
            f"code_lang='{code_lang or 'any'}' code_exts={code_exts or 'any'} "
            f"licenses={al} shuffle_buffer={shuffle_buffer}", flush=True
        )

    # Alternate forever; if one side is temporarily empty, yield from the other
    for tbatch, cbatch in itertools.zip_longest(gen_text, gen_code):
        if tbatch is not None:
            yield tbatch
        if cbatch is not None:
            yield cbatch