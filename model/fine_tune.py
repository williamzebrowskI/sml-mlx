#!/usr/bin/env python
"""
Fine-tune TinyGPLM into a general instruction / Q&A assistant using MLX.

This is intended as the FIRST SFT stage after FineWeb pretraining for your 333M base.
It trains on short single-turn (User -> Assistant) examples and masks loss so we
only learn to generate the Assistant portion (prevents the model from emitting "User:").

Default recipe: `allenai/tulu-v2-sft-mixture` (streaming).
Optional recipe: `mix` adds `HuggingFaceH4/ultrachat_200k` for extra chat variety.

Usage (333M base):

  PYTHONPATH=. python -m model.fine_tune \
    --ckpt model/checkpoints_333m/ckpt_final.safetensors \
    --spm-model tokenizer/fineweb_spm/spm.model \
    --config configs/config_mlx_slm_beta_v3_333m.json \
    --recipe tulu_v2 \
    --max-steps 20000 \
    --seq-len 3072 \
    --batch-size 1 \
    --lr 2e-5 \
    --wd 0.01 \
    --save-dir model/checkpoints_sft_333m \
    --save-every 2000
"""

import argparse
import os
import time
import math
import random
import importlib
import itertools
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# Import from your existing model module.
# We keep a handle to the module so globals like TOK update after set_tokenizer.
try:
    mdl = importlib.import_module("model.model")
except ImportError:
    mdl = importlib.import_module("model")

TinyGPLM = mdl.TinyGPLM
TinyGPConfig = mdl.TinyGPConfig
set_tokenizer = mdl.set_tokenizer
load_safetensors_model = mdl.load_safetensors_model
save_safetensors_model = mdl.save_safetensors_model


# ---------------------------------------------------------------------
# Dataset parsing: normalize many common chat/instruction formats into
# a single (user, assistant) pair.
# ---------------------------------------------------------------------

def _norm_role(role: str) -> Optional[str]:
    r = (role or "").strip().lower()
    if r in {"user", "human", "prompt"}:
        return "user"
    if r in {"assistant", "gpt", "bot", "model"}:
        return "assistant"
    if r == "system":
        return "system"
    return None


def _extract_messages(ex: Dict[str, Any]) -> Optional[List[Tuple[str, str]]]:
    raw = None
    if isinstance(ex.get("messages"), list):
        raw = ex.get("messages")
    elif isinstance(ex.get("conversations"), list):
        raw = ex.get("conversations")
    if raw is None:
        return None

    msgs: List[Tuple[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from") or m.get("speaker")
        content = m.get("content") or m.get("value") or m.get("text") or m.get("message")
        if not role or not content:
            continue
        r = _norm_role(str(role))
        if r is None:
            continue
        c = str(content).strip()
        if not c:
            continue
        msgs.append((r, c))
    return msgs or None


def _extract_pair_from_fields(ex: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    # Alpaca-ish
    inst = ex.get("instruction")
    out = ex.get("output")
    if isinstance(inst, str) and isinstance(out, str) and inst.strip() and out.strip():
        inp = ex.get("input")
        user = inst.strip()
        if isinstance(inp, str) and inp.strip():
            user = f"{user}\n{inp.strip()}"
        return user, out.strip()

    # Common single-turn keys
    candidates = [
        ("prompt", "completion"),
        ("prompt", "response"),
        ("question", "answer"),
        ("query", "response"),
        ("input", "output"),
    ]
    for uk, ak in candidates:
        u = ex.get(uk)
        a = ex.get(ak)
        if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip():
            return u.strip(), a.strip()

    return None


def _choose_user_assistant_pair(
    messages: List[Tuple[str, str]],
    *,
    strategy: str,
) -> Optional[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    last_user: Optional[str] = None
    for role, content in messages:
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user is not None:
            pairs.append((last_user, content))
            last_user = None
        else:
            continue

    if not pairs:
        return None
    if strategy == "first":
        return pairs[0]
    if strategy == "last":
        return pairs[-1]
    return random.choice(pairs)


def _extract_pair(ex: Dict[str, Any], *, pair_strategy: str) -> Optional[Tuple[str, str]]:
    msgs = _extract_messages(ex)
    if msgs:
        pair = _choose_user_assistant_pair(msgs, strategy=pair_strategy)
        if pair is not None:
            return pair
    return _extract_pair_from_fields(ex)


# ---------------------------------------------------------------------
# Streaming sources / recipes
# ---------------------------------------------------------------------

def _pair_stream(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    shuffle_buffer: int,
    seed: int,
    trust_remote_code: bool,
    pair_strategy: str,
) -> Iterator[Tuple[str, str]]:
    while True:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
            trust_remote_code=trust_remote_code,
        )
        if shuffle_buffer and shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        for ex in ds:
            pair = _extract_pair(ex, pair_strategy=pair_strategy)
            if pair is None:
                continue
            yield pair


def build_pair_iterator(
    recipe: str,
    *,
    shuffle_buffer: int,
    seed: int,
    trust_remote_code: bool,
    pair_strategy: str,
) -> Iterator[Tuple[str, str]]:
    recipe = recipe.strip().lower()
    if recipe == "tulu_v2":
        return _pair_stream(
            dataset_name="allenai/tulu-v2-sft-mixture",
            dataset_config=None,
            split="train",
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            trust_remote_code=trust_remote_code,
            pair_strategy=pair_strategy,
        )
    if recipe == "ultrachat":
        return _pair_stream(
            dataset_name="HuggingFaceH4/ultrachat_200k",
            dataset_config=None,
            split="train_sft",
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            trust_remote_code=trust_remote_code,
            pair_strategy=pair_strategy,
        )
    if recipe == "mix":
        return _mix_pair_iterator(
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            trust_remote_code=trust_remote_code,
            pair_strategy=pair_strategy,
        )
    raise ValueError(f"Unknown recipe: {recipe!r} (expected: tulu_v2, ultrachat, mix)")


def _mix_pair_iterator(
    *,
    shuffle_buffer: int,
    seed: int,
    trust_remote_code: bool,
    pair_strategy: str,
) -> Iterator[Tuple[str, str]]:
    """
    Weighted mixture of multiple pair streams.

    Important: this is a generator, but `build_pair_iterator` is not, so other
    recipes can safely `return _pair_stream(...)`.
    """
    tulu = build_pair_iterator(
        "tulu_v2",
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        trust_remote_code=trust_remote_code,
        pair_strategy=pair_strategy,
    )
    ultra = build_pair_iterator(
        "ultrachat",
        shuffle_buffer=shuffle_buffer,
        seed=seed + 1,
        trust_remote_code=trust_remote_code,
        pair_strategy=pair_strategy,
    )
    gens = [tulu, ultra]
    weights = [0.8, 0.2]
    while True:
        g = random.choices(gens, weights=weights, k=1)[0]
        yield next(g)

def _recipe_sources(recipe: str) -> List[Tuple[str, Optional[str], str]]:
    r = recipe.strip().lower()
    if r == "tulu_v2":
        return [("allenai/tulu-v2-sft-mixture", None, "train")]
    if r == "ultrachat":
        return [("HuggingFaceH4/ultrachat_200k", None, "train_sft")]
    if r == "mix":
        return [
            ("allenai/tulu-v2-sft-mixture", None, "train"),
            ("HuggingFaceH4/ultrachat_200k", None, "train_sft"),
        ]
    return []


def _debug_dataset_sample(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    trust_remote_code: bool,
    pair_strategy: str,
    n: int = 2,
):
    """
    Best-effort debug helper: print a few raw examples and whether we can extract a (user, assistant) pair.
    """
    try:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        print(f"[sft][debug] load_dataset failed for {dataset_name}/{dataset_config} split={split}: {e!r}", flush=True)
        return

    print(f"[sft][debug] sample from {dataset_name}/{dataset_config} split={split}:", flush=True)
    shown = 0
    try:
        for ex in ds:
            if shown >= n:
                break
            keys = list(ex.keys()) if isinstance(ex, dict) else [type(ex).__name__]
            print(f"[sft][debug] ex[{shown}] keys={keys}", flush=True)
            if isinstance(ex, dict):
                pair = _extract_pair(ex, pair_strategy=pair_strategy)
                if pair is None:
                    print("[sft][debug]  extracted_pair=None", flush=True)
                else:
                    u, a = pair
                    print(f"[sft][debug]  extracted_pair user_len={len(u)} assistant_len={len(a)}", flush=True)
            shown += 1
    except Exception as e:
        print(f"[sft][debug] iterating dataset failed: {e!r}", flush=True)
        return



# ---------------------------------------------------------------------
# Tokenization + assistant-only masking
# ---------------------------------------------------------------------

def _encode_plain(text: str) -> List[int]:
    # SentencePiece ids WITHOUT adding BOS/EOS (we add those explicitly).
    return mdl.TOK.sp.encode(text, out_type=int)


def build_xy_assistant_only(
    user_text: str,
    assistant_text: str,
    *,
    seq_len: int,
    max_user_chars: int = 0,
    max_assistant_chars: int = 0,
    min_assistant_tokens: int = 128,
    auto_truncate_user: bool = True,
    stats: Optional[Dict[str, int]] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    if stats is not None:
        stats["pairs_seen"] = stats.get("pairs_seen", 0) + 1

    if not user_text or not assistant_text:
        if stats is not None:
            stats["drop_empty"] = stats.get("drop_empty", 0) + 1
        return None

    user = user_text.strip()
    assistant = assistant_text.strip()
    if not user or not assistant:
        if stats is not None:
            stats["drop_empty"] = stats.get("drop_empty", 0) + 1
        return None

    if max_user_chars and len(user) > max_user_chars:
        user = user[:max_user_chars]
    if max_assistant_chars and len(assistant) > max_assistant_chars:
        assistant = assistant[:max_assistant_chars]

    # Ensure the prefix fits in the fixed training window while leaving
    # room for (at least) some assistant tokens.
    min_assistant_tokens = max(1, int(min_assistant_tokens))
    prefix_budget = max(1, int(seq_len) - min_assistant_tokens)

    truncated = False
    prefix = f"User: {user}\nAssistant: "
    prefix_ids = _encode_plain(prefix)

    if auto_truncate_user and len(prefix_ids) > prefix_budget:
        # Heuristic: repeatedly shrink the user text until the prefix token count fits.
        # Keep the *start* of the user message (often contains the instruction/question).
        for _ in range(8):
            if len(prefix_ids) <= prefix_budget:
                break
            # Scale user chars down roughly proportional to the token budget.
            scale = prefix_budget / max(1, len(prefix_ids))
            new_len = max(1, int(len(user) * scale * 0.95))
            if new_len >= len(user):
                new_len = max(1, len(user) - 1)
            user = user[:new_len]
            truncated = True
            prefix = f"User: {user}\nAssistant: "
            prefix_ids = _encode_plain(prefix)

    if len(prefix_ids) > prefix_budget:
        if stats is not None:
            stats["drop_prefix_too_long"] = stats.get("drop_prefix_too_long", 0) + 1
        return None

    full = prefix + assistant
    full_ids = _encode_plain(full)

    ids = [mdl.TOK.bos_id] + full_ids + [mdl.TOK.eos_id]
    boundary = 1 + len(prefix_ids)  # first assistant-answer token position in `ids`

    # Truncate to fixed window
    ids = ids[: seq_len + 1]
    if boundary >= len(ids):
        if stats is not None:
            stats["drop_prefix_too_long"] = stats.get("drop_prefix_too_long", 0) + 1
        return None

    x_ids = ids[:-1]
    y_ids = ids[1:]

    # Mask loss on everything up through the prefix (User + "Assistant: ")
    mask_upto = min(max(0, boundary - 1), len(y_ids))
    if mask_upto:
        y_ids[:mask_upto] = [mdl.PAD_ID_RUNTIME] * mask_upto

    # If we ended up masking everything (e.g., assistant got truncated away), drop it.
    if all(t == mdl.PAD_ID_RUNTIME for t in y_ids):
        if stats is not None:
            stats["drop_no_unmasked"] = stats.get("drop_no_unmasked", 0) + 1
        return None

    # Pad to seq_len
    if len(x_ids) < seq_len:
        pad = seq_len - len(x_ids)
        x_ids = x_ids + [mdl.PAD_ID_RUNTIME] * pad
        y_ids = y_ids + [mdl.PAD_ID_RUNTIME] * pad

    if stats is not None:
        stats["examples_kept"] = stats.get("examples_kept", 0) + 1
        if truncated:
            stats["user_truncated"] = stats.get("user_truncated", 0) + 1

    return x_ids, y_ids


def make_batch_iterator(
    pair_iter: Iterator[Tuple[str, str]],
    *,
    seq_len: int,
    batch_size: int,
    max_user_chars: int = 0,
    max_assistant_chars: int = 0,
    min_assistant_tokens: int = 128,
    auto_truncate_user: bool = True,
    stats: Optional[Dict[str, int]] = None,
    debug_every: int = 0,
):
    X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    filled = 0

    if stats is None:
        stats = {}

    for user, assistant in pair_iter:
        xy = build_xy_assistant_only(
            user,
            assistant,
            seq_len=seq_len,
            max_user_chars=max_user_chars,
            max_assistant_chars=max_assistant_chars,
            min_assistant_tokens=min_assistant_tokens,
            auto_truncate_user=auto_truncate_user,
            stats=stats,
        )
        if xy is None:
            if debug_every and stats.get("pairs_seen", 0) % debug_every == 0:
                print(
                    "[sft][data] "
                    f"seen={stats.get('pairs_seen', 0)} kept={stats.get('examples_kept', 0)} "
                    f"drop_empty={stats.get('drop_empty', 0)} "
                    f"drop_prefix_too_long={stats.get('drop_prefix_too_long', 0)} "
                    f"drop_no_unmasked={stats.get('drop_no_unmasked', 0)} "
                    f"user_truncated={stats.get('user_truncated', 0)}",
                    flush=True,
                )
            continue
        x_ids, y_ids = xy
        X[filled] = mx.array(x_ids, dtype=mx.int32)
        Y[filled] = mx.array(y_ids, dtype=mx.int32)
        filled += 1
        if filled == batch_size:
            yield X, Y
            X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            filled = 0


# ---------------------------------------------------------------------
# Fine-tune loop
# ---------------------------------------------------------------------

def finetune_general_sft(
    *,
    ckpt_path: str,
    spm_model: str,
    config_path: str,
    recipe: str,
    max_steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    wd: float,
    log_every: int,
    save_dir: str,
    save_every: int,
    lr_decay_start: Optional[int],
    min_lr_factor: float,
    shuffle_buffer: int,
    shuffle_seed: int,
    trust_remote_code: bool,
    pair_strategy: str,
    max_user_chars: int,
    max_assistant_chars: int,
    min_assistant_tokens: int,
    auto_truncate_user: bool,
    debug_data_every: int,
    resume_ckpt: Optional[str],
    resume_step: Optional[int],
    skip_batches: int,
):
    set_tokenizer(spm_model)

    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0 (got {max_steps})")

    print(
        f"[sft] recipe={recipe} pair_strategy={pair_strategy} seq_len={seq_len} "
        f"bs={batch_size} lr={lr:.2e} wd={wd:.2e} max_steps={max_steps}",
        flush=True,
    )
    print(
        f"[sft] dataset shuffle_buffer={shuffle_buffer} seed={shuffle_seed} trust_remote_code={trust_remote_code}",
        flush=True,
    )
    print(
        f"[sft] data auto_truncate_user={auto_truncate_user} min_assistant_tokens={min_assistant_tokens} "
        f"max_user_chars={max_user_chars} max_assistant_chars={max_assistant_chars}",
        flush=True,
    )

    # Build model config from JSON (required for 333M checkpoints)
    cfg = mdl.load_tinygplm_config(config_path, vocab_size=mdl.VOCAB_SIZE, max_seq=seq_len)
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    print(f"[sft] model initialized (vocab={cfg.vocab_size}, d_model={cfg.d_model}, layers={cfg.n_layers})")

    ckpt_to_load = resume_ckpt if resume_ckpt else ckpt_path
    if not os.path.exists(ckpt_to_load):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_to_load}")
    ok = load_safetensors_model(ckpt_to_load, model)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_to_load}")
    mx.eval(model.parameters())
    print(f"[sft] loaded checkpoint from {ckpt_to_load}")

    # LR schedule
    opt = optim.AdamW(lr, weight_decay=wd)
    base_lr = lr
    decay_start = lr_decay_start if lr_decay_start is not None else max_steps

    def get_lr(step: int) -> float:
        if step < decay_start or decay_start >= max_steps:
            return base_lr
        t = (step - decay_start) / max(1, max_steps - decay_start)
        t = min(max(t, 0.0), 1.0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))
        factor = min_lr_factor + (1.0 - min_lr_factor) * cos_factor
        return base_lr * factor

    def loss_fn(x, y):
        return model(x, y)["loss"]

    step_fn = nn.value_and_grad(model, loss_fn)

    pair_iter = build_pair_iterator(
        recipe,
        shuffle_buffer=shuffle_buffer,
        seed=shuffle_seed,
        trust_remote_code=trust_remote_code,
        pair_strategy=pair_strategy,
    )

    # Optional: sanity-check that the pair stream yields at least one example.
    if debug_data_every and debug_data_every > 0:
        try:
            pair_iter = iter(pair_iter)
            first_pair = next(pair_iter)
            u0, a0 = first_pair
            print(f"[sft][debug] first_pair user_chars={len(u0)} assistant_chars={len(a0)}", flush=True)

            tmp_stats: Dict[str, int] = {}
            ok_xy = build_xy_assistant_only(
                u0,
                a0,
                seq_len=seq_len,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                min_assistant_tokens=min_assistant_tokens,
                auto_truncate_user=auto_truncate_user,
                stats=tmp_stats,
            )
            print(
                "[sft][debug] first_pair_xy="
                + ("ok" if ok_xy is not None else "dropped")
                + f" stats={tmp_stats}",
                flush=True,
            )

            pair_iter = itertools.chain([first_pair], pair_iter)
        except Exception as e:
            print(f"[sft][debug] failed to fetch first_pair: {e!r}", flush=True)

    data_stats: Dict[str, int] = {}
    batch_iter = make_batch_iterator(
        pair_iter,
        seq_len=seq_len,
        batch_size=batch_size,
        max_user_chars=max_user_chars,
        max_assistant_chars=max_assistant_chars,
        min_assistant_tokens=min_assistant_tokens,
        auto_truncate_user=auto_truncate_user,
        stats=data_stats,
        debug_every=debug_data_every,
    )

    update_step = 0
    if resume_step is not None:
        update_step = max(0, int(resume_step))
    elif resume_ckpt:
        stem = os.path.splitext(os.path.basename(resume_ckpt))[0]
        parts = stem.split("_")
        if parts and parts[-1].isdigit():
            update_step = int(parts[-1])
    if update_step > 0:
        print(f"[sft] resuming at step {update_step}")

    if skip_batches > 0:
        skipped = 0
        for _ in range(skip_batches):
            try:
                next(batch_iter)
                skipped += 1
            except StopIteration:
                break
        print(f"[sft] skipped {skipped} batches to approximate dataset position")

    # Warm up the data pipeline so we fail loudly instead of silently doing 0 steps.
    batch_iter = iter(batch_iter)
    try:
        first_batch = next(batch_iter)
    except StopIteration as e:
        print(
            "[sft][data] "
            f"seen={data_stats.get('pairs_seen', 0)} kept={data_stats.get('examples_kept', 0)} "
            f"drop_empty={data_stats.get('drop_empty', 0)} "
            f"drop_prefix_too_long={data_stats.get('drop_prefix_too_long', 0)} "
            f"drop_no_unmasked={data_stats.get('drop_no_unmasked', 0)} "
            f"user_truncated={data_stats.get('user_truncated', 0)}",
            flush=True,
        )
        for ds_name, ds_cfg, ds_split in _recipe_sources(recipe):
            _debug_dataset_sample(
                dataset_name=ds_name,
                dataset_config=ds_cfg,
                split=ds_split,
                trust_remote_code=trust_remote_code,
                pair_strategy=pair_strategy,
                n=2,
            )
        raise RuntimeError(
            f"[sft] No training batches were produced from recipe={recipe!r}. "
            "This usually means the dataset stream couldn't be read, or no examples "
            "matched the expected schema, or your prompts are too long for the window. "
            "Try `--shuffle-buffer 0` and/or `--max-user-chars 2048`."
        ) from e

    last_log_time = time.time()
    trained_any = False
    for X, Y in itertools.chain([first_batch], batch_iter):
        if update_step >= max_steps:
            break

        opt.learning_rate = get_lr(update_step)

        loss, grads = step_fn(X, Y)
        mx.eval(loss, grads)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

        update_step += 1
        trained_any = True

        if update_step % log_every == 0:
            now = time.time()
            dt = now - last_log_time
            last_log_time = now
            toks = batch_size * seq_len
            toks_per_s = toks / max(1e-9, dt)
            ppl = float(mx.exp(loss).item()) if float(loss.item()) < 20 else float("inf")
            lr_cur = getattr(opt, "learning_rate", None)
            lr_str = f" lr={lr_cur:.2e}" if lr_cur is not None else ""
            print(f"[{update_step}] loss={loss.item():.4f} ppl={ppl:.2f}{lr_str} tok/s≈{toks_per_s:.0f}")

        if update_step > 0 and save_every > 0 and update_step % save_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"sft_{update_step:06d}.safetensors")
            ok = save_safetensors_model(out_path, model)
            if ok:
                print(f"[{update_step}] saved checkpoint to {out_path}")
            else:
                print(f"[{update_step}] ⚠️ failed to save checkpoint to {out_path}")

    if not trained_any:
        raise RuntimeError("[sft] No optimizer steps were run (unexpected).")

    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "sft_final.safetensors")
    ok = save_safetensors_model(final_path, model)
    if ok:
        print(f"[sft] saved final checkpoint to {final_path}")
    else:
        print(f"[sft] ⚠️ failed to save final checkpoint to {final_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Fine-tune TinyGPLM on general instruction/chat SFT (assistant-only loss)")
    p.add_argument("--ckpt", type=str, required=True, help="Base .safetensors checkpoint to fine-tune")
    p.add_argument("--spm-model", type=str, required=True, help="Path to SentencePiece model (spm.model)")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TinyGPLM JSON model config (must match the checkpoint).",
    )

    p.add_argument("--recipe", type=str, default="tulu_v2", help="Dataset recipe: tulu_v2 | ultrachat | mix")
    p.add_argument("--pair-strategy", type=str, default="random", help="Pair selection: first | last | random")
    p.add_argument("--shuffle-buffer", type=int, default=0, help="Streaming shuffle buffer (0 disables)")
    p.add_argument("--shuffle-seed", type=int, default=42, help="Shuffle seed")
    p.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to HF datasets (use --no-trust-remote-code to disable).",
    )

    p.add_argument("--max-steps", type=int, default=20_000, help="Total update steps for SFT")
    p.add_argument("--seq-len", type=int, default=3072, help="Sequence length for fine-tuning")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW")
    p.add_argument("--wd", type=float, default=0.01, help="Weight decay for AdamW")
    p.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    p.add_argument("--save-dir", type=str, default="model/checkpoints_sft_333m", help="Where to store SFT checkpoints")
    p.add_argument("--save-every", type=int, default=2000, help="Save checkpoint every N steps (0 = only final)")

    p.add_argument("--lr-decay-start", type=int, default=None, help="Step at which to start cosine LR decay (None = no decay)")
    p.add_argument("--min-lr-factor", type=float, default=0.1, help="Minimum LR fraction at end of decay")

    p.add_argument("--max-user-chars", type=int, default=0, help="Optional char cap for user text (0 = no cap)")
    p.add_argument("--max-assistant-chars", type=int, default=0, help="Optional char cap for assistant text (0 = no cap)")
    p.add_argument(
        "--min-assistant-tokens",
        type=int,
        default=128,
        help="When truncating long prompts, try to leave at least this many assistant tokens in-window.",
    )
    p.add_argument(
        "--auto-truncate-user",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-truncate overly-long user prompts so assistant tokens fit in the window.",
    )
    p.add_argument(
        "--debug-data-every",
        type=int,
        default=0,
        help="If >0, print data pipeline stats every N pairs while searching for batches.",
    )

    p.add_argument("--resume", type=str, default=None, help="Optional .safetensors checkpoint to resume from")
    p.add_argument("--resume-step", type=int, default=None, help="Override resume step (otherwise inferred from filename)")
    p.add_argument("--skip-batches", type=int, default=0, help="Best-effort: skip N streamed batches before training")

    args = p.parse_args()

    finetune_general_sft(
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        config_path=args.config,
        recipe=args.recipe,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        log_every=args.log_every,
        save_dir=args.save_dir,
        save_every=args.save_every,
        lr_decay_start=args.lr_decay_start,
        min_lr_factor=args.min_lr_factor,
        shuffle_buffer=args.shuffle_buffer,
        shuffle_seed=args.shuffle_seed,
        trust_remote_code=args.trust_remote_code,
        pair_strategy=args.pair_strategy,
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
        min_assistant_tokens=args.min_assistant_tokens,
        auto_truncate_user=args.auto_truncate_user,
        debug_data_every=args.debug_data_every,
        resume_ckpt=args.resume,
        resume_step=args.resume_step,
        skip_batches=args.skip_batches,
    )


if __name__ == "__main__":
    main()
