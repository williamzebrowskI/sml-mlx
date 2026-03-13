#!/usr/bin/env python3
"""Run a fixed set of basic prompts against an SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distributed_nanogpt_streaming.model import GPT2Config, GPT2LM


USER_START = "<|user_start|>\n"
USER_END = "\n<|user_end|>\n"
ASSISTANT_START = "<|assistant_start|>\n"
ASSISTANT_END = "\n<|assistant_end|>\n"
CONTROL_MARKERS = (
    "<|assistant_end|>",
    "<|assistant_start|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)

DEFAULT_QUESTIONS = [
    "What is 2 + 2?",
    "Who wrote Hamlet?",
    "What is the capital of France?",
    "Why do cars overheat?",
    "A good teacher should",
]


def _tree_assign(template: Any, flat: Dict[str, mx.array], prefix: str = "") -> Any:
    if isinstance(template, dict):
        return {k: _tree_assign(v, flat, f"{prefix}.{k}" if prefix else k) for k, v in template.items()}
    if isinstance(template, list):
        return [_tree_assign(v, flat, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(template)]
    if isinstance(template, tuple):
        return tuple(_tree_assign(v, flat, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(template))
    if isinstance(template, mx.array):
        key = prefix if prefix else "param"
        if key not in flat:
            return template
        value = flat[key]
        if value.shape != template.shape:
            raise ValueError(f"Shape mismatch for {key}: file={value.shape} model={template.shape}")
        return value
    return template


def _load_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    meta_path = Path(str(checkpoint_path) + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(question: str) -> str:
    return USER_START + question.strip() + USER_END + ASSISTANT_START


def _clean_output(text: str) -> str:
    stop_at = None
    for marker in CONTROL_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            stop_at = idx if stop_at is None else min(stop_at, idx)
    cleaned = text if stop_at is None else text[:stop_at]
    return " ".join(cleaned.split()).strip()


def _sample_next_token(logits: np.ndarray, *, temperature: float, top_k: int) -> int:
    if temperature <= 0.0 or top_k == 1:
        return int(np.argmax(logits))
    scaled = logits / max(temperature, 1e-5)
    if top_k > 0 and top_k < scaled.shape[0]:
        keep = np.argpartition(scaled, -top_k)[-top_k:]
        masked = np.full_like(scaled, -np.inf)
        masked[keep] = scaled[keep]
        scaled = masked
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / np.clip(probs.sum(), 1e-12, None)
    return int(np.random.choice(probs.shape[0], p=probs))


def _generate_answer(
    *,
    model: GPT2LM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt = _build_prompt(question)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if bos_id is not None:
        prompt_ids = [int(bos_id)] + prompt_ids
    if not prompt_ids:
        raise ValueError("Prompt tokenized to empty sequence")

    out_ids = list(int(x) for x in prompt_ids)
    generated_ids: List[int] = []
    assistant_end_ids = tokenizer.encode(ASSISTANT_END, add_special_tokens=False)

    for _ in range(max_new_tokens):
        ctx = out_ids[-model.config.block_size :]
        x = mx.array(np.asarray([ctx], dtype=np.int32), dtype=mx.int32)
        logits = model.logits(x)[:, -1, :].astype(mx.float32)
        mx.eval(logits)
        next_id = _sample_next_token(np.asarray(logits[0], dtype=np.float32), temperature=temperature, top_k=top_k)
        out_ids.append(next_id)
        generated_ids.append(next_id)
        if assistant_end_ids and generated_ids[-len(assistant_end_ids) :] == assistant_end_ids:
            break

    text = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)
    return _clean_output(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a GPT-2 SFT checkpoint with a few basic questions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/checkpoints/sft_snapshot_small_v1/step_0000250.safetensors",
    )
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--question", action="append", default=[])
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    metadata = _load_metadata(checkpoint_path)
    config_data = metadata.get("config") or metadata.get("model_args")
    if not config_data:
        raise RuntimeError("Checkpoint metadata is missing model config")

    model = GPT2LM(GPT2Config(**config_data))
    flat = mx.load(str(checkpoint_path))
    model.update(_tree_assign(model.parameters(), flat, prefix="model"))
    model.eval()
    mx.eval(model.parameters())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    questions = args.question or list(DEFAULT_QUESTIONS)
    print(f"[checkpoint] {checkpoint_path}")
    print(f"[tokenizer] {args.tokenizer_name}")
    print(f"[decode] temperature={args.temperature} top_k={args.top_k} max_new_tokens={args.max_new_tokens}")

    for i, question in enumerate(questions, start=1):
        answer = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print()
        print(f"[q{i}] {question}")
        print(f"[a{i}] {answer}")


if __name__ == "__main__":
    main()
