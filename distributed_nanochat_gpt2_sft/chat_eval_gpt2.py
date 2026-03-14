#!/usr/bin/env python3
"""Run a nanochat-style post-SFT eval suite on a GPT-2 MLX checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distributed_nanogpt_streaming.model import GPT2Config, GPT2LM
from distributed_nanochat_gpt2_sft.tasks import MMLU_LETTERS, SourceSpec, SpellingBeeCursor, render_mc


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
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
QA_PROBES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("What is 2 + 2?", ("4", "four")),
    ("Who wrote Hamlet?", ("william shakespeare", "shakespeare")),
    ("What is the capital of France?", ("paris",)),
    ("What planet do humans live on?", ("earth",)),
    ("What color is the sky on a clear day?", ("blue",)),
)


@dataclass
class TaskScore:
    name: str
    correct: int
    total: int
    accuracy: float
    samples: List[Dict[str, Any]]


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


def _resolve_checkpoint(checkpoint: str | None, checkpoint_dir: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if not checkpoint_dir:
        raise ValueError("Either --checkpoint or --checkpoint-dir must be provided.")
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    final_path = root / "final.safetensors"
    if final_path.exists():
        return final_path
    step_paths = sorted(root.glob("step_*.safetensors"))
    if not step_paths:
        raise FileNotFoundError(f"No step checkpoints found in {root}")
    return step_paths[-1]


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


def _encode_prompt(tokenizer: AutoTokenizer, question: str) -> List[int]:
    prompt = _build_prompt(question)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if bos_id is not None:
        prompt_ids = [int(bos_id)] + prompt_ids
    if not prompt_ids:
        raise ValueError("Prompt tokenized to empty sequence")
    return [int(x) for x in prompt_ids]


def _generate_answer(
    *,
    model: GPT2LM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    out_ids = _encode_prompt(tokenizer, question)
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


def _log_softmax_row(logits: mx.array) -> np.ndarray:
    row = np.asarray(logits[0], dtype=np.float32)
    row = row - np.max(row)
    row = row - math.log(np.exp(row).sum())
    return row


def _score_completion_tokens(
    *,
    model: GPT2LM,
    prompt_ids: Sequence[int],
    candidate_ids: Sequence[int],
) -> float:
    ctx = list(int(x) for x in prompt_ids)
    total = 0.0
    for token_id in candidate_ids:
        x = mx.array(np.asarray([ctx[-model.config.block_size :]], dtype=np.int32), dtype=mx.int32)
        logits = model.logits(x)[:, -1, :].astype(mx.float32)
        mx.eval(logits)
        total += float(_log_softmax_row(logits)[int(token_id)])
        ctx.append(int(token_id))
    return total


def _predict_mc_letter(
    *,
    model: GPT2LM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    prompt_ids = _encode_prompt(tokenizer, prompt)
    candidates = {}
    for letter in MMLU_LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if not ids:
            continue
        candidates[letter] = _score_completion_tokens(model=model, prompt_ids=prompt_ids, candidate_ids=ids)
    return max(candidates.items(), key=lambda item: item[1])[0]


def _extract_final_number(text: str) -> str | None:
    values = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if not values:
        return None
    return values[-1].replace(",", "")


def _extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    number = _extract_final_number(text)
    return number or ""


def _extract_spelling_bee_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    number = _extract_final_number(text)
    return number or ""


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _evaluate_mmlu(
    *,
    model: GPT2LM,
    tokenizer: AutoTokenizer,
    limit: int,
) -> TaskScore:
    ds = load_dataset("cais/mmlu", "all", split="test")
    total = min(limit, len(ds))
    correct = 0
    samples: List[Dict[str, Any]] = []
    for idx in range(total):
        row = ds[idx]
        prompt = render_mc(str(row["question"]).strip(), MMLU_LETTERS, [str(x).strip() for x in row["choices"]])
        pred = _predict_mc_letter(model=model, tokenizer=tokenizer, prompt=prompt)
        gold = MMLU_LETTERS[int(row["answer"])]
        is_correct = pred == gold
        correct += int(is_correct)
        if len(samples) < 3:
            samples.append({"question": row["question"], "pred": pred, "gold": gold, "correct": is_correct})
    return TaskScore("mmlu", correct, total, correct / max(1, total), samples)


def _evaluate_gsm8k(
    *,
    model: GPT2LM,
    tokenizer: AutoTokenizer,
    limit: int,
    max_new_tokens: int,
) -> TaskScore:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    total = min(limit, len(ds))
    correct = 0
    samples: List[Dict[str, Any]] = []
    for idx in range(total):
        row = ds[idx]
        pred_text = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=str(row["question"]).strip(),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
        )
        pred = _extract_gsm8k_answer(pred_text)
        gold = _extract_gsm8k_answer(str(row["answer"]))
        is_correct = bool(pred) and pred == gold
        correct += int(is_correct)
        if len(samples) < 3:
            samples.append(
                {
                    "question": row["question"],
                    "pred": pred,
                    "gold": gold,
                    "correct": is_correct,
                    "text": pred_text,
                }
            )
    return TaskScore("gsm8k", correct, total, correct / max(1, total), samples)


def _evaluate_spelling_bee(
    *,
    model: GPT2LM,
    tokenizer: AutoTokenizer,
    limit: int,
    max_new_tokens: int,
) -> TaskScore:
    spec = SourceSpec(kind="spelling_bee", split="test", weight=1)
    cursor = SpellingBeeCursor(spec, split="test")
    correct = 0
    samples: List[Dict[str, Any]] = []
    for _ in range(limit):
        row = cursor.next_conversation()
        question = str(row["messages"][0]["content"]).strip()
        gold_text = str(row["messages"][1]["content"][0]["text"])
        gold = _extract_spelling_bee_answer(gold_text)
        pred_text = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
        )
        pred = _extract_spelling_bee_answer(pred_text)
        is_correct = bool(pred) and pred == gold
        correct += int(is_correct)
        if len(samples) < 3:
            samples.append({"question": question, "pred": pred, "gold": gold, "correct": is_correct, "text": pred_text})
    return TaskScore("spelling_bee", correct, limit, correct / max(1, limit), samples)


def _evaluate_qa_probes(
    *,
    model: GPT2LM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> TaskScore:
    correct = 0
    samples: List[Dict[str, Any]] = []
    total = len(QA_PROBES)
    for question, answers in QA_PROBES:
        pred_text = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
        )
        normalized = _normalize_text(pred_text)
        is_correct = any(answer in normalized for answer in answers)
        correct += int(is_correct)
        samples.append({"question": question, "pred": pred_text, "gold": list(answers), "correct": is_correct})
    return TaskScore("qa_probes", correct, total, correct / max(1, total), samples)


def _chatcore_like(scores: Dict[str, TaskScore]) -> float:
    # Mirror nanochat's spirit: normalize multiple task scores and average them.
    mmlu = scores["mmlu"].accuracy
    mmlu_norm = max(0.0, min(1.0, (mmlu - 0.25) / 0.75))
    gsm = scores["gsm8k"].accuracy
    spelling = scores["spelling_bee"].accuracy
    qa = scores["qa_probes"].accuracy
    return 100.0 * float(np.mean([mmlu_norm, gsm, spelling, qa]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a nanochat-style eval suite on an MLX GPT-2 SFT checkpoint")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/checkpoints/sft_snapshot_large_v1",
    )
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--mmlu-limit", type=int, default=64)
    parser.add_argument("--gsm8k-limit", type=int, default=32)
    parser.add_argument("--spelling-limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint or None, args.checkpoint_dir or None)
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

    scores = {
        "mmlu": _evaluate_mmlu(model=model, tokenizer=tokenizer, limit=args.mmlu_limit),
        "gsm8k": _evaluate_gsm8k(model=model, tokenizer=tokenizer, limit=args.gsm8k_limit, max_new_tokens=args.max_new_tokens),
        "spelling_bee": _evaluate_spelling_bee(
            model=model,
            tokenizer=tokenizer,
            limit=args.spelling_limit,
            max_new_tokens=args.max_new_tokens,
        ),
        "qa_probes": _evaluate_qa_probes(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens),
    }
    chatcore_like = _chatcore_like(scores)

    report = {
        "checkpoint": str(checkpoint_path),
        "tokenizer_name": args.tokenizer_name,
        "limits": {
            "mmlu": args.mmlu_limit,
            "gsm8k": args.gsm8k_limit,
            "spelling_bee": args.spelling_limit,
            "qa_probes": len(QA_PROBES),
        },
        "scores": {name: asdict(score) for name, score in scores.items()},
        "chatcore_like": chatcore_like,
    }

    print(f"[checkpoint] {checkpoint_path}")
    for name, score in scores.items():
        print(f"[{name}] {score.correct}/{score.total} acc={score.accuracy:.4f}")
    print(f"[chatcore_like] {chatcore_like:.2f}")

    output_json = args.output_json.strip()
    if output_json:
        output_path = Path(output_json).expanduser().resolve()
    else:
        output_path = checkpoint_path.with_suffix(".chat_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"[report] {output_path}")


if __name__ == "__main__":
    main()
