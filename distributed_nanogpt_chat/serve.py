#!/usr/bin/env python3
"""nanochat-style web UI for the final GPT-2 MLX checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from distributed_nanogpt_streaming.model import GPT2Config, GPT2LM, count_parameters

MAX_MESSAGES_PER_REQUEST = 200
MAX_MESSAGE_LENGTH = 12000
MAX_TOTAL_CONVERSATION_LENGTH = 48000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 1024


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


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


def _apply_repetition_penalty_np(
    logits_np: np.ndarray,
    generated_ids: List[int],
    repetition_penalty: float,
    penalty_window: Optional[int],
) -> np.ndarray:
    if not generated_ids or repetition_penalty <= 1.0:
        return logits_np

    window_ids = generated_ids[-penalty_window:] if penalty_window else generated_ids
    counts = Counter(window_ids)
    for token_id, count in counts.items():
        if 0 <= token_id < logits_np.shape[0]:
            logits_np[token_id] /= repetition_penalty**count
    return logits_np


def _banned_tokens_for_no_repeat_ngram(sequence: List[int], ngram_size: int) -> set[int]:
    if ngram_size <= 1 or len(sequence) < ngram_size - 1:
        return set()
    prefix = tuple(sequence[-(ngram_size - 1) :])
    banned: set[int] = set()
    for i in range(len(sequence) - ngram_size + 1):
        ngram = sequence[i : i + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            banned.add(int(ngram[-1]))
    return banned


def _sample_next_token(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    penalty_window: Optional[int],
    generated_ids: List[int],
    no_repeat_ngram_size: int,
) -> int:
    logits_np = np.array(logits[0], dtype=np.float32)
    logits_np = _apply_repetition_penalty_np(
        logits_np,
        generated_ids=generated_ids,
        repetition_penalty=repetition_penalty,
        penalty_window=penalty_window,
    )

    for token_id in _banned_tokens_for_no_repeat_ngram(generated_ids, no_repeat_ngram_size):
        if 0 <= token_id < logits_np.shape[0]:
            logits_np[token_id] = -np.inf

    if temperature <= 0.0 or top_k == 1:
        return int(np.argmax(logits_np))

    scaled = logits_np / max(temperature, 1e-5)
    if top_k > 0 and top_k < scaled.shape[0]:
        keep = np.argpartition(scaled, -top_k)[-top_k:]
        masked = np.full_like(scaled, -np.inf)
        masked[keep] = scaled[keep]
        scaled = masked
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / np.clip(probs.sum(), 1e-12, None)
    return int(np.random.choice(probs.shape[0], p=probs))


def _load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    meta_path = Path(str(checkpoint_path) + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(messages: Iterable[ChatMessage]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "user":
            parts.append(f"User: {content}")
        elif message.role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def validate_chat_request(request: ChatRequest) -> None:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_MESSAGES_PER_REQUEST} messages per request")

    total_length = 0
    for index, message in enumerate(request.messages):
        if message.role not in {"user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"Message {index} has invalid role")
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {index} has empty content")
        if len(message.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {index} exceeds {MAX_MESSAGE_LENGTH} chars")
        total_length += len(message.content)
    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Conversation exceeds {MAX_TOTAL_CONVERSATION_LENGTH} chars")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")
    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}")


@dataclass
class GenerationSettings:
    checkpoint_path: Path
    tokenizer_name: str
    temperature: float
    top_k: int
    max_tokens: int
    repetition_penalty: float
    penalty_window: int
    no_repeat_ngram_size: int
    host: str
    port: int


class LocalCheckpointEngine:
    def __init__(self, settings: GenerationSettings):
        self.settings = settings
        self.lock = asyncio.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        metadata = _load_checkpoint_metadata(settings.checkpoint_path)
        model_args = metadata.get("model_args", {})
        self.model = GPT2LM(GPT2Config(**model_args))
        flat = mx.load(str(settings.checkpoint_path))
        self.model.update(_tree_assign(self.model.parameters(), flat, prefix="model"))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.metadata = metadata

    @property
    def stats(self) -> Dict[str, Any]:
        cfg = self.model.config
        return {
            "checkpoint": str(self.settings.checkpoint_path),
            "tokenizer": self.settings.tokenizer_name,
            "params_non_embedding": count_parameters(self.model, non_embedding=True),
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "iter_num": self.metadata.get("iter_num"),
            "world": self.metadata.get("world"),
        }

    def stream_completion(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float,
        top_k: int,
        max_tokens: int,
    ) -> Iterable[str]:
        prompt = _build_prompt(messages)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("Prompt tokenized to empty sequence")

        eos_token_id = self.tokenizer.eos_token_id
        generated_ids = list(prompt_ids)
        response_ids: List[int] = []
        last_clean_text = ""
        stop_markers = ("\nUser:", "\nAssistant:")

        for _ in range(max_tokens):
            ctx_ids = generated_ids[-self.model.config.block_size :]
            x = mx.array(np.asarray([ctx_ids], dtype=np.int32), dtype=mx.int32)
            logits = self.model.logits(x)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            next_id = _sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=self.settings.repetition_penalty,
                penalty_window=self.settings.penalty_window,
                generated_ids=generated_ids,
                no_repeat_ngram_size=self.settings.no_repeat_ngram_size,
            )

            if eos_token_id is not None and next_id == eos_token_id:
                break

            generated_ids.append(next_id)
            response_ids.append(next_id)

            current_text = self.tokenizer.decode(response_ids, clean_up_tokenization_spaces=False)
            if current_text.endswith("\ufffd"):
                continue

            stop_at = None
            for marker in stop_markers:
                idx = current_text.find(marker)
                if idx != -1:
                    stop_at = idx if stop_at is None else min(stop_at, idx)
            visible_text = current_text if stop_at is None else current_text[:stop_at]

            new_text = visible_text[len(last_clean_text) :]
            if new_text:
                yield f"data: {json.dumps({'token': new_text}, ensure_ascii=False)}\n\n"
                last_clean_text = visible_text

            if stop_at is not None:
                break

        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"


def create_app(settings: GenerationSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = LocalCheckpointEngine(settings)
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> HTMLResponse:
        ui_path = APP_DIR / "ui.html"
        with ui_path.open("r", encoding="utf-8") as f:
            html = f.read()
        html = html.replace("__DEFAULT_TEMPERATURE__", f"{settings.temperature:.2f}")
        html = html.replace("__DEFAULT_TOP_K__", str(settings.top_k))
        html = html.replace("__DEFAULT_MAX_TOKENS__", str(settings.max_tokens))
        html = html.replace("__MODEL_LABEL__", settings.checkpoint_path.name)
        return HTMLResponse(content=html)

    @app.get("/logo.svg")
    async def logo() -> FileResponse:
        return FileResponse(APP_DIR / "logo.svg", media_type="image/svg+xml")

    @app.post("/chat/completions")
    async def chat_completions(request: ChatRequest) -> StreamingResponse:
        validate_chat_request(request)
        engine: LocalCheckpointEngine = app.state.engine
        temperature = request.temperature if request.temperature is not None else settings.temperature
        top_k = request.top_k if request.top_k is not None else settings.top_k
        max_tokens = request.max_tokens if request.max_tokens is not None else settings.max_tokens

        async def event_stream() -> AsyncGenerator[str, None]:
            async with engine.lock:
                for chunk in engine.stream_completion(
                    request.messages,
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=max_tokens,
                ):
                    yield chunk
                    await asyncio.sleep(0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        engine: LocalCheckpointEngine = app.state.engine
        return {"status": "ok", "ready": True, **engine.stats}

    @app.get("/stats")
    async def stats() -> Dict[str, Any]:
        engine: LocalCheckpointEngine = app.state.engine
        return engine.stats

    return app


def parse_args() -> GenerationSettings:
    parser = argparse.ArgumentParser(description="Serve a nanochat-style UI for the final GPT-2 checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ROOT / "distributed_nanogpt_streaming" / "checkpoints" / "run_climbmix_shuffled_v3" / "final.safetensors",
    )
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=240)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--penalty-window", type=int, default=128)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    return GenerationSettings(
        checkpoint_path=args.checkpoint_path.resolve(),
        tokenizer_name=args.tokenizer_name,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        penalty_window=args.penalty_window,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        host=args.host,
        port=args.port,
    )


def main() -> None:
    settings = parse_args()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
