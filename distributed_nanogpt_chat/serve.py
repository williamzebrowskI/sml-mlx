#!/usr/bin/env python3
"""nanochat-style web UI for the final GPT-2 MLX checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
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
DEFAULT_SYSTEM_PROMPT = "You are a friendly, helpful AI assistant. Answer briefly, clearly, and directly. If the user simply greets you, reply with a short friendly greeting and ask how you can help. Do not repeat the user's question."
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model_id: Optional[str] = None
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
    parts: List[str] = [f"System: {DEFAULT_SYSTEM_PROMPT}"]
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


def _detect_conversation_format(metadata: Dict[str, Any]) -> str:
    cfg_path = str(metadata.get("args", {}).get("config", ""))
    if "distributed_nanochat_gpt2_sft" in cfg_path:
        return "sft"
    return "plain"


def _build_prompt_ids(
    messages: Iterable[ChatMessage],
    *,
    tokenizer: AutoTokenizer,
    conversation_format: str,
) -> List[int]:
    prompt_text = _build_prompt_text(messages, conversation_format=conversation_format)
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def _build_prompt_text(
    messages: Iterable[ChatMessage],
    *,
    conversation_format: str,
) -> str:
    if conversation_format == "plain":
        return _build_prompt(messages)

    parts: List[str] = []

    def add(text: str) -> None:
        parts.append(text)

    injected_system = False
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "user":
            if not injected_system:
                content = f"{DEFAULT_SYSTEM_PROMPT}\n\nUser request:\n{content}"
                injected_system = True
            add(USER_START)
            add(content)
            add(USER_END)
        elif message.role == "assistant":
            add(ASSISTANT_START)
            add(content)
            add(ASSISTANT_END)
    add(ASSISTANT_START)
    return "".join(parts)


def _truncate_control_text(text: str) -> str:
    stop_at = None
    for marker in CONTROL_MARKERS + ("\nUser:", "\nAssistant:"):
        idx = text.find(marker)
        if idx != -1:
            stop_at = idx if stop_at is None else min(stop_at, idx)
    return text if stop_at is None else text[:stop_at]


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
class ModelSettings:
    model_id: str
    label: str
    checkpoint_path: Path
    tokenizer_name: str
    conversation_format: str


@dataclass
class GenerationSettings:
    models: List[ModelSettings]
    default_model_id: str
    temperature: float
    top_k: int
    max_tokens: int
    repetition_penalty: float
    penalty_window: int
    no_repeat_ngram_size: int
    host: str
    port: int


class LocalCheckpointEngine:
    def __init__(self, model_settings: ModelSettings, generation_settings: GenerationSettings):
        self.model_settings = model_settings
        self.generation_settings = generation_settings
        self.request_count = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_settings.tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        metadata = _load_checkpoint_metadata(model_settings.checkpoint_path)
        model_args = metadata.get("model_args") or metadata.get("config") or {}
        self.model = GPT2LM(GPT2Config(**model_args))
        flat = mx.load(str(model_settings.checkpoint_path))
        self.model.update(_tree_assign(self.model.parameters(), flat, prefix="model"))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.metadata = metadata
        self.conversation_format = (
            model_settings.conversation_format
            if model_settings.conversation_format != "auto"
            else _detect_conversation_format(metadata)
        )
        print(
            (
                f"[model-loaded] model_id={self.model_settings.model_id} label={self.model_settings.label} "
                f"checkpoint={self.model_settings.checkpoint_path} "
                f"tokenizer={self.model_settings.tokenizer_name} format={self.conversation_format} "
                f"layers={self.model.config.n_layer} heads={self.model.config.n_head} "
                f"embd={self.model.config.n_embd} block={self.model.config.block_size} "
                f"params_non_embedding={count_parameters(self.model, non_embedding=True)}"
            ),
            flush=True,
        )

    def _next_request_id(self) -> int:
        self.request_count += 1
        return self.request_count

    def _log_request(
        self,
        *,
        request_id: int,
        messages: List[ChatMessage],
        temperature: float,
        top_k: int,
        max_tokens: int,
    ) -> None:
        print(f"[request {request_id}] messages={len(messages)}", flush=True)
        print(
            (
                f"[request-model {request_id}] model_id={self.model_settings.model_id} "
                f"label={self.model_settings.label} checkpoint={self.model_settings.checkpoint_path}"
            ),
            flush=True,
        )
        print(
            (
                f"[decode {request_id}] temperature={temperature} top_k={top_k} "
                f"max_tokens={max_tokens} repetition_penalty={self.generation_settings.repetition_penalty} "
                f"no_repeat_ngram_size={self.generation_settings.no_repeat_ngram_size}"
            ),
            flush=True,
        )
        for idx, message in enumerate(messages, start=1):
            wrapped = textwrap.shorten(message.content.replace("\n", "\\n"), width=500, placeholder="...")
            print(f"[message {request_id}.{idx}] role={message.role} content={wrapped}", flush=True)

    def _log_prompt(
        self,
        *,
        request_id: int,
        prompt_text: str,
        prompt_ids: List[int],
    ) -> None:
        print(
            (
                f"[prompt-meta {request_id}] format={self.conversation_format} "
                f"prompt_chars={len(prompt_text)} prompt_tokens={len(prompt_ids)}"
            ),
            flush=True,
        )
        print(f"[prompt {request_id}]\n{prompt_text}\n[/prompt {request_id}]", flush=True)

    def _log_response(
        self,
        *,
        request_id: int,
        visible_text: str,
        response_ids: List[int],
        stop_reason: str,
    ) -> None:
        print(
            (
                f"[response-meta {request_id}] response_chars={len(visible_text)} "
                f"response_tokens={len(response_ids)} stop_reason={stop_reason}"
            ),
            flush=True,
        )
        print(f"[response {request_id}]\n{visible_text}\n[/response {request_id}]", flush=True)

    @property
    def stats(self) -> Dict[str, Any]:
        cfg = self.model.config
        return {
            "model_id": self.model_settings.model_id,
            "label": self.model_settings.label,
            "checkpoint": str(self.model_settings.checkpoint_path),
            "tokenizer": self.model_settings.tokenizer_name,
            "params_non_embedding": count_parameters(self.model, non_embedding=True),
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "iter_num": self.metadata.get("iter_num"),
            "world": self.metadata.get("world"),
            "conversation_format": self.conversation_format,
        }

    def stream_completion(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float,
        top_k: int,
        max_tokens: int,
    ) -> Iterable[str]:
        request_id = self._next_request_id()
        self._log_request(
            request_id=request_id,
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        prompt_text = _build_prompt_text(
            messages,
            conversation_format=self.conversation_format,
        )
        prompt_ids = _build_prompt_ids(
            messages,
            tokenizer=self.tokenizer,
            conversation_format=self.conversation_format,
        )
        if not prompt_ids:
            raise ValueError("Prompt tokenized to empty sequence")

        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        if bos_id is not None:
            prompt_ids = [int(bos_id)] + prompt_ids
        self._log_prompt(request_id=request_id, prompt_text=prompt_text, prompt_ids=prompt_ids)

        eos_token_id = self.tokenizer.eos_token_id
        generated_ids = list(prompt_ids)
        response_ids: List[int] = []
        last_clean_text = ""
        assistant_end_ids = self.tokenizer.encode(ASSISTANT_END, add_special_tokens=False)
        stop_reason = "max_tokens"

        for _ in range(max_tokens):
            ctx_ids = generated_ids[-self.model.config.block_size :]
            x = mx.array(np.asarray([ctx_ids], dtype=np.int32), dtype=mx.int32)
            logits = self.model.logits(x)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            next_id = _sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=self.generation_settings.repetition_penalty,
                penalty_window=self.generation_settings.penalty_window,
                generated_ids=generated_ids,
                no_repeat_ngram_size=self.generation_settings.no_repeat_ngram_size,
            )

            if eos_token_id is not None and next_id == eos_token_id:
                stop_reason = "eos_token"
                break

            generated_ids.append(next_id)
            response_ids.append(next_id)
            if assistant_end_ids and response_ids[-len(assistant_end_ids) :] == assistant_end_ids:
                stop_reason = "assistant_end"
                break

            current_text = self.tokenizer.decode(response_ids, clean_up_tokenization_spaces=False)
            if current_text.endswith("\ufffd"):
                continue

            visible_text = _truncate_control_text(current_text)

            new_text = visible_text[len(last_clean_text) :]
            if new_text:
                yield f"data: {json.dumps({'token': new_text}, ensure_ascii=False)}\n\n"
                last_clean_text = visible_text

            if visible_text != current_text:
                stop_reason = "control_marker"
                break

        self._log_response(
            request_id=request_id,
            visible_text=last_clean_text,
            response_ids=response_ids,
            stop_reason=stop_reason,
        )
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"


def create_app(settings: GenerationSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engines = {
            model.model_id: LocalCheckpointEngine(model, settings) for model in settings.models
        }
        app.state.generation_lock = asyncio.Lock()
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
        models_payload = [
            {
                "id": model.model_id,
                "label": model.label,
                "checkpoint": str(model.checkpoint_path),
                "checkpoint_name": model.checkpoint_path.name,
            }
            for model in settings.models
        ]
        default_model = next(model for model in settings.models if model.model_id == settings.default_model_id)
        html = html.replace("__DEFAULT_TEMPERATURE__", f"{settings.temperature:.2f}")
        html = html.replace("__DEFAULT_TOP_K__", str(settings.top_k))
        html = html.replace("__DEFAULT_MAX_TOKENS__", str(settings.max_tokens))
        html = html.replace("__DEFAULT_MODEL_ID__", settings.default_model_id)
        html = html.replace("__DEFAULT_MODEL_LABEL__", default_model.label)
        html = html.replace("__MODEL_OPTIONS_JSON__", json.dumps(models_payload))
        return HTMLResponse(content=html)

    @app.get("/logo.svg")
    async def logo() -> FileResponse:
        return FileResponse(APP_DIR / "logo.svg", media_type="image/svg+xml")

    @app.post("/chat/completions")
    async def chat_completions(request: ChatRequest) -> StreamingResponse:
        validate_chat_request(request)
        model_id = request.model_id if request.model_id is not None else settings.default_model_id
        engines: Dict[str, LocalCheckpointEngine] = app.state.engines
        if model_id not in engines:
            raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")
        engine = engines[model_id]
        temperature = request.temperature if request.temperature is not None else settings.temperature
        top_k = request.top_k if request.top_k is not None else settings.top_k
        max_tokens = request.max_tokens if request.max_tokens is not None else settings.max_tokens

        async def event_stream() -> AsyncGenerator[str, None]:
            async with app.state.generation_lock:
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
        engines: Dict[str, LocalCheckpointEngine] = app.state.engines
        return {
            "status": "ok",
            "ready": True,
            "default_model_id": settings.default_model_id,
            "models": [engine.stats for engine in engines.values()],
        }

    @app.get("/models")
    async def models() -> Dict[str, Any]:
        engines: Dict[str, LocalCheckpointEngine] = app.state.engines
        return {
            "default_model_id": settings.default_model_id,
            "models": [engine.stats for engine in engines.values()],
        }

    @app.get("/stats")
    async def stats() -> Dict[str, Any]:
        engines: Dict[str, LocalCheckpointEngine] = app.state.engines
        return {
            "default_model_id": settings.default_model_id,
            "models": [engine.stats for engine in engines.values()],
        }

    return app


def parse_args() -> GenerationSettings:
    parser = argparse.ArgumentParser(description="Serve a nanochat-style UI for the final GPT-2 checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ROOT / "distributed_nanogpt_streaming" / "checkpoints" / "run_climbmix_shuffled_v3" / "final.safetensors",
    )
    parser.add_argument("--checkpoint-id", type=str, default="pretrain")
    parser.add_argument("--checkpoint-label", type=str, default="Pretrain")
    parser.add_argument("--alternate-checkpoint-path", type=Path, default=None)
    parser.add_argument("--alternate-id", type=str, default="sft")
    parser.add_argument("--alternate-label", type=str, default="SFT")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--alternate-tokenizer-name", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=240)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--penalty-window", type=int, default=128)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--conversation-format", type=str, default="auto", choices=["auto", "plain", "sft"])
    parser.add_argument("--alternate-conversation-format", type=str, default="auto", choices=["auto", "plain", "sft"])
    parser.add_argument("--default-model-id", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    models = [
        ModelSettings(
            model_id=args.checkpoint_id,
            label=args.checkpoint_label,
            checkpoint_path=args.checkpoint_path.resolve(),
            tokenizer_name=args.tokenizer_name,
            conversation_format=args.conversation_format,
        )
    ]
    if args.alternate_checkpoint_path is not None:
        models.append(
            ModelSettings(
                model_id=args.alternate_id,
                label=args.alternate_label,
                checkpoint_path=args.alternate_checkpoint_path.resolve(),
                tokenizer_name=args.alternate_tokenizer_name or args.tokenizer_name,
                conversation_format=args.alternate_conversation_format,
            )
        )

    ids = [model.model_id for model in models]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate model ids are not allowed: {ids}")

    default_model_id = args.default_model_id or models[0].model_id
    if default_model_id not in ids:
        raise ValueError(f"default_model_id must be one of {ids}, got {default_model_id}")

    return GenerationSettings(
        models=models,
        default_model_id=default_model_id,
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
