#!/usr/bin/env python
"""
Greedy-only chat server that reuses model.finetune_eval (unchanged) to generate responses.

- Calls the existing finetune_eval.eval_checkpoint under the hood and captures the greedy output.
- Serves a simple chat UI at /chat.
- Only greedy decoding; no sampling controls.

Run (from repo root):
    PYTHONPATH=. python -m chat.chat_service \
        --ckpt model/checkpoints_smolsmoltalk_sft/smolsmoltalk_sft_125000.safetensors \
        --spm-model tokenizer/fineweb_spm/spm.model \
        --seq-len 2048 \
        --host 0.0.0.0 \
        --port 8000

Env overrides also work:
    CHAT_CKPT_PATH=... CHAT_SPM_MODEL=... CHAT_SEQ_LEN=2048 uvicorn chat.chat_service:app --reload
"""

import argparse
import asyncio
import importlib
import io
import os
import contextlib
import json
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx  # type: ignore
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Keep handle to finetune_eval and model for tokenizer setup.
finetune_eval = importlib.import_module("model.finetune_eval")
try:
    mdl = importlib.import_module("model.model")
except ImportError:
    mdl = importlib.import_module("model")
set_tokenizer = mdl.set_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = os.getenv(
    "CHAT_CKPT_PATH",
    str(ROOT / "model" / "checkpoints_smolsmoltalk_sft" / "smolsmoltalk_sft_005000.safetensors"),
)
DEFAULT_SPM = os.getenv("CHAT_SPM_MODEL", str(ROOT / "tokenizer" / "fineweb_spm" / "spm.model"))
DEFAULT_SEQ_LEN = int(os.getenv("CHAT_SEQ_LEN", "2048"))
DEFAULT_CONFIG = os.getenv("CHAT_CONFIG", None)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


def run_finetune_eval_greedy(
    ckpt_path: str,
    spm_model: str,
    seq_len: int,
    max_new_tokens: int,
    prompt: str,
    config_path: str | None = None,
) -> str:
    """
    Invoke model.finetune_eval.eval_checkpoint (unchanged) and capture the greedy block.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        finetune_eval.eval_checkpoint(
            ckpt_path=ckpt_path,
            spm_model=spm_model,
            seq_len=seq_len,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
            config_path=config_path,
        )
    out = buf.getvalue()
    if "[greedy]" not in out:
        return out.strip()
    chunk = out.split("[greedy]", 1)[1]
    if "---" in chunk:
        chunk = chunk.split("---", 1)[0]
    return chunk.strip()


def _extract_assistant_only(full_text: str, prompt: str) -> str:
    """
    Extract only the assistant's reply from a decoded string that may include the prompt.
    Strategy:
      1) If there's an "Assistant:" marker, take everything after the last one.
      2) Else, drop the prompt prefix if it matches.
      3) Trim leading/trailing whitespace.
    """
    reply = full_text
    last_assistant = reply.rfind("Assistant:")
    if last_assistant != -1:
        reply = reply[last_assistant + len("Assistant:") :]
    elif prompt and reply.startswith(prompt):
        reply = reply[len(prompt) :]
    return reply.strip()


def build_app(
    ckpt_path: str = DEFAULT_CKPT,
    spm_model: str = DEFAULT_SPM,
    seq_len: int = DEFAULT_SEQ_LEN,
    config_path: str | None = DEFAULT_CONFIG,
) -> FastAPI:
    app = FastAPI(title="SmolSmolTalk Chat Service", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.ckpt_path = ckpt_path
    app.state.spm_model = spm_model
    app.state.seq_len = seq_len
    app.state.config_path = config_path

    web_dir = Path(__file__).parent
    if web_dir.exists():
        app.mount("/chat", StaticFiles(directory=web_dir, html=True), name="chat")

    @app.get("/")
    async def root():
        # Redirect root to the chat UI for convenience
        return RedirectResponse(url="/chat/")

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/apple-touch-icon.png")
    async def apple_touch_icon():
        return Response(status_code=204)

    @app.get("/apple-touch-icon-precomposed.png")
    async def apple_touch_icon_pre():
        return Response(status_code=204)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "ckpt": app.state.ckpt_path,
            "spm_model": app.state.spm_model,
            "max_seq": app.state.seq_len,
            "config": app.state.config_path,
        }

    @app.post("/generate")
    async def generate(body: GenerateRequest):
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            run_finetune_eval_greedy,
            app.state.ckpt_path,
            app.state.spm_model,
            app.state.seq_len,
            body.max_new_tokens,
            body.prompt,
            app.state.config_path,
        )
        return JSONResponse(
            {
                "prompt": body.prompt,
                "completion": text,
                "greedy": True,
            }
        )

    @app.websocket("/ws")
    async def websocket_chat(ws: WebSocket):
        await ws.accept()
        while True:
            try:
                init = await ws.receive_json()
            except WebSocketDisconnect:
                return
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Invalid payload: {e}"})
                continue

            message = init.get("message") or init.get("prompt") or ""
            if not message.strip():
                await ws.send_json({"type": "error", "message": "Message cannot be empty"})
                continue

            max_new_tokens = int(init.get("max_new_tokens", 200))

            await ws.send_json(
                {
                    "type": "ready",
                    "max_new_tokens": max_new_tokens,
                    "greedy": True,
                }
            )

            try:
                # Single-turn prompt (no history); append Assistant: if user didn't
                prompt = message.rstrip()
                if not prompt.endswith("Assistant:"):
                    if prompt.endswith("\n"):
                        prompt = prompt + "Assistant: "
                    else:
                        prompt = prompt + "\nAssistant: "

                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    run_finetune_eval_greedy,
                    app.state.ckpt_path,
                    app.state.spm_model,
                    app.state.seq_len,
                    max_new_tokens,
                    prompt,
                    app.state.config_path,
                )
                clean = _extract_assistant_only(text, prompt)
                if not clean:
                    clean = text.strip()
                await ws.send_json({"type": "token", "text": clean, "done": True})
            except WebSocketDisconnect:
                return
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            finally:
                await ws.send_json({"type": "done"})

    return app


# App instance for uvicorn chat.chat_service:app
app = build_app()


def parse_args():
    p = argparse.ArgumentParser("Serve Smol-SmolTalk SFT via FastAPI + WebSocket")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to .safetensors checkpoint")
    p.add_argument("--spm-model", type=str, default=DEFAULT_SPM, help="Path to SentencePiece model")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Max sequence length")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Optional path to JSON config")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args.ckpt, args.spm_model, args.seq_len, args.config)
    uvicorn.run(app, host=args.host, port=args.port)
