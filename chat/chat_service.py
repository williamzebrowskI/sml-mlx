#!/usr/bin/env python
"""
Chat server for SmolSmolTalk-100M using the SAME greedy decoding
logic as model.finetune_eval, but without appending '\\nAssistant:'.

Key behaviours:

- Loads model/model.py once (100M config via JSON).
- For each request, builds a prompt:
    - If your text starts with "User:", we send it as-is.
    - Otherwise we send: "User: <your text>" (no trailing "Assistant:").
- Greedy decoding uses:
    - mdl._greedy_with_repetition_penalty
    - repetition_penalty = 1.5
    - penalty_window     = 128
    - max_repeat         = 8
    - stop_phrases       = ["\\nUser:", "User:"]
- After decoding, we extract only the assistant's reply:
    - If there's an "Assistant:" marker, we take everything after the *first* one.
    - Otherwise, if the decoded string starts with the prompt, we strip that prefix.
    - Then we strip any trailing "\\nUser:" or "User:" at the end.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List

import mlx.core as mx  # type: ignore
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from model import model as mdl  # type: ignore

TinyGPLM = mdl.TinyGPLM
TinyGPConfig = mdl.TinyGPConfig
set_tokenizer = mdl.set_tokenizer
load_safetensors_model = mdl.load_safetensors_model

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CKPT = os.getenv(
    "CHAT_CKPT_PATH",
    str(
        ROOT
        / "model"
        / "checkpoints_smolsmoltalk_sft_100m"
        / "smolsmoltalk_sft_020000.safetensors"
    ),
)
DEFAULT_SPM = os.getenv(
    "CHAT_SPM_MODEL",
    str(ROOT / "tokenizer" / "fineweb_spm" / "spm.model"),
)
DEFAULT_SEQ_LEN = int(os.getenv("CHAT_SEQ_LEN", "3072"))
DEFAULT_CONFIG = os.getenv(
    "CHAT_CONFIG",
    str(ROOT / "configs" / "config_mlx_slm_beta_v2_100m.json"),
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80


# ---------- Model loading ----------


def load_model_once(
    ckpt_path: str,
    spm_model: str,
    seq_len: int,
    config_path: Optional[str],
) -> TinyGPLM:
    """
    Load tokenizer, config, and model once at startup.
    """
    set_tokenizer(spm_model)

    cfg_kwargs = {
        "vocab_size": mdl.TOK.vocab_size,
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
        "max_seq": seq_len,
        "max_grad_norm": 1.0,
    }

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        for k in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq", "max_grad_norm"):
            if k in cfg_json:
                cfg_kwargs[k] = cfg_json[k]
        if seq_len:
            cfg_kwargs["max_seq"] = min(seq_len, cfg_kwargs.get("max_seq", seq_len))

    cfg = TinyGPConfig(**cfg_kwargs)
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ok = load_safetensors_model(ckpt_path, model)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}")
    mx.eval(model.parameters())

    print(
        f"[serve] loaded {ckpt_path} "
        f"(vocab={cfg.vocab_size}, d_model={cfg.d_model}, "
        f"heads={cfg.n_heads}, layers={cfg.n_layers}, max_seq={cfg.max_seq})"
    )
    return model


# ---------- Prompt normalization (NO '\\nAssistant:') ----------


def normalize_chat_prompt_no_assistant(user_text: str) -> str:
    """
    Normalize any input into a one-line user turn *without* appending 'Assistant:'.

    Rules:
      - If it already starts with 'User:', keep it as-is.
      - Otherwise, prepend 'User: '.
    """
    txt = user_text.strip()
    if txt.startswith("User:"):
        return txt
    return f"User: {txt}"


# ---------- Greedy decoding (same as finetune_eval’s greedy) ----------


def generate_greedy_local(
    model: TinyGPLM,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float = 1.5,
    penalty_window: int = 128,
    max_repeat: int = 8,
    stop_phrases: Optional[List[str]] = None,
) -> str:
    """
    Greedy decode using mdl._greedy_with_repetition_penalty.

    Stop when:
      - eos_id
      - token repeats max_repeat times in a row
      - decoded text endswith any stop phrase
    """
    if stop_phrases is None:
        stop_phrases = ["\nUser:", "User:"]

    # Inference prompt encoding: BOS + tokens, no EOS.
    ids = mdl.encode_prompt(prompt)[: model.cfg.max_seq]
    out_ids = ids[:]
    last_id = None
    repeat_run = 0

    for _ in range(max_new_tokens):
        x_ids = mx.array([out_ids[: model.cfg.max_seq]], dtype=mx.int32)
        logits = mdl.TinyGPLM.logits(model, x_ids)  # call instance method explicitly
        last_logits = logits[:, -1, :]

        next_id = mdl._greedy_with_repetition_penalty(
            last_logits,
            generated_ids=out_ids,
            repetition_penalty=repetition_penalty,
            penalty_window=penalty_window,
        )

        out_ids.append(next_id)
        repeat_run = repeat_run + 1 if next_id == last_id else 1
        last_id = next_id

        text_so_far = mdl.TOK.decode(out_ids)
        if (
            next_id == mdl.TOK.eos_id
            or repeat_run >= max_repeat
            or any(text_so_far.endswith(s) for s in stop_phrases)
        ):
            break

    return mdl.TOK.decode(out_ids)


def extract_assistant_only(full_text: str, prompt: str) -> str:
    """
    Extract only the assistant's reply from full decoded text.

    For outputs like:
      'User: ...\\nAssistant: name? Assistant: The capital of France is Paris.\\nUser:'

    Strategy:
      1) Find the FIRST 'Assistant:'; take everything after that.
      2) If not found, but full_text starts with the prompt, strip the prompt.
      3) Strip trailing '\\nUser:' or 'User:' at the end.
      4) Trim whitespace.
    """
    txt = full_text.strip()

    # 1) Try to find the first "Assistant:"
    first = txt.find("Assistant:")
    if first != -1:
        reply = txt[first + len("Assistant:") :]
    else:
        # 2) Fallback: strip the prompt prefix if it matches
        if prompt and txt.startswith(prompt):
            reply = txt[len(prompt) :]
        else:
            reply = txt

    # 3) Strip trailing User markers if they appear at the end or near the end
    for marker in ["\nUser:", "User:"]:
        idx = reply.find(marker)
        if idx != -1:
            reply = reply[:idx]
            break

    return reply.strip()


# ---------- FastAPI app ----------


def build_app(
    ckpt_path: str = DEFAULT_CKPT,
    spm_model: str = DEFAULT_SPM,
    seq_len: int = DEFAULT_SEQ_LEN,
    config_path: Optional[str] = DEFAULT_CONFIG,
) -> FastAPI:
    app = FastAPI(title="SmolSmolTalk-100M Chat Service", version="0.3.0")

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
    app.state.model = load_model_once(ckpt_path, spm_model, seq_len, config_path)

    web_dir = Path(__file__).parent
    if web_dir.exists():
        app.mount("/chat", StaticFiles(directory=web_dir, html=True), name="chat")

    @app.get("/")
    async def root():
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
        max_new = min(body.max_new_tokens, 200)
        user_text = body.prompt.strip()

        # Normalize to "User: ..." *without* appending "Assistant:"
        prompt = normalize_chat_prompt_no_assistant(user_text)

        full = generate_greedy_local(
            app.state.model,
            prompt,
            max_new_tokens=max_new,
            repetition_penalty=1.5,
            penalty_window=128,
        )
        reply = extract_assistant_only(full, prompt)
        print(f"[chat] full_decoded={full!r}")

        return JSONResponse(
            {
                "prompt": prompt,
                "completion": reply,
                "full_decoded": full,
                "mode": "greedy_1.5",
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
                await ws.send_multipart(
                    [{"type": "error", "message": "Message cannot be empty"}]
                )
                continue

            max_new_tokens = min(int(init.get("max_new_tokens", 80)), 200)

            await ws.send_json(
                {
                    "type": "ready",
                    "max_new_tokens": max_new_tokens,
                    "mode": "greedy_1.5",
                }
            )

            try:
                user_text = message.strip()
                prompt = normalize_chat_prompt_no_assistant(user_text)

                full = generate_greedy_local(
                    app.state.model,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.5,
                    penalty_window=128,
                )
                reply = extract_assistant_only(full, prompt)
                print(f"[chat] full_decoded={full!r}")
                if not reply:
                    reply = full.strip()

                await ws.send_json({"type": "token", "text": reply, "done": True})
            except WebSocketDisconnect:
                return
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            finally:
                await ws.send_json({"type": "done"})

    return app


def parse_args():
    p = argparse.ArgumentParser("Serve SmolSmolTalk-100M SFT via FastAPI + WebSocket")
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
else:
    app = build_app()
