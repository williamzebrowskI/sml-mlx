#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
PY="$ROOT/.venv/bin/python"

exec "$PY" "$ROOT/distributed_nanogpt_chat/serve.py" \
  --checkpoint-path "$ROOT/distributed_nanogpt_streaming/checkpoints/run_climbmix_shuffled_v3/final.safetensors" \
  --tokenizer-name gpt2 \
  --temperature 0.8 \
  --top-k 40 \
  --max-tokens 240 \
  --repetition-penalty 1.2 \
  --no-repeat-ngram-size 3 \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
