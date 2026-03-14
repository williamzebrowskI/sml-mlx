#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
PY="$ROOT/.venv/bin/python"

exec "$PY" "$ROOT/distributed_nanogpt_chat/serve.py" \
  --checkpoint-path "$ROOT/distributed_nanochat_gpt2_sft/checkpoints/sft_snapshot_large_v1/final.safetensors" \
  --checkpoint-id "sft" \
  --checkpoint-label "SFT final" \
  --alternate-checkpoint-path "$ROOT/distributed_nanogpt_streaming/checkpoints/run_climbmix_shuffled_v3/final.safetensors" \
  --alternate-id "pretrain" \
  --alternate-label "Pretrain final" \
  --default-model-id "sft" \
  --tokenizer-name gpt2 \
  --conversation-format auto \
  --alternate-conversation-format auto \
  --temperature 0.2 \
  --top-k 10 \
  --max-tokens 64 \
  --repetition-penalty 1.35 \
  --no-repeat-ngram-size 4 \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
