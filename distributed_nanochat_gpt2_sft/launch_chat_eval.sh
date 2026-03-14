#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
exec "$ROOT/.venv/bin/python" \
  "$ROOT/distributed_nanochat_gpt2_sft/chat_eval_gpt2.py" \
  --checkpoint-dir "$ROOT/distributed_nanochat_gpt2_sft/checkpoints/sft_snapshot_large_v1" \
  "$@"
