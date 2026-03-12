#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
exec "$ROOT/.venv/bin/mlx.launch" \
  --hostfile "$ROOT/distributed_nanogpt_pretrain/hosts.json" \
  --backend ring \
  --python "$ROOT/.venv/bin/python" \
  -- "$ROOT/distributed_nanogpt_pretrain/train_gpt2.py" \
  --config "$ROOT/distributed_nanogpt_pretrain/smoke_config.json" \
  "$@"
