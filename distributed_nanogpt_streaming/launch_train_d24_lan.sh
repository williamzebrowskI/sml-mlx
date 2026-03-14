#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
exec "$ROOT/.venv/bin/mlx.launch" \
  --hostfile "$ROOT/distributed_nanogpt_streaming/hosts_lan.json" \
  --backend ring \
  --python "$ROOT/.venv/bin/python" \
  -- "$ROOT/distributed_nanogpt_streaming/train_gpt2_streaming.py" \
  --config "$ROOT/distributed_nanogpt_streaming/config_d24.json" \
  "$@"
