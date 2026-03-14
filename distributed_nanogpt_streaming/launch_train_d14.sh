#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
exec "$ROOT/.venv/bin/mlx.launch" \
  --hostfile "$ROOT/distributed_nanogpt_streaming/hosts.json" \
  --backend ring \
  --python "$ROOT/.venv/bin/python" \
  -- "$ROOT/distributed_nanogpt_streaming/train_gpt2_streaming.py" \
  --config "$ROOT/distributed_nanogpt_streaming/config_d14.json" \
  --data-warmup-batches 2 \
  --grad-allreduce-sync-mb 1 \
  "$@"
