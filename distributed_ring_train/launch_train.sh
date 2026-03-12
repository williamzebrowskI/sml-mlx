#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
HOSTFILE="$ROOT/distributed_ring_train/hosts.json"
CONFIG="$ROOT/distributed_ring_train/config.json"

exec "$ROOT/.venv/bin/mlx.launch" \
  --hostfile "$HOSTFILE" \
  --backend ring \
  --python "$ROOT/.venv/bin/python" \
  -- "$ROOT/train/train.py" \
  --config "$CONFIG" \
  "$@"
