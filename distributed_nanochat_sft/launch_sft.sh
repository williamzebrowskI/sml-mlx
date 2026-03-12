#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
exec "$ROOT/.venv/bin/mlx.launch" \
  --hostfile "$ROOT/distributed_nanochat_sft/hosts.json" \
  --backend ring \
  --python "$ROOT/.venv/bin/python" \
  -- "$ROOT/distributed_nanochat_sft/train_sft.py" \
  --config "$ROOT/distributed_nanochat_sft/config.json" \
  "$@"
