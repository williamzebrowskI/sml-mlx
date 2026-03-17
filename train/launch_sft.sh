#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
exec "$ROOT/.venv/bin/python" \
  "$ROOT/train/train_sft.py" \
  --config "$ROOT/train/sft_config.json" \
  "$@"
