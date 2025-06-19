#!/usr/bin/env python3
# convert_npz_to_safetensors_mlx.py
"""
Convert MLX LoRA adapters (adapters.npz) → PEFT-style adapter_model.safetensors
without touching PyTorch.  Requires:
  pip install safetensors mlx numpy
"""

import argparse, json, os, numpy as np, mlx.core as mx
from safetensors.numpy import save_file            # NumPy backend

DTYPE_TABLE = {
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
    "fp32": mx.float32,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",  required=True, help="path/to/adapters.npz")
    ap.add_argument("--out",  required=True, help="output folder")
    ap.add_argument("--dtype", default="fp16", choices=DTYPE_TABLE,
                    help="precision to store in safetensors (default: fp16)")
    ap.add_argument("--base-model", default="mlx_smallLM",
                    help="identifier of the base checkpoint")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (r)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw = np.load(args.npz)

    # cast with MLX and return to NumPy for safetensors
    target_dtype = DTYPE_TABLE[args.dtype]
    state = {
    k: np.asarray(               # NumPy converts via __array__ protocol
        mx.array(raw[k]).astype(target_dtype)
    )
    for k in raw.files
    }
    save_file(state, os.path.join(args.out, "adapter_model.safetensors"))

    # minimal PEFT-style metadata
    cfg = {
        "base_model_name_or_path": args.base_model,
        "peft_type": "LORA",
        "r": args.rank,
        "lora_alpha": args.rank * 2,
        "target_modules": sorted({k.rsplit(".", 2)[0] for k in state}),
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    with open(os.path.join(args.out, "adapter_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"✅  LoRA adapter written to: {args.out}")

if __name__ == "__main__":
    main()