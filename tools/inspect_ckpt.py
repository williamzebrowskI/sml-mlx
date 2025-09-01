#!/usr/bin/env python3
# tools/inspect_ckpt.py
import argparse, json, re
from pathlib import Path
from typing import Optional, Tuple
from safetensors import safe_open

STEP_RE = re.compile(r"ckpt_(\d{1,12})")

def find_latest_ckpt(path: Path) -> Path:
    if path.is_file() and path.suffix == ".safetensors":
        return path
    if path.is_dir():
        cands = list(path.glob("ckpt_*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"No ckpt_*.safetensors in {path}")
        def step(p: Path) -> int:
            m = STEP_RE.search(p.name)
            return int(m.group(1)) if m else -1
        return max(cands, key=step)
    raise FileNotFoundError(f"Path not found: {path}")

def guess_emb_key(keys) -> Optional[str]:
    # common names in this codebase
    if "emb.weight" in keys: return "emb.weight"
    # fallback: look for something embedding-like
    cand = [k for k in keys if ("emb" in k or "embed" in k) and k.endswith(".weight")]
    return cand[0] if cand else None

def layer_indices(keys) -> Tuple[int,int,int]:
    """Return (min_layer, max_layer, count) from keys like 'layers.<i>.*'."""
    idxs = set()
    for k in keys:
        m = re.match(r"layers\.(\d+)\.", k)
        if m: idxs.add(int(m.group(1)))
    if not idxs:
        return (0, -1, 0)
    return (min(idxs), max(idxs), len(idxs))

def read_shape(f, key):
    try:
        return f.get_tensor(key).shape  # loads just this tensor
    except Exception:
        return None

def find_one(keys, pattern: str) -> Optional[str]:
    rex = re.compile(pattern)
    for k in keys:
        if rex.fullmatch(k): return k
    return None

def derive_heads_pair(d_model: int) -> Tuple[int,int]:
    """Best-guess a (num_heads, head_dim) pair from d_model, favoring 64 head_dim."""
    if d_model % 64 == 0:
        return (d_model // 64, 64)
    # otherwise prefer the smallest head_dim in {128, 96, 80, 72, 48, 32} that divides d_model
    for hd in (128, 96, 80, 72, 48, 32):
        if d_model % hd == 0:
            return (d_model // hd, hd)
    # fallback
    return (8, d_model // 8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to a .safetensors file OR a directory containing ckpt_*.safetensors")
    ap.add_argument("--config", help="Optional path to model JSON to compare against")
    args = ap.parse_args()

    ckpt_path = find_latest_ckpt(Path(args.ckpt))
    print(f"→ Using checkpoint: {ckpt_path}")

    with safe_open(str(ckpt_path), framework="numpy") as f:
        keys = list(f.keys())

        # 1) vocab_size & d_model from embedding
        emb_key = guess_emb_key(keys)
        if not emb_key:
            raise RuntimeError("Could not find embedding weight key (e.g., 'emb.weight').")
        emb_shape = read_shape(f, emb_key)
        if not emb_shape or len(emb_shape) != 2:
            raise RuntimeError(f"Unexpected embedding shape for {emb_key}: {emb_shape}")
        vocab_size_ckpt, d_model_ckpt = emb_shape
        print(f"• emb: {emb_key} shape={emb_shape} → vocab_size={vocab_size_ckpt}, d_model={d_model_ckpt}")

        # 2) number of transformer layers
        lmin, lmax, lcount = layer_indices(keys)
        if lcount == 0:
            raise RuntimeError("Could not infer layer count (no 'layers.<i>.' keys found).")
        num_layers_ckpt = lmax + 1
        print(f"• layers: indices {lmin}…{lmax} → num_layers={num_layers_ckpt}")

        # 3) sanity: attention + ffn representative shapes
        qkv_key = find_one(keys, r"layers\.0\.attn\.qkv\.weight")
        oproj_key = find_one(keys, r"layers\.0\.attn\.o_proj\.weight")
        ffn_in_key = find_one(keys, r"layers\.0\.ffn\.proj_in\.weight")
        ffn_out_key = find_one(keys, r"layers\.0\.ffn\.proj_out\.weight")

        if qkv_key:
            qkv_shape = read_shape(f, qkv_key)
            if qkv_shape and len(qkv_shape) == 2:
                print(f"• attn.qkv: {qkv_key} shape={qkv_shape} (expect (~3*d_model, d_model))")
                if qkv_shape[1] != d_model_ckpt:
                    print(f"  ⚠ qkv second dim {qkv_shape[1]} != d_model {d_model_ckpt}")
        if oproj_key:
            oproj_shape = read_shape(f, oproj_key)
            if oproj_shape:
                print(f"• attn.o_proj: {oproj_key} shape={oproj_shape} (expect (d_model, d_model))")
        if ffn_in_key:
            ffn_in_shape = read_shape(f, ffn_in_key)
            if ffn_in_shape:
                print(f"• ffn.proj_in: {ffn_in_key} shape={ffn_in_shape} (expect (ffn_hidden, d_model))")
        if ffn_out_key:
            ffn_out_shape = read_shape(f, ffn_out_key)
            if ffn_out_shape:
                print(f"• ffn.proj_out: {ffn_out_key} shape={ffn_out_shape} (expect (d_model, ffn_hidden))")

        # A reasonable heads/head_dim guess (cannot be recovered uniquely from weights)
        n_heads_guess, head_dim_guess = derive_heads_pair(d_model_ckpt)
        print(f"• inferred (num_heads, head_dim) guess = ({n_heads_guess}, {head_dim_guess}) "
              f"since num_heads*head_dim must equal d_model={d_model_ckpt}")

        # 4) Optional: compare to config
        if args.config:
            cfg = json.loads(Path(args.config).read_text())
            def g(name, default=None): return cfg.get(name, default)

            problems = []
            print("\nComparing against config:", args.config)
            # vocab (often set by tokenizer at runtime; still useful to compare)
            if g("vocab_size") is not None and g("vocab_size") != vocab_size_ckpt:
                problems.append(f"vocab_size: config={g('vocab_size')} vs ckpt={vocab_size_ckpt}")

            if g("model_dim") is not None and g("model_dim") != d_model_ckpt:
                problems.append(f"model_dim: config={g('model_dim')} vs ckpt={d_model_ckpt}")

            if g("num_transformer_layers") is not None and g("num_transformer_layers") != num_layers_ckpt:
                problems.append(f"num_transformer_layers: config={g('num_transformer_layers')} vs ckpt={num_layers_ckpt}")

            # heads_x_dim consistency
            if g("num_heads") is not None and g("head_dim") is not None:
                hx = g("num_heads") * g("head_dim")
                if hx != d_model_ckpt:
                    problems.append(f"num_heads*head_dim must equal d_model: {g('num_heads')}*{g('head_dim')}={hx} "
                                    f"!= ckpt d_model {d_model_ckpt}")

            if problems:
                print("❌ MISMATCHES found:")
                for p in problems: print("   -", p)
            else:
                print("✅ Config matches checkpoint-critical shapes.")

        # 5) Print a minimal config block that will load this checkpoint
        print("\nMinimal shape-safe config fields (set these to resume):")
        print(json.dumps({
            "vocab_size": int(vocab_size_ckpt),
            "num_transformer_layers": int(num_layers_ckpt),
            "model_dim": int(d_model_ckpt),
            "num_heads": int(n_heads_guess),
            "head_dim": int(head_dim_guess)
        }, indent=2))

if __name__ == "__main__":
    main()