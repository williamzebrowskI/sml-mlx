# #!/usr/bin/env python3
# """
# Eval harness for TinyStories-100M with configurable sampling:

# • Loads a safetensors checkpoint
# • Supports “greedy”, “top-k” or “nucleus (top-p)” sampling via MLX APIs
# • Applies repetition penalty and n-gram blocking
# • Prints a single generation under a header
# """

# import argparse
# import json
# import mlx.core as mx

# from model.model import Transformer, TransformerConfig
# from tokenizer import sp_tokenizer


# def load_model(checkpoint_path: str, config_path: str) -> Transformer:
#     """Load config, instantiate model, load weights, and set to eval mode."""
#     cfg = TransformerConfig(**json.load(open(config_path)))
#     model = Transformer(cfg)
#     model.load_weights(checkpoint_path)
#     model.eval()
#     return model


# def next_token_greedy(logits: mx.array) -> int:
#     """Return argmax token (greedy decoding)."""
#     return int(mx.argmax(logits).item())


# def run_mlx_sample(
#     model: Transformer,
#     tokenizer,
#     prompt: str,
#     max_new_tokens: int = 50,
#     sampling_method: str = "top-k",  # one of ["greedy","top-k","top-p"]
#     temperature: float = 1.0,
#     top_k: int = 40,
#     top_p: float = 0.9,
#     rep_penalty: float = 1.2,
#     ngram_block: int = 2,
#     key_seed: int = 0,
#     seq_len: int = 512,
# ) -> str:
#     # 1) PRNG key setup
#     key = mx.random.key(key_seed)

#     # 2) Encode prompt → token IDs
#     ids = tokenizer.encode(prompt)

#     # 3) Track seen n-grams
#     seen_ngrams = set()

#     for _ in range(max_new_tokens):
#         # 4) Build input tensor (1, L)
#         x = mx.array(ids[-seq_len:], dtype=mx.int32)[None, :]

#         # 5) Forward pass → logits (vocab,)
#         logits = model(x)[0, -1]

#         # 6) Make mutable copy
#         logits = mx.array(logits)
#         last = ids[-1]
#         logits[last] = -1e9

#         # 7) Repetition penalty
#         for t in set(ids):
#             logits[t] = logits[t] / rep_penalty

#         # 8) N-gram blocking
#         if len(ids) >= ngram_block:
#             prev = tuple(ids[-(ngram_block - 1):])
#             for tok_id in range(int(logits.shape[0])):
#                 if (*prev, tok_id) in seen_ngrams:
#                     logits[tok_id] = -1e9

#         # 9) Decoding method
#         if sampling_method == "greedy" or temperature <= 0:
#             next_id = next_token_greedy(logits)

#         elif sampling_method == "top-k":
#             # 9a) Temperature scaling
#             scaled = logits / temperature

#             # 9b) ascending sort → take last top_k indices
#             sorted_idx = mx.argsort(scaled)
#             topk_idx = sorted_idx[-top_k:]
#             topk_logits = scaled[topk_idx]

#             # 9c) Sample
#             key, subkey = mx.random.split(key)
#             rel = int(mx.random.categorical(topk_logits, key=subkey).item())
#             next_id = int(topk_idx[rel].item())

#         elif sampling_method == "top-p":
#             # 9a) Temperature scaling
#             scaled = logits / temperature

#             # 9b) Convert to probabilities via softmax: p = exp(scaled) / sum(exp(scaled))
#             lse = mx.logsumexp(scaled)
#             probs = mx.exp(scaled - lse)

#             # 9c) Sort ascending and compute cumulative sum
#             sorted_idx = mx.argsort(probs)
#             sorted_probs = probs[sorted_idx]
#             cum = mx.cumsum(sorted_probs)

#             # 9d) Determine cutoff where cumulative > top_p
#             cutoff = int(mx.argmax(cum > top_p).item()) + 1
#             top_p_idx = sorted_idx[-cutoff:]
#             top_p_probs = probs[top_p_idx]

#             # 9e) Renormalize
#             total = mx.sum(top_p_probs)
#             normed = top_p_probs / total

#             # 9f) Sample
#             key, subkey = mx.random.split(key) 
#             rel = int(mx.random.categorical(normed, key=subkey).item())
#             next_id = int(top_p_idx[rel].item())

#         else:
#             raise ValueError(f"Unknown sampling_method={sampling_method}")

#         # 10) Append & record n-gram
#         ids.append(next_id)
#         if len(ids) >= ngram_block:
#             seen_ngrams.add(tuple(ids[-ngram_block:]))

#         # 11) Stop on EOS
#         if next_id == tokenizer.eos_id:
#             break

#     # 12) Decode → text
#     return tokenizer.decode(ids)


# def main():
#     p = argparse.ArgumentParser("Generate from TinyStories-100M")
#     p.add_argument("--checkpoint",    required=True, help="Path to .safetensors checkpoint")
#     p.add_argument("--tokenizer",     required=True, help="SentencePiece .model file")
#     p.add_argument("--config",        default="model/config.json", help="Path to config.json")
#     p.add_argument("--prompt",        default="Once upon a time", help="Initial prompt")
#     p.add_argument("--max-new-tokens", type=int, default=50, help="Number of tokens to generate")
#     p.add_argument("--sampling-method", choices=["greedy","top-k","top-p"], default="top-k")
#     p.add_argument("--temperature",    type=float, default=0.1, help="Sampling temperature")
#     p.add_argument("--top-k",          type=int,   default=40,  help="Top-k for top-k sampling")
#     p.add_argument("--top-p",          type=float, default=0.9, help="Top-p for nucleus sampling")
#     p.add_argument("--rep-penalty",    type=float, default=1.2, help="Repetition penalty factor")
#     p.add_argument("--ngram-block",    type=int,   default=2,   help="N-gram blocking length")
#     p.add_argument("--key-seed",       type=int,   default=42,  help="PRNG seed")
#     p.add_argument("--seq-len",        type=int,   default=512, help="Max context length")
#     args = p.parse_args()

#     tok   = sp_tokenizer.load(args.tokenizer)
#     model = load_model(args.checkpoint, args.config)

#     output = run_mlx_sample(
#         model, tok,
#         prompt=args.prompt,
#         max_new_tokens=args.max_new_tokens,
#         sampling_method=args.sampling_method,
#         temperature=args.temperature,
#         top_k=args.top_k,
#         top_p=args.top_p,
#         rep_penalty=args.rep_penalty,
#         ngram_block=args.ngram_block,
#         key_seed=args.key_seed,
#         seq_len=args.seq_len,
#     )

#     print("\n=== Generation ===\n")
#     print(output)
#     print("\n==================\n")


# if __name__ == "__main__":
#     main()


"""
eval.py – text-generation harness for an MLX Transformer checkpoint.

Highlights
----------
• Loads a .safetensors checkpoint and its JSON config.
• Supports “greedy”, “top-k” or “nucleus (top-p)” sampling.
• Adds temperature scaling, repetition penalty (PaLM style),
  and n-gram blocking.
• Reuses an MXNet-style PRNG key for reproducible stochastic decoding.

Usage example
-------------
python eval.py \
  --checkpoint  model/checkpoints/ckpt_093000.safetensors \
  --config      model/config.json \
  --tokenizer   tokenizer/spm.model \
  --prompt      "Valkyria III is commonly referred to as " \
  --sampling-method top-p \
  --temperature 0.8 \
  --top-p       0.98 \
  --rep-penalty 2.5 \
  --ngram-block 5 \
  --max-new-tokens 30
"""

import argparse
import json
import mlx.core as mx

from model.model import Transformer, TransformerConfig
from tokenizer   import sp_tokenizer


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: str, config_path: str) -> Transformer:
    """Instantiate Transformer and load weights from a safetensors file."""
    cfg   = TransformerConfig(**json.load(open(config_path)))
    model = Transformer(cfg)
    model.load_weights(checkpoint_path)
    model.eval()
    return model


def greedy_id(logits: mx.array) -> int:
    """Arg-max sampling."""
    return int(mx.argmax(logits).item())


def softmax(x: mx.array) -> mx.array:
    """Numerically stable softmax."""
    return mx.exp(x - mx.logsumexp(x))


# --------------------------------------------------------------------------- #
#  Sampling loop                                                              #
# --------------------------------------------------------------------------- #
def run_mlx_sample(
    model: Transformer,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 50,
    sampling_method: str = "top-k",      # ["greedy","top-k","top-p"]
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    rep_penalty: float = 1.2,
    ngram_block: int = 2,
    key_seed: int = 0,
    seq_len: int = 512,
) -> str:
    """Generate text from *prompt* using the chosen decoding strategy."""
    key = mx.random.key(key_seed)
    ids = tokenizer.encode(prompt)

    seen_ngrams = set()

    V = model.out_proj.weight.shape[0]  # vocab size once, for speed

    for _ in range(max_new_tokens):
        x = mx.array(ids[-seq_len:], dtype=mx.int32)[None, :]  # (1, L)
        logits = model(x)[0, -1]  # (V,)

        # ------------------------------------------------------------------ #
        #  Post-logit transforms                                             #
        # ------------------------------------------------------------------ #
        logits = mx.array(logits)

        # 1. Temperature scaling
        temp = max(temperature, 1e-4)  # avoid div-by-zero
        logits = logits / temp

        # 2. Repetition penalty (PaLM / NeoX style)
        for tok in set(ids):
            if logits[tok] > 0:
                logits[tok] /= rep_penalty
            else:
                logits[tok] *= rep_penalty

        # 3. N-gram blocking
        if len(ids) >= ngram_block:
            prev = tuple(ids[-(ngram_block - 1):])
            # collect once, then mask in a vectorised way
            forbid = [
                tok_id for tok_id in range(V)
                if (*prev, tok_id) in seen_ngrams
            ]
            if forbid:
                logits[forbid] = -1e9

        # ------------------------------------------------------------------ #
        #  Choose the next token                                             #
        # ------------------------------------------------------------------ #
        if sampling_method == "greedy" or temp == 0.0:
            next_id = greedy_id(logits)

        elif sampling_method == "top-k":
            topk = min(top_k, V)
            idx  = mx.argsort(logits)[-topk:]  # indices of k largest logits
            probs = softmax(logits[idx])

            key, subkey = mx.random.split(key)
            rel = int(mx.random.categorical(probs, key=subkey).item())
            next_id = int(idx[rel].item())

        elif sampling_method == "top-p":
            probs = softmax(logits)
            idx   = mx.argsort(probs)          # ascending
            probs = probs[idx]
            cumsum = mx.cumsum(probs)

            # first position where cumulative mass exceeds top_p
            cutoff = int(mx.argmax(cumsum > top_p).item()) + 1
            cutoff = max(1, cutoff)            # safety: at least 1 token

            idx   = idx[-cutoff:]
            probs = probs[-cutoff:]
            probs = probs / mx.sum(probs)      # re-normalise

            key, subkey = mx.random.split(key)
            rel = int(mx.random.categorical(probs, key=subkey).item())
            next_id = int(idx[rel].item())

        else:
            raise ValueError(f"sampling_method must be greedy, top-k or top-p")

        # ------------------------------------------------------------------ #
        #  Book-keeping                                                      #
        # ------------------------------------------------------------------ #
        ids.append(next_id)
        if len(ids) >= ngram_block:
            seen_ngrams.add(tuple(ids[-ngram_block:]))

        # EOS sentinel (if available)
        if hasattr(tokenizer, "eos_id") and next_id == tokenizer.eos_id:
            break

    return tokenizer.decode(ids)


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser("MLX text generation")
    parser.add_argument("--checkpoint", required=True, help="*.safetensors")
    parser.add_argument("--config",     default="model/config.json")
    parser.add_argument("--tokenizer",  required=True)
    parser.add_argument("--prompt",     default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=50)

    parser.add_argument("--sampling-method",
                        choices=["greedy", "top-k", "top-p"],
                        default="top-k")
    parser.add_argument("--temperature",  type=float, default=1.0)
    parser.add_argument("--top-k",        type=int,   default=40)
    parser.add_argument("--top-p",        type=float, default=0.9)
    parser.add_argument("--rep-penalty",  type=float, default=1.2)
    parser.add_argument("--ngram-block",  type=int,   default=2)
    parser.add_argument("--key-seed",     type=int,   default=42)
    parser.add_argument("--seq-len",      type=int,   default=512)
    args = parser.parse_args()

    tok   = sp_tokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, args.config)

    text = run_mlx_sample(
        model, tok,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        sampling_method=args.sampling_method,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        ngram_block=args.ngram_block,
        key_seed=args.key_seed,
        seq_len=args.seq_len,
    )

    print("\n=== Generation ===\n")
    print(text)
    print("\n==================\n")


if __name__ == "__main__":
    main()