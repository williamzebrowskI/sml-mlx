
# #!/usr/bin/env python3
# """
# interact_lora.py

# Interactive REPL for your LoRA‐fine‐tuned Transformer using pure MLX sampling.

# Usage:
#   python interact_lora.py \
#     --checkpoint model/checkpoints_owt/ckpt_best.safetensors \
#     --adapter    model/lora_adapters/adapters.npz \
#     --config     model/config.json \
#     --tokenizer  tokenizer/spm.model \
#     --max_new_tokens 10 \
#     --sampling greedy

# Type your prompt at >>> and press Enter.
# """

# import argparse, json, os, sys

# import mlx.core as mx                                        #  [oai_citation:0‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
# import mlx.nn   as nn                                        #  [oai_citation:1‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)

# # ensure project root is on import path
# sys.path.insert(0, os.getcwd())

# from model.model       import Transformer, TransformerConfig
# from tokenizer         import sp_tokenizer

# def sample_top_p_mlx(logits: mx.array, top_p: float = 0.9, temp: float = 1.0) -> int:
#     """
#     Nucleus (top-p) sampling using MLX only.
#     logits: 1D MLX array of shape (vocab_size,)
#     """
#     # 1) apply temperature and softmax
#     scaled = logits / temp
#     probs  = nn.softmax(scaled) 

#     # 2) sort descending by probability: MLX argsort does not accept 'direction', so sort by negative
#     sorted_idx = mx.argsort(-probs)  # descending order indices
#     sorted_p   = mx.take(probs, sorted_idx) 

#     # 3) cumulative sum to find cutoff where cumulative ≥ top_p
#     cumsum     = mx.cumsum(sorted_p, axis=-1)  
#     mask       = cumsum >= top_p
#     # find first index where mask is True
#     # MLX array: use argmax on mask (boolean cast to int) to get first True index
#     cutoff     = int(mx.argmax(mask)) + 1  

#     # 4) restrict to top candidates and renormalize
#     top_candidates  = sorted_idx[:cutoff] 
#     candidate_p     = mx.take(probs, top_candidates)
#     normalized_p    = candidate_p / mx.sum(candidate_p) 

#     # 5) draw from categorical distribution
#     # MLX categorical takes log-probabilities or probabilities? It expects logits; we supply log normalized_p
#     log_p           = mx.log(normalized_p) 
#     # sample index among top_candidates
#     idx_in_top      = int(mx.random.categorical(log_p)) 
#     next_id         = int(top_candidates[idx_in_top])
#     return next_id

# def build_parser():
#     parser = argparse.ArgumentParser("LoRA‐tuned REPL")
#     parser.add_argument("--checkpoint",    required=True, help="Base model safetensors")
#     parser.add_argument("--adapter",       required=True, help="LoRA adapter .npz")
#     parser.add_argument("--config",        required=True, help="TransformerConfig JSON")
#     parser.add_argument("--tokenizer",     required=True, help="SentencePiece model")
#     parser.add_argument("--max_new_tokens","-m", type=int, default=10,
#                         help="Max tokens to generate")
#     parser.add_argument("--sampling", choices=["greedy","top-p"], default="greedy",
#                         help="Decoding method")
#     parser.add_argument("--temp",   type=float, default=0.8, help="Temperature for sampling")
#     parser.add_argument("--top_p",  type=float, default=0.9, help="Nucleus sampling p-value")
#     return parser

# def main():
#     args = build_parser().parse_args()

#     # Load config + model
#     cfg   = TransformerConfig(**json.load(open(args.config)))  #  [oai_citation:11‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#     model = Transformer(cfg)
#     model.load_weights(args.checkpoint)                       #  [oai_citation:12‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#     model.load_weights(args.adapter, strict=False)             # load LoRA adapters  [oai_citation:13‡arxiv.org](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com)
#     model.eval()

#     # Load tokenizer & special IDs
#     tok = sp_tokenizer.load(args.tokenizer)                    #  [oai_citation:14‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#     bos = tok.sp.bos_id()                                      # SentencePiece BOS ID  [oai_citation:15‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#     eos = tok.sp.eos_id()                                      # SentencePiece EOS ID  [oai_citation:16‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)

#     print("Loaded model & adapters. Empty prompt to exit.")
#     while True:
#         prompt = input(">>> ").strip()
#         if not prompt:
#             break

#         # Convert “What is X?” → “X is ” to match possible fine-tune style
#         text = prompt
#         if prompt.lower().startswith("what is ") and prompt.endswith("?"):
#             subj = prompt[8:-1].strip()
#             if subj:
#                 text = subj[0].upper() + subj[1:] + " is "
#         elif not text.endswith(" "):
#             text += " "

#         # Encode with BOS token
#         ids = [bos] + tok.encode(text)  #  [oai_citation:17‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#         gen = []

#         for _ in range(args.max_new_tokens):
#             # Prepare context: last context_size tokens
#             context_ids = ids[-cfg.context_size:] if hasattr(cfg, "context_size") else ids
#             x = mx.array([context_ids])  # batch size 1  [oai_citation:18‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#             logits = model(x)[0, -1]     # MLX array of shape (vocab,)  [oai_citation:19‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)

#             # Choose next token
#             if args.sampling == "greedy":
#                 next_id = int(mx.argmax(logits))  #  [oai_citation:20‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#             else:
#                 next_id = sample_top_p_mlx(logits,
#                                            top_p=args.top_p,
#                                            temp=args.temp)

#             if next_id == eos:
#                 break
#             ids.append(next_id)
#             gen.append(next_id)

#         # Decode generated tokens
#         if gen:
#             print(tok.decode(gen))  #  [oai_citation:21‡ml-explore.github.io](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html?utm_source=chatgpt.com)
#         else:
#             print("[no output]")
#         print()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
interact_lora.py  –  MLX interactive loop with
  • greedy / top-p / top-k
  • repetition-penalty (CTRL-style)
Stop generation on SentencePiece EOS.
"""
import argparse, json, os, sys, math
import mlx.core as mx
import mlx.nn   as nn

sys.path.insert(0, os.getcwd())
from model.model import Transformer, TransformerConfig
from tokenizer    import sp_tokenizer


# ───────── nucleus and top-k helpers ─────────────────────────────────
def sample_top_p(logits: mx.array, *, p=0.9, temp=1.0) -> int:
    probs = nn.softmax(logits / temp)
    idx   = mx.argsort(-probs)          # descending
    cum   = mx.cumsum(mx.take(probs, idx))
    cut   = int(mx.argmax(cum >= p)) + 1
    cand  = idx[:cut]
    probs = probs[cand] / mx.sum(probs[cand])
    return int(cand[int(mx.random.categorical(mx.log(probs)))])


def sample_top_k(logits: mx.array, *, k=40, temp=1.0) -> int:
    probs = nn.softmax(logits / temp)
    idx   = mx.argsort(-probs)[:k]
    probs = probs[idx] / mx.sum(probs[idx])
    return int(idx[int(mx.random.categorical(mx.log(probs)))])


# ───────── CLI ──────────────────────────────────────────────────────
ap = argparse.ArgumentParser("LoRA REPL")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--adapter",    required=True)
ap.add_argument("--config",     required=True)
ap.add_argument("--tokenizer",  required=True)
ap.add_argument("--max_new_tokens","-m", type=int, default=25)
ap.add_argument("--sampling", choices=["greedy","top-p","top-k"], default="greedy")
ap.add_argument("--temp",   type=float, default=0.8)
ap.add_argument("--top_p",  type=float, default=0.9)
ap.add_argument("--top_k",  type=int,   default=40)
ap.add_argument("--rep_penalty", type=float, default=1.0,
                help=">1.0 penalises recently generated tokens")
args = ap.parse_args()

# ───────── load model + tokenizer ───────────────────────────────────
cfg   = TransformerConfig(**json.load(open(args.config)))
model = Transformer(cfg)
model.load_weights(args.checkpoint)
model.load_weights(args.adapter, strict=False)
model.eval()

tok  = sp_tokenizer.load(args.tokenizer)
BOS, EOS = tok.sp.bos_id(), tok.sp.eos_id()

print("Loaded model & adapters. Empty prompt to exit.")
while True:
    prompt = input(">>> ").strip()
    if not prompt:
        break

    ids = [BOS] + tok.encode(prompt)
    past_tokens = set()                      # track for repetition penalty

    for _ in range(args.max_new_tokens):
        ctx = ids[-getattr(cfg,"context_size",512):]
        logits = model(mx.array([ctx]))[0, -1]

        # apply repetition penalty
        if args.rep_penalty > 1.0:
            for t in past_tokens:
                logits[t] /= args.rep_penalty

        # choose next token
        if args.sampling == "greedy":
            next_id = int(mx.argmax(logits))
        elif args.sampling == "top-k":
            next_id = sample_top_k(logits, k=args.top_k, temp=args.temp)
        else:                                # top-p
            next_id = sample_top_p(logits, p=args.top_p, temp=args.temp)

        if next_id == EOS:
            break
        ids.append(next_id)
        past_tokens.add(next_id)

    out = tok.decode(ids[1:])   # strip BOS
    print(out[len(prompt):] if out.startswith(prompt) else out, "\n")