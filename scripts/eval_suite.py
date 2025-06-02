# #!/usr/bin/env python3
# """
# Evaluate *one* checkpoint against a small prompt suite.

# It prints a Markdown table that you can paste into your training-notes
# or commit message.

# Usage
# -----
# PYTHONPATH=. python scripts/eval_suite.py \
#     --checkpoint model/checkpoints/ckpt_041000.safetensors \
#     --config     model/config.json \
#     --tokenizer  tokenizer/spm.model
# """
# import argparse, json, textwrap, pathlib
# import mlx.core as mx

# from model.model   import Transformer, TransformerConfig
# from tokenizer     import sp_tokenizer
# from model.eval    import run_mlx_sample   # reuse the sampler

# # --------------------------------------------------------------------------- #
# # 10-prompt evaluation suite                                                  #
# # --------------------------------------------------------------------------- #

# # --------------------------------------------------------------------------- #
# # 10-prompt PYTHON-ONLY evaluation suite (StackExchange-Code oriented)        #
# # --------------------------------------------------------------------------- #
# SUITE = [
#     # 1. idiomatic comprehension vs. loop
#     dict(
#         prompt="Create a random python function that outputs a list of integers:\n\n"
#                "\nAnswer:\n",
#         args=dict(sampling_method="top-p", top_p=0.92, temperature=0.4,
#                   rep_penalty=1.15, ngram_block=3, max_new_tokens=50, key_seed=101)
#     ),

#     # 2. simple decorator from scratch
#     dict(
#         prompt="Implement a timing decorator called **measure_time** that prints the run-time "
#                "of the wrapped function.\n\n```python\n# Your code here\n```",
#         args=dict(sampling_method="top-p", top_k=40, temperature=0.55,
#                   rep_penalty=1.15, ngram_block=4, max_new_tokens=120, key_seed=102)
#     ),

#     # 3. custom context-manager
#     dict(
#         prompt="Write a context manager **suppress_stdout()** that temporarily silences "
#                "`stdout` inside a *with* block.",
#         args=dict(sampling_method="top-p", top_k=50, temperature=0.6,
#                   rep_penalty=1.20, ngram_block=4, max_new_tokens=120, key_seed=103)
#     ),

#     # 4. generator vs. list memory
#     dict(
#         prompt="In one paragraph, explain why a generator expression\n"
#                "`(x*x for x in big_list)` is more memory-efficient than\n"
#                "`[x*x for x in big_list]`.",
#         args=dict(sampling_method="top-p", top_p=0.9, temperature=0.45,
#                   rep_penalty=1.15, ngram_block=3, max_new_tokens=80, key_seed=104)
#     ),

#     # 5. async / await toy example
#     dict(
#         prompt="Show a minimal **asyncio** example that concurrently fetches three URLs "
#                "and prints their status codes.",
#         args=dict(sampling_method="top-p", top_k=60, temperature=0.7,
#                   rep_penalty=1.1, ngram_block=4, max_new_tokens=140, key_seed=105)
#     ),

#     # 6. type-hints & mypy
#     dict(
#         prompt="Add PEP-484 type annotations to the following function and briefly explain "
#                "one benefit of doing so.\n\n```python\ndef merge(a, b):\n    return {**a, **b}\n```",
#         args=dict(sampling_method="top-p", top_p=0.88, temperature=0.45,
#                   rep_penalty=1.2, ngram_block=3, max_new_tokens=90, key_seed=106)
#     ),

#     # 7. pandas groupby aggregation
#     dict(
#         prompt="Using **pandas**, given a DataFrame `df` with columns *city*, *year*, *sales*, "
#                "show how to compute the total sales per city.",
#         args=dict(sampling_method="top-p", top_k=30, temperature=0.5,
#                   rep_penalty=1.15, ngram_block=3, max_new_tokens=80, key_seed=107)
#     ),

#     # 8. secure subprocess call
#     dict(
#         prompt="Demonstrate a safe way to run the shell command `ls -l /tmp` from Python "
#                "without invoking the shell or risking injection.",
#         args=dict(sampling_method="top-p", temperature=0,
#                   rep_penalty=1.25, ngram_block=4, max_new_tokens=60, key_seed=108)
#     ),

#     # 9. itertools permutations
#     dict(
#         prompt="Use **itertools.permutations** to print all orderings of the list "
#                "`['A', 'B', 'C']`.",
#         args=dict(sampling_method="top-p", top_k=20, temperature=0.6,
#                   rep_penalty=1.1, ngram_block=3, max_new_tokens=60, key_seed=109)
#     ),

#     # 10. lambda vs. def quick note
#     dict(
#         prompt="Explain in two sentences when a **lambda** is preferable to defining a "
#                "full function with *def* in Python.",
#         args=dict(sampling_method="top-p", top_p=0.85, temperature=0.5,
#                   rep_penalty=1.05, ngram_block=3, max_new_tokens=40, key_seed=110)
#     ),
# ]
# SEQ_LEN = 512   # context length fed into the model

# # --------------------------------------------------------------------------- #
# # helpers                                                                     #
# # --------------------------------------------------------------------------- #
# def load_model(ckpt: str, cfg_path: str):
#     cfg   = TransformerConfig(**json.load(open(cfg_path)))
#     model = Transformer(cfg)
#     model.load_weights(ckpt)
#     model.eval()
#     return model


# def main() -> None:
#     ap = argparse.ArgumentParser("Mini evaluation suite")
#     ap.add_argument("--checkpoint", required=True)
#     ap.add_argument("--config",     default="model/config.json")
#     ap.add_argument("--tokenizer",  required=True)
#     args = ap.parse_args()

#     tok   = sp_tokenizer.load(args.tokenizer)
#     model = load_model(args.checkpoint, args.config)

#     rows = []
#     for item in SUITE:
#         gen = run_mlx_sample(
#             model, tok,
#             prompt   = item["prompt"],
#             seq_len  = SEQ_LEN,
#             **item["args"],
#         )
#         excerpt = textwrap.shorten(
#             gen[len(item["prompt"]):].strip(),
#             width=120, placeholder=" …"
#         )
#         rows.append((item["prompt"].split("\n")[0][:32] + "…", excerpt))

#     ck_name = pathlib.Path(args.checkpoint).stem
#     print(f"### Sample outputs – `{ck_name}`\n")
#     print("| prompt | completion |")
#     print("|--------|------------|")
#     for ptxt, comp in rows:
#         print(f"| *{ptxt}* | {comp} |")


# if __name__ == "__main__":
#     main()




    #!/usr/bin/env python3
"""
Evaluate one checkpoint on a prompt suite and print *full* completions.

The output is still Markdown-flavoured, but each completion is rendered in
its own fenced block so multilines and code remain readable when pasted into
notes or a commit message.

Usage
-----
PYTHONPATH=. python scripts/eval_suite.py \
    --checkpoint model/checkpoints/ckpt_xxxxxx.safetensors \
    --config     model/config.json \
    --tokenizer  tokenizer/spm.model
"""
import argparse, json, pathlib, textwrap
import mlx.core as mx

from model.model    import Transformer, TransformerConfig
from tokenizer      import sp_tokenizer
from model.eval     import run_mlx_sample   # ← your sampler

SUITE = [
    # 1. create & print a simple list
    dict(
        prompt="What is the python programming language?",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.10, ngram_block=3, max_new_tokens=120, key_seed=501),
    ),

    # 2. sum a list of numbers
    dict(
        prompt="Write a one-line expression that sums the numbers in the list `[3, 7, 2, 9]`.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.15, ngram_block=4, max_new_tokens=120, key_seed=502),
    ),

    # 3. check if a number is even
    dict(
        prompt="Implement a function `is_even(n)` that returns `True` if *n* is even.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.12, ngram_block=3, max_new_tokens=120, key_seed=503),
    ),

    # 4. reverse a string
    dict(
        prompt="Show two different ways to reverse the string `'hello'` in Python.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.18, ngram_block=4, max_new_tokens=120, key_seed=504),
    ),

    # 5. dictionary comprehension
    dict(
        prompt="Using a **dictionary comprehension**, map the integers 0-4 to their squares.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.14, ngram_block=3, max_new_tokens=120, key_seed=505),
    ),

    # 6. read file lines into a list
    dict(
        prompt="In one short snippet, open the file `data.txt` and read all lines into a list.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.16, ngram_block=4, max_new_tokens=120, key_seed=506),
    ),

    # 7. use enumerate in a loop
    dict(
        prompt="Demonstrate iterating over the list `['a', 'b', 'c']` with **enumerate** "
               "so that both the index and value are printed.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.09, ngram_block=3, max_new_tokens=120, key_seed=507),
    ),

    # 8. sort a list of dictionaries
    dict(
        prompt="Given `people = [{'name':'Bob','age':25}, {'name':'Ann','age':19}]`, "
               "sort the list by the `age` key.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.13, ngram_block=4, max_new_tokens=120, key_seed=508),
    ),

    # 9. lambda with filter
    dict(
        prompt="Use **filter** and a `lambda` to keep only even numbers from the list `range(10)`.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.11, ngram_block=3, max_new_tokens=120, key_seed=509),
    ),

    # 10. transpose a 2-D list with zip
    dict(
        prompt="Using `zip`, transpose the matrix `[[1,2,3],[4,5,6]]`.",
        args=dict(sampling_method="greedy", temperature=0,
                  rep_penalty=1.17, ngram_block=4, max_new_tokens=120, key_seed=510),
    ),
]
SEQ_LEN = 512          # max context length fed into the model
WRAP   = 120           # soften very long *single* lines for Markdown width

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_model(ckpt: str, cfg_path: str):
    cfg   = TransformerConfig(**json.load(open(cfg_path)))
    mdl   = Transformer(cfg)
    mdl.load_weights(ckpt)
    mdl.eval()
    return mdl


def is_multiline(text: str) -> bool:
    """True if the completion contains a newline OR back-ticks that imply code."""
    return ("\n" in text) or ("```" in text)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("Mini evaluation suite (full outputs)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config",     default="model/config.json")
    ap.add_argument("--tokenizer",  required=True)
    args = ap.parse_args()

    tok   = sp_tokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, args.config)

    ck_name = pathlib.Path(args.checkpoint).stem
    print(f"## Sample outputs – `{ck_name}`\n")

    for idx, item in enumerate(SUITE, 1):
        prompt = item["prompt"]
        completion = run_mlx_sample(
            model, tok,
            prompt   = prompt,
            seq_len  = SEQ_LEN,
            **item["args"],
        )[len(prompt):]  # strip the prompt from the raw generation

        # Soft-wrap *single-line* very long completions for nicer width
        if not is_multiline(completion):
            completion = textwrap.fill(completion, WRAP)

        # Markdown section for this example
        first_line = prompt.split("\n")[0][:48] + ("…" if len(prompt.split("\n")[0]) > 48 else "")
        print(f"### {idx}. {first_line}\n")
        print("**Prompt**")
        print("```text")
        print(prompt.rstrip())
        print("```")
        print("**Completion**")
        fence = "```python" if "```python" in completion or completion.lstrip().startswith(("def ", "class ")) else "```"
        print(fence)
        print(completion.rstrip())
        print("```")
        print()     # blank line between entries


if __name__ == "__main__":
    main()