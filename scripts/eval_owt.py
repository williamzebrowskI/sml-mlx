

#!/usr/bin/env python3
"""
Evaluate one checkpoint on a prompt suite of general‐knowledge questions
and print *full* completions. Each completion is fenced so multiline
and/or code‐like answers remain readable in Markdown.

Usage
-----
PYTHONPATH=. python scripts/eval_owt.py \
    --checkpoint model/checkpoints_owt/ckpt_008000.safetensors \
    --config     model/config.json \
    --tokenizer  tokenizer/spm.model
"""
import argparse
import json
import pathlib
import textwrap

from model.model    import Transformer, TransformerConfig
from tokenizer      import sp_tokenizer
from model.eval     import run_mlx_sample   # ← your sampler

# ————————————————————————————————————————————————————————————————————————————————
# SUITE OF “GENERAL/WORLD‐KNOWLEDGE” QUESTIONS (30 total, mixing sampling methods)
# ————————————————————————————————————————————————————————————————————————————————
SUITE = [
    # 1. Capital of France
    dict(
        prompt="The capital of France is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=60,
            key_seed=1001,
        ),
    ),

    # 2. Author of a famous novel
    dict(
        prompt="The author of \"Pride and Prejudice\" is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=60,
            key_seed=1002,
        ),
    ),

    # 3. How photosynthesis works
    dict(
        prompt="Photosynthesis in plants is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=120,
            key_seed=1003,
        ),
    ),

    # 4. Largest ocean on Earth
    dict(
        prompt="The largest ocean on Earth is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=80,
            key_seed=1004,
        ),
    ),

    # 5. Nobel Prize in Literature
    dict(
        prompt="The first recipient of the Nobel Prize in Literature is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=100,
            key_seed=1005,
        ),
    ),

    # 6. Speed of light question
    dict(
        prompt="What is the speed of light in vacuum, and why is it important in physics? ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.10,
            ngram_block=3,
            max_new_tokens=80,
            key_seed=1006,
        ),
    ),

    # 7. Country with largest population
    dict(
        prompt="Which country has the largest population in the world, and approximately how many people live there? ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.10,
            ngram_block=4,
            max_new_tokens=80,
            key_seed=1007,
        ),
    ),

    # 8. Causes of the seasons
    dict(
        prompt="Explain what causes the four seasons on Earth. ",
        args=dict(
            sampling_method="top-k",
            temperature=0.6,
            rep_penalty=1.08,
            ngram_block=3,
            max_new_tokens=100,
            key_seed=1008,
        ),
    ),

    # 9. Pythagorean theorem
    dict(
        prompt="What is the Pythagorean theorem, and provide an example application. ",
        args=dict(
            sampling_method="top-k",
            temperature=0.6,
            rep_penalty=1.08,
            ngram_block=3,
            max_new_tokens=100,
            key_seed=1009,
        ),
    ),

    # 10. Human genome project
    dict(
        prompt="Briefly describe the Human Genome Project and its significance. ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.09,
            ngram_block=3,
            max_new_tokens=100,
            key_seed=1010,
        ),
    ),

    # 11. Tallest mountain
    dict(
        prompt="The tallest mountain in the world is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=60,
            key_seed=1011,
        ),
    ),

    # 12. First person on the Moon
    dict(
        prompt="The first person to walk on the Moon was ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=60,
            key_seed=1012,
        ),
    ),

    # 13. Chemical symbol for water
    dict(
        prompt="The chemical symbol for water is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=40,
            key_seed=1013,
        ),
    ),

    # 14. Artist of the Mona Lisa
    dict(
        prompt="The artist who painted the Mona Lisa is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=60,
            key_seed=1014,
        ),
    ),

    # 15. Theory of relativity
    dict(
        prompt="The theory of relativity was developed by ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.10,
            ngram_block=3,
            max_new_tokens=80,
            key_seed=1015,
        ),
    ),

    # 16. Largest desert
    dict(
        prompt="The largest desert in the world is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=60,
            key_seed=1016,
        ),
    ),

    # 17. What causes rainbows
    dict(
        prompt="Rainbows are caused by ",
        args=dict(
            sampling_method="top-k",
            temperature=0.6,
            rep_penalty=1.08,
            ngram_block=3,
            max_new_tokens=80,
            key_seed=1017,
        ),
    ),

    # 18. Significance of DNA
    dict(
        prompt="DNA is significant because ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.09,
            ngram_block=3,
            max_new_tokens=80,
            key_seed=1018,
        ),
    ),

    # 19. Discovery of penicillin
    dict(
        prompt="Penicillin was discovered by ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=3,
            max_new_tokens=60,
            key_seed=1019,
        ),
    ),

    # 20. Greenhouse effect
    dict(
        prompt="The greenhouse effect refers to ",
        args=dict(
            sampling_method="top-k",
            temperature=0.6,
            rep_penalty=1.08,
            ngram_block=3,
            max_new_tokens=80,
            key_seed=1020,
        ),
    ),

    # ————————————————————————————————————————————————————————————————————————————————
    # 10 Additional Unique Questions (making 30 total)
    # ————————————————————————————————————————————————————————————————————————————————

    # 21. The capital of France (again, under different sampling)
    dict(
        prompt="The capital of France is ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.10,
            ngram_block=3,
            max_new_tokens=20,
            key_seed=1021,
        ),
    ),

    # 22. Largest planet in our solar system
    dict(
        prompt="The largest planet in our solar system is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=2,
            max_new_tokens=20,
            key_seed=1022,
        ),
    ),

    # 23. Author of 'To Kill a Mockingbird'
    dict(
        prompt="The author of \"To Kill a Mockingbird\" is ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.10,
            ngram_block=3,
            max_new_tokens=20,
            key_seed=1023,
        ),
    ),

    # 24. Chemical symbol for table salt
    dict(
        prompt="The chemical symbol for table salt (sodium chloride) is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=2,
            max_new_tokens=10,
            key_seed=1024,
        ),
    ),

    # 25. Tallest building in the world (as of 2025)
    dict(
        prompt="The tallest building in the world (as of 2025) is ",
        args=dict(
            sampling_method="top-k",
            temperature=0.6,
            rep_penalty=1.08,
            ngram_block=3,
            max_new_tokens=20,
            key_seed=1025,
        ),
    ),

    # 26. Inventor of the telephone
    dict(
        prompt="The inventor of the telephone was ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=2,
            max_new_tokens=15,
            key_seed=1026,
        ),
    ),

    # 27. Primary language spoken in Brazil
    dict(
        prompt="The primary language spoken in Brazil is ",
        args=dict(
            sampling_method="top-p",
            temperature=0.6,
            rep_penalty=1.08,
            ngram_block=3,
            max_new_tokens=15,
            key_seed=1027,
        ),
    ),

    # 28. Square root of 144
    dict(
        prompt="The square root of 144 is ",
        args=dict(
            sampling_method="greedy",
            temperature=0.0,
            rep_penalty=1.05,
            ngram_block=2,
            max_new_tokens=10,
            key_seed=1028,
        ),
    ),

    # 29. Current President of the United States
    dict(
        prompt="The current President of the United States is ",
        args=dict(
            sampling_method="top-p",
            temperature=0.7,
            rep_penalty=1.10,
            ngram_block=3,
            max_new_tokens=20,
            key_seed=1029,
        ),
    ),

    # 30. Process by which trees convert CO₂ into oxygen
    dict(
        prompt="The process by which trees convert carbon dioxide into oxygen is called ",
        args=dict(
            sampling_method="top-k",
            temperature=0.6,
            rep_penalty=1.08,
            ngram_block=3,
            max_new_tokens=25,
            key_seed=1030,
        ),
    ),
]

SEQ_LEN = 512          # max context length fed into the model
WRAP    = 100          # soften very long single lines for Markdown width

# ————————————————————————————————————————————————————————————————————————————————
# HELPERS
# ————————————————————————————————————————————————————————————————————————————————
def load_model(ckpt: str, cfg_path: str):
    cfg = TransformerConfig(**json.load(open(cfg_path)))
    mdl = Transformer(cfg)
    mdl.load_weights(ckpt)
    mdl.eval()
    return mdl

def is_multiline(text: str) -> bool:
    """Return True if completion contains newline or code markers."""
    return ("\n" in text) or ("```" in text)

# ————————————————————————————————————————————————————————————————————————————————
# MAIN
# ————————————————————————————————————————————————————————————————————————————————
def main() -> None:
    ap = argparse.ArgumentParser("OpenWebText Eval Suite (full outputs)")
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to safetensors checkpoint (e.g. model/checkpoints_owt/ckpt_008000.safetensors)"
    )
    ap.add_argument(
        "--config",
        default="model/config.json",
        help="Path to TransformerConfig JSON"
    )
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="Path to SentencePiece model (e.g. tokenizer/spm.model)"
    )
    args = ap.parse_args()

    # Load tokenizer + model
    tok   = sp_tokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, args.config)

    ck_name = pathlib.Path(args.checkpoint).stem
    print(f"## Sample outputs – `{ck_name}`\n")

    for idx, item in enumerate(SUITE, 1):
        prompt        = item["prompt"]
        sampling_args = item["args"]

        # Validate sampling_method
        sm = sampling_args.get("sampling_method", "").lower()
        if sm not in ("greedy", "top-k", "top-p"):
            raise ValueError(f"sampling_method must be 'greedy', 'top-k' or 'top-p', got '{sm}'")

        completion = run_mlx_sample(
            model, tok,
            prompt          = prompt,
            seq_len         = SEQ_LEN,
            sampling_method = sm,
            temperature     = sampling_args["temperature"],
            rep_penalty     = sampling_args["rep_penalty"],
            ngram_block     = sampling_args["ngram_block"],
            max_new_tokens  = sampling_args["max_new_tokens"],
            key_seed        = sampling_args["key_seed"],
        )
        # strip the prompt prefix so we only show the generated continuation
        generated = completion[len(prompt):]

        # Soft‐wrap single‐line very long completions
        if not is_multiline(generated):
            generated = textwrap.fill(generated, WRAP)

        # Markdown section
        first_line = prompt.split("\n")[0][:48]
        if len(prompt.split("\n")[0]) > 48:
            first_line += "…"
        print(f"### {idx}. {first_line}\n")
        print("**Prompt**")
        print("```text")
        print(prompt.rstrip())
        print("```")
        print("**Completion**")
        # Always use generic fencing for general‐knowledge responses
        print("```")
        print(generated.rstrip())
        print("```")
        print()  # blank line between entries

if __name__ == "__main__":
    main()

