# # scripts/sample_fineweb.py
# import json
# import sentencepiece as spm  # only for counting tokens
# from datasets import load_dataset

# TARGET_TOKENS = 20_000_000
# OUTFILE       = "fineweb_sample.jsonl"

# # (Optional) if you already have a rough tokenizer, load it to count tokens,
# # otherwise just approximate by words or characters.
# sp = spm.SentencePieceProcessor()  
# sp.load("tokenizer/spm.model")  # you can skip/count by splitting on whitespace

# def count_tokens(text: str) -> int:
#     return len(sp.encode(text, out_type=int))  # or len(text.split())

# def main():
#     ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
#     total = 0
#     with open(OUTFILE, "w", encoding="utf-8") as f:
#         for ex in ds:
#             text = ex["text"]
#             n = count_tokens(text)
#             if n == 0:
#                 continue
#             f.write(json.dumps({"text": text}) + "\n")
#             total += n
#             if total >= TARGET_TOKENS:
#                 break
#     print(f"→ dumped ~{total:,} tokens to {OUTFILE}")

# if __name__ == "__main__":
#     main()

