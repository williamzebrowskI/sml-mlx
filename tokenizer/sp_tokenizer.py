
# import sentencepiece as spm

# class SPTokenizer:
#     def __init__(self, model_path):
#         self.sp = spm.SentencePieceProcessor(model_file=model_path)
#         self.eos_id = self.sp.eos_id()

#     def encode(self, text):
#         return self.sp.encode(text, out_type=int)

# def load(model_path):
#     return SPTokenizer(model_path)

import sentencepiece as spm
from typing import List

class SPTokenizer:
    def __init__(self, model_path: str):
        # load the SentencePiece model
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        # store the EOS ID for generation stopping
        self.eos_id = self.sp.eos_id()

    def encode(self, text: str) -> List[int]:
        """Encode a string to a list of token IDs."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string."""
        # If you want to strip off padding or anything, you can post-process here.
        return self.sp.decode(ids)

def load(model_path: str) -> SPTokenizer:
    """Helper to load the tokenizer."""
    return SPTokenizer(model_path)