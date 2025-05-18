
import sentencepiece as spm

class SPTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.eos_id = self.sp.eos_id()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

def load(model_path):
    return SPTokenizer(model_path)
