
PYTHON := python

.PHONY: data tokenizer train eval clean

data:
	$(PYTHON) data/scripts/fetch_stack.py --output data/raw
	$(PYTHON) data/scripts/clean_code.py --input data/raw --output data/clean_py.jsonl

tokenizer:
	$(PYTHON) tokenizer/train_tokenizer.py --input data/clean_py.jsonl --model-dir tokenizer

train:
	$(PYTHON) model/train.py --config model/config.json --tokenizer tokenizer/py50k.model --dataset data/clean_py.jsonl --out model/checkpoints

eval:
	$(PYTHON) model/eval.py --checkpoint model/checkpoints/latest.safetensors --tokenizer tokenizer/py50k.model

clean:
	rm -rf model/checkpoints
