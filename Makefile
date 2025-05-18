
PYTHON := python

.PHONY: data tokenizer train eval clean

data:
	python data/scripts/fetch_tinystories.py \
		--output data/raw/tiny_stories.jsonl
	python data/scripts/clean_code.py \
		--input data/raw/tiny_stories.jsonl \
		--output data/clean_text.jsonl

tokenizer:   ## trains tokenizer/py50k_bpe.{model,vocab}
	$(PYTHON) tokenizer/train_tokenizer.py \
	           --input data/clean_text.jsonl \
	           --model-dir tokenizer \
	           --model-type bpe \
	           --vocab-size 50096 
train:
	PYTHONPATH=. \
	$(PYTHON) -m model.train \
	    --config model/config.json \
	    --out model/checkpoints \
	    --seq_len $(SEQ_LEN) \
	    --batch_size $(BATCH_SIZE) \
	    --total_steps $(TOTAL_STEPS)

eval:
	$(PYTHON) model/eval.py --checkpoint model/checkpoints/latest.safetensors --tokenizer tokenizer/py50k.model

encode:
	PYTHONPATH=. $(PYTHON) scripts/encode_jsonl.py \
	           data/clean_text.jsonl tokenizer/py50k_bpe.model \
	           > data/encoded.txt

clean:
	rm -rf model/checkpoints
