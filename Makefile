
PYTHON := python
LATEST_CKPT := $(shell ls -1 model/checkpoints_so/ckpt_121000.safetensors 2>/dev/null | sort | tail -n1)

.PHONY: data tokenizer train eval clean

data:
	python data/scripts/fetch_openwebtext.py \
		--dataset       Skylion007/openwebtext \
		--split         train \
		--token_budget  2000000000 \
		--output        data/raw/openwebtext_2B.jsonl

tokenizer:   ## trains tokenizer/py50k_bpe.{model,vocab}
	python tokenizer/train_tokenizer.py \
		--input data/raw/wikitext_train.jsonl \
		--model-dir tokenizer \
		--vocab-size 50096 \
		--model-type bpe
train:
	python model/train.py \
	  --config       model/config.json \
	  --dataset      data/raw/wikitext_train.jsonl \
	  --tokenizer    tokenizer/spm.model \
	  --out          model/checkpoints \
	  --context_size $(CONTEXT) \
	  --batch_size   $(BATCH) \
	  --total_steps  $(STEPS) \
	  --lr           $(LR) \
	  --warmup       $(WARMUP) \
	  --steps_report $(REPORT) \
	  --steps_checkpoint $(CKPT)

eval:
	PYTHONPATH=. \
	python -m model.eval \
	    --checkpoint model/checkpoints/ckpt_final.safetensors \
	    --tokenizer tokenizer/spm.model \
		$(ARGS)

encode:
	PYTHONPATH=. $(PYTHON) scripts/encode_jsonl.py \
	           data/clean_text.jsonl tokenizer/py50k_bpe.model \
	           > data/encoded.txt

test:   ## run eval_suite.py on the newest checkpoint
	PYTHONPATH=. \
	python scripts/eval_suite.py \
	    --checkpoint $(LATEST_CKPT) \
	    --config     model/config.json \
	    --tokenizer  tokenizer/spm.model

clean:
	rm -rf model/checkpoints
