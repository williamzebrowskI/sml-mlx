# # model/utils.py

# import time
# import pathlib
# import mlx.core as mx
# import mlx.nn   as nn
# import sentencepiece as spm

# from model.model  import OpenELM, SMLMConfig

# # _CFG_PATH = pathlib.Path(__file__).parent.parent / "model" / "config.json"
# _CFG_PATH = pathlib.Path(__file__).parent / "config.json"
# _cfg      = SMLMConfig.from_json(str(_CFG_PATH))

# _tok = spm.SentencePieceProcessor(model_file=_cfg.tokenizer_path)
# _model = OpenELM(_cfg)

# # locate latest checkpoint
# _ckpt_dir = pathlib.Path(_cfg.checkpoint_dir)
# _ckpts    = list(_ckpt_dir.glob("ckpt_*.safetensors"))
# if not _ckpts:
#     raise FileNotFoundError(f"No checkpoints found in {_ckpt_dir}")
# def _step(p: pathlib.Path) -> int:
#     return int(p.stem.split("_")[-1])
# _latest = max(_ckpts, key=_step)

# print(f"[utils] loading latest checkpoint → {_latest.name}")
# _model.load_weights(str(_latest))
# mx.eval(_model.parameters())

# # 5) persistent PRNG key
# _key = mx.random.key(int(time.time()*1e6))

# def _encode(text: str) -> mx.array:
#     ids = _tok.encode(text, out_type=int, add_bos=True, add_eos=False)
#     return mx.array(ids, dtype=mx.int32)

# def _decode(ids: mx.array) -> str:
#     return _tok.decode([int(i) for i in ids])

# # def generate_stream(
# #     prompt: str,
# #     max_new:   int   = 128,
# #     top_k:     int   = 40,
# #     temp:      float = 0.7,
# # ):
# #     """
# #     Yield one token at a time as soon as it’s sampled.
# #     """
# #     global _key
# #     ids = _encode(prompt)
# #     yield prompt  # emit the user prompt first, if you like

# #     for _ in range(max_new):
# #         _key, subkey = mx.random.split(_key)
# #         logits = _model(mx.expand_dims(ids,0))[0,-1] / max(temp,1e-6)

# #         if 0 < top_k < logits.size:
# #             vals = mx.topk(logits, k=top_k)
# #             kth  = vals[-1] if vals.ndim else vals
# #             logits = mx.where(logits < kth, -mx.inf, logits)

# #         probs   = nn.softmax(logits, axis=-1)
# #         next_id = int(mx.random.categorical(probs, key=subkey))

# #         # append & decode just this one token
# #         ids = mx.concatenate([ids, mx.array([next_id],dtype=mx.int32)], axis=0)
# #         tok = _tok.decode([next_id])
# #         yield tok

# #         if next_id == _tok.eos_id():
# #             break