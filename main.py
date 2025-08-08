# from mlx_lm import load
# model, tok = load("openai/gpt-oss-20b")

# def show(name, obj):
#     try:
#         v = eval(name, {}, {"model": model})
#         if hasattr(v, "__len__"):
#             print(f"{name}: len={len(v)} type={type(v)}")
#         else:
#             print(f"{name}: type={type(v)}")
#     except Exception as e:
#         print(f"{name}: !! {type(e).__name__}: {e}")

# print("=== probing ===")
# for cand in [
#     "model.model.layers",
#     "model.layers",
#     "layers",
# ]:
#     show(cand, model)

# for cand in [
#     "model.model.embed_tokens",
#     "model.embed_tokens",
#     "embed_tokens",
# ]:
#     show(cand, model)

# for cand in [
#     "model.lm_head",
#     "lm_head",
#     "model.model.lm_head",
# ]:
#     show(cand, model)

# # If there is a final norm:
# for cand in [
#     "model.model.norm",
#     "model.norm",
# ]:
#     show(cand, model)