from huggingface_hub import HfApi, HfFolder

# 1) Authenticate
token = HfFolder.get_token()
api   = HfApi()
repo_id = "wzebrowski/slm-mlx-v4-100m"

# 2) Only the minimal files
to_upload = [
    ("model/checkpoints/ckpt_final.safetensors",
     "model/checkpoints/ckpt_final.safetensors"),
    ("model/config.json", "model/config.json"),
    ("tokenizer/spm.model", "tokenizer/spm.model"),
    ("tokenizer/spm.vocab", "tokenizer/spm.vocab"),
]

# 3) Push them
for local_path, hf_path in to_upload:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Uploaded {local_path} → {hf_path}")