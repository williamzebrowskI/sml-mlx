from huggingface_hub import HfApi, HfFolder

# 1) Authenticate
token = HfFolder.get_token()
api   = HfApi()
repo_id = "wzebrowski/mlx_slm_ft"

# 2) Only the minimal files
to_upload = [
    ("mlx_lora_peft/adapter_model.safetensors",
     "adapter_model.safetensors"),
    ("mlx_lora_peft/adapter_config.json", "adapter_config.json"),
    ("tokenizer/spm.model", "spm.model"),
    ("tokenizer/spm.vocab", "spm.vocab"),
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