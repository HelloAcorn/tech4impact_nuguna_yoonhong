from huggingface_hub import snapshot_download

# Model path on Hugging Face
model_name = "defog/llama-3-sqlcoder-8b"

# Download the model
snapshot_download(repo_id=model_name)
