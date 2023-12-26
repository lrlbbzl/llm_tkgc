export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download --local-dir-use-symlinks False EleutherAI/gpt-neox-20b --local-dir ./models/gpt-neox-20b

