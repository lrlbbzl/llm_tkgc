export CUDA_VISIBLE_DEVICES=0
python inference.py --dataset WIKI \
    --half True \
    --add-prefix \
    --history-length 20 \
    --output-dir './outputs/WIKI/llama_prefix_float16_paged_nokbit' \
    --lora-weights-path './outputs/WIKI/llama_prefix_float16_paged_nokbit'