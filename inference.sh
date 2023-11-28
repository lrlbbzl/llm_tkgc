export CUDA_VISIBLE_DEVICES=6,7
python inference.py --dataset YAGO \
    --half True \
    --history-length 30 \
    --check-example \
    --output-dir './outputs/YAGO/llama_len30_paged' \
    --lora-weights-path './outputs/YAGO/llama_len30_paged'