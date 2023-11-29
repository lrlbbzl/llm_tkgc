export CUDA_VISIBLE_DEVICES=5
python inference.py --dataset ICEWS05-15 \
    --half True \
    --history-length 10 \
    --output-dir './outputs/ICEWS18/llama_len10_paged' \
    --lora-weights-path './outputs/ICEWS18/llama_len10_paged'