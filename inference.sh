export CUDA_VISIBLE_DEVICES=5
python inference.py --dataset ICEWS14 \
    --half True \
    --add-prefix \
    --output-dir './outputs/ICEWS14/llama_gcn_prefix_float16_paged_nokbit' \
    --lora-weights-path './outputs/ICEWS14/llama_gcn_prefix_float16_paged_nokbit'