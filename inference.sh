export CUDA_VISIBLE_DEVICES=7
python inference.py --dataset ICEWS14 \
    --half True \
    --output-dir './outputs/noend_float16_paged_nokbit' \
    --lora-weights-path './outputs/noend_float16_paged_nokbit'