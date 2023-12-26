export CUDA_VISIBLE_DEVICES=6
python inference.py --dataset ICEWS05-15 \
    --history-length 10 \
    --inference-direction 'right' \
    --base-model Llama-2-7b-ms \
    --partial-num 0 \
    --output-dir './outputs/ICEWS05-15/llama_len10_bi_noaug' \
    --lora-weights-path './outputs/ICEWS05-15/llama_len10_bi_aug'