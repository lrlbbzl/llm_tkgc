export CUDA_VISIBLE_DEVICES=5
python inference.py --dataset ICEWS14 \
    --history-length 10 \
    --inference-direction 'left' \
    --data-augment \
    --output-dir './outputs/ICEWS14/llama_len10_bi_aug' \
    --lora-weights-path './outputs/ICEWS14/llama_len10_bi_aug'