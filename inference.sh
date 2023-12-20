export CUDA_VISIBLE_DEVICES=4
python inference.py --dataset ICEWS14 \
    --history-length 20 \
    --inference-direction 'left' \
    --data-augment \
    --output-dir './outputs/ICEWS14/llama_len20_bi_aug' \
    --lora-weights-path './outputs/ICEWS14/llama_len20_bi_aug'