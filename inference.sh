export CUDA_VISIBLE_DEVICES=0
python inference.py --dataset YAGO \
    --history-length 30 \
    --inference-direction 'left' \
    --base-model-path '/mnt/data/lrl23/models/modelscope' \
    --base-model Llama-2-7b-ms \
    --partial-num 0 \
    --output-dir './outputs/ICEWS14/llama_len20_bi_aug' \
    --lora-weights-path './outputs/ICEWS14/llama_len20_bi_aug'