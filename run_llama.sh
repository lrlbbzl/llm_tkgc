export CUDA_VISIBLE_DEVICES=0
python main.py --dataset ICEWS14 \
    --n-global-epoch 100 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --lora-rank 8 \
    --history-length 20 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --data-augment \
    --inference-direction 'bi' \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name icews14_len20_bi_aug \
    --output-dir './outputs/ICEWS14/llama_len20_bi_aug'