export CUDA_VISIBLE_DEVICES=3,5,6
python main.py --dataset ICEWS05-15 \
    --n-global-epoch 100 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 8 \
    --sm-batch-size 1 \
    --lora-rank 8 \
    --history-length 30 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --data-augment \
    --inference-direction 'bi' \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name icews05-15_4090_len30_bi_aug \
    --output-dir './outputs/ICEWS05-15/llama_len30_bi_aug'