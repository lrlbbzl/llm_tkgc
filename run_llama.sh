export CUDA_VISIBLE_DEVICES=7
python main.py --dataset ICEWS14 \
    --n-global-epoch 50 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --use-gnn False \
    --lora-rank 8 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name noend_lr2e-5_rank8_paged_nokbit \
    --output-dir './outputs/noend_float16_paged_nokbit'