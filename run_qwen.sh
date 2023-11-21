export CUDA_VISIBLE_DEVICES=5
python main.py --dataset ICEWS14 \
    --n-global-epoch 50 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --use-gnn False \
    --lora-rank 32 \
    --base-model-path ./models/modelscope \
    --base-model Qwen-7B-Chat \
    --run-name no_gnn_lr2e-5_rank32_qwen