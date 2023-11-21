export CUDA_VISIBLE_DEVICES=5
python main.py --dataset ICEWS14 \
    --n-global-epoch 50 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --use-gnn False \
    --lora-rank 8 \
    --n-ft-epoch 2 \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name lr2e-5_rank8_adamwhf_nokbit \
    --output-dir './outputs/float16_adamwhf_nokbit'