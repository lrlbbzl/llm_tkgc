export CUDA_VISIBLE_DEVICES=2,3
python main.py --dataset YAGO \
    --n-global-epoch 200 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --lora-rank 8 \
    --history-length 30 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name yago_len30_paged \
    --output-dir './outputs/YAGO/llama_len30_paged'