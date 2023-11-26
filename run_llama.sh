export CUDA_VISIBLE_DEVICES=4
python main.py --dataset ICEWS18 \
    --n-global-epoch 200 \
    --hidden-size 500 \
    --do-pretrain \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --lora-rank 8 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name icews18_lr2e-5_rank8_paged_nokbit \
    --output-dir './outputs/ICEWS18/llama_float16_paged_nokbit'