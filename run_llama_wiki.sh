export CUDA_VISIBLE_DEVICES=5
python main.py --dataset WIKI \
    --n-global-epoch 300 \
    --hidden-size 500 \
    --do-pretrain \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --lora-rank 8 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --history-length 20 \
    --add-prefix \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name wiki_prefix_lr2e-5_rank8_paged_nokbit \
    --output-dir './outputs/WIKI/llama_prefix_float16_paged_nokbit'