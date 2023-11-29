export CUDA_VISIBLE_DEVICES=3
python main.py --dataset WIKI \
    --n-global-epoch 100 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --lora-rank 8 \
    --history-length 80 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --base-model-path ./models/modelscope \
    --base-model Llama-2-7b-ms \
    --run-name WIKI_len10_paged \
    --output-dir './outputs/WIKI/llama_len10_paged'