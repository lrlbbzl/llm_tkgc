export CUDA_VISIBLE_DEVICES=0,1
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
    --ft-direction 'bi' \
    --base-model-path /mnt/data/lrl23/models/modelscope \
    --base-model Llama-2-13b-ms \
    --run-name icews14_13b_len20_bi_aug \
    --output-dir './outputs/ICEWS14/llama_13b_len20_bi_aug'