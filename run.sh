export CUDA_VISIBLE_DEVICES=0
python main.py --dataset ICEWS14 \
    --n-global-epoch 200 \
    --hidden-size 500 \
    --do-pretrain \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1