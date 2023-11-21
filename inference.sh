export CUDA_VISIBLE_DEVICES=5
python inference.py --dataset ICEWS14 \
    --output-dir './outputs/float16_adamwhf_nokbit' \
    --lora-weights-path './outputs/float16_adamwhf_nokbit'