#!/bin/bash

# Set Python path
export PYTHONPATH="/home/fit02/dien-workspace/viamr/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Running training script..."

# Choose GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Run training
torchrun --nproc_per_node=2 -m src.train_sft \
    --dataset1_path "/home/fit02/dien-workspace/viamr/src/data/train_amr_1.txt" \
    --dataset2_path "/home/fit02/dien-workspace/viamr/src/data/train_amr_2.txt" \
    --output_dir "/home/fit02/dien-workspace/viamr/outputs/Qwen-1.7B-SFT-2" \
    --model_name "Qwen/Qwen3-1.7B" \
    --deepspeed_path "/home/fit02/dien-workspace/viamr/src/config/ds_zero2.json" \
    --learning_rate 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_input_length 2048 \
    --max_target_length 2048 \
    --num_train_epochs 20 \
    --save_steps 500 \
    --eval_steps 10 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --use_lora 0 \
    2>&1 | tee Qwen-SFT.log