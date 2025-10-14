# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.


# For example: bash scripts/10_optimize_rotation.sh /data/share/Llama-3.2-3B 4 4 4
# For example: bash scripts/10_optimize_rotation.sh /data/share/Llama-3.2-1B-Instruct 4 4 4

# For example: bash scripts/10_optimize_rotation.sh /data/share/Llama-3.2-3B-Instruct 4 4 4
# For example: bash scripts/10_optimize_rotation.sh /data/share/Qwen2.5-3B-Instruct 4 4 4
# For example: bash scripts/10_optimize_rotation.sh /data/share/SmolLM2-1.7B-Instruct 4 4 4
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nnodes=1 --nproc_per_node=4 -m train.train \
--input_model $1  \
--output_rotation_path "/data/zjh/tensor/SmolLM_it_8_8_16_0_01_0_01_256_wp64.bin" \
--output_dir "/data/zjh/tensor/outputs" \
--logging_dir "/data/zjh/tensor/logs" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--max_steps 256 \
--logging_steps 1 \
--learning_rate 0.01 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--save_safetensors False \
--report_to "tensorboard" \
--mode "static" \
--granularity "per_tensor" \
--need_sample_for_static_init 16 \
--warmup_step 64 \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_mse \
--no-a_sym \
--no-k_sym \
--no-v_sym \
--k_groupsize 128 \
--v_groupsize 128 \
