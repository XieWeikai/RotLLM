# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.


# For example: bash scripts/spinquant/train.sh /data/share/Llama-3.2-3B 32 8 8
# For example: bash scripts/spinquant/train.sh /data/share/Llama-3.2-3B-Instruct 32 8 8
# For example: bash scripts/spinquant/train.sh /data/share/Qwen2.5-3B-Instruct 32 8 8
# For example: bash scripts/spinquant/train.sh /data/share/SmolLM2-1.7B-Instruct 32 8 8
# For example: bash scripts/spinquant/train.sh /data/share/Qwen3-1.7B 32 8 8
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nnodes=1 --nproc_per_node=4 --master_port=45678 -m runner.runner \
--stage "train" \
--input_model $1  \
--output_rotation_path "/data/zjh/tensor_0413/spinquant/Llama3.2_3B_32_4_4.bin" \
--output_dir "/data/zjh/tensor_0413/spinquant/outputs" \
--logging_dir "/data/zjh/tensor_0413/spinquant/logs" \
--save_strategy "no" \
--model_max_length 2048 \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 2 \
--max_steps 100 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--save_safetensors False \
--report_to "tensorboard" \
--mode "dynamic" \
--granularity "per_channel" \
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