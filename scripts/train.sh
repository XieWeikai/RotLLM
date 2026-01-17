# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.


# For example: bash scripts/train.sh /data/share/Llama-3.2-3B-Instruct 8 8 32 32
# For example: bash scripts/train.sh /data/share/Qwen2.5-3B-Instruct 8 8 32 32
# For example: bash scripts/train.sh /data/share/SmolLM2-1.7B-Instruct 8 8 32 32
# For example: bash scripts/train.sh /data/share/Qwen3-1.7B 8 8 32 32
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nnodes=1 --nproc_per_node=4 --master_port=45678 -m runner.runner \
--stage "train" \
--input_model $1  \
--output_rotation_path "/data/zjh/tensor_final/Qwen2_4_8_32_32_01_001_1024_512_i_per_channel_weight_a16down_100.bin" \
--output_dir "/data/zjh/tensor_final/outputs" \
--logging_dir "/data/zjh/tensor_final/logs" \
--save_strategy "no" \
--model_max_length 2048 \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--max_steps 1024 \
--logging_steps 10 \
--learning_rate 0.1 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--save_safetensors False \
--report_to "tensorboard" \
--mode "static" \
--granularity "per_tensor" \
--need_sample_for_static_init 16 \
--warmup_step 512 \
--w_bits $2 \
--a_bits $3 \
--q_bits $4 \
--k_bits $4 \
--v_bits $4 \
--oa_bits $5 \
--no-a_sym \
--no-oa_sym \
--no-q_sym \
--no-k_sym \
--no-v_sym \
--a_init_type "maxmin" \
--oa_init_type "maxmin" \
--no-adaptive_down_input_activation_16bits \
--no-adaptive_online_rotation_R4 \
--no-adaptive_mixed_precision \
--adapt_R4_percentage 1.0 \
--adapt_activation_percentage 0.06 \
--adapt_need_sample 4 \
--max_grad_norm 1.0 \