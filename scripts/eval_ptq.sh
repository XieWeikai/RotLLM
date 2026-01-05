# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.


# For example: bash scripts/eval_ptq.sh /data/share/Llama-3.2-3B-Instruct 8 8 32 32
# For example: bash scripts/eval_ptq.sh /data/share/Qwen2.5-3B-Instruct 8 8 32 32
# For example: bash scripts/eval_ptq.sh /data/share/SmolLM2-1.7B-Instruct 8 8 32 32
# For example: bash scripts/eval_ptq.sh /data/share/Qwen3-1.7B 8 8 32 32
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=6 torchrun --nnodes=1 --nproc_per_node=1 --master_port=60001 -m runner.runner \
--stage "eval" \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--mode "static" \
--granularity "per_tensor" \
--no-trainable_scale \
--no-trainable_R \
--need_sample_for_static_init 16 \
--a_init_type "mean" \
--w_init_type "mean" \
--oa_init_type "maxmin" \
--q_init_type "maxmin" \
--k_init_type "maxmin" \
--v_init_type "mean" \
--w_bits $2 \
--a_bits $3 \
--q_bits $4 \
--k_bits $4 \
--v_bits $4 \
--oa_bits $5 \
--no-a_sym \
--q_sym \
--k_sym \
--v_sym \
--no-oa_sym \
--no-task \
--no-adaptive_online_rotation_R4 \
--no-adaptive_mixed_precision \
--adapt_R4_percentage 0.1 \
--adapt_activation_percentage 0.06 \
--adapt_need_sample 4 \