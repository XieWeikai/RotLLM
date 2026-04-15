# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

# For example: bash scripts/spinquant/eval.sh /data/share/Llama-3.2-3B 8 8 8
# For example: bash scripts/spinquant/eval.sh /data/share/Llama-3.2-3B-Instruct 8 8 8
# For example: bash scripts/spinquant/eval.sh /data/share/Qwen2.5-3B-Instruct 8 8 8
# For example: bash scripts/spinquant/eval.sh /data/share/SmolLM2-1.7B-Instruct 8 8 8
# For example: bash scripts/spinquant/eval.sh /data/share/Qwen3-1.7B 8 8 8
# For example: bash scripts/spinquant/eval.sh /data/share/Qwen3-VL-2B-Instruct 8 8 8
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=4 torchrun --nnodes=1 --nproc_per_node=1 --master_port=60008 -m runner.runner \
--stage "eval" \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--bf16 False \
--save_safetensors False \
--mode "dynamic" \
--granularity "per_channel" \
--trainable_R \
--no-w_rtn \
--need_sample_for_static_init 16 \
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
--output_rotation_path "/data/zjh/tensor_0413/spinquant/Llama3.2_3B_32_4_4.bin" \
--no-task \