# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
# For example: bash scripts/10_optimize_rotation.sh /data/share/Llama-3.2-3B-Instruct 16 4 4
torchrun --nnodes=1 --nproc_per_node=4 -m train.train \
--input_model $1  \
--output_rotation_path "/data/zjh" \
--output_dir "/data/zjh/outputs" \
--logging_dir "/data/zjh/logs" \
--model_max_length 512 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--max_steps 800 \
--logging_steps 10 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing False \
--save_strategy "no" \
--save_safetensors False \
--report_to "tensorboard" \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
