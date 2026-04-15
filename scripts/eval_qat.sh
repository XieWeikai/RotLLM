# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.


# For example: bash scripts/eval_qat.sh /data/share/Llama-3.2-3B-Instruct 8 8 8 8
# For example: bash scripts/eval_qat.sh /data/share/Qwen2.5-3B-Instruct 8 8 8 8
# For example: bash scripts/eval_qat.sh /data/share/SmolLM2-1.7B-Instruct 8 8 8 8
# For example: bash scripts/eval_qat.sh /data/share/Qwen3-1.7B 8 8 8 8
# For example: bash scripts/eval_qat.sh /data/share/TinyLlama-1.1B-Chat-v1.0 8 8 8 8
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
--mode "static" \
--granularity "per_tensor" \
--trainable_scale \
--trainable_R \
--need_sample_for_static_init 16 \
--oa_init_type "maxmin" \
--q_init_type "maxmin" \
--k_init_type "maxmin" \
--v_init_type "maxmin" \
--w_bits $2 \
--a_bits $3 \
--q_bits $4 \
--k_bits $4 \
--v_bits $4 \
--oa_bits $5 \
--no-a_sym \
--no-q_sym \
--k_sym \
--v_sym \
--no-oa_sym \
--output_rotation_path "/data/zjh/tensor_final_1/Llama_4_8_32_32_01_001_512_256_i_per_channel_weight_a16down_100.bin" \
--executorch \
--no-task \
# --convert_model_path "/data/zjh/model_pth/Qwen2-3B-it-rotated.pth" \