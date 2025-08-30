# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from dataclasses import dataclass, field
from typing import Optional
import argparse
import transformers

from train.config import AllQuantizeConfigs

# TODO:
# npu 能否做 clip 操作，不同 linear 量化的 bit 数是否不同
# 是否分布式
# bias 怎么量化
# quantizer.py 中，batchsize 中不同样本要不同的 scale，还是共用一个 scale
# 一共六个 scale、zero_point ，都有哪些需要学
# 最后一次输出激活需要量化吗，为什么会有 FP16 的计算
# npu 为什么出来也是 int8，出来也做一次量化吗，那这样岂不是对激活做了两次同样的量化吗
# 计算图是否会循环依赖
# KV cache 里存的应该是量化后的吗？旋转后的吗？


@dataclass
class ModelArguments:
    input_model: Optional[str] = field(
        default="test-input", metadata={"help": "Input model"}
    )
    output_rotation_path: Optional[str] = field(
        default="test-output", metadata={"help": "Output rotation checkpoint path"}
    )  


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )


def parser_gen():
    parser = argparse.ArgumentParser()      

    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )

    # Rotation Arguments
    parser.add_argument(        
        "--rotate",
        action=argparse.BooleanOptionalAction,   
        default=False,
        help="""Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys""",
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )

    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=-1,     
        help="Random Seed for generating random matrix!!",
    )
    parser.add_argument(
        "--fp32_had",   
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_clip_ratio",
        type=float,
        default=1.0,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation""",    
    )
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )

    args, unknown = parser.parse_known_args()
    return args, unknown





def process_args_ptq():
    ptq_args = None

    ptq_args, unknown_args = parser_gen()

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)


    # 创建默认配置实例
    all_qconfigs = AllQuantizeConfigs()

    # activation
    all_qconfigs.activation.num_bits = getattr(ptq_args, "a_bits")
    all_qconfigs.activation.is_symmetric = not getattr(ptq_args, "a_asym")
    all_qconfigs.activation.groupsize = getattr(ptq_args, "a_groupsize")
    all_qconfigs.activation.clip = getattr(ptq_args, "a_clip_ratio")

    # weight
    all_qconfigs.weight.num_bits = getattr(ptq_args, "w_bits")
    all_qconfigs.weight.is_symmetric = not getattr(ptq_args, "w_asym")
    all_qconfigs.weight.groupsize = getattr(ptq_args, "w_groupsize") 
    all_qconfigs.weight.clip = getattr(ptq_args, "w_clip_ratio")

    # TODO: Bias (添加相关命令行参数适用于 RotLLM)
    all_qconfigs.bias.num_bits = 16
    all_qconfigs.bias.is_symmetric = False
    all_qconfigs.bias.groupsize = -1
    all_qconfigs.bias.clip = 1.0

    # key 
    all_qconfigs.key.num_bits = getattr(ptq_args, "k_bits")
    all_qconfigs.key.is_symmetric = not getattr(ptq_args, "k_asym")
    all_qconfigs.key.groupsize = getattr(ptq_args, "k_groupsize")
    all_qconfigs.key.clip = getattr(ptq_args, "k_clip_ratio")

    # value
    all_qconfigs.value.num_bits = getattr(ptq_args, "v_bits")
    all_qconfigs.value.is_symmetric = not getattr(ptq_args, "v_asym")
    all_qconfigs.value.groupsize = getattr(ptq_args, "v_groupsize")
    all_qconfigs.value.clip = getattr(ptq_args, "v_clip_ratio")

    return model_args, training_args, ptq_args, all_qconfigs
