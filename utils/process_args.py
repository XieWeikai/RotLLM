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
    cache_dir: Optional[str] = field(default="/data/zjh/tokenizer")
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
        "--w_mse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )
    parser.add_argument(
        "--w_rtn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
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

    # Path
    parser.add_argument(
        "--optimized_rotation_path",
        type=str,
        default=None,
        help="Load the optimized R1 and R2 from the specified path!",
    )

    args, unknown = parser.parse_known_args()
    return args, unknown





def process_args_ptq():
    ptq_args = None

    ptq_args, unknown_args = parser_gen()

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)

    ptq_args.bsz = training_args.per_device_eval_batch_size
    print("training_args.per_device_eval_batch_size", training_args.per_device_eval_batch_size)


    # Create default config instance
    all_qconfigs = AllQuantizeConfigs()

    # activation
    all_qconfigs.activation.num_bits = getattr(ptq_args, "a_bits")
    all_qconfigs.activation.is_symmetric = not getattr(ptq_args, "a_asym")
    all_qconfigs.activation.groupsize = getattr(ptq_args, "a_groupsize")
    all_qconfigs.activation.clip = getattr(ptq_args, "a_clip_ratio")

    all_qconfigs.activation.int8_down_proj = getattr(ptq_args, "int8_down_proj")

    # weight
    all_qconfigs.weight.num_bits = getattr(ptq_args, "w_bits")
    all_qconfigs.weight.is_symmetric = not getattr(ptq_args, "w_asym")
    all_qconfigs.weight.groupsize = getattr(ptq_args, "w_groupsize") 
    all_qconfigs.weight.clip = getattr(ptq_args, "w_clip_ratio")

    all_qconfigs.weight.mse = getattr(ptq_args, "w_mse")
    all_qconfigs.weight.int8_down_proj = getattr(ptq_args, "int8_down_proj")
    all_qconfigs.weight.rtn = getattr(ptq_args, "w_rtn")
    all_qconfigs.weight.nsamples = getattr(ptq_args, "nsamples")
    all_qconfigs.weight.percdamp = getattr(ptq_args, "percdamp")
    all_qconfigs.weight.act_order = getattr(ptq_args, "act_order")

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
