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
import torch

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
    # Use for train
    parser.add_argument(
        "--adaptive_mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""If it is false, you can customize the W-A-KV quantization precision. 
        If it is true, the default 4-4-16 quantization is used, 
        automatically selecting certain important positions to improve quantization precision.""",
    )
    parser.add_argument(
        "--adapt_need_sample",
        type=int,
        default=0,
        help="Number of samples used for analysis in adaptive mixed-precision quantization.",
    )
    parser.add_argument(
        "--adapt_activation_percentage",
        type=float,
        default=1.0,
        help="The percentage of all activated positions that increased from 4-bit quantization to 8-bit quantization.",
    )


    # Use for eval
    parser.add_argument(
        "--trainable_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Valid only when mode="static". 
        If true, it indicates that static quantization includes a trainable scale; 
        If false, it indicates that dynamic quantization only trains R, and uses the static quantization to initialize the scale for evaluation.""",
    )
    parser.add_argument(
        "--trainable_R",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Valid only when mode="static". 
        If true, it indicates using the rotation matrix R optimized by training; 
        If false, it indicates using the Hadamard matrix as the rotation matrix.""",
    )
    parser.add_argument(
        "--task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Decide whether to test all tasks. If true, test all task metrics; if false, only test PPL.""",
    )
    parser.add_argument(
        "--sageattn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Decide whether to use the SageAttention quantization method. 
        Note: sageattn=true and k_bits, v_bits < 16 cannot both be true at the same time, otherwise it will cause duplicate quantization.""",
    )

    # Used for static quantization
    parser.add_argument(
        "--mode",
        type=str,
        default="static",
        help="Static quantization or Dynamic quatization",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="per_tensor",
        help="per_tensor or per_channel",
    )
    parser.add_argument(
        "--need_sample_for_static_init",
        type=int,
        default=0,
        help="The number of samples required for static quantization to initialize activations and weights.",
    )
    parser.add_argument(
        "--warmup_step",
        type=int,
        default=0,
        help="The number of steps used to adjust the activated quantization parameters during the initial training phase.",
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
        "--a_sym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )
    parser.add_argument(
        "--a_init_type",
        type=str,
        default="mean",
        help="mean or maxmin",
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
        "--w_sym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_clip_ratio",
        type=float,
        default=1.0,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--w_init_type",
        type=str,
        default="mean",
        help="mean or maxmin",
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
    parser.add_argument(
        "--v_groupsize", 
        type=int, 
        default=-1,
    )
    parser.add_argument(
        "--v_sym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )
    parser.add_argument(
        "--v_init_type",
        type=str,
        default="mean",
        help="mean or maxmin",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument(
        "--k_groupsize", 
        type=int, 
        default=-1,
    )
    parser.add_argument(
        "--k_sym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )
    parser.add_argument(
        "--k_init_type",
        type=str,
        default="mean",
        help="mean or maxmin",
    )

    args, unknown = parser.parse_known_args()
    return args, unknown





def process_args_ptq():
    ptq_args = None

    ptq_args, unknown_args = parser_gen()

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)

    ptq_args.bsz = training_args.per_device_eval_batch_size

    # Create default config instance
    all_qconfigs = AllQuantizeConfigs()

    # activation
    all_qconfigs.activation.mode = getattr(ptq_args, "mode")
    all_qconfigs.activation.granularity = getattr(ptq_args, "granularity")
    all_qconfigs.activation.need_sample_for_static_init = getattr(ptq_args, "need_sample_for_static_init")
    all_qconfigs.activation.warmup_step = torch.tensor(getattr(ptq_args, "warmup_step"))
    all_qconfigs.activation.init_type = getattr(ptq_args, "a_init_type")

    all_qconfigs.activation.num_bits = getattr(ptq_args, "a_bits")
    all_qconfigs.activation.is_symmetric = getattr(ptq_args, "a_sym")
    all_qconfigs.activation.groupsize = getattr(ptq_args, "a_groupsize")
    all_qconfigs.activation.clip_ratio = getattr(ptq_args, "a_clip_ratio")

    all_qconfigs.activation.int8_down_proj = getattr(ptq_args, "int8_down_proj")

    # weight
    all_qconfigs.weight.mode = getattr(ptq_args, "mode")
    all_qconfigs.weight.granularity = getattr(ptq_args, "granularity")
    all_qconfigs.weight.need_sample_for_static_init = getattr(ptq_args, "need_sample_for_static_init")
    all_qconfigs.weight.init_type = getattr(ptq_args, "w_init_type")

    all_qconfigs.weight.num_bits = getattr(ptq_args, "w_bits")
    all_qconfigs.weight.is_symmetric = getattr(ptq_args, "w_sym")
    all_qconfigs.weight.groupsize = getattr(ptq_args, "w_groupsize") 
    all_qconfigs.weight.clip_ratio = getattr(ptq_args, "w_clip_ratio")

    all_qconfigs.weight.mse = getattr(ptq_args, "w_mse")
    all_qconfigs.weight.int8_down_proj = getattr(ptq_args, "int8_down_proj")
    all_qconfigs.weight.rtn = getattr(ptq_args, "w_rtn")
    all_qconfigs.weight.nsamples = getattr(ptq_args, "nsamples")
    all_qconfigs.weight.percdamp = getattr(ptq_args, "percdamp")
    all_qconfigs.weight.act_order = getattr(ptq_args, "act_order")

    # Bias: 不做任何量化，但保留该接口
    all_qconfigs.bias.num_bits = 16

    # key 
    all_qconfigs.key.mode = getattr(ptq_args, "mode")
    all_qconfigs.key.granularity = getattr(ptq_args, "granularity")
    all_qconfigs.key.need_sample_for_static_init = getattr(ptq_args, "need_sample_for_static_init")
    all_qconfigs.key.warmup_step = torch.tensor(getattr(ptq_args, "warmup_step"))
    all_qconfigs.key.init_type = getattr(ptq_args, "k_init_type")

    all_qconfigs.key.num_bits = getattr(ptq_args, "k_bits")
    all_qconfigs.key.is_symmetric = getattr(ptq_args, "k_sym")
    all_qconfigs.key.groupsize = getattr(ptq_args, "k_groupsize")
    all_qconfigs.key.clip_ratio = getattr(ptq_args, "k_clip_ratio")

    # value
    all_qconfigs.value.mode = getattr(ptq_args, "mode")
    all_qconfigs.value.granularity = getattr(ptq_args, "granularity")
    all_qconfigs.value.need_sample_for_static_init = getattr(ptq_args, "need_sample_for_static_init")
    all_qconfigs.value.warmup_step = torch.tensor(getattr(ptq_args, "warmup_step"))
    all_qconfigs.value.init_type = getattr(ptq_args, "v_init_type")

    all_qconfigs.value.num_bits = getattr(ptq_args, "v_bits")
    all_qconfigs.value.is_symmetric = getattr(ptq_args, "v_sym")
    all_qconfigs.value.groupsize = getattr(ptq_args, "v_groupsize")
    all_qconfigs.value.clip_ratio = getattr(ptq_args, "v_clip_ratio")

    return model_args, training_args, ptq_args, all_qconfigs
