import torch
import torch.nn as nn
from typing import Dict, Optional, List
import transformers
import copy


from .config import AllQuantizeConfigs
from .train_model import RotationQuantLinear, RotationEmbedding
from utils.fuse_norm_utils import fuse_layer_norms
from utils.rotation_utils import get_orthogonal_matrix
from modeling.qwen2 import apply_R3R4_change_qwen2_model
from modeling.llama import apply_R3R4_change_llama_model
from .train_parameter import LearnRotateModule, NoLearnRotateModule, FakeQuantizer


def untie_word_embeddings(model):
    if model.config.tie_word_embeddings:
        model.config.tie_word_embeddings = False

        # create a new weight for lm_head
        new_weight = torch.empty_like(model.model.embed_tokens.weight)
        new_weight.copy_(model.model.embed_tokens.weight)

        # copy from model.model.embed_tokens.weight
        model.lm_head.weight = nn.Parameter(new_weight)

        # ensure that the ptr of weight of lm_head is not the same as ptr of the weight of embed_tokens
        assert model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()

def build_rotation_map(
        num_layers, 
        hidden_size, 
        R1: Optional[LearnRotateModule] = None, 
        R2: Optional[List[LearnRotateModule]] = None, 
        R4: Optional[List[NoLearnRotateModule]] = None
):
    """
    You can flexibly control whether to introduce a rotation matrix at each position(pre, post, around(means pre and post))
    """
    rotation_map = {}

    rotation_map["model.embed_tokens"] = (
        None,
        R1,
        "post"
    )

    rotation_map["lm_head"] = (
        R1,
        None,
        "pre"
    )

    for i in range(num_layers):
        # Attention
        rotation_map[f"model.layers.{i}.self_attn.q_proj"] = (
            R1,
            None,
            "pre"
        )
        rotation_map[f"model.layers.{i}.self_attn.k_proj"] = (
            R1,
            None,
            "pre"
        )
        rotation_map[f"model.layers.{i}.self_attn.v_proj"] = (
            R1,
            R2[i],
            "around"
        )
        rotation_map[f"model.layers.{i}.self_attn.o_proj"] = (
            R2[i],
            R1,
            "around"
        )

        # MLP
        rotation_map[f"model.layers.{i}.mlp.gate_proj"] = (
            R1, 
            None,
            "pre"
        )
        rotation_map[f"model.layers.{i}.mlp.up_proj"] = (
            R1, 
            None,
            "pre"
        )
        rotation_map[f"model.layers.{i}.mlp.down_proj"] = (
            R4[i],
            R1, 
            "around"
        )

    return rotation_map


def replace_linear_with_rotation_quant(
    model: nn.Module,
    quant_configs: AllQuantizeConfigs,
    rotation_map: dict = None,
    prefix: str = ""  # 记录父路径
):
    """
    替换模型中所有 nn.Linear 为 RotationQuantLinear

    Args:
        model (nn.Module): 原始模型
        quant_configs (AllQuantizeConfigs): activation/weight/bias/key/value 量化配置
        rotation_map (dict): key=模块名, value=(R_pre, R_post, rotation_pos)
        prefix (str): 记录父路径，以便从 rotation_map 提取旋转配置
    Returns:
        nn.Module: 替换完成的模型
    """

    # 遍历模型模块，记录父模块和名字
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # 如果子模块是 nn.Linear，替换
        if isinstance(module, nn.Linear):
            R_pre, R_post, rotation_pos = None, None, "none"
            if rotation_map and full_name in rotation_map:
                R_pre, R_post, rotation_pos = rotation_map[full_name]

            # 构建 RotationQuantLinear
            new_module = RotationQuantLinear(
                config=quant_configs,
                linear=module,
                rotation_pos=rotation_pos,
                R_pre=R_pre,
                R_post=R_post
            )

            # 替换父模块中的子模块
            setattr(model, name, new_module)

        else:
            # 如果不是 Linear，递归处理子模块
            replace_linear_with_rotation_quant(module, quant_configs, rotation_map, prefix=full_name)

    return model



def replace_embedding_with_rotation_embedding(model: nn.Module, rotation_map: dict = None, prefix: str = ""):
    """
    替换模型中 nn.Embedding 为 RotationEmbedding，并设置旋转矩阵
    """
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # 如果子模块是 nn.Embedding，替换
        if isinstance(module, nn.Embedding):
            R_pre, R_post, rotation_pos = None, None, "none"
            if rotation_map and full_name in rotation_map:
                R_pre, R_post, rotation_pos = rotation_map[full_name]
            else:
                print("No!",full_name)

            # 替换父模块中的子模块
            setattr(model, name, RotationEmbedding(embedding=module, rotation_pos=rotation_pos, R_pre=R_pre, R_post=R_post))
            break
        else:
            # 如果不是 Embedding，递归处理子模块
            replace_embedding_with_rotation_embedding(module, rotation_map, prefix=full_name)
    return model


def collect_fakequant_configs(model):
    """
    找到 model 中所有 FakeQuantizer，便于 set_special_quantization_configuration 修改 FakeQuantizer 中的量化配置 config
    """
    fq_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizer):
            fq_dict[name] = module.config
    return fq_dict

def model_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, "groupsize should be greater than 1!"

    if model.config.intermediate_size % groupsize == 0:
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, "Invalid groupsize for llama!"

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, "Invalid groupsize for down_proj!"
    return down_proj_groupsize


def set_special_quantization_configuration(model, ptq_args):
    subset = collect_fakequant_configs(model)
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size
    head_dim = dim // num_heads

    for name in subset:
        # weight:
        if ptq_args.w_bits < 16 and "weightQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 16
            if ptq_args.int8_down_proj and "down_proj" in name:
                subset[name].num_bits = 8

        # activation:
        if ptq_args.a_bits < 16 and "actQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 16
            if "down_proj" in name:
                if ptq_args.int8_down_proj:
                    subset[name].num_bits = 8
                if ptq_args.a_groupsize > 0:
                    down_proj_groupsize = model_down_proj_groupsize(model, ptq_args.a_groupsize)
                    subset[name].groupsize = down_proj_groupsize
            if "o_proj" in name:
                subset[name].groupsize = head_dim

        # value:
        if ptq_args.v_bits < 16 and "vQuant" in name:
            if "v_proj" in name:
                subset[name].groupsize = head_dim
    return model 


def prepare_model(model, batch: torch.Tensor, quant_configs: AllQuantizeConfigs, ptq_args):
    transformers.set_seed(ptq_args.seed)
    device = model.device

    # untie embedding and lm_head
    untie_word_embeddings(model)
    # 模型预处理
    fuse_layer_norms(model)

    # 设置训练参数
    for param in model.parameters():
        param.requires_grad = False

    model.config.use_cache = False

    # 准备旋转矩阵
    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = dim // num_heads
    hidden_dim = model.config.intermediate_size

    # 生成 Hadamard 旋转矩阵
    R1 = LearnRotateModule(get_orthogonal_matrix(dim, mode="hadamard", device=device))
    R2 = [LearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]
    R3 = [NoLearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]
    R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  

    # TODO: 修改代码使该位置统一
    # 添加 online 旋转矩阵 R3、R4
    # apply_R3R4_change_qwen2_model(model, R3, R4, copy.deepcopy(quant_configs.key), copy.deepcopy(quant_configs.value))
    apply_R3R4_change_llama_model(model, R3, R4, copy.deepcopy(quant_configs.key), copy.deepcopy(quant_configs.value))


    # 准备旋转矩阵、旋转位置
    rotation_map = build_rotation_map(num_layers, dim, R1, R2, R4)

    # 调用替换函数，替换线性层，添加旋转矩阵
    model = replace_linear_with_rotation_quant(
        model,
        quant_configs=quant_configs,
        rotation_map=rotation_map
    )

    # 调用替换函数，替换 Embedding 层，添加旋转矩阵
    model = replace_embedding_with_rotation_embedding(
        model,
        rotation_map=rotation_map
    )

    # 使用校准集中样本 batch 初始化所有 quantizer
    model.eval()
    with torch.no_grad(): 
        model(batch)

    # 处理特殊层的 quantizer 的设置，改变 config
    model = set_special_quantization_configuration(model, ptq_args)    


    # 整合可训练参数
    R_trainable_parameters = [R1.weight] + [r.weight for r in R2]
    q_trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]

    new_q_trainable_parameters = []
    for p in q_trainable_parameters:
        if all(p is not r for r in R_trainable_parameters):
            new_q_trainable_parameters.append(p)
    q_trainable_parameters = new_q_trainable_parameters

    return model, R_trainable_parameters, q_trainable_parameters
    