import torch
import torch.nn as nn
from typing import Optional, List
from tqdm import tqdm
import importlib


from .config import AllQuantizeConfigs
from .train_model import RotationQuantLinear, RotationEmbedding
from utils.fuse_norm_utils import fuse_layer_norms
from utils.rotation_utils import get_orthogonal_matrix
from .train_parameter import LearnRotateModule, NoLearnRotateModule, FakeQuantizer
from modeling.monkeypatch import add_qkv_rotation_quant
from utils.utils import get_local_rank, log
from utils.adapt_mix_precision import adapt_modify_quantization_precision, collect_fakequant_configs
from attention.core import Quant_scaled_dot_product_attention
from utils.adapt_online_rotation import adapt_choose_online_rotation


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
    prefix: str = ""  # Record parent path
):
    """
    Replace all nn.Linear in the model with RotationQuantLinear.

    Args:
        model (nn.Module): original model
        quant_configs (AllQuantizeConfigs): activation/weight/bias/key/value quantitative config
        rotation_map (dict): key=module name, value=(R_pre, R_post, rotation_pos)
        prefix (str): Record the parent path in order to extract the rotation configuration from the rotation_map
    Returns:
        nn.Module: The completed model after replacement
    """

    # Traverse the model module, recording the parent module and name
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # If the submodule is nn.Linear, replace it
        if isinstance(module, nn.Linear):
            R_pre, R_post, rotation_pos = None, None, "none"
            if rotation_map and full_name in rotation_map:
                R_pre, R_post, rotation_pos = rotation_map[full_name]

            # Build RotationQuantLinear
            new_module = RotationQuantLinear(
                config=quant_configs,
                linear=module,
                rotation_pos=rotation_pos,
                R_pre=R_pre,
                R_post=R_post
            )

            # Replace the submodule in the parent module
            setattr(model, name, new_module)

        else:
            # If not linear, recursively process submodules
            replace_linear_with_rotation_quant(module, quant_configs, rotation_map, prefix=full_name)

    return model



def replace_embedding_with_rotation_embedding(model: nn.Module, rotation_map: dict = None, prefix: str = ""):
    """
    Replace all nn.Embedding in the model with RotationEmbedding
    """
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # If the submodule is nn.Embedding, replace it
        if isinstance(module, nn.Embedding):
            R_pre, R_post, rotation_pos = None, None, "none"
            if rotation_map and full_name in rotation_map:
                R_pre, R_post, rotation_pos = rotation_map[full_name]

            # Replace the submodule in the parent module
            setattr(model, name, RotationEmbedding(embedding=module, rotation_pos=rotation_pos, R_pre=R_pre, R_post=R_post))
            break
        else:
            # If not embedding, recursively process submodules.
            replace_embedding_with_rotation_embedding(module, rotation_map, prefix=full_name)
    return model


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

        # out_activation:
        if ptq_args.oa_bits < 16 and "outActQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 16
    return model 


def share_linear_scale_and_zero_point(model, local_rank=None):
    if local_rank is None:
        local_rank = 0
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Sharing parameter scale and zero_point (qkv_proj and gateup)", disable=not (local_rank == 0)):
        layer = layers[i]
        shared_parameter = {
            "qkv_scale": None,    # q_proj, k_proj, v_proj 共享
            "qkv_zero_point": None,
            "gateup_scale": None, # gate_proj, up_proj 共享
            "gateup_zero_point": None,
        }
        for name, linear in layer.named_modules():
            if isinstance(linear, RotationQuantLinear):
                # 共享 q/k/v 的激活量化 scale
                if any(key in name for key in ["q_proj", "k_proj", "v_proj"]):
                    if shared_parameter["qkv_scale"] is None:
                        shared_parameter["qkv_scale"] = linear.actQuant.scale
                        shared_parameter["qkv_zero_point"] = linear.actQuant.zero_point
                    else:
                        # 引用同一个 nn.Parameter 对象
                        linear.actQuant.scale = shared_parameter["qkv_scale"]
                        linear.actQuant.zero_point = shared_parameter["qkv_zero_point"]
                    # 由于该处 scale、zero_point 由3个 linear 共享，因此计算 LSQ+ activation initialization loss 时需要除以3，避免重复累积 loss
                    linear.actQuant.config.warmup_share_parameter_num = 3

                # 共享 gate/up 的激活量化 scale
                elif any(key in name for key in ["gate_proj", "up_proj"]):
                    if shared_parameter["gateup_scale"] is None:
                        shared_parameter["gateup_scale"] = linear.actQuant.scale
                        shared_parameter["gateup_zero_point"] = linear.actQuant.zero_point
                    else:
                        linear.actQuant.scale = shared_parameter["gateup_scale"]
                        linear.actQuant.zero_point = shared_parameter["gateup_zero_point"]
                    # 由于该处 scale、zero_point 由2个 linear 共享，因此计算 LSQ+ activation initialization loss 时需要除以2，避免重复累积 loss
                    linear.actQuant.config.warmup_share_parameter_num = 2


def prepare_model(model, quant_configs: AllQuantizeConfigs, ptq_args, batch: Optional[torch.Tensor] = None):
    device = model.device
    model.eval()
    local_rank = get_local_rank()

    # untie embedding and lm_head
    untie_word_embeddings(model)
    # Model preprocessing
    fuse_layer_norms(model)

    # Set training parameters
    for param in model.parameters():
        param.requires_grad = False

    model.config.use_cache = False

    # Prepare rotation matrix
    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = dim // num_heads
    hidden_dim = model.config.intermediate_size

    # Generate Hadamard rotation matrix
    R1 = LearnRotateModule(get_orthogonal_matrix(dim, mode="hadamard", device=device))
    R2 = [LearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]
    R3 = [NoLearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]

    if ptq_args.adaptive_online_rotation_R4:
        R4_hadamard = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  
        R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="identity", device=device)) for _ in range(num_layers)]  
    else:
        R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  


    # Prepare the rotation matrix and the rotation position
    rotation_map = build_rotation_map(num_layers, R1, R2, R4)

    # Call the replacement function, replace the linear layer, and add the rotation matrix and quantizer
    model = replace_linear_with_rotation_quant(
        model,
        quant_configs=quant_configs,
        rotation_map=rotation_map
    )

    # Call the replacement function, replace the Embedding layer, and add the rotation matrix and quantizer
    model = replace_embedding_with_rotation_embedding(
        model,
        rotation_map=rotation_map
    )

    # 添加在线旋转矩阵 R4
    model_type = model.config.model_type
    # 动态导入对应的 modeling 模块
    try:
        modeling_module = importlib.import_module(f"modeling.{model_type}")
    except ModuleNotFoundError:
        raise ImportError(f"Cannot find modeling module for '{model_type}' (expected modeling/{model_type}.py)")

    # 检查模块中是否定义了 apply_R4_change_model
    if not hasattr(modeling_module, "apply_R4_change_model"):
        raise AttributeError(f"'modeling.{model_type}' does not define function 'apply_R4_change_model'")

    # 调用函数
    func = getattr(modeling_module, "apply_R4_change_model")
    if local_rank == 0:
        log.info(f"✅ Found 'apply_R4_change_model' in modeling.{model_type}, now calling it...")
    func(model, R4, local_rank)

    
    # 必须要先将所有的 linear 替换成 RotationQuantLinear，然后再调用下面函数为 Value 添加量化操作，同时还对 Query、Key 添加在线旋转 R3、量化操作
    add_qkv_rotation_quant(model, R3, quant_configs.query, quant_configs.key, quant_configs.value, local_rank=local_rank)
    

    if ptq_args.sageattn:
        torch.nn.functional.scaled_dot_product_attention = Quant_scaled_dot_product_attention

    # Adjust the settings of the quantizer for special layers, change the config.
    model = set_special_quantization_configuration(model, ptq_args) 

    # 默认每层都需要在线旋转矩阵 R4
    adaptive_R4 = {i: True for i in range(num_layers)}
    # Initialize all quantizers using the calibration set.
    if quant_configs.weight.mode == "static":
        assert batch is not None, "We need to prepare the initial sample set required for static quantization."
        model.eval()

        if ptq_args.adaptive_mixed_precision or ptq_args.adaptive_online_rotation_R4:
            batch_find_threshold = batch[-ptq_args.adapt_need_sample:]
            batch = batch[:ptq_args.need_sample_for_static_init]

            if ptq_args.adaptive_online_rotation_R4:
                # 将 donw_proj 层的 activation 修改为 8 bit
                from utils.adapt_online_rotation import modify_down_proj_activation_8bit
                model = modify_down_proj_activation_8bit(model)

        if local_rank == 0:
            collect_fakequant_configs(model, "./txt/one.txt", True)

        with torch.no_grad(): 
            for i in tqdm(range(batch.size(0)), desc="Init scale and zero_point for static quant", disable=not (local_rank == 0)):
                sample = batch[i].unsqueeze(0)  # 保持 batch 维度
                model(sample)
            if local_rank == 0:
                log.info(f"✅ Init scale and zero_point ok!")

        if ptq_args.adaptive_mixed_precision or ptq_args.adaptive_online_rotation_R4:
            assert batch_find_threshold is not None, "batch_find_threshold should not be empty."
            
            if ptq_args.adaptive_online_rotation_R4:
                # 选择在线旋转矩阵 R4
                model, adaptive_R4 = adapt_choose_online_rotation(model, R4_hadamard, batch, batch_find_threshold, ptq_args, local_rank)

            if ptq_args.adaptive_mixed_precision:
                model = adapt_modify_quantization_precision(model, adaptive_R4, batch, batch_find_threshold, ptq_args, local_rank)


    # 将 q_proj、k_proj、v_proj 前的激活量化器中的 scale、zero_point 共享同一个参数
    # 将 gate_proj、up_proj 前的激活量化器中的 scale、zero_point 共享同一个参数
    share_linear_scale_and_zero_point(model, local_rank)

    # Integration of trainable parameters
    R_trainable_parameters = [R1.weight] + [r.weight for r in R2]
    q_trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]

    new_q_trainable_parameters = []
    for p in q_trainable_parameters:
        if all(p is not r for r in R_trainable_parameters):
            new_q_trainable_parameters.append(p)
    q_trainable_parameters = new_q_trainable_parameters

    return model, adaptive_R4, R_trainable_parameters, q_trainable_parameters
    