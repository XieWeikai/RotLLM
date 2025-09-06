import torch
import torch.nn as nn
from typing import Optional, List
import transformers


from .config import AllQuantizeConfigs
from .train_model import RotationQuantLinear, RotationEmbedding
from utils.fuse_norm_utils import fuse_layer_norms
from utils.rotation_utils import get_orthogonal_matrix
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
    替换模型中 nn.Embedding 为 RotationEmbedding，并设置旋转矩阵
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


def collect_fakequant_configs(model):
    """
    Find all FakeQuantizers in the model to facilitate the modification of the 
    quantization configuration in FakeQuantizer using set_special_quantization_configuration.
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
    R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  


    # Add online rotation matrix R3 and R4
    if model.config.model_type == "llama": 
        from modeling.llama import apply_R3R4_change_model 
        print("from modeling.llama import apply_R3R4_change_model")
    elif model.config.model_type == "qwen2": 
        from modeling.qwen2 import apply_R3R4_change_model
        print("from modeling.qwen2 import apply_R3R4_change_model")
    else:
        raise NotImplementedError(f"Unsupported model type {model.config.model_type}")

    apply_R3R4_change_model(model, R3, R4, quant_configs.key, quant_configs.value)


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

    # Initialize all quantizers using the calibration set.
    if quant_configs.weight.mode == "static":
        model.eval()
        with torch.no_grad(): 
            model(batch)

    # Adjust the settings of the quantizer for special layers, change the config.
    model = set_special_quantization_configuration(model, ptq_args)    


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

    return model, R_trainable_parameters, q_trainable_parameters
    