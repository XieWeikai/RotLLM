import re
import torch
import torch.nn as nn
from typing import Optional, List


from train.train_parameter import LearnRotateModule, NoLearnRotateModule
from utils.utils import set_config_attribute


def untie_word_embeddings(model):
    """
    Use the Hugging Face official API to dynamically obtain the input and output embedding layers, avoiding hardcoded paths.
    """

    original_states = set_config_attribute(model, "tie_word_embeddings", False)
    is_tied = (original_states[0] is True) or (original_states[1] is True)

    if is_tied:
        # 动态获取 input_embeddings 和 output_embeddings (lm_head)
        in_embed = model.get_input_embeddings()
        out_embed = model.get_output_embeddings()

        # create a new weight for lm_head
        new_weight = torch.empty_like(in_embed.weight)
        new_weight.copy_(in_embed.weight)

        # copy from in_embed.weight
        out_embed.weight = nn.Parameter(new_weight)

        # ensure that the ptr of weight is not the same
        assert in_embed.weight.data_ptr() != out_embed.weight.data_ptr()



def build_rotation_map(
        model,  
        R1: Optional[LearnRotateModule] = None, 
        R2: Optional[List[LearnRotateModule]] = None, 
        R4: Optional[List[NoLearnRotateModule]] = None
):
    rotation_map = {}

    embed_name = next(name for name, mod in model.named_modules() if mod is model.get_input_embeddings())
    head_name = next(name for name, mod in model.named_modules() if mod is model.get_output_embeddings())
    
    rotation_map[embed_name] = (None, R1, "post")
    rotation_map[head_name] = (R1, None, "pre")


    parent_path = embed_name.rsplit('.', 1)[0]
    text_layers_prefix = f"{parent_path}.layers."

    for name, module in model.named_modules():
        if not name.startswith(text_layers_prefix):
            continue

        # 通过正则提取当前层的索引 (比如 "model.layers.5.self_attn.q_proj" -> 提取出 5)
        match = re.search(r'\.(\d+)\.', name)
        if not match:
            continue
            
        layer_idx = int(match.group(1))

        R2_idx = R2[layer_idx]
        R4_idx = R4[layer_idx]


        # Attention
        if name.endswith(".q_proj") or name.endswith(".k_proj"):
            rotation_map[name] = (R1, None, "pre")
        elif name.endswith(".v_proj"):
            rotation_map[name] = (R1, R2_idx, "around")
        elif name.endswith(".o_proj"):
            rotation_map[name] = (R2_idx, R1, "around")

        # MLP
        elif name.endswith(".gate_proj") or name.endswith(".up_proj"):
            rotation_map[name] = (R1, None, "pre")
        elif name.endswith(".down_proj"):
            rotation_map[name] = (R4_idx, R1, "around")

    return rotation_map