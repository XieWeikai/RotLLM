import torch
import torch.nn as nn
from typing import Optional, List


from train.train_parameter import LearnRotateModule, NoLearnRotateModule


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