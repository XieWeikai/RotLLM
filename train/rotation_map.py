from typing import Optional, List


from train.train_parameter import LearnRotateModule, NoLearnRotateModule

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