import torch
import torch.nn as nn


from train.train_utils import build_rotation_map


def add_rotation_to_linear(model: nn.Module, rotation_map: dict):
    """
    Merge all mergeable rotation matrices into the specific layers defined in rotation_map.

    Args:
        model (nn.Module): original model
        rotation_map (dict): key=module name, value=(R_pre, R_post, rotation_pos)
    """
    if not rotation_map:
        return

    # 直接遍历白名单字典
    for full_name, (R_pre, R_post, rotation_pos) in rotation_map.items():
        try:
            # 根据字符串路径，直接从模型中抓取该层的实例
            module = model.get_submodule(full_name)
        except AttributeError:
            print(f"Warning: Module {full_name} not found in model. Skipping.")
            continue

        if isinstance(module, nn.Linear):
            rotate_linear(module, R_pre, R_post, rotation_pos)
        elif isinstance(module, nn.Embedding):
            rotate_embedding(module, R_pre, R_post, rotation_pos)


def rotate_linear(module: nn.Linear, R_pre, R_post, rotation_pos):
    w = module.weight.data
    b = module.bias.data if module.bias is not None else None
    if rotation_pos in ["pre", "around"]:
        assert w.shape[1] % R_pre.shape[0] == 0, "Input dim should be multiple of R_pre dim"
        num_blocks = w.shape[1] // R_pre.shape[0]

        w_dtype = w.dtype
        w_device = w.device
        w = w.view(w.shape[0], num_blocks, R_pre.shape[0])
        w = (w.to(R_pre.dtype) @ R_pre.to(device=w_device)).to(dtype=w_dtype)    
        w = w.view(w.shape[0], num_blocks * R_pre.shape[0])

    if rotation_pos in ["post", "around"]:
        assert w.shape[0] % R_post.shape[0] == 0, "Output dim(weight) should be multiple of R_post dim"
        num_blocks = w.shape[0] // R_post.shape[0]

        w_dtype = w.dtype
        w_device = w.device
        w = w.T
        w = w.view(w.shape[0], num_blocks, R_post.shape[0])
        w = (w.to(R_post.dtype) @ R_post.to(device=w_device)).to(dtype=w_dtype)
        w = w.view(w.shape[0], num_blocks * R_post.shape[0])
        w = w.T
        if b is not None:
            assert b.shape[0] % R_post.shape[0] == 0, "Output dim(bias) should be multiple of R_post dim"
            b_dtype = b.dtype
            b_device = b.device
            b = (b.to(R_post.dtype).view(num_blocks, -1) @ R_post.to(device=b_device)).to(dtype=b_dtype)
            b = b.view(-1)
    module.weight.data = w
    if b is not None:
        module.bias.data = b


def rotate_embedding(module: nn.Embedding, R_pre, R_post, rotation_pos):
    w = module.weight.data

    assert rotation_pos not in ["pre", "around"], "An error occurred in the rotation position of the embedding layer."
        
    if rotation_pos in ["post"]:
        assert w.shape[-1] == R_post.shape[0], "R should be same size as dim of output activation"
        w_dtype = w.dtype
        w_device = w.device
        w = (w.to(dtype=R_post.dtype) @ R_post.to(device=w_device)).to(dtype=w_dtype)
    module.weight.data = w


def rotate_model(model, R1, R2, R4):
    num_layers = model.config.num_hidden_layers
    rotation_map = build_rotation_map(model, R1, R2, R4)
    add_rotation_to_linear(model, rotation_map)
   