import torch
import torch.nn as nn


from train.prepare_model import build_rotation_map


def add_rotation_to_linear(
    model: nn.Module,
    rotation_map: dict = None,
    prefix: str = ""  # Record parent path
):
    """
    Merge all mergeable rotation matrices into all nn.Linear layers of the model

    Args:
        model (nn.Module): original model
        rotation_map (dict): key=module name, value=(R_pre, R_post, rotation_pos)
        prefix (str): Record the parent path in order to extract the rotation configuration from the rotation_map.
    Returns:
        nn.Module: Merged model completed
    """

    # Traverse the model module, recording the parent module and name.
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # If the submodules are nn.Linear and nn.embedding, replace them.
        if isinstance(module, nn.Linear):
            R_pre, R_post, rotation_pos = None, None, "none"
            if rotation_map and full_name in rotation_map:
                R_pre, R_post, rotation_pos = rotation_map[full_name]

            # Rotating weights
            rotate_linear(module, R_pre, R_post, rotation_pos)
        elif isinstance(module, nn.Embedding):
            R_pre, R_post, rotation_pos = None, None, "none"
            if rotation_map and full_name in rotation_map:
                R_pre, R_post, rotation_pos = rotation_map[full_name]

            # Rotating weights
            rotate_embedding(module, R_pre, R_post, rotation_pos)
        else:
            # Recursive processing of submodules
            add_rotation_to_linear(module, rotation_map, prefix=full_name)



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


def rotate_model(model, ptq_args, R4):
    assert ptq_args.optimized_rotation_path is not None, "We must give the optimized_rotation_path in the command line."
    R_path = ptq_args.optimized_rotation_path
    R1 = torch.load(R_path)["model.embed_tokens.R_post"].cuda().to(torch.float32)
  
    layers = [layer for layer in model.model.layers]
    R2 = []  
    for idx, layer in enumerate(layers):
        key = f"model.layers.{idx}.self_attn.v_proj.R_post"
        R_post = torch.load(R_path)[key].cuda().to(torch.float32)
        R2.append(R_post)

    num_layers = model.config.num_hidden_layers
    rotation_map = build_rotation_map(num_layers, R1, R2, R4)
    add_rotation_to_linear(model, rotation_map)
   