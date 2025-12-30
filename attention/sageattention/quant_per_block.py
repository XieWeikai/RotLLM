import torch
import torch.nn.functional as F

def quant_per_block(input, config, BLKQ=128, BLKK=64):
    """
    input: [B, H, L, C]
    returns: scale: [B, H, num_blocks, 1]
    """
    input = input.transpose(1, 2)

    B, H, L, C = input.shape
    num_blocks = (L + BLKQ - 1) // BLKQ
    pad_len = num_blocks * BLKQ - L

    # --- padding ---
    if pad_len > 0:
        x = F.pad(input, (0, 0, 0, pad_len))
    else:
        x = input

    x = x.float()  

    # [B, H, num_blocks, BLK, C]
    x_blocks = x.view(B, H, num_blocks, BLKQ, C)

    if config.is_symmetric:
        min_val = -(2 ** (config.num_bits - 1))
        max_val =  (2 ** (config.num_bits - 1)) - 1
    else:
        min_val = 0
        max_val = (2 ** config.num_bits) - 1


    # --- scale 计算方式 ---
    if config.init_type == "maxmin":
        # max abs
        xmax = x_blocks.amax(dim=(3, 4), keepdim=True)
        xmin = x_blocks.amin(dim=(3, 4), keepdim=True)
    elif config.init_type == "mean":
        # mean ± 3sigma → 转成 max(abs)
        mean = x_blocks.mean(dim=(3, 4), keepdim=True)
        std = x_blocks.std(dim=(3, 4), keepdim=True)
        xmax = mean + 3 * std
        xmin = mean - 3 * std
    else:
        raise ValueError(f"Unknown scale method: {config.init_type}")


    if config.is_symmetric:
        xmax = torch.maximum(torch.abs(xmin), torch.abs(xmax)).clamp(min=1e-5)
        scale = xmax / max_val
        zero_point = None
    else:
        scale = (xmax - xmin).clamp(min=1e-5) / (max_val - min_val)
        zero_point = torch.round(min_val - xmin / scale)
        
    scale = scale.to(torch.float32)
    zero_point = zero_point.to(torch.float32) if zero_point is not None else None

    # scale: [B, H, num_blocks, 1]
    scale = scale.view(B, H, num_blocks, 1)
    zero_point = zero_point.view(B, H, num_blocks, 1) if zero_point is not None else None

    scale = scale.transpose(1, 2)
    zero_point = zero_point.transpose(1, 2) if zero_point is not None else None

    return scale, zero_point 